"""Stage 4b - PyTorch neural network as a second model implementation.

Why PyTorch and not TensorFlow: TensorFlow publishes no cp314 wheels at the time
of writing, so on Python 3.14 it cannot be installed at all. PyTorch 2.12 works,
and nothing in this design depends on which framework is used.

The network is deliberately wrapped in a scikit-learn-compatible estimator
(``fit``/``predict_proba``/``get_params``). That single decision means the neural
model flows through *exactly* the same calibration, threshold selection,
evaluation, registry and serving code as the gradient booster - no parallel
serving path, no second set of metric definitions, and a genuinely
like-for-like comparison. Swapping which one is served is one config value.

On tabular data of this size a well-tuned gradient booster is the strong
favourite, and the honest expectation is that it wins. The value of building the
network anyway is that the comparison is measured rather than asserted.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import Settings, TorchSettings
from . import evaluate as ev
from .registry import save_model
from .train_sklearn import _build_preprocessor

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Feed-forward network for tabular binary classification.

    BatchNorm after each linear layer keeps activations well-scaled given the
    wide dynamic range of the monetary features; dropout provides the
    regularisation that matters most on a dataset this small.
    """

    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        # Single logit; the sigmoid lives in BCEWithLogitsLoss for numerical
        # stability, and is applied explicitly at inference.
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TorchClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn estimator wrapping :class:`MLP`.

    Implementing the estimator protocol is what lets ``CalibratedClassifierCV``,
    ``FrozenEstimator`` and the shared evaluation code treat this exactly like any
    other classifier.
    """

    def __init__(
        self,
        n_features: int | None = None,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    # -- training ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> TorchClassifier:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = MLP(X.shape[1], list(self.hidden_dims), self.dropout).to(self.device_)

        # Re-weight the positive class by the negative/positive ratio so the
        # minority class is not drowned out by the loss.
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device_)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        use_early_stopping = X_valid is not None and y_valid is not None
        best_score = -np.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        self.history_: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimiser.zero_grad()
                loss = criterion(self.model_(batch_x), batch_y)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch_x)
            epoch_loss /= len(loader.dataset)

            if not use_early_stopping:
                self.history_.append({"epoch": epoch, "train_loss": epoch_loss})
                continue

            # Early stopping tracks validation PR-AUC rather than loss: PR-AUC is
            # the metric the model is selected on, and with a re-weighted loss the
            # two can move in opposite directions.
            valid_score = float(average_precision_score(y_valid, self._raw_proba(X_valid)))
            self.history_.append(
                {"epoch": epoch, "train_loss": epoch_loss, "valid_pr_auc": valid_score}
            )

            if valid_score > best_score:
                best_score = valid_score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(
                        "early stopping",
                        extra={"epoch": epoch, "best_valid_pr_auc": best_score},
                    )
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_valid_score_ = float(best_score) if use_early_stopping else float("nan")
        self.epochs_trained_ = len(self.history_)
        return self

    # -- inference ---------------------------------------------------------

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device_)
        with torch.no_grad():
            return torch.sigmoid(self.model_(tensor)).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = self._raw_proba(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._raw_proba(X) >= 0.5).astype(int)

    # -- serialisation -----------------------------------------------------
    #
    # Persist weights as a plain state_dict rather than pickling the nn.Module.
    # A pickled module embeds the class definition and is brittle across torch
    # and code versions; a state_dict plus the architecture parameters can always
    # be rebuilt, which is what a model that has to load inside a container
    # months later actually needs.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        state.pop("device_", None)
        if model is not None:
            state["_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state_dict = state.pop("_state_dict", None)
        self.__dict__.update(state)
        self.device_ = torch.device("cpu")  # inference target is CPU on Cloud Run
        if state_dict is not None:
            self.model_ = MLP(self.n_features_in_, list(self.hidden_dims), self.dropout)
            self.model_.load_state_dict(state_dict)
            self.model_.to(self.device_)
            self.model_.eval()


def train(
    settings: Settings,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Train, calibrate, threshold and register the PyTorch model."""
    target = settings.model.target
    cost = settings.model.cost
    cfg: TorchSettings = settings.model.torch

    # Neural networks need scaled inputs and cannot consume NaN, so this branch
    # uses the imputing/scaling preprocessor. Fitted on train only.
    preprocessor = _build_preprocessor(feature_names, scale=True, impute=True)
    x_train = preprocessor.fit_transform(train_df[feature_names])
    x_valid = preprocessor.transform(valid_df[feature_names])
    x_test = preprocessor.transform(test_df[feature_names])

    y_train = train_df[target].to_numpy()
    y_valid = valid_df[target].to_numpy()
    y_test = test_df[target].to_numpy()

    logger.info(
        "training pytorch model",
        extra={"features": x_train.shape[1], "rows": x_train.shape[0], "device": "cpu"},
    )

    classifier = TorchClassifier(
        hidden_dims=tuple(cfg.hidden_dims),
        dropout=cfg.dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        random_state=settings.model.random_state,
    )
    classifier.fit(x_train, y_train, X_valid=x_valid, y_valid=y_valid)

    # Same calibration and threshold procedure as the scikit-learn path, so the
    # comparison is like-for-like.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(classifier), method=settings.model.sklearn.calibration_method
    )
    calibrated.fit(x_valid, y_valid)

    valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    threshold, _ = ev.optimal_threshold(y_valid, valid_prob, cost)

    test_prob = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = ev.evaluate(y_test, test_prob, cost, threshold)
    valid_metrics = ev.evaluate(y_valid, valid_prob, cost, threshold)

    # The preprocessor is bundled with the calibrated model so serving applies
    # byte-identical transformations to those used in training.
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    servable = Pipeline([("preprocess", preprocessor), ("model", calibrated)])

    metadata = save_model(
        settings,
        servable,
        flavor="torch",
        feature_names=feature_names,
        threshold=threshold,
        metrics={"valid": valid_metrics, "test": test_metrics},
        extra={
            "architecture": {
                "hidden_dims": list(cfg.hidden_dims),
                "dropout": cfg.dropout,
                "optimiser": "AdamW",
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "loss": "BCEWithLogitsLoss(pos_weight=n_neg/n_pos)",
            },
            "epochs_trained": classifier.epochs_trained_,
            "best_valid_pr_auc": classifier.best_valid_score_,
            "calibration_method": settings.model.sklearn.calibration_method,
        },
    )

    return {
        "model": servable,
        "metadata": metadata,
        "test_metrics": test_metrics,
        "valid_metrics": valid_metrics,
        "history": classifier.history_,
        "test_prob": test_prob,
    }
