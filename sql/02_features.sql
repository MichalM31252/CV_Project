-- ---------------------------------------------------------------------------
-- Stage: clean -> features
--
-- Same portability contract as 01_clean.sql: BigQuery standard SQL and DuckDB,
-- unchanged. No SAFE_DIVIDE (use `x / NULLIF(y, 0)`), no FARM_FINGERPRINT, no
-- FLOAT64/DOUBLE casts (use `* 1.0`), and no references to select-list aliases
-- within the same SELECT - BigQuery forbids that, so the work is built up
-- through CTEs instead.
--
-- Month indexing throughout: 1 = most recent (September 2005) .. 6 = oldest
-- (April 2005).
--
-- TEMPORAL ALIGNMENT - the one genuinely easy thing to get wrong here.
-- `pay_amt_i` is the payment made during month i, and it settles the statement
-- issued the month before, i.e. `bill_amt_(i+1)`. Dividing pay_amt_i by
-- bill_amt_i would compare a payment against a bill that had not been issued
-- when the payment was made - a subtle lookahead that inflates offline metrics
-- and then evaporates in production. Ratios below therefore run i = 1..5, since
-- month 6 has no preceding statement in the window.
--
-- Feature families:
--   utilisation  - balance against credit limit, its level, peak and trend
--   payment      - how much of the outstanding statement the client actually pays
--   delinquency  - depth, breadth and direction of arrears
--   dynamics     - volatility and growth of the balance over the window
--
-- All of it is computed in the warehouse rather than in pandas: the same SQL
-- serves 30k rows here and would serve 30M without change, and the feature
-- definitions live in one auditable place shared by training and batch scoring.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE {features_table} AS
WITH per_month AS (
    SELECT
        client_id,
        limit_bal,
        sex,
        education,
        marriage,
        age,
        default_next_month,

        pay_status_1, pay_status_2, pay_status_3,
        pay_status_4, pay_status_5, pay_status_6,

        bill_amt_1, bill_amt_2, bill_amt_3,
        bill_amt_4, bill_amt_5, bill_amt_6,

        pay_amt_1, pay_amt_2, pay_amt_3,
        pay_amt_4, pay_amt_5, pay_amt_6,

        -- Credit utilisation per month: what fraction of the limit is drawn.
        bill_amt_1 * 1.0 / NULLIF(limit_bal, 0) AS utilization_1,
        bill_amt_2 * 1.0 / NULLIF(limit_bal, 0) AS utilization_2,
        bill_amt_3 * 1.0 / NULLIF(limit_bal, 0) AS utilization_3,
        bill_amt_4 * 1.0 / NULLIF(limit_bal, 0) AS utilization_4,
        bill_amt_5 * 1.0 / NULLIF(limit_bal, 0) AS utilization_5,
        bill_amt_6 * 1.0 / NULLIF(limit_bal, 0) AS utilization_6,

        -- Payment coverage: payment in month i over the statement it settles.
        pay_amt_1 * 1.0 / NULLIF(bill_amt_2, 0) AS payment_ratio_1,
        pay_amt_2 * 1.0 / NULLIF(bill_amt_3, 0) AS payment_ratio_2,
        pay_amt_3 * 1.0 / NULLIF(bill_amt_4, 0) AS payment_ratio_3,
        pay_amt_4 * 1.0 / NULLIF(bill_amt_5, 0) AS payment_ratio_4,
        pay_amt_5 * 1.0 / NULLIF(bill_amt_6, 0) AS payment_ratio_5
    FROM {clean_table}
),

aggregated AS (
    SELECT
        p.*,

        -- ---- utilisation ------------------------------------------------
        (utilization_1 + utilization_2 + utilization_3
         + utilization_4 + utilization_5 + utilization_6) / 6.0 AS avg_utilization,

        GREATEST(utilization_1, utilization_2, utilization_3,
                 utilization_4, utilization_5, utilization_6) AS max_utilization,

        -- Direction of travel: positive means the balance grew across the window
        -- relative to the limit. A rising trend at a given level is worse than a
        -- flat one, which a point-in-time utilisation figure cannot express.
        utilization_1 - utilization_6 AS utilization_trend,

        -- Remaining headroom on the most recent statement. Negative = over limit.
        (limit_bal - bill_amt_1) * 1.0 / NULLIF(limit_bal, 0) AS credit_headroom,

        -- ---- payment behaviour -------------------------------------------
        (COALESCE(payment_ratio_1, 0) + COALESCE(payment_ratio_2, 0)
         + COALESCE(payment_ratio_3, 0) + COALESCE(payment_ratio_4, 0)
         + COALESCE(payment_ratio_5, 0)) / 5.0 AS avg_payment_ratio,

        -- Whole-window coverage. Less noisy than the mean of monthly ratios when
        -- individual statements are near zero.
        --
        -- NULL when no statement was issued across months 2-6, i.e. a dormant
        -- account. That NULL is left in place rather than defaulted to 0: "paid
        -- nothing against a bill" and "had no bill to pay" are different states,
        -- and collapsing them would be wrong. The companion flag below carries
        -- the distinction explicitly for models that cannot consume NULL.
        (pay_amt_1 + pay_amt_2 + pay_amt_3 + pay_amt_4 + pay_amt_5 + pay_amt_6) * 1.0
            / NULLIF(bill_amt_2 + bill_amt_3 + bill_amt_4
                     + bill_amt_5 + bill_amt_6, 0) AS overall_payment_ratio,

        -- Dormancy indicator. Empirically these accounts default at ~31.6%
        -- against a ~22.1% base rate, so the *absence* of billing history is
        -- itself a risk signal and must survive imputation.
        CASE
            WHEN (bill_amt_2 + bill_amt_3 + bill_amt_4
                  + bill_amt_5 + bill_amt_6) <> 0 THEN 1
            ELSE 0
        END AS has_billing_history,

        (CASE WHEN pay_amt_1 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_amt_2 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_amt_3 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_amt_4 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_amt_5 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_amt_6 = 0 THEN 1 ELSE 0 END) AS months_zero_payment,

        (pay_amt_1 + pay_amt_2 + pay_amt_3
         + pay_amt_4 + pay_amt_5 + pay_amt_6) AS total_paid_6m,

        -- ---- delinquency ---------------------------------------------------
        -- Most recent repayment status. Single strongest signal in this dataset:
        -- arrears now are the best predictor of arrears next month.
        pay_status_1 AS recent_delinquency,

        GREATEST(pay_status_1, pay_status_2, pay_status_3,
                 pay_status_4, pay_status_5, pay_status_6) AS max_delinquency,

        (CASE WHEN pay_status_1 >= 1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_2 >= 1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_3 >= 1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_4 >= 1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_5 >= 1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_6 >= 1 THEN 1 ELSE 0 END) AS months_delinquent,

        -- Rising = arrears deepening towards the observation date.
        pay_status_1 - pay_status_6 AS delinquency_trend,

        -- Revolving (status 0) is distinct from paying duly (-1): the client is
        -- servicing the minimum and carrying debt, a materially higher risk state.
        (CASE WHEN pay_status_1 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_2 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_3 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_4 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_5 = 0 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_6 = 0 THEN 1 ELSE 0 END) AS months_revolving,

        (CASE WHEN pay_status_1 = -1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_2 = -1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_3 = -1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_4 = -1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_5 = -1 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_6 = -1 THEN 1 ELSE 0 END) AS months_paid_duly,

        -- Status -2 = no consumption. A dormant card looks like a perfect payer
        -- on every ratio above, so it needs its own feature to stay separable.
        (CASE WHEN pay_status_1 = -2 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_2 = -2 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_3 = -2 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_4 = -2 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_5 = -2 THEN 1 ELSE 0 END
         + CASE WHEN pay_status_6 = -2 THEN 1 ELSE 0 END) AS months_no_consumption,

        -- ---- balance dynamics ----------------------------------------------
        (CASE WHEN bill_amt_1 > limit_bal THEN 1 ELSE 0 END
         + CASE WHEN bill_amt_2 > limit_bal THEN 1 ELSE 0 END
         + CASE WHEN bill_amt_3 > limit_bal THEN 1 ELSE 0 END
         + CASE WHEN bill_amt_4 > limit_bal THEN 1 ELSE 0 END
         + CASE WHEN bill_amt_5 > limit_bal THEN 1 ELSE 0 END
         + CASE WHEN bill_amt_6 > limit_bal THEN 1 ELSE 0 END) AS months_over_limit,

        (bill_amt_1 + bill_amt_2 + bill_amt_3
         + bill_amt_4 + bill_amt_5 + bill_amt_6) / 6.0 AS avg_bill_amt,

        -- Second moment, kept separate so volatility can be derived downstream
        -- without a self-join or a non-portable STDDEV over pivoted columns.
        (bill_amt_1 * 1.0 * bill_amt_1 + bill_amt_2 * 1.0 * bill_amt_2
         + bill_amt_3 * 1.0 * bill_amt_3 + bill_amt_4 * 1.0 * bill_amt_4
         + bill_amt_5 * 1.0 * bill_amt_5 + bill_amt_6 * 1.0 * bill_amt_6) / 6.0
            AS avg_bill_amt_sq
    FROM per_month AS p
),

enriched AS (
    SELECT
        a.*,

        -- Population standard deviation of the balance across the window,
        -- via E[x^2] - E[x]^2. GREATEST(..., 0) guards the tiny negative values
        -- floating-point cancellation can produce before SQRT.
        SQRT(GREATEST(avg_bill_amt_sq - avg_bill_amt * avg_bill_amt, 0)) AS bill_volatility,

        CASE WHEN months_over_limit > 0 THEN 1 ELSE 0 END AS ever_over_limit,
        CASE WHEN months_delinquent > 0 THEN 1 ELSE 0 END AS ever_delinquent,

        -- Debt burden relative to the granted limit.
        (bill_amt_1 - pay_amt_1) * 1.0 / NULLIF(limit_bal, 0) AS net_balance_ratio
    FROM aggregated AS a
)

SELECT
    client_id,

    -- demographics
    limit_bal,
    sex,
    education,
    marriage,
    age,

    -- raw repayment status (tree models exploit the ordinal scale directly)
    pay_status_1, pay_status_2, pay_status_3,
    pay_status_4, pay_status_5, pay_status_6,

    -- utilisation
    avg_utilization,
    max_utilization,
    utilization_trend,
    credit_headroom,

    -- payment behaviour
    avg_payment_ratio,
    overall_payment_ratio,
    has_billing_history,
    months_zero_payment,
    total_paid_6m,

    -- delinquency
    recent_delinquency,
    max_delinquency,
    months_delinquent,
    delinquency_trend,
    months_revolving,
    months_paid_duly,
    months_no_consumption,
    ever_delinquent,

    -- balance dynamics
    avg_bill_amt,
    bill_volatility,
    months_over_limit,
    ever_over_limit,
    net_balance_ratio,

    default_next_month,

    -- Deterministic split on client id.
    --
    -- Pure integer arithmetic rather than a hash function, because BigQuery's
    -- FARM_FINGERPRINT and DuckDB's HASH return different values - the split
    -- would silently differ between the two backends and a model validated
    -- locally would be evaluated on different rows in production.
    --
    -- The multiplier is Knuth's; coprime with the modulus, so residues are
    -- distributed evenly. Assignment depends only on client_id, so it is stable
    -- across reruns and as new clients arrive: a client never migrates from test
    -- into train and leaks.
    CASE
        WHEN MOD(client_id * {hash_multiplier}, {hash_modulus}) <= {train_max} THEN 'train'
        WHEN MOD(client_id * {hash_multiplier}, {hash_modulus}) <= {valid_max} THEN 'valid'
        ELSE 'test'
    END AS split_name
FROM enriched;
