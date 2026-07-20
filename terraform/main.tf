# ---------------------------------------------------------------------------
# Infrastructure for the credit default risk pipeline.
#
# Everything the project needs on GCP is declared here so the environment can be
# rebuilt from scratch and reviewed in code, rather than assembled by clicking
# through the console and forgotten.
#
#   terraform init
#   terraform plan  -var-file=terraform.tfvars
#   terraform apply -var-file=terraform.tfvars
#
# Cost note: as declared this sits inside the GCP free tier for a portfolio-scale
# workload. Cloud Run scales to zero (no idle charge), BigQuery gives 1 TB of
# free queries per month against a dataset measured in megabytes, and the two
# buckets hold well under the 5 GB free allowance.
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# APIs must be enabled before any resource that depends on them. Terraform infers
# ordering from the explicit depends_on entries further down.
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "cloudscheduler.googleapis.com",
  ])
  service = each.key

  # Leave APIs enabled on destroy: other things in the project may depend on
  # them, and disabling is disruptive and slow to reverse.
  disable_on_destroy = false
}

# --------------------------- storage ---------------------------------------

resource "google_storage_bucket" "raw" {
  name     = "${var.project_id}-${var.name_prefix}-raw"
  location = var.region

  # Prevents a stray object ACL from making ingested data public.
  uniform_bucket_level_access = true
  force_destroy               = var.allow_bucket_destroy

  versioning {
    enabled = true
  }

  # Raw landings are immutable audit records, but they need not be kept forever
  # at standard-storage prices.
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "artifacts" {
  name     = "${var.project_id}-${var.name_prefix}-artifacts"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = var.allow_bucket_destroy

  # Model artifacts are versioned by the registry under versions/<id>/, but
  # object versioning additionally protects the mutable latest/ pointer.
  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required]
}

# --------------------------- bigquery --------------------------------------

resource "google_bigquery_dataset" "credit_risk" {
  dataset_id  = var.bq_dataset
  location    = var.region
  description = "Credit default risk - raw, cleaned, feature and monitoring tables"

  # Guards against `terraform destroy` silently deleting populated tables.
  delete_contents_on_destroy = var.allow_dataset_destroy

  labels = {
    project     = var.name_prefix
    managed_by  = "terraform"
    environment = var.environment
  }

  depends_on = [google_project_service.required]
}

# Prediction log schema is declared rather than autodetected: this table is
# written continuously by the API, and an inferred schema can drift between
# loads (an all-null batch changing a column's type, for instance).
resource "google_bigquery_table" "prediction_log" {
  dataset_id          = google_bigquery_dataset.credit_risk.dataset_id
  table_id            = "prediction_log"
  deletion_protection = var.environment == "prod"

  # Partitioning keeps monitoring queries cheap: a drift check reads recent days
  # rather than scanning the full history.
  time_partitioning {
    type  = "DAY"
    field = "predicted_at"
  }
  clustering = ["model_version", "decision"]

  schema = jsonencode([
    { name = "predicted_at", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "client_id", type = "INTEGER", mode = "REQUIRED" },
    { name = "model_version", type = "STRING", mode = "REQUIRED" },
    { name = "default_probability", type = "FLOAT", mode = "REQUIRED" },
    { name = "decision", type = "STRING", mode = "REQUIRED" },
    { name = "risk_band", type = "STRING", mode = "NULLABLE" },
    { name = "threshold", type = "FLOAT", mode = "NULLABLE" },
    { name = "limit_bal", type = "FLOAT", mode = "NULLABLE" },
    { name = "age", type = "INTEGER", mode = "NULLABLE" },
    { name = "pay_status_1", type = "INTEGER", mode = "NULLABLE" },
    { name = "bill_amt_1", type = "FLOAT", mode = "NULLABLE" },
    { name = "pay_amt_1", type = "FLOAT", mode = "NULLABLE" },
  ])
}

resource "google_bigquery_table" "drift_metrics" {
  dataset_id          = google_bigquery_dataset.credit_risk.dataset_id
  table_id            = "drift_metrics"
  deletion_protection = var.environment == "prod"

  time_partitioning {
    type  = "DAY"
    field = "computed_at"
  }

  schema = jsonencode([
    { name = "computed_at", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "feature", type = "STRING", mode = "REQUIRED" },
    { name = "psi", type = "FLOAT", mode = "NULLABLE" },
    { name = "status", type = "STRING", mode = "NULLABLE" },
    { name = "baseline_mean", type = "FLOAT", mode = "NULLABLE" },
    { name = "current_mean", type = "FLOAT", mode = "NULLABLE" },
    { name = "baseline_n", type = "INTEGER", mode = "NULLABLE" },
    { name = "current_n", type = "INTEGER", mode = "NULLABLE" },
  ])
}

# --------------------------- artifact registry ------------------------------

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = var.name_prefix
  format        = "DOCKER"
  description   = "Container images for the credit risk API"

  depends_on = [google_project_service.required]
}

# --------------------------- service account --------------------------------

# Dedicated identity for the API. The default Compute Engine service account is
# broadly privileged; a purpose-built account keeps the grants below meaningful.
resource "google_service_account" "api" {
  account_id   = "${var.name_prefix}-api"
  display_name = "Credit Risk API (Cloud Run)"
  description  = "Runtime identity for the prediction service"
}

# Least privilege. The API writes predictions and reads model artifacts, so it
# gets exactly those rights and nothing more - notably not bigquery.admin, and
# not project-level storage access.
resource "google_bigquery_dataset_iam_member" "api_data_editor" {
  dataset_id = google_bigquery_dataset.credit_risk.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.api.email}"
}

# Running a query bills the project, which is a project-level permission and
# cannot be granted on the dataset alone.
resource "google_project_iam_member" "api_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_artifact_reader" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# --------------------------- cloud run --------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name     = var.service_name
  location = var.region

  # Route only through the load balancer / public URL, not internal-only.
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email

    # Scale to zero when idle: this is what keeps a portfolio deployment free.
    # max_instance_count caps the blast radius of a traffic spike or a runaway
    # client, both of which otherwise translate directly into cost.
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
        # CPU is throttled outside requests. Model inference is bursty and
        # request-scoped, so paying for always-allocated CPU is waste.
        cpu_idle = true
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "CR__BACKEND"
        value = "gcp"
      }
      env {
        name  = "CR__GCP__PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "CR__GCP__BQ_DATASET"
        value = google_bigquery_dataset.credit_risk.dataset_id
      }
      env {
        name  = "CR__GCP__ARTIFACT_BUCKET"
        value = google_storage_bucket.artifacts.name
      }
      env {
        name  = "CR__GCP__RAW_BUCKET"
        value = google_storage_bucket.raw.name
      }
      env {
        name  = "CR__GCP__REGION"
        value = var.region
      }

      # Liveness: restart a wedged container. Points at /health, which never
      # touches the model, so a slow model cannot trigger a restart loop.
      liveness_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 30
        period_seconds        = 30
        timeout_seconds       = 5
        failure_threshold     = 3
      }

      # Readiness: hold traffic until the model is actually loaded. Generous
      # failure_threshold because model load plus warm-up takes a few seconds.
      startup_probe {
        http_get {
          path = "/ready"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        timeout_seconds       = 5
        failure_threshold     = 12
      }
    }

    # Model inference is CPU-bound; beyond this, concurrent requests contend for
    # the same cores and tail latency degrades faster than throughput improves.
    max_instance_request_concurrency = var.concurrency
    timeout                          = "60s"
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.required,
    google_bigquery_dataset_iam_member.api_data_editor,
  ]
}

# Public access. For a real lender this would be removed and callers would
# authenticate with an ID token; it is enabled here so the portfolio endpoint is
# reachable without credentials.
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_public_access ? 1 : 0
  name     = google_cloud_run_v2_service.api.name
  location = google_cloud_run_v2_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --------------------------- monitoring -------------------------------------

# Drift checks log at ERROR severity with a "drift alert" message; this turns
# that log line into a real alert.
resource "google_monitoring_alert_policy" "drift_alert" {
  count        = var.enable_alerts ? 1 : 0
  display_name = "Credit risk - feature drift detected"
  combiner     = "OR"

  conditions {
    display_name = "PSI above alert threshold"
    condition_matched_log {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        resource.labels.service_name="${var.service_name}"
        severity>=ERROR
        jsonPayload.message="drift alert"
      EOT
    }
  }

  # Log-based conditions require a notification rate limit.
  alert_strategy {
    notification_rate_limit {
      period = "3600s"
    }
    auto_close = "604800s"
  }
}

resource "google_monitoring_alert_policy" "error_rate" {
  count        = var.enable_alerts ? 1 : 0
  display_name = "Credit risk - elevated 5xx rate"
  combiner     = "OR"

  conditions {
    display_name = "5xx responses over threshold"
    condition_threshold {
      filter = <<-EOT
        resource.type="cloud_run_revision"
        AND resource.labels.service_name="${var.service_name}"
        AND metric.type="run.googleapis.com/request_count"
        AND metric.labels.response_code_class="5xx"
      EOT
      comparison      = "COMPARISON_GT"
      threshold_value = var.error_rate_threshold
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  alert_strategy {
    auto_close = "86400s"
  }
}
