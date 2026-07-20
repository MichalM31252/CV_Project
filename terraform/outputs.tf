output "service_url" {
  description = "Public URL of the prediction API"
  value       = google_cloud_run_v2_service.api.uri
}

output "raw_bucket" {
  description = "Bucket holding raw ingested data"
  value       = google_storage_bucket.raw.name
}

output "artifact_bucket" {
  description = "Bucket holding versioned model artifacts"
  value       = google_storage_bucket.artifacts.name
}

output "bq_dataset" {
  description = "BigQuery dataset id"
  value       = google_bigquery_dataset.credit_risk.dataset_id
}

output "artifact_registry" {
  description = "Docker repository URI for pushing images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.containers.repository_id}"
}

output "service_account" {
  description = "Runtime service account for the API"
  value       = google_service_account.api.email
}

# Convenience: the exact environment variables that point a local run at the
# infrastructure just created.
output "local_env" {
  description = "Environment variables to run the pipeline against these resources"
  value       = <<-EOT
    export CR__BACKEND=gcp
    export CR__GCP__PROJECT_ID=${var.project_id}
    export CR__GCP__REGION=${var.region}
    export CR__GCP__BQ_DATASET=${google_bigquery_dataset.credit_risk.dataset_id}
    export CR__GCP__RAW_BUCKET=${google_storage_bucket.raw.name}
    export CR__GCP__ARTIFACT_BUCKET=${google_storage_bucket.artifacts.name}
  EOT
}
