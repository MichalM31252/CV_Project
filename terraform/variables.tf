variable "project_id" {
  description = "GCP project id that owns and is billed for these resources"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, BigQuery and Cloud Storage. Keep them aligned: BigQuery refuses cross-region reads, and co-location avoids egress charges."
  type        = string
  default     = "europe-west1"
}

variable "name_prefix" {
  description = "Prefix applied to bucket, repository and service-account names"
  type        = string
  default     = "credit-risk"
}

variable "environment" {
  description = "Deployment environment. 'prod' turns on BigQuery table deletion protection."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "bq_dataset" {
  description = "BigQuery dataset name"
  type        = string
  default     = "credit_risk"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "credit-risk-api"
}

variable "container_image" {
  description = "Fully-qualified image URI, e.g. europe-west1-docker.pkg.dev/PROJECT/credit-risk/api:v1"
  type        = string
}

# --- scaling and sizing ------------------------------------------------------

variable "min_instances" {
  description = "Minimum Cloud Run instances. 0 scales to zero (free when idle) at the cost of cold starts; set to 1 to eliminate them."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum Cloud Run instances. Caps cost during a traffic spike."
  type        = number
  default     = 4
}

variable "cpu" {
  description = "vCPU per instance"
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Memory per instance. 1Gi comfortably holds the model, DuckDB and the Python runtime; below ~512Mi the container risks OOM during model load."
  type        = string
  default     = "1Gi"
}

variable "concurrency" {
  description = "Concurrent requests per instance"
  type        = number
  default     = 40
}

# --- access and safety -------------------------------------------------------

variable "allow_public_access" {
  description = "Grant roles/run.invoker to allUsers. Convenient for a portfolio demo; disable for anything handling real data."
  type        = bool
  default     = true
}

variable "allow_bucket_destroy" {
  description = "Permit `terraform destroy` to delete non-empty buckets"
  type        = bool
  default     = false
}

variable "allow_dataset_destroy" {
  description = "Permit `terraform destroy` to delete a dataset containing tables"
  type        = bool
  default     = false
}

variable "enable_alerts" {
  description = "Create Cloud Monitoring alert policies"
  type        = bool
  default     = true
}

variable "error_rate_threshold" {
  description = "5xx responses per second that trigger the error-rate alert"
  type        = number
  default     = 0.1
}
