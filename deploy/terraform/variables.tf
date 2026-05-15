variable "region" {
  description = "AWS region"
  default     = "ap-southeast-1"
}

variable "project_name" {
  description = "Project name (used for resource naming)"
  default     = "healthcare-fl"
}

variable "num_supernodes" {
  description = "Number of FL SuperNode (client) instances"
  default     = 2
}

variable "superlink_instance_type" {
  description = "EC2 instance type for SuperLink (server)"
  default     = "t3.large" # 2 vCPU, 8GB — aggregation is CPU-bound
}

variable "supernode_instance_type" {
  description = "EC2 instance type for SuperNodes (clients)"
  default     = "g4dn.xlarge" # 1 GPU, 4 vCPU, 16GB — for training
}

variable "supernode_instance_type_cpu" {
  description = "CPU-only fallback for SuperNodes"
  default     = "t3.xlarge" # 4 vCPU, 16GB
}

variable "use_gpu" {
  description = "Use GPU instances for SuperNodes"
  type        = bool
  default     = false
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "data_s3_bucket" {
  description = "S3 bucket containing training data"
  type        = string
}

variable "data_s3_prefix" {
  description = "S3 prefix for training data"
  default     = "flower_data/"
}

variable "experiment" {
  description = "Experiment name (e.g. IID, FedProx_Mu0.1_Alpha_0.5)"
  default     = "IID"
}

variable "num_rounds" {
  description = "Number of FL rounds"
  default     = 100
}

variable "task" {
  description = "Task: sepsis or chest_xray"
  default     = "sepsis"
  validation {
    condition     = contains(["sepsis", "chest_xray"], var.task)
    error_message = "Must be 'sepsis' or 'chest_xray'."
  }
}

variable "enable_tls" {
  description = "Enable TLS between SuperLink and SuperNodes"
  type        = bool
  default     = true
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH"
  type        = list(string)
  default     = [] # Empty = no SSH access. Set to ["YOUR_IP/32"]
}
