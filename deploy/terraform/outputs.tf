output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.fl.repository_url
}

output "superlink_public_ip" {
  description = "SuperLink public IP (for SSH / flwr run)"
  value       = aws_instance.superlink.public_ip
}

output "superlink_private_ip" {
  description = "SuperLink private IP (used by SuperNodes)"
  value       = aws_instance.superlink.private_ip
}

output "supernode_private_ips" {
  description = "SuperNode private IPs"
  value       = aws_instance.supernode[*].private_ip
}

output "push_command" {
  description = "Command to push Docker image"
  value       = <<-EOT
    aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${aws_ecr_repository.fl.repository_url}
    docker tag healthcare-fl:latest ${aws_ecr_repository.fl.repository_url}:latest
    docker push ${aws_ecr_repository.fl.repository_url}:latest
  EOT
}

output "run_experiment_command" {
  description = "SSH to SuperLink and run an experiment"
  value       = <<-EOT
    ssh -i ~/.ssh/${var.key_name}.pem ec2-user@${aws_instance.superlink.public_ip} \
      "docker exec superlink flower-superlink --help"
  EOT
}
