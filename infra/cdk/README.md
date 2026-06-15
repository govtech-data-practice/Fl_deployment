# AWS CDK Stacks

Placeholder for AWS CDK infrastructure-as-code stacks.

The current infrastructure provisioning uses Terraform (see `infra/terraform/` or `deploy/terraform/`).
CDK stacks will be added as an alternative for teams that prefer CDK.

## Terraform (current)

```bash
cd infra/terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform plan
terraform apply
```
