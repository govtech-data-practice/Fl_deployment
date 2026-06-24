# Tutorial 9: Infrastructure with Terraform

**Time:** 30 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 8](08-distributed-deployment.md), Terraform installed

## What You'll Learn

- Provision FL infrastructure on AWS with Terraform
- Configure VPC, security groups, and EC2 instances
- Deploy containers to provisioned infrastructure

## Step 1: Configure Terraform Variables

```bash
cd deploy/terraform/
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
key_name        = "your-ssh-keypair"
data_s3_bucket  = "your-fl-data-bucket"
num_supernodes  = 3
use_gpu         = true
enable_tls      = true
allowed_ssh_cidrs = ["YOUR_IP/32"]
```

> **Prerequisite:** Create the S3 bucket specified in `data_s3_bucket` before running `terraform apply`. The bucket must be in the same AWS region as your deployment.

> **Cost warning:** Running `terraform apply` will provision AWS resources (EC2 instances, VPC, etc.) that incur ongoing costs. Remember to run `terraform destroy` (Step 6) when you are done to avoid unexpected charges.

## Step 2: Initialise and Plan

```bash
terraform init
terraform plan
```

Review the plan. Terraform will create:
- VPC with public and private subnets
- Security groups (gRPC port 9092, SSH)
- 1 SuperLink (coordinator) EC2 instance
- N SuperNode (client) EC2 instances
- S3 access for data and model artefacts

## Step 3: Apply

```bash
terraform apply
```

Note the outputs:
```
superlink_public_ip = "54.x.x.x"
supernode_public_ips = ["54.x.x.x", "54.x.x.y", "54.x.x.z"]
```

## Step 4: Deploy Containers

Use the Terraform outputs to deploy Docker containers:

```bash
# On coordinator
ssh ec2-user@<superlink_ip> "docker run -d --name fl-coordinator \
  -p 9092:9092 healthcare-fl:v1.0.0 \
  python3 runners/run_ec2.py fraud --distributed"

# On each client
ssh ec2-user@<supernode_ip> "docker run -d --name fl-client \
  healthcare-fl:v1.0.0 \
  python3 runners/run_client.py --server <coordinator_private_ip>:9092"
```

## Step 5: Key Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `region` | `ap-southeast-1` | AWS region |
| `num_supernodes` | `2` | Number of client instances |
| `superlink_instance_type` | `t3.large` | Coordinator instance type |
| `supernode_instance_type` | `g4dn.xlarge` | Client instance type (GPU) |
| `use_gpu` | `false` | Enable GPU instances |
| `key_name` | (required) | SSH key pair |
| `data_s3_bucket` | (required) | S3 bucket for data |
| `enable_tls` | `true` | Enable mTLS |

## Step 6: Tear Down

```bash
terraform destroy
```

## Next Steps

- [Tutorial 10: Vertical FL & PSA](10-vertical-fl.md) — train across vertically partitioned data
