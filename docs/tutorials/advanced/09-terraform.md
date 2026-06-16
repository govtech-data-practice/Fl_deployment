# Tutorial 9: Infrastructure with Terraform

**Time:** 30 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 8](08-distributed-deployment.md), Terraform installed

## What You'll Learn

- Provision FL infrastructure on AWS with Terraform
- Configure VPC, security groups, and EC2 instances
- Enable TLS and GPU support
- Manage infrastructure lifecycle

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

Type `yes` to confirm. Note the outputs:
```
superlink_public_ip = "54.x.x.x"
supernode_public_ips = ["10.0.1.10", "10.0.1.11", "10.0.1.12"]
```

## Step 4: Configure cluster.env

Use the Terraform outputs to populate your cluster config:

```bash
cd ../..
cat > cluster.env << EOF
FL_SERVER_HOST=$(cd deploy/terraform && terraform output -raw superlink_public_ip)
FL_CLIENT_HOSTS="$(cd deploy/terraform && terraform output -json supernode_public_ips | python3 -c 'import sys,json;print(" ".join(json.load(sys.stdin)))')"
FL_NUM_CLIENTS=3
FL_SSH_KEY=~/.ssh/your-keypair.pem
EOF
```

## Step 5: Deploy and Run

```bash
# Generate certs, distribute image, start services
./deploy/distributed/deploy.sh up

# Run a smoke test on the real infrastructure
python run_ec2.py fraud --synthetic

# Verify everything is healthy
./deploy/health_check.sh
```

## Step 6: Key Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `region` | `ap-southeast-1` | AWS region |
| `num_supernodes` | `2` | Number of client instances |
| `superlink_instance_type` | `t3.large` | Coordinator instance type |
| `supernode_instance_type` | `g4dn.xlarge` | Client instance type (GPU) |
| `supernode_instance_type_cpu` | `t3.xlarge` | CPU fallback |
| `use_gpu` | `false` | Enable GPU instances |
| `key_name` | (required) | SSH key pair |
| `data_s3_bucket` | (required) | S3 bucket for data |
| `enable_tls` | `true` | Enable mTLS |
| `allowed_ssh_cidrs` | `[]` | SSH access CIDR blocks |

## Step 7: Tear Down

```bash
cd deploy/terraform/
terraform destroy
```

Type `yes` to confirm. All AWS resources are removed.

## What You Learned

- Terraform automates the full infrastructure lifecycle
- One `terraform apply` provisions the complete FL cluster
- Outputs feed directly into `cluster.env` for deployment scripts

## Next Steps

- [Tutorial 10: Vertical FL & PSI](10-vertical-fl.md) — train across vertically partitioned data
