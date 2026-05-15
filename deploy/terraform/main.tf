terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
    tls = { source = "hashicorp/tls", version = "~> 4.0" }
  }
}

provider "aws" {
  region = var.region
}

locals {
  name           = var.project_name
  supernode_type = var.use_gpu ? var.supernode_instance_type : var.supernode_instance_type_cpu
  superlink_ami  = data.aws_ami.amazon_linux.id
  supernode_ami  = var.use_gpu ? data.aws_ami.deep_learning.id : data.aws_ami.amazon_linux.id
  data_dir       = var.task == "sepsis" ? "/data/flower_data" : "/data"
}

# ---------- AMIs ----------

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning AMI (Ubuntu 22.04) *"]
  }
}

# ---------- VPC ----------

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.region}a", "${var.region}b"]
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.10.0/24", "10.0.11.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = { Project = local.name }
}

# ---------- Security Groups ----------

resource "aws_security_group" "superlink" {
  name_prefix = "${local.name}-superlink-"
  vpc_id      = module.vpc.vpc_id

  # Fleet API (SuperNodes connect here)
  ingress {
    from_port       = 9092
    to_port         = 9092
    protocol        = "tcp"
    security_groups = [aws_security_group.supernode.id]
  }

  # ServerAppIo API
  ingress {
    from_port       = 9091
    to_port         = 9091
    protocol        = "tcp"
    security_groups = [aws_security_group.supernode.id]
  }

  # Control API
  ingress {
    from_port       = 9093
    to_port         = 9093
    protocol        = "tcp"
    security_groups = [aws_security_group.supernode.id]
  }

  # SSH (optional)
  dynamic "ingress" {
    for_each = length(var.allowed_ssh_cidrs) > 0 ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.allowed_ssh_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-superlink-sg" }
}

resource "aws_security_group" "supernode" {
  name_prefix = "${local.name}-supernode-"
  vpc_id      = module.vpc.vpc_id

  # SSH (optional)
  dynamic "ingress" {
    for_each = length(var.allowed_ssh_cidrs) > 0 ? [1] : []
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.allowed_ssh_cidrs
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-supernode-sg" }
}

# ---------- ECR ----------

resource "aws_ecr_repository" "fl" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration { scan_on_push = true }
}

# ---------- IAM ----------

resource "aws_iam_role" "ec2" {
  name = "${local.name}-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecr_s3" {
  name = "${local.name}-ecr-s3"
  role = aws_iam_role.ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.data_s3_bucket}",
          "arn:aws:s3:::${var.data_s3_bucket}/*",
        ]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${local.name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# ---------- TLS Certificates (self-signed) ----------

resource "tls_private_key" "ca" {
  count     = var.enable_tls ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "tls_self_signed_cert" "ca" {
  count           = var.enable_tls ? 1 : 0
  private_key_pem = tls_private_key.ca[0].private_key_pem

  subject { common_name = "${local.name}-ca" }

  is_ca_certificate     = true
  validity_period_hours = 8760 # 1 year
  allowed_uses          = ["cert_signing", "crl_signing"]
}

resource "tls_private_key" "server" {
  count     = var.enable_tls ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_cert_request" "server" {
  count           = var.enable_tls ? 1 : 0
  private_key_pem = tls_private_key.server[0].private_key_pem

  subject { common_name = "superlink.${local.name}.internal" }

  dns_names    = ["superlink", "localhost"]
  ip_addresses = ["127.0.0.1"]
}

resource "tls_locally_signed_cert" "server" {
  count              = var.enable_tls ? 1 : 0
  cert_request_pem   = tls_cert_request.server[0].cert_request_pem
  ca_private_key_pem = tls_private_key.ca[0].private_key_pem
  ca_cert_pem        = tls_self_signed_cert.ca[0].cert_pem

  validity_period_hours = 8760
  allowed_uses          = ["digital_signature", "key_encipherment", "server_auth"]
}

# Store certs in SSM Parameter Store
resource "aws_ssm_parameter" "ca_cert" {
  count = var.enable_tls ? 1 : 0
  name  = "/${local.name}/tls/ca-cert"
  type  = "SecureString"
  value = tls_self_signed_cert.ca[0].cert_pem
}

resource "aws_ssm_parameter" "server_cert" {
  count = var.enable_tls ? 1 : 0
  name  = "/${local.name}/tls/server-cert"
  type  = "SecureString"
  value = tls_locally_signed_cert.server[0].cert_pem
}

resource "aws_ssm_parameter" "server_key" {
  count = var.enable_tls ? 1 : 0
  name  = "/${local.name}/tls/server-key"
  type  = "SecureString"
  value = tls_private_key.server[0].private_key_pem
}

# Add SSM read permissions to IAM role
resource "aws_iam_role_policy" "ssm" {
  count = var.enable_tls ? 1 : 0
  name  = "${local.name}-ssm"
  role  = aws_iam_role.ec2.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameter", "ssm:GetParameters"]
      Resource = "arn:aws:ssm:${var.region}:*:parameter/${local.name}/tls/*"
    }]
  })
}

# ---------- SuperLink Instance ----------

resource "aws_instance" "superlink" {
  ami                    = local.superlink_ami
  instance_type          = var.superlink_instance_type
  key_name               = var.key_name
  subnet_id              = module.vpc.public_subnets[0]
  vpc_security_group_ids = [aws_security_group.superlink.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/userdata_superlink.sh.tpl", {
    region       = var.region
    ecr_repo     = aws_ecr_repository.fl.repository_url
    image_tag    = "latest"
    project_name = local.name
    enable_tls   = var.enable_tls
  })

  tags = { Name = "${local.name}-superlink" }
}

# ---------- SuperNode Instances ----------

resource "aws_instance" "supernode" {
  count = var.num_supernodes

  ami                    = local.supernode_ami
  instance_type          = local.supernode_type
  key_name               = var.key_name
  subnet_id              = module.vpc.private_subnets[count.index % length(module.vpc.private_subnets)]
  vpc_security_group_ids = [aws_security_group.supernode.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/userdata_supernode.sh.tpl", {
    region         = var.region
    ecr_repo       = aws_ecr_repository.fl.repository_url
    image_tag      = "latest"
    project_name   = local.name
    superlink_ip   = aws_instance.superlink.private_ip
    partition_id   = count.index
    num_clients    = var.num_supernodes
    data_s3_bucket = var.data_s3_bucket
    data_s3_prefix = var.data_s3_prefix
    data_dir       = local.data_dir
    task           = var.task
    enable_tls     = var.enable_tls
    use_gpu        = var.use_gpu
  })

  tags = { Name = "${local.name}-supernode-${count.index}" }

  depends_on = [aws_instance.superlink]
}
