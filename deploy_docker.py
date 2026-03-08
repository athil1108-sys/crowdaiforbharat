import os
import boto3
import base64
import subprocess
from dotenv import load_dotenv

load_dotenv()

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
REPO_NAME = "crowdai-dashboard"
IMAGE_TAG = "latest"

print(f"🚀 Starting Docker Container Deployment to AWS ECR ({REGION})...")

# 1. Get AWS Account ID
sts = boto3.client('sts', region_name=REGION)
account_id = sts.get_caller_identity()["Account"]
registry_url = f"{account_id}.dkr.ecr.{REGION}.amazonaws.com"
full_image_name = f"{registry_url}/{REPO_NAME}:{IMAGE_TAG}"

print(f"   ℹ️  Target Registry: {registry_url}")

# 2. Ensure ECR Repository exists
ecr = boto3.client('ecr', region_name=REGION)
try:
    ecr.describe_repositories(repositoryNames=[REPO_NAME])
    print(f"   ✅ ECR Repository '{REPO_NAME}' already exists.")
except ecr.exceptions.RepositoryNotFoundException:
    print(f"   ⏳ Creating ECR Repository '{REPO_NAME}'...")
    ecr.create_repository(repositoryName=REPO_NAME)
    print(f"   ✅ Created ECR Repository.")

# 3. Authenticate Docker with ECR
print(f"🔐 Authenticating Docker daemon with AWS ECR using AWS CLI...")

# We use the standard AWS CLI login command pipe
login_cmd = f"aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {registry_url}"
result = subprocess.run(login_cmd, shell=True, capture_output=True, text=True)

if result.returncode != 0:
    print(f"   ❌ Docker login failed: {result.stderr}")
    exit(1)
print("   ✅ Docker authenticated successfully.")

# 4. Build the Docker Image
print(f"🐳 Building Docker image '{REPO_NAME}:{IMAGE_TAG}'...")
build_cmd = f"docker build -t {REPO_NAME}:{IMAGE_TAG} ."
result = subprocess.run(build_cmd, shell=True)
if result.returncode != 0:
    print("   ❌ Docker build failed.")
    exit(1)
print("   ✅ Docker image built.")

# 5. Tag and Push the Docker Image
print(f"🏷️  Tagging image for ECR...")
subprocess.run(f"docker tag {REPO_NAME}:{IMAGE_TAG} {full_image_name}", shell=True, check=True)

print(f"📤 Pushing image to AWS ECR: {full_image_name}")
# Note: Pushing a 1GB+ image might take a few minutes depending on internet speed
push_result = subprocess.run(f"docker push {full_image_name}", shell=True)
if push_result.returncode != 0:
    print("   ❌ Docker push failed.")
    exit(1)

print("\n🎉 Container successfully pushed to AWS ECR!")
print(f"URI: {full_image_name}")
