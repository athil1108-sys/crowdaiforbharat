import os
import zipfile
import boto3
import json
import time
from dotenv import load_dotenv

load_dotenv()

FUNCTION_NAME = "CrowdAIPredictor"
ROLE_NAME = "CrowdAILambdaExecutionRole"
REGION = "us-east-1"
ZIP_FILE = "lambda_deployment.zip"

print(f"🚀 Starting Phase 2: Lambda Deployment for {FUNCTION_NAME}...")

# 1. Create a deployment package (ZIP)
print(f"📦 Packaging Lambda function...")
if os.path.exists(ZIP_FILE):
    os.remove(ZIP_FILE)

with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add the handler script at the root of the zip
    zipf.write(os.path.join('src', 'lambda_handler.py'), 'lambda_handler.py')
    print("   ✅ Added lambda_handler.py")
    
# We will rely on AWS Lambda layers for numpy/scikit-learn later.
# For now, deploying the bare script.

# 2. Setup IAM Role for Lambda
print(f"🔐 Setting up IAM Role '{ROLE_NAME}'...")
iam = boto3.client('iam', region_name=REGION)
role_arn = None

# Assume Role Policy document for Lambda
assume_role_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

try:
    response = iam.get_role(RoleName=ROLE_NAME)
    role_arn = response['Role']['Arn']
    print(f"   ✅ Role already exists: {role_arn}")
except iam.exceptions.NoSuchEntityException:
    print(f"   ⏳ Creating role...")
    response = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy)
    )
    role_arn = response['Role']['Arn']
    
    # Attach basic execution constraints and full S3/DynamoDB access for simplicity in demo
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn='arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess')
    print(f"   ✅ Role created and policies attached: {role_arn}")
    print(f"   ⏳ Waiting 10s for IAM propagation...")
    time.sleep(10)

# 3. Create or Update Lambda Function
print(f"⚡ Deploying Lambda Function '{FUNCTION_NAME}'...")
lambda_client = boto3.client('lambda', region_name=REGION)

with open(ZIP_FILE, 'rb') as f:
    zip_bytes = f.read()

try:
    lambda_client.get_function(FunctionName=FUNCTION_NAME)
    print(f"   ⏳ Function exists, updating code...")
    lambda_client.update_function_code(
        FunctionName=FUNCTION_NAME,
        ZipFile=zip_bytes
    )
    print(f"   ✅ Lambda code updated successfully.")
except lambda_client.exceptions.ResourceNotFoundException:
    print(f"   ⏳ Creating new function...")
    
    response = lambda_client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime='python3.10',
        Role=role_arn,
        Handler='lambda_handler.handler',  
        Code={'ZipFile': zip_bytes},
        Timeout=15,
        MemorySize=512,
        Layers=[
            'arn:aws:lambda:us-east-1:336392222285:layer:AWSSDKPandas-Python310:13'
        ],
        Environment={
            'Variables': {
                'CROWDAI_S3_BUCKET': os.environ.get('CROWDAI_S3_BUCKET', 'crowdai-models-unique'),
                'AWS_DEFAULT_REGION': 'us-east-1'
            }
        }
    )
    print(f"   ✅ Lambda function created successfully: {response['FunctionArn']}")

print("\n🚀 Phase 2 Lambda Deployment Complete!")
