
import os
import joblib
from src.model import load_model, save_model, MODEL_PATH, SCALER_PATH
from src.aws_storage import upload_model_to_s3

print("📤 Uploading model artifacts to S3...")
# Just call the upload part of save_model logic
model, scaler = load_model()
u1 = upload_model_to_s3(MODEL_PATH, "congestion_model.pkl")
u2 = upload_model_to_s3(SCALER_PATH, "scaler.pkl")

if u1 and u2:
    print("✅ Model artifacts uploaded to Amazon S3 successfully!")
else:
    print("❌ S3 upload failed. Check your bucket permissions.")
