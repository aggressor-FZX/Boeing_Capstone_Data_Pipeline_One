from huggingface_hub import snapshot_download
import os

print("Starting model download...")
print("This will take 10-20 minutes for ~50GB")

model_path = snapshot_download(
    repo_id="OpenGVLab/InternVL2_5-26B",
    local_dir="./models/InternVL2_5-26B",
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"✅ Model downloaded to: {os.path.abspath(model_path)}")
