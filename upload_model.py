import os
from huggingface_hub import HfApi, login, create_repo, upload_file

# === SET YOUR TOKEN HERE ===
token = os.getenv("HF_TOKEN")
if not token:
    token = input("Paste your HF token here: ").strip()

login(token=token)

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/regretzero-model"

print(f"Creating repo: {repo_id}")
create_repo(repo_id, exist_ok=True)

files = [
    "model/regret_ppo.pt",
    "demo/regret_demo.py",
    "README.md"
]

for file_path in files:
    if os.path.exists(file_path):
        print(f"Uploading {file_path}...")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.split("/")[-1],
            repo_id=repo_id
        )
        print(f"✅ Uploaded {file_path}")
    else:
        print(f"⚠️  {file_path} not found - skipping")

print(f"\n🎉 Upload complete!")
print(f"View your model at: https://huggingface.co/{repo_id}")
