import os
from huggingface_hub import login, create_repo, upload_folder

token = os.environ.get("HF_TOKEN")
repo_id = os.environ.get("REPO_ID")

login(token=token)

upload_folder(
    folder_path=os.environ.get("FOLDER_PATH"),
    repo_id=repo_id,
    repo_type="model",
)