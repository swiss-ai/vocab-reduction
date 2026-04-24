import os
from huggingface_hub import login, create_repo, upload_folder

token = os.environ.get("HF_TOKEN")
repo_id = os.environ.get("REPO_ID")

login(token=token)
create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)

upload_folder(
    folder_path=os.environ.get("FOLDER_PATH"),
    repo_id=repo_id,
    repo_type="model",
)