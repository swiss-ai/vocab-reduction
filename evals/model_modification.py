import re
from huggingface_hub import HfApi

repo_id = "faruk-zahiragic/apertus-300m-vanilla-baseline"
api = HfApi()
base_path = "/users/faruk_zahiragic/miniconda/envs/apertus-1p5-evals/lib/python3.13/site-packages/transformers/models/apertus"

def fix_and_upload(filename):
    print(f"Fixing {filename}...")
    with open(f"{base_path}/{filename}", "r") as f:
        content = f.read()

    # 1. Fix the relative imports we addressed earlier
    content = re.sub(r"from \.\.\.", "from transformers.", content)
    content = re.sub(r"from \.\.", "from transformers.", content)
    
    # 2. Fix the PreTrainedConfig typo (lowercase 't' is correct)
    content = content.replace("PreTrainedConfig", "PretrainedConfig")
    
    # 3. Ensure PreTrainedModel is correct (uppercase 'T' is correct for models)
    # This is usually already correct in the source, but good to ensure.
    
    fixed_name = f"fixed_{filename}"
    with open(fixed_name, "w") as f:
        f.write(content)

    api.upload_file(
        path_or_fileobj=fixed_name,
        path_in_repo=filename,
        repo_id=repo_id
    )

fix_and_upload("configuration_apertus.py")
fix_and_upload("modeling_apertus.py")
print("Done! Typo fixed and files re-uploaded.")