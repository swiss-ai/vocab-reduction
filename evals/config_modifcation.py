import json
from huggingface_hub import HfApi, hf_hub_download

repo_id = "faruk-zahiragic/apertus-300m-vanilla-baseline"
api = HfApi()

# 1. Update config.json with the auto_map
print("Updating config.json...")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Ensure the type and mapping are correct
config["model_type"] = "apertus"
config["auto_map"] = {
    "AutoConfig": "configuration_apertus.ApertusConfig",
    "AutoModelForCausalLM": "modeling_apertus.ApertusForCausalLM"
}
config["trust_remote_code"] = True

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

api.upload_file(path_or_fileobj="config.json", path_in_repo="config.json", repo_id=repo_id)

# 2. Overwrite the 24MB tokenizer_config.json with a tiny standard one
print("Fixing tokenizer_config.json...")
tokenizer_config = {
    "tokenizer_class": "LlamaTokenizerFast", 
    "clean_up_tokenization_spaces": False,
    "model_max_length": 8192,
    "padding_side": "left"
}

with open("tokenizer_config.json", "w") as f:
    json.dump(tokenizer_config, f, indent=2)

api.upload_file(path_or_fileobj="tokenizer_config.json", path_in_repo="tokenizer_config.json", repo_id=repo_id)

print("Metadata repair complete! Your repo is now 'Apertus-aware'.")