from huggingface_hub import HfApi
import json

repo_id = "faruk-zahiragic/apertus-300m-vanilla-baseline"
api = HfApi()

# Standard config for Llama-based models using tokenizer.json
# By NOT including an 'auto_map', we stop the search for tokenizer.py
standard_config = {
    "tokenizer_class": "LlamaTokenizerFast", 
    "clean_up_tokenization_spaces": False,
    "model_max_length": 8192,
    "padding_side": "left",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>"
}

with open("tokenizer_config.json", "w") as f:
    json.dump(standard_config, f, indent=2)

api.upload_file(
    path_or_fileobj="tokenizer_config.json",
    path_in_repo="tokenizer_config.json",
    repo_id=repo_id
)

print("Standardized tokenizer_config.json uploaded!")