"""
Utility for pruning tokenizer/config artifacts from a multimodal checkpoint into a
smaller modality-specific checkpoint.

Expected input files inside model_dir
=====================================

config.json
-----------
{
  "architectures": [
    "ApertusForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "dtype": "bfloat16",
  "eos_token_id": 2,
  "hidden_act": "xielu",
  "hidden_dropout": 0.0,
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 21504,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "apertus",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pad_token_id": 3,
  "post_norm": false,
  "qk_norm": true,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.0",
  "use_cache": true,
  "vocab_size": 131072
}

generation_config.json
----------------------
{
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "transformers_version": "4.57.0"
}

tokenizer.json
--------------
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": {
    "__type__": "list",
    "__length__": 136368,
    "__sample__": {
      "first": {
        "id": 0,
        "content": "<unk>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      },
      "second": {
        "id": 1,
        "content": "<s>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      },
      "...": "skipped entries",
      "last": {
        "id": 266439,
        "content": "<|audio token 4095|>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true
      }
    }
  },
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
  },
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {
        "SpecialToken": {
          "id": "<s>",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      }
    ],
    "pair": [
      {
        "SpecialToken": {
          "id": "<s>",
          "type_id": 0
        }
      },
      {
        "Sequence": {
          "id": "A",
          "type_id": 0
        }
      },
      {
        "SpecialToken": {
          "id": "<s>",
          "type_id": 1
        }
      },
      {
        "Sequence": {
          "id": "B",
          "type_id": 1
        }
      }
    ],
    "special_tokens": {
      "</s>": {
        "id": "</s>",
        "ids": [
          2
        ],
        "tokens": [
          "</s>"
        ]
      },
      "<s>": {
        "id": "<s>",
        "ids": [
          1
        ],
        "tokens": [
          "<s>"
        ]
      }
    }
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": true,
    "vocab": {
      "__type__": "object",
      "__entries__": 131072,
      "__sample__": {
        "first": {
          "<unk>": 0
        },
        "second": {
          "<s>": 1
        },
        "...": "skipped entries",
        "last": {
          "åĲİæ±īä¹¦": 131071
        }
      }
    },
    "merges": {
      "__type__": "list",
      "__length__": 269443,
      "__sample__": {
        "first": [
          "Ġ",
          "Ġ"
        ],
        "second": [
          "Ġ",
          "t"
        ],
        "...": "skipped entries",
        "last": [
          "åĲİ",
          "æ±īä¹¦"
        ]
      }
    }
  }
}

tokenizer_config.json
---------------------
{
  "add_bos_token": true,
  "add_eos_token": false,
  "add_prefix_space": false,
  "added_tokens_count": 131272,
  "added_tokens_decoder": {
    "__type__": "object",
    "__entries__": 136368,
    "__sample__": {
      "first": {
        "0": {
          "content": "<unk>",
          "lstrip": false,
          "normalized": false,
          "rstrip": false,
          "single_word": false,
          "special": true
        }
      },
      "second": {
        "1": {
          "content": "<s>",
          "lstrip": false,
          "normalized": false,
          "rstrip": false,
          "single_word": false,
          "special": true
        }
      },
      "...": "skipped entries",
      "last": {
        "266439": {
          "content": "<|audio token 4095|>",
          "lstrip": false,
          "normalized": false,
          "rstrip": false,
          "single_word": false,
          "special": true
        }
      }
    }
  },
  "additional_special_tokens": {
    "__type__": "list",
    "__length__": 4096,
    "__sample__": {
      "first": "<|audio token 0|>",
      "second": "<|audio token 1|>",
      "...": "skipped entries",
      "last": "<|audio token 4095|>"
    }
  },
  "audio_tokenizer": {
    "codebook_size": 4096,
    "type": "wavtokenizer"
  },
  "base_vocab_size": 131072,
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "</s>",
  "extra_special_tokens": {},
  "model_input_names": [
    "input_ids",
    "attention_mask"
  ],
  "model_max_length": 1000000000000000019884624838656,
  "omnimodal_config": {
    "modalities": [
      {
        "name": "vision",
        "offset": 131272,
        "vocab_size": 131072
      },
      {
        "name": "audio",
        "offset": 262344,
        "vocab_size": 4096
      }
    ],
    "omni_special_token_offset": 131072
  },
  "pad_token": "<pad>",
  "padding_side": "left",
  "tokenizer_class": "PreTrainedTokenizerFast",
  "unk_token": "<unk>",
  "vision_tokenizer": {
    "codebook_size": 131072,
    "path": "/capstor/store/cscs/swissai/infra01/MLLM/Emu3.5-VisionTokenizer",
    "type": "Emu3.5"
  },
  "vocab_size": 131072
}

special_tokens_map.json
-----------------------
{
  "additional_special_tokens": {
    "__type__": "list",
    "__length__": 4096,
    "__sample__": {
      "first": "<|audio token 0|>",
      "second": "<|audio token 1|>",
      "...": "skipped entries",
      "last": "<|audio token 4095|>"
    }
  },
  "bos_token": {
    "content": "<s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "eos_token": {
    "content": "</s>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "pad_token": {
    "content": "<pad>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  },
  "unk_token": {
    "content": "<unk>",
    "lstrip": false,
    "normalized": false,
    "rstrip": false,
    "single_word": false
  }
}

Important assumptions
=====================

1. Text tokens occupy [0, base_vocab_size).

2. Omni-special tokens, if present, start at:
   tokenizer_config.json["omnimodal_config"]["omni_special_token_offset"]

3. Each modality occupies one contiguous block described by:
   {
     "name": "...",
     "offset": <int>,
     "vocab_size": <int>
   }

4. Special token ids in config.json and generation_config.json refer to the
   original token id space before pruning.

5. token_to_old_id is built from:
   - tokenizer.json["model"]["vocab"]
   - tokenizer.json["added_tokens"]

6. tokenizer.json is rewritten into a compact vocabulary space 0..N-1.

7. config.json["vocab_size"] becomes the full compact model vocab size.

8. tokenizer_config.json["base_vocab_size"] and ["vocab_size"] are treated as the
   kept base text vocabulary size.

9. index_map.pt stores:
   new_id -> old_id

This module only rewrites tokenizer/config artifacts.
Actual model tensor slicing should be done separately using plan.keep_ids.
"""

import argparse
import copy
import gc
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the current Python process.

    If logging has not been configured yet, this sets a simple global format.
    If handlers already exist, only the root log level is updated.

    Args:
        level: Standard logging level such as logging.INFO or logging.DEBUG.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


class Modality(str, Enum):
    """
    Public modality enum used by callers of this module.

    Values:
        TEXT: Keep the base text vocabulary block.
        VISION: Keep the vision token block.
        AUDIO: Keep the audio token block.

    The enum values intentionally match tokenizer_config.json modality names.
    """

    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


@dataclass(frozen=True)
class TokenSpan:
    """
    Half-open token id interval [start, end).

    Example:
        TokenSpan("text", 0, 131072) contains ids 0 through 131071.
    """

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        """Number of ids in the span."""
        return self.end - self.start

    def contains(self, token_id: int) -> bool:
        """Return True if token_id lies inside this span."""
        return self.start <= token_id < self.end

    def ids(self) -> range:
        """Return the full id range represented by this span."""
        return range(self.start, self.end)


@dataclass(frozen=True)
class VocabularyLayout:
    """
    Logical token layout of the source tokenizer/model.

    Fields:
        text:
            Contiguous span for the text vocabulary.
        omni_special:
            Optional span for omni-special tokens that sit between text and
            the first modality block.
        modalities:
            Mapping from modality name, such as "vision" or "audio", to its
            contiguous token span.
    """

    text: TokenSpan
    omni_special: Optional[TokenSpan]
    modalities: Dict[str, TokenSpan]

    @classmethod
    def from_tokenizer_config(cls, tokenizer_config: Mapping[str, Any]) -> "VocabularyLayout":
        """
        Construct a VocabularyLayout from tokenizer_config.json.

        Assumptions:
            - base_vocab_size marks the end of the text block
            - omnimodal_config["omni_special_token_offset"] marks the start of
              the omni-special block
            - the first modality offset marks the end of the omni-special block
            - each modality occupies one contiguous block of ids
        """
        base_vocab_size = tokenizer_config["base_vocab_size"]
        omni_cfg = tokenizer_config["omnimodal_config"]
        modality_specs = sorted(omni_cfg["modalities"], key=lambda item: item["offset"])
        omni_special_offset = omni_cfg["omni_special_token_offset"]

        text = TokenSpan(name="text", start=0, end=base_vocab_size)

        omni_special = None
        if modality_specs and modality_specs[0]["offset"] > omni_special_offset:
            omni_special = TokenSpan(
                name="omni_special",
                start=omni_special_offset,
                end=modality_specs[0]["offset"],
            )

        modalities: Dict[str, TokenSpan] = {}
        for spec in modality_specs:
            name = spec["name"]
            start = spec["offset"]
            end = start + spec["vocab_size"]
            modalities[name] = TokenSpan(name=name, start=start, end=end)

        logger.info(
            "Vocabulary layout inferred: text=%s, omni_special=%s, modalities=%s",
            (text.start, text.end),
            None if omni_special is None else (omni_special.start, omni_special.end),
            {k: (v.start, v.end) for k, v in modalities.items()},
        )
        return cls(text=text, omni_special=omni_special, modalities=modalities)


@dataclass(frozen=True)
class ModalitySelection:
    """
    User selection describing which modalities should be retained.
    """

    modalities: Tuple[Modality, ...]
    keep_omni_special_tokens: Optional[bool] = None

    @classmethod
    def from_sequence(
        cls,
        modalities: Sequence[Modality],
        keep_omni_special_tokens: Optional[bool] = None,
    ) -> "ModalitySelection":
        """
        Build a selection from an input sequence while preserving order and
        removing duplicates.
        """
        deduped = tuple(dict.fromkeys(modalities))
        return cls(
            modalities=deduped,
            keep_omni_special_tokens=keep_omni_special_tokens,
        )

    @property
    def keep_text(self) -> bool:
        """Return True if the text block should be kept."""
        return Modality.TEXT in self.modalities

    @property
    def kept_config_modalities(self) -> Tuple[str, ...]:
        """
        Return modality names that should be looked up in tokenizer_config.json.

        TEXT is excluded because it is represented by the base vocab span.
        """
        return tuple(
            modality.value
            for modality in self.modalities
            if modality in (Modality.VISION, Modality.AUDIO)
        )

    def resolved_keep_omni_special_tokens(self) -> bool:
        """
        Resolve whether omni-special tokens should be kept.

        Default:
            keep them whenever at least one non-text modality is kept.
        """
        if self.keep_omni_special_tokens is not None:
            return self.keep_omni_special_tokens
        return len(self.kept_config_modalities) > 0


@dataclass(frozen=True)
class RemapPlan:
    """
    Full plan for converting old token ids into a new compact token space.
    """

    selection: ModalitySelection
    required_special_ids: Tuple[int, ...]
    keep_ids: List[int]
    old_to_new: Dict[int, int]
    base_text_kept_count: int
    omni_special_kept_count: int
    modality_kept_counts: Dict[str, int]
    new_modality_offsets: Dict[str, int]
    total_vocab_size: int

    @classmethod
    def build(
        cls,
        layout: VocabularyLayout,
        selection: ModalitySelection,
        required_special_ids: Sequence[int],
    ) -> "RemapPlan":
        """
        Build the old-id -> new-id remap plan.
        """
        unknown_modalities = [
            name for name in selection.kept_config_modalities if name not in layout.modalities
        ]
        if unknown_modalities:
            raise ValueError("Unknown modalities requested: %s" % unknown_modalities)

        keep_set: Set[int] = set()

        if selection.keep_text:
            keep_set.update(layout.text.ids())

        if selection.resolved_keep_omni_special_tokens() and layout.omni_special is not None:
            keep_set.update(layout.omni_special.ids())

        for modality_name in selection.kept_config_modalities:
            keep_set.update(layout.modalities[modality_name].ids())

        keep_set.update(required_special_ids)

        keep_ids = sorted(keep_set)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

        base_text_kept_count = sum(1 for token_id in keep_ids if layout.text.contains(token_id))

        omni_special_kept_count = 0
        if layout.omni_special is not None:
            omni_special_kept_count = sum(
                1 for token_id in keep_ids if layout.omni_special.contains(token_id)
            )

        modality_kept_counts: Dict[str, int] = {}
        for name, span in layout.modalities.items():
            modality_kept_counts[name] = sum(1 for token_id in keep_ids if span.contains(token_id))

        new_modality_offsets: Dict[str, int] = {}
        cursor = base_text_kept_count + omni_special_kept_count
        for name in selection.kept_config_modalities:
            new_modality_offsets[name] = cursor
            cursor += modality_kept_counts[name]

        plan = cls(
            selection=selection,
            required_special_ids=tuple(sorted(set(required_special_ids))),
            keep_ids=keep_ids,
            old_to_new=old_to_new,
            base_text_kept_count=base_text_kept_count,
            omni_special_kept_count=omni_special_kept_count,
            modality_kept_counts=modality_kept_counts,
            new_modality_offsets=new_modality_offsets,
            total_vocab_size=len(keep_ids),
        )

        logger.info(
            "Remap plan built: modalities=%s, keep_text=%s, keep_omni_special=%s, total_vocab_size=%d",
            [m.value for m in selection.modalities],
            selection.keep_text,
            selection.resolved_keep_omni_special_tokens(),
            plan.total_vocab_size,
        )
        return plan


@dataclass(frozen=True)
class SourceArtifacts:
    """
    Parsed source JSON artifacts loaded from the original model directory.
    """

    model_dir: Path
    tokenizer_json: Dict[str, Any]
    tokenizer_config: Dict[str, Any]
    special_tokens_map: Dict[str, Any]
    config: Dict[str, Any]
    generation_config: Dict[str, Any]

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "SourceArtifacts":
        """
        Load all expected JSON artifacts from a model directory.
        """
        model_dir_path = Path(model_dir)

        with (model_dir_path / "tokenizer.json").open("r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        with (model_dir_path / "tokenizer_config.json").open("r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        with (model_dir_path / "special_tokens_map.json").open("r", encoding="utf-8") as f:
            special_tokens_map = json.load(f)
        with (model_dir_path / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        with (model_dir_path / "generation_config.json").open("r", encoding="utf-8") as f:
            generation_config = json.load(f)

        artifacts = cls(
            model_dir=model_dir_path,
            tokenizer_json=tokenizer_json,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            config=config,
            generation_config=generation_config,
        )

        artifacts.warn_if_expected_structure_differs()
        logger.info("Loaded artifacts from %s", model_dir_path)
        return artifacts

    def warn_if_expected_structure_differs(self) -> None:
        """
        Emit warnings if loaded files differ from the originally provided structure.
        """
        expected_top_level = {
            "tokenizer_json": {
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            },
            "tokenizer_config": {
                "add_bos_token",
                "add_eos_token",
                "add_prefix_space",
                "added_tokens_count",
                "added_tokens_decoder",
                "additional_special_tokens",
                "audio_tokenizer",
                "base_vocab_size",
                "bos_token",
                "clean_up_tokenization_spaces",
                "eos_token",
                "extra_special_tokens",
                "model_input_names",
                "model_max_length",
                "omnimodal_config",
                "pad_token",
                "padding_side",
                "tokenizer_class",
                "unk_token",
                "vision_tokenizer",
                "vocab_size",
            },
            "special_tokens_map": {
                "additional_special_tokens",
                "bos_token",
                "eos_token",
                "pad_token",
                "unk_token",
            },
            "config": {
                "architectures",
                "attention_bias",
                "attention_dropout",
                "bos_token_id",
                "dtype",
                "eos_token_id",
                "hidden_act",
                "hidden_dropout",
                "hidden_size",
                "initializer_range",
                "intermediate_size",
                "max_position_embeddings",
                "mlp_bias",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "pad_token_id",
                "post_norm",
                "qk_norm",
                "rms_norm_eps",
                "rope_scaling",
                "rope_theta",
                "tie_word_embeddings",
                "transformers_version",
                "use_cache",
                "vocab_size",
            },
            "generation_config": {
                "_from_model_config",
                "bos_token_id",
                "eos_token_id",
                "transformers_version",
            },
        }

        actual = {
            "tokenizer_json": self.tokenizer_json,
            "tokenizer_config": self.tokenizer_config,
            "special_tokens_map": self.special_tokens_map,
            "config": self.config,
            "generation_config": self.generation_config,
        }

        for name, expected_keys in expected_top_level.items():
            actual_keys = set(actual[name].keys())
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)

            if missing:
                logger.warning("%s missing expected keys: %s", name, missing)
            if extra:
                logger.warning("%s has extra keys: %s", name, extra)

        tokenizer_model_keys = {
            "type",
            "dropout",
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "byte_fallback",
            "ignore_merges",
            "vocab",
            "merges",
        }
        actual_model_keys = set(self.tokenizer_json["model"].keys())
        missing = sorted(tokenizer_model_keys - actual_model_keys)
        extra = sorted(actual_model_keys - tokenizer_model_keys)

        if missing:
            logger.warning("tokenizer_json.model missing expected keys: %s", missing)
        if extra:
            logger.warning("tokenizer_json.model has extra keys: %s", extra)

    def build_token_to_old_id_map(self) -> Dict[str, int]:
        """
        Build token_string -> old_token_id mapping from tokenizer.json.
        """
        token_to_old_id: Dict[str, int] = {}

        for token, old_id in self.tokenizer_json["model"]["vocab"].items():
            token_to_old_id[token] = old_id

        for entry in self.tokenizer_json["added_tokens"]:
            token_to_old_id[entry["content"]] = entry["id"]

        return token_to_old_id

    def collect_mandatory_special_token_ids(self) -> Set[int]:
        """
        Collect all token ids that must be preserved even if the selected
        modalities would otherwise drop them.
        """
        token_to_old_id = self.build_token_to_old_id_map()
        required: Set[int] = set()

        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            required.add(self.config[key])

        for key in ("bos_token_id", "eos_token_id"):
            required.add(self.generation_config[key])

        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = self.special_tokens_map[key]
            token_str = value if isinstance(value, str) else value["content"]
            if token_str in token_to_old_id:
                required.add(token_to_old_id[token_str])

        logger.info("Collected %d required special ids", len(required))
        return required

    def infer_vocabulary_layout(self) -> VocabularyLayout:
        """
        Infer the original token layout from tokenizer_config.json.
        """
        return VocabularyLayout.from_tokenizer_config(self.tokenizer_config)


@dataclass(frozen=True)
class PreparedArtifacts:
    """
    Final rewritten artifacts plus the layout and remap plan that produced them.
    """

    tokenizer_json: Dict[str, Any]
    tokenizer_config: Dict[str, Any]
    special_tokens_map: Dict[str, Any]
    config: Dict[str, Any]
    generation_config: Dict[str, Any]
    plan: RemapPlan
    layout: VocabularyLayout

    @property
    def keep_ids(self) -> List[int]:
        """Convenience accessor for the old token ids that are kept."""
        return self.plan.keep_ids

    @property
    def old_to_new(self) -> Dict[int, int]:
        """Convenience accessor for old_id -> new_id mapping."""
        return self.plan.old_to_new

    def save_to_directory(self, out_dir: str, save_index_map: bool = True) -> None:
        """
        Save rewritten artifacts to an output directory.

        Files written:
            - tokenizer.json
            - tokenizer_config.json
            - special_tokens_map.json
            - config.json
            - generation_config.json
            - optionally index_map.pt
        """
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        files_to_save = {
            "tokenizer.json": self.tokenizer_json,
            "tokenizer_config.json": self.tokenizer_config,
            "special_tokens_map.json": self.special_tokens_map,
            "config.json": self.config,
            "generation_config.json": self.generation_config,
        }

        for filename, data in files_to_save.items():
            output_path = out_dir_path / filename
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved %s", output_path)

        if save_index_map:
            index_map_path = out_dir_path / "index_map.pt"
            torch.save(torch.tensor(self.plan.keep_ids, dtype=torch.long), index_map_path)
            logger.info("Saved %s", index_map_path)

        logger.info("All output artifacts saved to %s", out_dir_path)


def build_pruned_tokenizer_json(
    tokenizer_json: Mapping[str, Any],
    plan: RemapPlan,
) -> Dict[str, Any]:
    """
    Rewrite tokenizer.json into the new compact token-id space.
    """
    updated = copy.deepcopy(tokenizer_json)
    old_to_new = plan.old_to_new

    old_vocab = updated["model"]["vocab"]
    new_vocab_items = [
        (token, old_to_new[old_id])
        for token, old_id in old_vocab.items()
        if old_id in old_to_new
    ]
    new_vocab_items.sort(key=lambda item: item[1])
    updated["model"]["vocab"] = {token: new_id for token, new_id in new_vocab_items}

    if "vocab_size" in updated["model"]:
        updated["model"]["vocab_size"] = plan.base_text_kept_count

    kept_vocab_tokens = set(updated["model"]["vocab"].keys())
    filtered_merges = []

    for rule in updated["model"]["merges"]:
        if isinstance(rule, list):
            left, right = rule
        else:
            left, right = rule.split()

        if left in kept_vocab_tokens and right in kept_vocab_tokens and left + right in kept_vocab_tokens:
            filtered_merges.append(rule)

    updated["model"]["merges"] = filtered_merges
    logger.info(
        "Filtered tokenizer merges from %d to %d",
        len(tokenizer_json["model"]["merges"]),
        len(filtered_merges),
    )

    new_added_tokens = []
    for entry in updated["added_tokens"]:
        old_id = entry["id"]
        if old_id in old_to_new:
            new_entry = copy.deepcopy(entry)
            new_entry["id"] = old_to_new[old_id]
            new_added_tokens.append(new_entry)
    new_added_tokens.sort(key=lambda entry: entry["id"])
    updated["added_tokens"] = new_added_tokens

    if updated["padding"] is not None and "pad_id" in updated["padding"]:
        old_pad_id = updated["padding"]["pad_id"]
        updated["padding"]["pad_id"] = old_to_new[old_pad_id]

    for token_spec in updated["post_processor"]["special_tokens"].values():
        token_spec["ids"] = [old_to_new[old_id] for old_id in token_spec["ids"]]

    logger.info(
        "Built pruned tokenizer.json: base_text_vocab=%d, total_vocab_size=%d",
        plan.base_text_kept_count,
        plan.total_vocab_size,
    )
    return updated


def build_pruned_tokenizer_config(
    tokenizer_config: Mapping[str, Any],
    token_to_old_id: Mapping[str, int],
    plan: RemapPlan,
) -> Dict[str, Any]:
    """
    Rewrite tokenizer_config.json to match the new pruned token layout.
    """
    updated = copy.deepcopy(tokenizer_config)
    old_to_new = plan.old_to_new

    updated["vocab_size"] = plan.base_text_kept_count
    updated["base_vocab_size"] = plan.base_text_kept_count
    updated["added_tokens_count"] = plan.base_text_kept_count + plan.omni_special_kept_count

    new_decoder = {}
    for old_id_str, spec in updated["added_tokens_decoder"].items():
        old_id = int(old_id_str)
        if old_id in old_to_new:
            new_decoder[str(old_to_new[old_id])] = copy.deepcopy(spec)
    updated["added_tokens_decoder"] = dict(
        sorted(new_decoder.items(), key=lambda item: int(item[0]))
    )

    filtered_specials = []
    for token in updated["additional_special_tokens"]:
        if token in token_to_old_id and token_to_old_id[token] in old_to_new:
            filtered_specials.append(token)
    updated["additional_special_tokens"] = filtered_specials

    if plan.selection.kept_config_modalities or plan.omni_special_kept_count > 0:
        modalities = []
        for name in plan.selection.kept_config_modalities:
            modalities.append(
                {
                    "name": name,
                    "offset": plan.new_modality_offsets[name],
                    "vocab_size": plan.modality_kept_counts[name],
                }
            )

        updated["omnimodal_config"] = {
            "modalities": modalities,
            "omni_special_token_offset": plan.base_text_kept_count,
        }
    else:
        del updated["omnimodal_config"]

    if "vision" not in plan.selection.kept_config_modalities and "vision_tokenizer" in updated:
        del updated["vision_tokenizer"]

    if "audio" not in plan.selection.kept_config_modalities and "audio_tokenizer" in updated:
        del updated["audio_tokenizer"]

    logger.info(
        "Built pruned tokenizer_config.json: base_vocab_size=%d, kept_modalities=%s",
        updated["base_vocab_size"],
        plan.selection.kept_config_modalities,
    )
    return updated


def build_pruned_special_tokens_map(
    special_tokens_map: Mapping[str, Any],
    token_to_old_id: Mapping[str, int],
    plan: RemapPlan,
) -> Dict[str, Any]:
    """
    Rewrite special_tokens_map.json after pruning.
    """
    updated = copy.deepcopy(special_tokens_map)
    old_to_new = plan.old_to_new

    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = updated[key]
        token_str = value if isinstance(value, str) else value["content"]
        if token_str in token_to_old_id and token_to_old_id[token_str] not in old_to_new:
            raise ValueError("Special token %s=%r was removed from the vocabulary" % (key, token_str))

    filtered_specials = []
    for token in updated["additional_special_tokens"]:
        if token in token_to_old_id and token_to_old_id[token] in old_to_new:
            filtered_specials.append(token)
    updated["additional_special_tokens"] = filtered_specials

    logger.info("Built pruned special_tokens_map.json")
    return updated


def build_pruned_model_configs(
    config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    plan: RemapPlan,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Rewrite config.json and generation_config.json into the new compact token-id space.
    """
    old_to_new = plan.old_to_new

    new_config = copy.deepcopy(config)
    new_generation_config = copy.deepcopy(generation_config)

    new_config["vocab_size"] = plan.total_vocab_size

    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        new_config[key] = old_to_new[new_config[key]]

    for key in ("bos_token_id", "eos_token_id"):
        new_generation_config[key] = old_to_new[new_generation_config[key]]

    logger.info("Built pruned config.json and generation_config.json")
    return new_config, new_generation_config


def build_pruned_artifacts_from_source(
    source: SourceArtifacts,
    selection: ModalitySelection,
) -> PreparedArtifacts:
    """
    End-to-end preparation starting from already loaded source artifacts.
    """
    layout = source.infer_vocabulary_layout()
    required_special_ids = source.collect_mandatory_special_token_ids()
    plan = RemapPlan.build(
        layout=layout,
        selection=selection,
        required_special_ids=sorted(required_special_ids),
    )

    token_to_old_id = source.build_token_to_old_id_map()

    tokenizer_json = build_pruned_tokenizer_json(source.tokenizer_json, plan)
    tokenizer_config = build_pruned_tokenizer_config(source.tokenizer_config, token_to_old_id, plan)
    special_tokens_map = build_pruned_special_tokens_map(source.special_tokens_map, token_to_old_id, plan)
    config, generation_config = build_pruned_model_configs(
        source.config,
        source.generation_config,
        plan,
    )

    return PreparedArtifacts(
        tokenizer_json=tokenizer_json,
        tokenizer_config=tokenizer_config,
        special_tokens_map=special_tokens_map,
        config=config,
        generation_config=generation_config,
        plan=plan,
        layout=layout,
    )


def build_pruned_artifacts_from_model_dir(
    model_dir: str,
    modalities: Sequence[Modality] = (Modality.TEXT,),
    keep_omni_special_tokens: Optional[bool] = None,
) -> PreparedArtifacts:
    """
    Convenience entry point for artifact rewriting only.
    """
    source = SourceArtifacts.from_model_dir(model_dir)
    selection = ModalitySelection.from_sequence(
        modalities=modalities,
        keep_omni_special_tokens=keep_omni_special_tokens,
    )
    prepared = build_pruned_artifacts_from_source(source, selection)

    logger.info(
        "Prepared artifacts for modalities=%s, total_vocab_size=%d",
        [m.value for m in prepared.plan.selection.modalities],
        prepared.plan.total_vocab_size,
    )
    return prepared


def prune_and_save_checkpoint(
    model_dir: str,
    out_dir: str,
    modalities: Sequence[Modality] = (Modality.TEXT,),
    keep_omni_special_tokens: Optional[bool] = None,
    device: str = "cuda:0",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> PreparedArtifacts:
    """
    End-to-end pruning entry point.

    This function:
      1. Loads source tokenizer/config artifacts.
      2. Builds the remap plan.
      3. Loads the model checkpoint.
      4. Prunes vocab-sized model tensors:
         - input embeddings
         - output embeddings / lm_head
         - lm_head bias, if present
      5. Saves the pruned model and rewritten artifacts.
    """
    logger.info("Starting checkpoint pruning")
    logger.info("Source model_dir=%s", model_dir)
    logger.info("Output out_dir=%s", out_dir)
    logger.info("Requested modalities=%s", [m.value for m in modalities])

    source = SourceArtifacts.from_model_dir(model_dir)
    selection = ModalitySelection.from_sequence(
        modalities=modalities,
        keep_omni_special_tokens=keep_omni_special_tokens,
    )
    prepared = build_pruned_artifacts_from_source(source, selection)

    logger.info(
        "Prepared remap plan with total_vocab_size=%d",
        prepared.plan.total_vocab_size,
    )

    logger.info("Loading model weights from %s", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    if not isinstance(input_embeddings, torch.nn.Embedding):
        raise TypeError(
            "Expected model.get_input_embeddings() to return torch.nn.Embedding, "
            f"got {type(input_embeddings).__name__}"
        )

    if not isinstance(output_embeddings, torch.nn.Linear):
        raise TypeError(
            "Expected model.get_output_embeddings() to return torch.nn.Linear, "
            f"got {type(output_embeddings).__name__}"
        )

    keep_ids = prepared.keep_ids
    keep_index = torch.tensor(keep_ids, dtype=torch.long, device=input_embeddings.weight.device)

    old_input_vocab_size = input_embeddings.weight.shape[0]
    old_output_vocab_size = output_embeddings.weight.shape[0]

    if keep_index.numel() == 0:
        raise ValueError("The remap plan produced an empty keep_ids list")

    if int(keep_index.max().item()) >= old_input_vocab_size:
        raise ValueError(
            f"keep_ids contains id {int(keep_index.max().item())}, "
            f"but input embedding vocab size is only {old_input_vocab_size}"
        )

    if int(keep_index.max().item()) >= old_output_vocab_size:
        raise ValueError(
            f"keep_ids contains id {int(keep_index.max().item())}, "
            f"but output embedding vocab size is only {old_output_vocab_size}"
        )

    logger.info(
        "Pruning input embeddings from %d -> %d rows",
        old_input_vocab_size,
        len(keep_ids),
    )
    logger.info(
        "Pruning output embeddings from %d -> %d rows",
        old_output_vocab_size,
        len(keep_ids),
    )

    new_input_weight = input_embeddings.weight.detach().index_select(0, keep_index).clone()
    new_output_weight = output_embeddings.weight.detach().index_select(0, keep_index).clone()

    new_output_bias = None
    if output_embeddings.bias is not None:
        new_output_bias = output_embeddings.bias.detach().index_select(0, keep_index).clone()

    new_input_embeddings = torch.nn.Embedding(
        num_embeddings=len(keep_ids),
        embedding_dim=input_embeddings.embedding_dim,
        device=new_input_weight.device,
        dtype=new_input_weight.dtype,
    )
    new_input_embeddings.weight.data.copy_(new_input_weight)

    new_output_embeddings = torch.nn.Linear(
        in_features=output_embeddings.in_features,
        out_features=len(keep_ids),
        bias=(new_output_bias is not None),
        device=new_output_weight.device,
        dtype=new_output_weight.dtype,
    )
    new_output_embeddings.weight.data.copy_(new_output_weight)

    if new_output_bias is not None:
        new_output_embeddings.bias.data.copy_(new_output_bias)

    model.set_input_embeddings(new_input_embeddings)
    model.set_output_embeddings(new_output_embeddings)

    if getattr(model.config, "tie_word_embeddings", False):
        logger.info("Retieing word embeddings because tie_word_embeddings=True")
        model.tie_weights()

    for key, value in prepared.config.items():
        setattr(model.config, key, value)

    if getattr(model, "generation_config", None) is not None:
        for key, value in prepared.generation_config.items():
            setattr(model.generation_config, key, value)

    del input_embeddings
    del output_embeddings
    del new_input_weight
    del new_output_weight
    del new_output_bias
    gc.collect()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving pruned model weights to %s", out_dir)
    model.save_pretrained(out_dir)

    prepared.save_to_directory(out_dir, save_index_map=True)

    logger.info("Checkpoint pruning finished successfully")
    return prepared


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Modes:
        - artifact-only mode:
            rewrite and save tokenizer/config artifacts only

        - full pruning mode:
            rewrite/save artifacts and prune model vocab-sized weights
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prune a multimodal Apertus checkpoint to selected modalities by rewriting "
            "tokenizer/config artifacts, optionally pruning model vocab-sized weights too."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the source model directory containing tokenizer/config files and model weights.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where rewritten artifacts and optionally pruned model weights will be saved.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        required=True,
        choices=[modality.value for modality in Modality],
        help=(
            "Modalities to keep. Valid values: text vision audio. "
            "Examples: --modalities text ; --modalities vision ; --modalities text vision"
        ),
    )

    omni_group = parser.add_mutually_exclusive_group()
    omni_group.add_argument(
        "--keep-omni-special-tokens",
        dest="keep_omni_special_tokens",
        action="store_true",
        help="Force keeping omni-special tokens.",
    )
    omni_group.add_argument(
        "--drop-omni-special-tokens",
        dest="drop_omni_special_tokens",
        action="store_true",
        help="Force dropping omni-special tokens when possible.",
    )

    parser.add_argument(
        "--prune-model-weights",
        action="store_true",
        help=(
            "Also prune model vocab-sized tensors (input embeddings, output embeddings, "
            "and lm_head bias if present). If omitted, only tokenizer/config artifacts are rewritten."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for loading and pruning model weights. Used only with --prune-model-weights.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype used when loading model weights. Used only with --prune-model-weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True when loading the model.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


def parse_modalities(modality_names: Sequence[str]) -> List[Modality]:
    """
    Convert CLI modality strings into Modality enums.
    """
    return [Modality(name) for name in modality_names]


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    """
    Convert CLI dtype string into torch.dtype.
    """
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def resolve_keep_omni_special_tokens(args: argparse.Namespace) -> Optional[bool]:
    """
    Resolve omni-special token behavior from CLI flags.

    Returns:
        True if --keep-omni-special-tokens was passed
        False if --drop-omni-special-tokens was passed
        None otherwise, meaning default selection logic should be used
    """
    if args.keep_omni_special_tokens:
        return True
    if args.drop_omni_special_tokens:
        return False
    return None


def main() -> None:
    """
    Command-line entry point.
    """
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))

    modalities = parse_modalities(args.modalities)
    keep_omni_special_tokens = resolve_keep_omni_special_tokens(args)

    if args.prune_model_weights:
        prune_and_save_checkpoint(
            model_dir=args.model_dir,
            out_dir=args.out_dir,
            modalities=modalities,
            keep_omni_special_tokens=keep_omni_special_tokens,
            device=args.device,
            torch_dtype=parse_torch_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
        )
    else:
        prepared = build_pruned_artifacts_from_model_dir(
            model_dir=args.model_dir,
            modalities=modalities,
            keep_omni_special_tokens=keep_omni_special_tokens,
        )
        prepared.save_to_directory(args.out_dir, save_index_map=True)


if __name__ == "__main__":
    main()