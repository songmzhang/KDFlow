import os
from typing import List, Dict, Any

import torch
from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoProcessor, AutoTokenizer

from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def blending_datasets(
    datasets,
    probabilities=None,
    strategy=None,
    seed=42,
    max_count=1e8,
    stopping_strategy="all_exhausted",
    dataset_split="train",
):
    """Blend multiple datasets with optional probability sampling.

    Args:
        datasets (str): Comma-separated list of dataset paths
        probabilities (str, optional): Comma-separated list of probabilities for sampling.
            If None, datasets will be concatenated without probability sampling.
        strategy: Training strategy object
        seed (int): Random seed
        max_count (int): Maximum number of samples per dataset
    """
    datasets = datasets.split(",")
    if probabilities is not None:
        probabilities = list(map(float, probabilities.split(",")))
        assert len(probabilities) == len(datasets)

    data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet", ".arrow"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        elif strategy.args.data.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        # Select dataset
        if dataset_split and dataset_split in data:
            data = data[dataset_split]
        data = data.select(range(min(max_count, len(data))))
        data_list.append(data)

    # merge datasets
    if strategy.is_rank_0():
        print(data_list)

    # If probabilities is None, concatenate datasets directly
    if probabilities is None:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(data_list)
    else:
        dataset = interleave_datasets(
            data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    return dataset


def get_tokenizer_or_processor(model_name_or_path, model=None, padding_side="right", use_fast=True, need_processor=False):
    if need_processor:
        return get_processor(model_name_or_path, model, padding_side, use_fast)
    else:
        return get_tokenizer(model_name_or_path, model, padding_side, use_fast)


def get_processor(model_name_or_path, model=None, padding_side="right", use_fast=True):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=use_fast)
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        logger.info("Detect no pad_token in tokenizer, set it to eos_token.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        if model is not None:
            model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id

    return processor


def get_tokenizer(model_name_or_path, model=None, padding_side="right", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        logger.info("Detect no pad_token in tokenizer, set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    from torch.nn.utils.rnn import pad_sequence
    assert side in ("left", "right")
    sequences = [seq.squeeze(0) for seq in sequences]
    if side == "left":
        sequences = [seq.flip(dims=0) for seq in sequences]
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=value
    )
    if side == "left":
        padded_sequences = torch.flip(padded_sequences, dims=[1])
    return padded_sequences


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor, return_tensors: bool = True):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        ids = ids[mask.bool()] if return_tensors else ids[mask.bool()].tolist()
        no_padding_batch.append(ids)
    return no_padding_batch


# ShareGPT role mapping to OpenAI roles
SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _is_sharegpt_format(data):
    """Check if data is in ShareGPT format (list of dicts with 'from'/'value' keys)."""
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    return "from" in data[0] and "value" in data[0]


def _is_openai_format(data):
    """Check if data is already in OpenAI messages format (list of dicts with 'role'/'content' keys)."""
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    return "role" in data[0] and "content" in data[0]


def _is_alpaca_format(data):
    """Check if data is in Alpaca format (dict with 'instruction' key)."""
    if not isinstance(data, dict):
        return False
    return "instruction" in data


def _convert_sharegpt(messages):
    """Convert ShareGPT format messages to OpenAI messages format.
    
    ShareGPT: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    OpenAI:   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    converted = []
    for msg in messages:
        role = SHAREGPT_ROLE_MAP.get(msg["from"], msg["from"])
        content = msg.get("value") or ""
        converted.append({"role": role, "content": content})
    return converted


def _convert_alpaca(data):
    """Convert Alpaca format to OpenAI messages format.
    
    Alpaca format fields:
        - instruction (required): the main instruction/question
        - input (optional): additional input context
        - output (optional): the expected response
        - system (optional): system prompt
        - history (optional): list of [user_msg, assistant_msg] pairs
    
    Returns:
        List of OpenAI messages: [{"role": "...", "content": "..."}, ...]
    """
    messages = []

    # System prompt
    system_prompt = data.get("system", "")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # History turns
    history = data.get("history", [])
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})

    # Current instruction + optional input
    instruction = data.get("instruction", "")
    extra_input = data.get("input", "")
    if extra_input:
        user_content = f"{instruction}\n{extra_input}"
    else:
        user_content = instruction
    messages.append({"role": "user", "content": user_content})

    # Output (response)
    output = data.get("output", "")
    if output:
        messages.append({"role": "assistant", "content": output})

    return messages


def convert_to_openai_messages(data, role="user", expand_image=False):
    """Unified converter: auto-detect data format and convert to OpenAI messages.
    
    Supported input formats:
        1. OpenAI messages (already): [{"role": "user", "content": "..."}] -> returned as-is
        2. ShareGPT: [{"from": "human", "value": "..."}] -> converted
        3. Alpaca: {"instruction": "...", "input": "...", "output": "...", ...} -> converted
        4. Plain string: "..." -> wrapped as [{"role": role, "content": "..."}]
    
    Args:
        data: Input data in any supported format.
        role: Role to assign when data is a plain string. Defaults to "user".
        expand_image: Whether to expand <image> placeholders into multi-part content.
            Only set to True for multimodal (image) scenarios. Defaults to False.
    
    Returns:
        List of dicts in OpenAI messages format: [{"role": "...", "content": "..."}, ...]
    """
    if data is None:
        raise ValueError("convert_to_openai_messages received None input.")

    if isinstance(data, str):
        messages = [{"role": role, "content": data}]
    elif isinstance(data, list):
        if not data:
            raise ValueError("convert_to_openai_messages received an empty list.")
        if _is_openai_format(data):
            messages = data
        elif _is_sharegpt_format(data):
            messages = _convert_sharegpt(data)
        else:
            raise ValueError(
                f"Unsupported data format. Expected OpenAI messages, ShareGPT, Alpaca, or plain string. "
                f"Got: {type(data)} with keys/content: {list(data[0].keys()) if data else data}"
            )
    elif isinstance(data, dict):
        if _is_alpaca_format(data):
            messages = _convert_alpaca(data)
        else:
            raise ValueError(
                f"Unsupported data format. Expected OpenAI messages, ShareGPT, Alpaca, or plain string. "
                f"Got: {type(data)} with keys/content: {data}"
            )
    else:
        raise ValueError(
            f"Unsupported data format. Expected OpenAI messages, ShareGPT, Alpaca, or plain string. "
            f"Got: {type(data)}"
        )

    if expand_image:
        messages = expand_image_placeholders(messages)
    return messages


def expand_image_placeholders(messages, placeholder="<image>"):
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str) or placeholder not in content:
            continue
        parts = content.split(placeholder)
        content_list = []
        for i, part in enumerate(parts):
            if part:
                content_list.append({"type": "text", "text": part})
            if i < len(parts) - 1:
                content_list.append({"type": "image"})
        msg["content"] = content_list
    return messages
