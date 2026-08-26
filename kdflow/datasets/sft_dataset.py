from typing import Callable, Optional, Dict, List, Any

import torch
from torch.utils.data import Dataset
from datasets import Image as ImageFeature
from PIL import Image

from kdflow.datasets.utils import (
    convert_to_openai_messages,
    get_tokenizer_or_processor,
    validate_dataset_columns,
    zero_pad_sequences,
)
from kdflow.models.utils import TokenizerCompareResult

class SFTDataset(Dataset):
    """
    Dataset for SFT and Off-Policy KD.
    
    Args:
        dataset: dataset for SFT and Off-Policy KD
        strategy: training strategy object
        tokenizer_info: result of tokenizer comparison (template_identical, vocab_identical)
        max_data_num: maximum number of data to load
        input_template: optional template for formatting input
        num_processors: number of processors for parallel data loading
    """
    
    def __init__(
        self,
        dataset,
        strategy,
        tokenizer_info: Optional[TokenizerCompareResult] = None,
        max_data_num: int = -1,
        input_template: Optional[str] = None,
        num_processors: int = 8,
    ) -> None:
        super().__init__()
        self.args = strategy.args
        self.strategy = strategy
        self.max_length = self.args.data.max_len
        self.tokenizer_info = tokenizer_info
        self.template_identical = True if tokenizer_info is None else self.tokenizer_info.template_identical
        self.vocab_identical = True if tokenizer_info is None else self.tokenizer_info.vocab_identical

        self.input_template = input_template
        self.input_key = getattr(self.args.data, "input_key", None)
        self.teacher_input_key = getattr(self.args.data, "teacher_input_key", None) or self.input_key
        self.output_key = getattr(self.args.data, "output_key", None)
        self.apply_chat_template = getattr(self.args.data, "apply_chat_template", False)
        self.enable_thinking = getattr(self.args.model, "enable_thinking", False)

        self.image_key = getattr(self.args.data, "image_key", None)
        self.same_tokenizer = True if tokenizer_info is None else self.tokenizer_info.is_identical
        self.teacher_student_share_input = self.same_tokenizer and (self.teacher_input_key == self.input_key)

        # Validate that all required columns exist in the dataset
        required_columns = {"input_key": self.input_key}
        if self.output_key:
            required_columns["output_key"] = self.output_key
        if self.image_key:
            required_columns["image_key"] = self.image_key
        has_teacher = (
            self.args.model.teacher_name_or_path is not None
            or self.args.kd.multi_teacher_config is not None
        )
        if has_teacher and self.teacher_input_key != self.input_key:
            required_columns["teacher_input_key"] = self.teacher_input_key
        if self.args.kd.multi_teacher_config:
            required_columns["teacher_routing_key"] = self.args.data.teacher_routing_key
        validate_dataset_columns(dataset, **required_columns)
        
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path, 
            need_processor=self.image_key is not None,
        )
        # extract tokenizer in hf processor
        self.student_tokenizer = getattr(self.student_processor, "tokenizer", self.student_processor)
        self.teacher_processors = {}
        if self.args.kd.multi_teacher_config:
            for teacher_key, teacher_path in self.args.kd.multi_teacher_config.items():
                self.teacher_processors[teacher_key] = get_tokenizer_or_processor(
                    teacher_path, need_processor=self.image_key is not None,
                )
        elif self.args.model.teacher_name_or_path is not None:
            self.teacher_processors["default"] = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )

        if max_data_num > 0 and max_data_num < len(dataset):
            self.strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        # Adjust worker count based on dataset size to speed up preprocessing.
        num_processors = min(
            num_processors,
            max(1, len(dataset) // 2000),
        )
        self.processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
            load_from_cache_file=False,
            desc="Processing data",
        )
        if self.image_key:
            self.processed_dataset = self.processed_dataset.cast_column("images", [ImageFeature()])

        original_len = len(self.processed_dataset)
        strategy.log(f"Before length filter: {len(self.processed_dataset)}")
        self.processed_dataset = self.processed_dataset.filter(
            lambda stu_input_len: stu_input_len <= self.max_length,
            input_columns=["stu_input_len"],
            desc="Filtering overlang samples",
        )
        if len(self.processed_dataset) < original_len:
            self.strategy.log(
                f"Filtered {original_len - len(self.processed_dataset)} samples "
                f"exceeding max_length={self.max_length} "
                f"({original_len} -> {len(self.processed_dataset)})"
            )
        
        self._print_sample()

    def _print_sample(self) -> None:
        sample = self.processed_dataset[0]
        self.strategy.print("Student prompt + response:")
        self.strategy.print(sample["stu_prompt"] + sample["stu_response"])
        if not self.teacher_student_share_input:
            self.strategy.print("Teacher prompt + response:")
            self.strategy.print(sample["tea_prompt"] + sample["tea_response"])

    def process_data(self, data: Dict) -> Dict[str, Any]:
        stu_chat_template_fn = self.student_processor.apply_chat_template if self.apply_chat_template else None
        stu_prompt, stu_response = self.preprocess_data(
            data, self.input_template, self.input_key, self.output_key,
            apply_chat_template=stu_chat_template_fn,
        )
        stu_eos_token = self.student_tokenizer.eos_token
        if not stu_response.endswith(stu_eos_token):
            stu_response += " " + stu_eos_token

        result = {"stu_prompt": stu_prompt, "stu_response": stu_response}
        result["stu_resp_len"] = self._compute_token_length(
            self.student_processor, stu_response
        )
        result["stu_input_len"] = self._compute_token_length(
            self.student_processor, stu_prompt
        ) + result["stu_resp_len"]

        if self.image_key is not None:
            result["images"] = self.load_images(data[self.image_key])

        if self.args.model.teacher_name_or_path is not None \
        or self.args.kd.multi_teacher_config is not None:
            if not self.teacher_student_share_input:
                # Select the appropriate teacher processor
                if self.args.kd.multi_teacher_config is not None:
                    routing_key = self.args.data.teacher_routing_key
                    assert routing_key is not None, "`--teacher_routing_key` must be specified when using multi_teacher_config"
                    assert routing_key in data, f"Routing key '{routing_key}' not found in data"
                    teacher_key = data[routing_key]
                    if teacher_key not in self.teacher_processors:
                        raise ValueError(
                            f"Teacher routing key '{teacher_key}' not found in multi_teacher_config. "
                            f"Available keys: {list(self.teacher_processors.keys())}."
                        )
                    teacher_processor = self.teacher_processors[teacher_key]
                else:
                    teacher_processor = self.teacher_processors.get("default", self.student_processor)
                cur_teacher_tokenizer = getattr(teacher_processor, "tokenizer", teacher_processor)

                tea_chat_template_fn = teacher_processor.apply_chat_template if self.apply_chat_template else None
                tea_prompt, tea_response = self.preprocess_data(
                    data, self.input_template, self.teacher_input_key, self.output_key,
                    apply_chat_template=tea_chat_template_fn,
                )
                tea_eos_token = cur_teacher_tokenizer.eos_token
                if not tea_response.endswith(tea_eos_token):
                    tea_response += " " + tea_eos_token
                result["tea_prompt"] = tea_prompt
                result["tea_response"] = tea_response
                result["tea_resp_len"] = self._compute_token_length(teacher_processor, tea_response)
            else:
                result["tea_prompt"] = stu_prompt
                result["tea_response"] = stu_response
                result["tea_resp_len"] = result["stu_resp_len"]

        # load teacher routing info of each data for multi-teacher distillation
        if self.args.kd.multi_teacher_config is not None:
            assert self.args.data.teacher_routing_key is not None, "`--teacher_routing_key` must be specified when using multi_teacher_config"
            assert self.args.data.teacher_routing_key in data, f"Routing key '{self.args.data.teacher_routing_key}' not found in data"
            result["teacher_routing_key"] = data[self.args.data.teacher_routing_key]

        return result
    
    def preprocess_data(
        self,
        data: Dict,
        input_template: Optional[str] = None,
        input_key: str = "input",
        output_key: Optional[str] = None,
        apply_chat_template: Optional[Callable] = None,
    ) -> tuple:
        if not apply_chat_template:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
            assert output_key is not None
            return prompt, data[output_key]

        has_image = self.image_key is not None
        if output_key:
            messages = convert_to_openai_messages(data[input_key], expand_image=has_image) + \
                convert_to_openai_messages(data[output_key], role="assistant", expand_image=has_image)
        else:
            messages = convert_to_openai_messages(data[input_key], expand_image=has_image)

        chat_template_kwargs = {}
        if "enable_thinking" in str(apply_chat_template.__self__.chat_template):
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
        prompt = apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs,
        )
        full_text = apply_chat_template(messages, tokenize=False, **chat_template_kwargs)
        response = full_text[len(prompt):].removeprefix("<think>\n\n</think>\n\n").rstrip()
        return prompt, response

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.processed_dataset[idx]

    def load_images(self, image_content):
        """Load image(s) from various input formats. Always returns a list."""
        if isinstance(image_content, Image.Image):
            return [image_content]
        if isinstance(image_content, str):
            return [Image.open(image_content).convert("RGB")]
        if isinstance(image_content, list):
            return [self._load_single_image(img) for img in image_content]
        return []

    @staticmethod
    def _load_single_image(img):
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        return None

    def _compute_token_length(self, processor, text):
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _encode_single(self, processor, text, images=None) -> Dict[str, Any]:
        """Single-sample processor call."""
        kwargs = {"text": [text], "padding": False, "truncation": False,
                  "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        enc = processor(**kwargs)
        skip = {"input_ids", "attention_mask", "mm_token_type_ids"}
        multi_modal_inputs = {k: v for k, v in enc.items() if k not in skip}
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "multi_modal_inputs": multi_modal_inputs or None,
        }

    @staticmethod
    def _build_loss_mask(attn_mask, resp_lens):
        bs, seq_len = attn_mask.shape
        loss_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
        for i in range(bs):
            real_end = attn_mask[i].sum().item()
            start = max(real_end - resp_lens[i] - 1, 0)
            loss_mask[i, start:real_end - 1] = True
        return loss_mask

    def collate_fn(self, item_list: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(item_list)
        per_sample_images = (
            [item["images"] for item in item_list]
            if self.image_key else [None] * bsz
        )

        stu_full = [item["stu_prompt"] + item["stu_response"] for item in item_list]
        stu_encs = [
            self._encode_single(self.student_processor, stu_full[i], per_sample_images[i])
            for i in range(bsz)
        ]
        stu_pad_id = self.student_tokenizer.pad_token_id
        stu_input_ids = zero_pad_sequences(
            [e["input_ids"] for e in stu_encs], side="right", value=stu_pad_id,
        )
        stu_attn_mask = zero_pad_sequences(
            [e["attention_mask"] for e in stu_encs], side="right", value=0,
        ).long()
        stu_loss_mask = self._build_loss_mask(
            stu_attn_mask, [item["stu_resp_len"] for item in item_list],
        )

        batch: Dict[str, Any] = {
            "stu_input_ids": stu_input_ids,
            "stu_attn_mask": stu_attn_mask,
            "stu_loss_mask": stu_loss_mask,
        }
        if any(e["multi_modal_inputs"] is not None for e in stu_encs):
            batch["stu_multi_modal_inputs"] = [e["multi_modal_inputs"] for e in stu_encs]

        if "tea_prompt" in item_list[0]:
            if not self.teacher_student_share_input:
                tea_full = [item["tea_prompt"] + item["tea_response"] for item in item_list]
                tea_encs = []
                for i in range(bsz):
                    if self.args.kd.multi_teacher_config and "teacher_routing_key" in item_list[i]:
                        proc = self.teacher_processors[item_list[i]["teacher_routing_key"]]
                    else:
                        proc = self.teacher_processors.get("default", self.student_processor)
                    tokenizer = getattr(proc, "tokenizer", proc)
                    tea_encs.append(self._encode_single(tokenizer, tea_full[i]))
                if self.args.kd.multi_teacher_config:
                    tea_pad_id = self.student_tokenizer.pad_token_id
                else:
                    teacher_processor = self.teacher_processors.get("default", self.student_processor)
                    teacher_tokenizer = getattr(teacher_processor, "tokenizer", teacher_processor)
                    tea_pad_id = teacher_tokenizer.pad_token_id
            else:
                if self.image_key:
                    tea_encs = [
                        self._encode_single(self.student_tokenizer, text)
                        for text in stu_full
                    ]
                else:
                    tea_encs = stu_encs
                tea_pad_id = self.student_tokenizer.pad_token_id

            tea_input_ids = zero_pad_sequences(
                [e["input_ids"] for e in tea_encs], side="right", value=tea_pad_id,
            )
            tea_attn_mask = zero_pad_sequences(
                [e["attention_mask"] for e in tea_encs], side="right", value=0,
            ).long()
            batch["tea_input_ids"] = tea_input_ids
            batch["tea_attn_mask"] = tea_attn_mask
            batch["tea_loss_mask"] = self._build_loss_mask(
                tea_attn_mask, [item["tea_resp_len"] for item in item_list],
            )

        if self.image_key:
            batch["images"] = per_sample_images
            
        if "teacher_routing_key" in item_list[0]:
            batch["teacher_routing_key"] = [item["teacher_routing_key"] for item in item_list]

        return batch
