from typing import Callable, Optional, Dict, List, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from kdflow.datasets.utils import convert_to_openai_messages, get_tokenizer_or_processor
from kdflow.model.utils import TokenizerCompareResult

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
        self.same_tokenizer = self.template_identical and self.vocab_identical
        self.teacher_student_share_input = self.same_tokenizer and (self.teacher_input_key == self.input_key)
        
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path, 
            need_processor=self.image_key is not None,
        )
        self.teacher_processor = None
        if self.args.model.teacher_name_or_path is not None:
            self.teacher_processor = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )

        if max_data_num > 0 and max_data_num < len(dataset):
            self.strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        self.processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
            load_from_cache_file=False,
            desc="Processing data",
        )

        original_len = len(self.processed_dataset)
        strategy.log(f"Before length filter: {len(self.processed_dataset)}")
        self.processed_dataset = self.processed_dataset.filter(
            lambda x: x["stu_input_len"] <= self.max_length,
            num_proc=num_processors,
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
        stu_chat_template_fn = self.student_processor.apply_chat_template
        stu_prompt, stu_response = self.preprocess_data(
            data, self.input_template, self.input_key, self.output_key,
            apply_chat_template=stu_chat_template_fn,
        )
        stu_eos_token = self.get_eos_token(self.student_processor)
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

        if self.args.model.teacher_name_or_path is not None:
            if not self.teacher_student_share_input:
                tea_chat_template_fn = self.teacher_processor.apply_chat_template
                tea_prompt, tea_response = self.preprocess_data(
                    data, self.input_template, self.teacher_input_key, self.output_key,
                    apply_chat_template=tea_chat_template_fn,
                )
                tea_eos_token = self.get_eos_token(self.teacher_processor)
                if not tea_response.endswith(tea_eos_token):
                    tea_response += " " + tea_eos_token
                result["tea_prompt"] = tea_prompt
                result["tea_response"] = tea_response
                result["tea_resp_len"] = self._compute_token_length(
                    self.teacher_processor, tea_response
                )
            else:
                result["tea_prompt"] = stu_prompt
                result["tea_response"] = stu_response
                result["tea_resp_len"] = result["stu_resp_len"]

        return result
    
    def get_eos_token(self, processor):
        return processor.tokenizer.eos_token if hasattr(processor, "tokenizer") else processor.eos_token

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

        prompt = apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        full_text = apply_chat_template(messages, tokenize=False, enable_thinking=self.enable_thinking)
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

    def _encode_batch(self, processor, full_texts, images=None):
        kwargs = {"text": full_texts, "padding": "longest", "truncation": False,
                  "max_length": self.max_length, "return_tensors": "pt"}
        if images is not None:
            kwargs["images"] = images
        return processor(**kwargs)

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
        stu_full = [item["stu_prompt"] + item["stu_response"] for item in item_list]
        images = [item["images"] for item in item_list] if self.image_key else None

        stu_enc = self._encode_batch(self.student_processor, stu_full, images)

        batch = {f"mm_{k}": v for k, v in stu_enc.items() if k not in ("input_ids", "attention_mask")}
        batch["stu_input_ids"] = stu_enc["input_ids"]
        batch["stu_attn_mask"] = stu_enc["attention_mask"]
        batch["stu_loss_mask"] = self._build_loss_mask(
            stu_enc["attention_mask"],
            [item["stu_resp_len"] for item in item_list],
        )

        if "tea_prompt" in item_list[0]:
            if not self.teacher_student_share_input:
                tea_full = [item["tea_prompt"] + item["tea_response"] for item in item_list]
                tea_enc = self._encode_batch(self.teacher_processor, tea_full, images)
                batch["tea_input_ids"] = tea_enc["input_ids"]
                batch["tea_attn_mask"] = tea_enc["attention_mask"]
                batch["tea_loss_mask"] = self._build_loss_mask(
                    tea_enc["attention_mask"],
                    [item["tea_resp_len"] for item in item_list],
                )
            else:
                batch["tea_input_ids"] = batch["stu_input_ids"]
                batch["tea_attn_mask"] = batch["stu_attn_mask"]
                batch["tea_loss_mask"] = batch["stu_loss_mask"]

        if images is not None:
            batch["images"] = images

        if "tea_prompt" in item_list[0]:
            batch["tea_full_texts"] = [
                item["tea_prompt"] + item["tea_response"] for item in item_list
            ]

        return batch
