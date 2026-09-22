from typing import Optional, Dict, Any, List

from torch.utils.data import Dataset

from kdflow.datasets.utils import (
    convert_to_openai_messages,
    get_tokenizer_or_processor,
    validate_dataset_columns,
)
from kdflow.models.utils import TokenizerCompareResult


class PromptDataset(Dataset):
    """
    Dataset for On-Policy Distillation

    Args:
        dataset: dataset for on-policy distillation
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
        tokenizer_info,
        max_data_num: int = None,
        input_template: Optional[str] = None,
        num_processors: int = 8,
    ) -> None:
        super().__init__()
        self.args = strategy.args
        self.strategy = strategy
        self.tokenizer_info = tokenizer_info or TokenizerCompareResult()
        self.template_identical = self.tokenizer_info.template_identical
        self.vocab_identical = self.tokenizer_info.vocab_identical
        self.same_tokenizer = self.tokenizer_info.is_identical
        self.input_template = input_template
        
        # Config from strategy
        self.input_key = getattr(self.args.data, "input_key", None)
        self.teacher_input_key = getattr(self.args.data, "teacher_input_key", None) or self.input_key
        self.label_key = getattr(self.args.data, "label_key", None)
        self.apply_chat_template = getattr(self.args.data, "apply_chat_template", False)
        self.enable_thinking = getattr(self.args.model, "enable_thinking", False)
        self.prompt_max_len = getattr(self.args.data, "prompt_max_len", 0)

        self.image_key = getattr(self.args.data, "image_key", None)

        # Validate that all required columns exist in the dataset
        required_columns = {"input_key": self.input_key}
        if self.teacher_input_key != self.input_key:
            required_columns["teacher_input_key"] = self.teacher_input_key
        if self.label_key:
            required_columns["label_key"] = self.label_key
        if self.args.kd.multi_teacher_config:
            required_columns["teacher_routing_key"] = self.args.data.teacher_routing_key
        validate_dataset_columns(dataset, **required_columns)

        # Load processor if multimodal
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )
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

        # Truncate dataset if max_data_num is specified
        if max_data_num is not None and max_data_num > 0 and max_data_num < len(dataset):
            strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        num_processors = 1
        strategy.log(
            "Set num_processors to 1 for faster processing.",
            level="warning",
        )
        self.processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors if num_processors > 1 else None,
            load_from_cache_file=False,
            desc="Processing data",
        )
        self.processed_dataset = self.processed_dataset.map(
            self._compute_prompt_lengths,
            input_columns=["stu_prompt"],
            batched=True,
            batch_size=512,
            load_from_cache_file=False,
            desc="Computing prompt lengths",
        )
        if self.prompt_max_len > 0:
            original_len = len(self.processed_dataset)
            self.processed_dataset = self.processed_dataset.filter(
                lambda prompt_len: prompt_len <= self.prompt_max_len,
                input_columns=["prompt_len"],
                load_from_cache_file=False,
                desc="Filtering long prompts",
            )
            filtered_count = original_len - len(self.processed_dataset)
            strategy.log(f"Filtered {filtered_count} samples exceeding prompt_max_len={self.prompt_max_len}.")

        self._print_sample()

    def _print_sample(self) -> None:
        """Print sample data for debugging."""
        self.strategy.print(f"Total samples: {len(self.processed_dataset)}")
        if len(self.processed_dataset) == 0:
            return
        self.strategy.print(f"Sample student prompt:\n{self.processed_dataset[0]['stu_prompt']}")
        if not self.template_identical or self.teacher_input_key != self.input_key:
            self.strategy.print(f"Sample teacher prompt:\n{self.processed_dataset[0]['tea_prompt']}")

    def process_data(self, data: Dict) -> Dict[str, Any]:
        """Build prompts for each sample and keep raw image paths."""
        stu_prompt = self._build_prompt(data, self.student_processor, self.input_key)
        routing_key = self.args.data.teacher_routing_key if self.args.kd.multi_teacher_config else None
        if routing_key:
            teacher_key = data[routing_key]
            if teacher_key not in self.teacher_processors:
                raise ValueError(
                    f"Teacher routing key '{teacher_key}' not found in multi_teacher_config. "
                    f"Available keys: {list(self.teacher_processors)}."
                )
            teacher_processor = self.teacher_processors[teacher_key]
            tea_prompt = self._build_prompt(data, teacher_processor, self.teacher_input_key)
        elif self.same_tokenizer and self.input_key == self.teacher_input_key:
            tea_prompt = stu_prompt
        else:
            teacher_processor = self.teacher_processors.get("default", self.student_processor)
            tea_prompt = self._build_prompt(data, teacher_processor, self.teacher_input_key)

        result = {
            "stu_prompt": stu_prompt,
            "tea_prompt": tea_prompt,
            "label": data.get(self.label_key, "") if self.label_key else "",
            "datasource": data.get("datasource", "default"),
        }
        if self.image_key:
            images = data.get(self.image_key) or []
            result["images"] = [images] if isinstance(images, str) else images
        if routing_key:
            result["teacher_routing_key"] = teacher_key
        return result

    def _compute_prompt_lengths(self, prompts: List[str]) -> Dict[str, List[int]]:
        """Compute the prompt_len column with batch-level tokenization."""
        tokenizer = getattr(self.student_processor, "tokenizer", self.student_processor)
        encoded = tokenizer(
            prompts, add_special_tokens=True, padding=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )
        return {"prompt_len": [len(input_ids) for input_ids in encoded["input_ids"]]}

    def _build_prompt(self, data: Dict, processor_or_tokenizer, input_key: str) -> str:
        """Build prompt from data with optional chat template or input template.
        
        Args:
            data: The data dict containing input
            processor_or_tokenizer: The processor or tokenizer to use for apply_chat_template
            input_key: The key to extract input from data
            
        Returns:
            Formatted prompt string
        """
        if self.apply_chat_template:
            chat = convert_to_openai_messages(data[input_key], expand_image=self.image_key is not None)
            while chat and chat[-1].get("role", "user") == "assistant":
                chat.pop()
            chat_template_kwargs = {}
            if "enable_thinking" in str(getattr(processor_or_tokenizer, "chat_template", "")):
                chat_template_kwargs["enable_thinking"] = self.enable_thinking
            return processor_or_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        
        prompt = data[input_key]
        return self.input_template.format(prompt) if self.input_template else prompt

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Returns:
            Dict with keys: datasource, stu_prompt, tea_prompt, label, images (optional)
        """
        item = self.processed_dataset[idx]
        result = {
            "datasource": item["datasource"],
            "stu_prompt": item["stu_prompt"],
            "tea_prompt": item["tea_prompt"],
            "label": item["label"],
        }
        if "images" in item:
            result["images"] = item["images"]
        if "teacher_routing_key" in item:
            result["teacher_routing_key"] = item["teacher_routing_key"]
        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collate function that simply returns the list of dicts.
        
        DataLoader will pass a list of dicts, we just return it as-is
        since rollout method expects a list of dicts.
        """
        return batch
