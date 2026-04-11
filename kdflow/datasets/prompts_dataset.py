from typing import Callable, Optional, Dict, Any, List

from torch.utils.data import Dataset
from PIL import Image

from kdflow.datasets.utils import convert_to_openai_messages, get_tokenizer_or_processor
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
        tokenizer_info: Optional[TokenizerCompareResult] = None,
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
        self.same_tokenizer = self.template_identical and self.vocab_identical
        self.strategy = strategy
        self.input_template = input_template
        
        # Config from strategy
        self.input_key = getattr(self.args.data, "input_key", None)
        self.teacher_input_key = getattr(self.args.data, "teacher_input_key", None) or self.input_key
        self.label_key = getattr(self.args.data, "label_key", None)
        self.apply_chat_template = getattr(self.args.data, "apply_chat_template", False)
        self.enable_thinking = getattr(self.args.model, "enable_thinking", False)
        self.prompt_max_len = getattr(self.args.data, "prompt_max_len", 0)

        self.image_key = getattr(self.args.data, "image_key", None)
        self.same_tokenizer = self.template_identical and self.vocab_identical

        # Load processor if multimodal
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

        # Truncate dataset if max_data_num is specified
        if max_data_num > 0 and max_data_num < len(dataset):
            strategy.log(f"Truncating dataset from {len(dataset)} to {max_data_num}")
            dataset = dataset.select(range(max_data_num))

        # Parallel loading datasets
        self.processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
            load_from_cache_file=False,
            desc="Processing data",
        )

        # Filter by prompt_max_len
        original_len = len(self.processed_dataset)
        if self.prompt_max_len > 0:
            strategy.log(f"Before prompt_max_len filter: {len(self.processed_dataset)}")
            self.processed_dataset = self.processed_dataset.filter(
                lambda x: x["prompt_len"] <= self.prompt_max_len,
                num_proc=num_processors,
                desc="Filtering overlang samples",
            )
            strategy.log(f"After prompt_max_len filter: {len(self.processed_dataset)}")
            if len(self.processed_dataset) < original_len:
                self.strategy.log(
                    f"Filtered {original_len - len(self.processed_dataset)} samples "
                    f"exceeding prompt_max_len={self.prompt_max_len} "
                    f"({original_len} -> {len(self.processed_dataset)})"
                )
                
        self._print_sample()

    def _print_sample(self) -> None:
        """Print sample data for debugging."""
        self.strategy.print(f"Total samples: {len(self.processed_dataset)}")
        self.strategy.print(f"Sample student prompt:\n{self.processed_dataset[0]['stu_prompt']}")
        if not self.template_identical or self.teacher_input_key != self.input_key:
            self.strategy.print(f"Sample teacher prompt:\n{self.processed_dataset[0]['tea_prompt']}")

    def process_data(self, data: Dict) -> Dict[str, Any]:
        """Process a single data sample."""
        # Build student prompt
        stu_prompt = self._build_prompt(data, self.student_processor, self.input_key)
        
        # Build teacher prompt
        if self.same_tokenizer and self.input_key == self.teacher_input_key:
            tea_prompt = stu_prompt
        else:
            tea_prompt = self._build_prompt(data, self.teacher_processor, self.teacher_input_key)
        
        # Compute prompt token length for filtering
        tokenizer = self.student_processor.tokenizer if hasattr(self.student_processor, "tokenizer") else self.student_processor
        prompt_len = len(tokenizer.encode(stu_prompt))

        result = {
            "stu_prompt": stu_prompt,
            "tea_prompt": tea_prompt,
            "prompt": stu_prompt,
            "prompt_len": prompt_len,
            "label": data.get(self.label_key, "") if self.label_key else "",
            "datasource": data.get("datasource", "default"),
        }
        # Load images if multimodal
        if self.image_key:
            result["images"] = self._load_images(data.get(self.image_key))
        return result
    
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
            return processor_or_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        
        prompt = data[input_key]
        return self.input_template.format(prompt) if self.input_template else prompt

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
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
        return result

    @staticmethod
    def _load_images(image_content) -> list:
        """Load image(s) from various input formats. Always returns a list."""
        if image_content is None:
            return []
        if isinstance(image_content, Image.Image):
            return [image_content]
        if isinstance(image_content, str):
            return [Image.open(image_content).convert("RGB")]
        if isinstance(image_content, list):
            result = []
            for img in image_content:
                if isinstance(img, Image.Image):
                    result.append(img)
                elif isinstance(img, str):
                    result.append(Image.open(img).convert("RGB"))
            return result
        return []

    @staticmethod
    def collate_fn(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Collate function that simply returns the list of dicts.
        
        DataLoader will pass a list of dicts, we just return it as-is
        since rollout method expects a list of dicts.
        """
        return batch
