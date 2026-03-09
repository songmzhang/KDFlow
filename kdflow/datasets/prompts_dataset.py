from typing import Callable, Optional, Dict, Any, List

from torch.utils.data import Dataset

from kdflow.datasets.utils import convert_to_openai_messages
from kdflow.models.utils import TokenizerCompareResult


class PromptDataset(Dataset):
    """
    Dataset for On-Policy Distillation

    Args:
        dataset: dataset for on-policy distillation
        student_tokenizer: tokenizer for student model
        strategy: training strategy object
        teacher_tokenizer: optional tokenizer for teacher model
        tokenizer_info: result of tokenizer comparison (template_identical, vocab_identical)
        input_template: optional template for formatting input
        num_processors: number of processors for parallel data loading
    """

    def __init__(
        self,
        dataset,
        student_tokenizer: Callable,
        strategy,
        teacher_tokenizer: Optional[Callable] = None,
        tokenizer_info: Optional[TokenizerCompareResult] = None,
        max_data_num: int = None,
        input_template: Optional[str] = None,
        num_processors: int = 8,
    ) -> None:
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer if teacher_tokenizer else student_tokenizer
        self.tokenizer_info = tokenizer_info or TokenizerCompareResult()
        self.template_identical = self.tokenizer_info.template_identical
        self.vocab_identical = self.tokenizer_info.vocab_identical
        self.strategy = strategy
        self.input_template = input_template
        
        # For backward compatibility
        self.tokenizer = student_tokenizer
        
        # Config from strategy
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.teacher_input_key = getattr(self.strategy.args.data, "teacher_input_key", None) or self.input_key
        self.label_key = getattr(self.strategy.args.data, "label_key", None)
        self.apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)
        self.image_key = getattr(self.strategy.args.data, "image_key", None)
        self.enable_thinking = self.strategy.args.model.enable_thinking

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
        
        self._print_sample()

    def _print_sample(self) -> None:
        """Print sample data for debugging."""
        self.strategy.print(f"Total samples: {len(self.processed_dataset)}")
        self.strategy.print(f"Sample student prompt:\n{self.processed_dataset[0]['stu_prompt']}")
        if not self.template_identical:
            self.strategy.print(f"Sample teacher prompt:\n{self.processed_dataset[0]['tea_prompt']}")

    def _build_prompt(self, data: Dict, tokenizer: Callable, input_key: str) -> str:
        """Build prompt from data with optional chat template or input template.
        
        Args:
            data: The data dict containing input
            tokenizer: The tokenizer to use for apply_chat_template
            input_key: The key to extract input from data
            
        Returns:
            Formatted prompt string
        """
        if self.apply_chat_template:
            chat = convert_to_openai_messages(data[input_key])
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
        
        prompt = data[input_key]
        return self.input_template.format(prompt) if self.input_template else prompt

    def process_data(self, data: Dict) -> Dict[str, Any]:
        """Process a single data sample."""
        # Build student prompt
        stu_prompt = self._build_prompt(data, self.student_tokenizer, self.input_key)
        
        # Build teacher prompt
        # Use different prompt if: different template, different input_key (self-distillation)
        if self.template_identical and self.input_key == self.teacher_input_key:
            tea_prompt = stu_prompt
        else:
            tea_prompt = self._build_prompt(data, self.teacher_tokenizer, self.teacher_input_key)
        
        result = {
            "stu_prompt": stu_prompt,
            "tea_prompt": tea_prompt,
            "prompt": stu_prompt,
            "label": data.get(self.label_key, "") if self.label_key else "",
            "datasource": data.get("datasource", "default"),
        }
        if self.image_key:
            image_paths = data.get(self.image_key)
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            result["image_paths"] = image_paths or []
        return result

    def __len__(self) -> int:
        return len(self.processed_dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get item by index.
        
        Returns:
            Dict with keys: datasource, stu_prompt, tea_prompt, label
        """
        item = self.processed_dataset[idx]
        result = {
            "datasource": item["datasource"],
            "stu_prompt": item["stu_prompt"],
            "tea_prompt": item["tea_prompt"],
            "label": item["label"],
        }
        if "image_paths" in item:
            result["image_paths"] = item["image_paths"]
        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Collate function that simply returns the list of dicts.
        
        DataLoader will pass a list of dicts, we just return it as-is
        since rollout method expects a list of dicts.
        """
        return batch
