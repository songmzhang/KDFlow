import json
import os
from typing import Any, Dict, List, Optional

import torch

from kdflow.datasets.utils import get_tokenizer_or_processor
from kdflow.utils.utils import zero_pad_sequences


class RolloutDataProcessor:
    """Convert raw rollout outputs into training micro-batches."""

    def __init__(self, strategy, is_same_tokenizer: bool):
        self.args = strategy.args
        self.is_same_tokenizer = is_same_tokenizer
        self.image_key = getattr(self.args.data, "image_key", None)

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
        elif self.args.model.teacher_name_or_path and not self.is_same_tokenizer:
            self.teacher_processors["default"] = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )

    def process(
        self,
        stu_prompts: List[str],
        tea_prompts: List[str],
        outputs: List[Dict[str, Any]],
        labels: List[Any],
        sampling_params: Dict[str, Any],
        global_step: int,
        mode: str,
        images: Optional[List] = None,
        teacher_routing_keys: Optional[List] = None,
    ) -> tuple[List[dict], Dict[str, float]]:
        """Save, tokenize and collate raw rollout outputs."""
        self._save_rollout_data(stu_prompts, outputs, labels, global_step, mode)

        sample_list = [
            self._build_rollout_sample(
                stu_prompt=stu_prompts[index],
                tea_prompt=tea_prompts[index],
                output=output,
                label=labels[index],
                images=images[index] if images and images[index] else None,
                teacher_routing_key=(
                    teacher_routing_keys[index]
                    if teacher_routing_keys and teacher_routing_keys[index]
                    else None
                ),
            )
            for index, output in enumerate(outputs)
        ]
        rollout_metrics = self._compute_rollout_metrics(sample_list, sampling_params, mode)

        if mode == "train" and self.args.rollout.print_rollout_sample:
            print(sample_list[0]["stu_prompts"][0] + sample_list[0]["stu_responses"][0])

        micro_batches = self._collate_micro_batches(
            sample_list, self.args.train.micro_train_batch_size
        )
        return micro_batches, rollout_metrics

    def _save_rollout_data(
        self,
        prompts: List[str],
        outputs: List[Dict[str, Any]],
        labels: List[Any],
        global_step: int,
        mode: str,
    ) -> None:
        rollout_dir = os.path.join(self.args.train.save_path, "rollout_data")
        if mode == "eval":
            rollout_dir = os.path.join(rollout_dir, "val")
        os.makedirs(rollout_dir, exist_ok=True)

        rollout_path = os.path.join(rollout_dir, f"{global_step}.jsonl")
        with open(rollout_path, "w") as file:
            for prompt, output, label in zip(prompts, outputs, labels):
                rollout_record = {"prompt": prompt, "output": output["text"]}
                if "reward_result" in output:
                    rollout_record["reward_result"] = output["reward_result"]
                if label is not None and label != "":
                    rollout_record["label"] = label
                file.write(json.dumps(rollout_record, ensure_ascii=False) + "\n")

    @staticmethod
    def _compute_rollout_metrics(
        sample_list: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
        mode: str,
    ) -> Dict[str, float]:
        length_names = ("prompt_length", "response_length", "total_length")
        length_values = {
            name: [sample.pop(name).item() for sample in sample_list]
            for name in length_names
        }
        metric_prefix = "rollout" if mode == "train" else "eval"
        rollout_metrics = {}
        for name, values in length_values.items():
            rollout_metrics[f"{metric_prefix}/{name}/mean"] = sum(values) / len(values)
            rollout_metrics[f"{metric_prefix}/{name}/max"] = max(values)

        max_response_length = sampling_params["max_new_tokens"]
        response_lengths = length_values["response_length"]
        rollout_metrics[f"{metric_prefix}/response_clip_ratio"] = sum(
            length >= max_response_length for length in response_lengths
        ) / len(response_lengths)
        return rollout_metrics

    @staticmethod
    def _collate_values(values: list):
        value = values[0]
        if isinstance(value, torch.Tensor):
            return zero_pad_sequences(values, side="right", value=0)
        if isinstance(value, list):
            return sum(values, [])
        if value is None:
            return None
        return values

    def _collate_micro_batches(
        self, sample_list: List[Dict], batch_size: int
    ) -> List[Dict]:
        """Collate single samples into micro-batches."""
        micro_batch_list = []
        for index in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[index : index + batch_size]
            micro_batch = {
                key: self._collate_values([sample[key] for sample in batch_samples])
                for key in batch_samples[0]
            }
            micro_batch_list.append(micro_batch)
        return micro_batch_list

    def _tokenize_sample(
        self,
        prompt: str,
        response: str,
        processor,
        prefix: str,
        images=None,
        response_ids: Optional[List[int]] = None,
        append_eos: bool = False,
    ) -> Dict[str, Any]:
        """Tokenize prompt and response for a single sample."""
        tokenizer = getattr(processor, "tokenizer", processor)
        if response_ids is not None:
            model_input = {"text": prompt}
            if images:
                model_input["images"] = images
            tokens = processor(
                **model_input, return_tensors="pt", add_special_tokens=False
            )
            prompt_length = tokens["input_ids"].shape[1]
            response_input_ids = tokens["input_ids"].new_tensor(response_ids)
            input_ids = torch.cat(
                [tokens["input_ids"][0], response_input_ids]
            )
            attention_mask = torch.cat(
                [
                    tokens["attention_mask"][0],
                    tokens["attention_mask"][0].new_ones(len(response_ids)),
                ]
            )
            target_length = len(response_ids)
        else:
            response_tokens = tokenizer(
                response, return_tensors="pt", add_special_tokens=False
            )
            response_length = response_tokens["input_ids"].shape[1]
            model_input = {"text": prompt + response}
            if images:
                model_input["images"] = images
            tokens = processor(
                **model_input, return_tensors="pt", add_special_tokens=False
            )
            prompt_length = tokens["input_ids"].shape[1] - response_length
            input_ids = tokens["input_ids"][0]
            attention_mask = tokens["attention_mask"][0]
            target_length = response_length
            if append_eos:
                input_ids = torch.cat(
                    [input_ids, input_ids.new_tensor([tokenizer.eos_token_id])]
                )
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(1)]
                )
                target_length += 1

        loss_mask = torch.tensor(
            [False] * (prompt_length - 1)
            + [True] * target_length
            + [False],
            device=input_ids.device,
        )

        result = {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attention_mask,
            f"{prefix}_loss_mask": loss_mask,
            f"_{prefix}_prompt_length": prompt_length,
        }
        multi_modal_inputs = {
            key: torch.as_tensor(value)
            for key, value in tokens.items()
            if key not in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        if multi_modal_inputs:
            result[f"_{prefix}_multi_modal_inputs"] = multi_modal_inputs
        return result

    def _get_teacher_processor(self, teacher_routing_key=None):
        """Get the teacher processor for the given routing key."""
        if teacher_routing_key and teacher_routing_key in self.teacher_processors:
            return self.teacher_processors[teacher_routing_key]
        if "default" in self.teacher_processors:
            return self.teacher_processors["default"]
        return self.student_processor

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output: Dict[str, Any],
        label: Any,
        images=None,
        teacher_routing_key=None,
    ) -> Dict[str, Any]:
        """Build a rollout sample with student and teacher tokenizations."""
        response_ids = output["output_ids"]
        response_text = output["text"]
        output_token_logprobs = output["meta_info"]["output_token_logprobs"]
        output_log_probs = [item[0] for item in output_token_logprobs]
        output_logprob_ids = [item[1] for item in output_token_logprobs]
        student_tokenizer = getattr(
            self.student_processor, "tokenizer", self.student_processor
        )
        response_has_eos = (
            bool(response_ids) and response_ids[-1] == student_tokenizer.eos_token_id
        )

        stu_tokens = self._tokenize_sample(
            stu_prompt,
            response_text,
            self.student_processor,
            "stu",
            images=images,
            response_ids=response_ids,
        )
        stu_loss_mask = stu_tokens["stu_loss_mask"].bool()
        stu_label_ids = stu_tokens["stu_input_ids"].roll(shifts=-1)[
            stu_loss_mask
        ]
        if (
            output_logprob_ids != response_ids
            or output_logprob_ids != stu_label_ids.tolist()
        ):
            raise ValueError("Rollout and student token IDs are not aligned")

        rollout_log_probs = torch.zeros_like(
            stu_tokens["stu_input_ids"], dtype=torch.float32
        )
        rollout_log_probs[stu_loss_mask] = torch.tensor(
            output_log_probs, dtype=torch.float32
        )

        teacher_processor = self._get_teacher_processor(teacher_routing_key)
        teacher_tokenizer = getattr(
            teacher_processor, "tokenizer", teacher_processor
        )
        tea_tokens = self._tokenize_sample(
            tea_prompt,
            response_text,
            teacher_tokenizer,
            "tea",
            response_ids=response_ids if self.is_same_tokenizer else None,
            append_eos=response_has_eos,
        )

        prompt_length = stu_tokens["_stu_prompt_length"]
        response_length = len(response_ids)
        total_length = stu_tokens["stu_attn_mask"].sum().item()

        sample = {
            **{key: value for key, value in tea_tokens.items() if not key.startswith("_")},
            **{key: value for key, value in stu_tokens.items() if not key.startswith("_")},
            "rollout_log_probs": rollout_log_probs,
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "prompt_length": torch.FloatTensor([[prompt_length]]),
            "response_length": torch.FloatTensor([[response_length]]),
            "total_length": torch.FloatTensor([[total_length]]),
        }
        stu_multi_modal_inputs = stu_tokens.get("_stu_multi_modal_inputs")
        if stu_multi_modal_inputs is not None:
            sample["stu_multi_modal_inputs"] = [stu_multi_modal_inputs]
        if images:
            sample["images"] = [images]
        if teacher_routing_key is not None:
            sample["teacher_routing_key"] = teacher_routing_key
        return sample
