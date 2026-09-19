"""Run in the KDFlow training environment with two GPUs:

torchrun --standalone --nproc_per_node=2 -m unittest discover -s tests -p test_fsdp_checkpoint.py
"""

import os
import random
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from kdflow.arguments import AllArguments
from kdflow.backend.fsdp import FSDP2Strategy, checkpoint
from kdflow.models import DistillModel


def local_copy(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu().clone()


class CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        directory = [tempfile.mkdtemp(prefix=".test-checkpoint-", dir=Path.cwd()) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(directory, src=0)
        cls.root = Path(directory[0])
        cls.pretrained = cls.root / "initial_model"
        if dist.get_rank() == 0:
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=32, attention_dropout=0.0,
                bos_token_id=1, eos_token_id=2, pad_token_id=0,
            ))
            model.save_pretrained(cls.pretrained)
            vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
            vocabulary.update({f"token_{i}": i for i in range(4, 32)})
            PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
                pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
            ).save_pretrained(cls.pretrained)
        dist.barrier()

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(cls.root)
        dist.destroy_process_group()

    def setUp(self):
        self.save_path = self.root / self._testMethodName
        self.prepare_training()

    def prepare_training(self):
        args = AllArguments()
        args.model.student_name_or_path = str(self.pretrained)
        args.model.attn_implementation = "eager"
        args.data.packing_samples = False
        args.train.bf16 = False
        args.ckpt.save_path = str(self.save_path)
        args.ckpt.save_training_state = True
        args.ckpt.max_ckpt_num = 2
        self.strategy = FSDP2Strategy(
            args=args, bf16=False, micro_train_batch_size=1, train_batch_size=dist.get_world_size(),
        )
        self.strategy.setup_distributed()
        self.model = self.strategy.prepare(DistillModel(self.strategy))
        self.optimizer = self.strategy.create_optimizer(self.model, lr=1e-3, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.8)
        data = torch.arange(dist.get_world_size() * 4 * 8).reshape(-1, 8) % 32
        self.loader = self.strategy.setup_dataloader(data, batch_size=1, shuffle=True)
        self.loader.sampler.set_epoch(0)

    def update(self, batch):
        self.model.train()
        tokens = batch.cuda()
        hidden = self.model(tokens, attention_mask=torch.ones_like(tokens))["hidden_states"][-1]
        self.strategy.backward(hidden.float().sin().mean(), self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

    def save(self, step):
        return self.strategy.save_checkpoint(
            self.model, 1, step, optimizer=self.optimizer, scheduler=self.scheduler,
            trainer_state={"data_loader_state_dict": self.loader.state_dict()},
            extra_state={"rank": dist.get_rank()},
        )

    def load(self, path):
        state = self.strategy.load_checkpoint(
            self.model, path, optimizer=self.optimizer, scheduler=self.scheduler,
        )
        self.loader.load_state_dict(state["trainer_state"]["data_loader_state_dict"])
        return state

    def weights(self):
        return {name: local_copy(param) for name, param in self.model.named_parameters()}

    def test_resume_matches_next_update(self):
        iterator = iter(self.loader)
        self.update(next(iterator))
        random.seed(100 + dist.get_rank())
        np.random.seed(100 + dist.get_rank())
        torch.manual_seed(100 + dist.get_rank())
        saved = self.save(1)
        expected_weights = self.weights()
        expected_moments = {
            name: {key: local_copy(value) for key, value in self.optimizer.state[param].items()}
            for name, param in self.model.named_parameters() if param in self.optimizer.state
        }
        expected_rng = (random.random(), np.random.random(), torch.rand(3), torch.rand(3, device="cuda"))
        expected_scheduler = self.scheduler.state_dict()
        next_batch = next(iterator)
        self.update(next_batch)
        expected_next_weights = self.weights()

        self.prepare_training()
        state = self.load(saved)
        self.assertEqual((state["epoch"], state["global_step"], state["epoch_end"]), (1, 1, False))
        self.assertEqual(state["extra_state"], {"rank": dist.get_rank()})
        self.assertEqual(self.scheduler.state_dict(), expected_scheduler)
        self.assertEqual(random.random(), expected_rng[0])
        self.assertEqual(np.random.random(), expected_rng[1])
        torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(3, device="cuda"), expected_rng[3], rtol=0, atol=0)
        for name, param in self.model.named_parameters():
            torch.testing.assert_close(local_copy(param), expected_weights[name], rtol=0, atol=0)
            for key, expected in expected_moments.get(name, {}).items():
                torch.testing.assert_close(local_copy(self.optimizer.state[param][key]), expected, rtol=0, atol=0)
        restored_batch = next(iter(self.loader))
        torch.testing.assert_close(restored_batch, next_batch, rtol=0, atol=0)
        self.update(restored_batch)
        for name, value in self.weights().items():
            torch.testing.assert_close(value, expected_next_weights[name], rtol=1e-5, atol=1e-6)

    def test_retention_overwrite_and_final_export(self):
        old = self.save(99)
        self.prepare_training()
        first, second = self.save(1), self.save(2)
        self.load(first)
        self.assertEqual(self.save(2), second)
        third = self.save(3)
        self.assertFalse(first.exists())
        self.assertTrue(old.exists())
        self.assertEqual(self.strategy.checkpoint_names, [second.name, third.name])
        self.assertEqual(checkpoint.resolve_resume_checkpoint(self.save_path, resume_training=True), third)
        contents = {path: path.read_bytes() for path in self.save_path.rglob("*") if path.is_file()}
        self.strategy.save_model(self.model, self.save_path)
        for path, content in contents.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.strategy.checkpoint_names, [second.name, third.name])
        dist.barrier()
        if dist.get_rank() == 0:
            exported = LlamaForCausalLM.from_pretrained(self.save_path, local_files_only=True)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(self.save_path, local_files_only=True)
            self.assertEqual(exported.config.vocab_size, 32)
            self.assertEqual(len(tokenizer), 32)


if __name__ == "__main__":
    unittest.main()
