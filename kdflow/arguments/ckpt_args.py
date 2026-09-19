from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CheckpointArguments:
    """Arguments for checkpoint saving and restoration."""

    save_path: str = field(
        default="./ckpt/",
        metadata={"help": "Root directory for checkpoints."}
    )
    save_training_state: bool = field(
        default=False,
        metadata={"help": "Save training state in training_state/ for resuming training."}
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint directory under save_path to resume from; enables resume_training."}
    )
    resume_training: bool = field(
        default=False,
        metadata={"help": "Resume from resume_from or the latest checkpoint in save_path."}
    )
    save_steps: int = field(
        default=-1,
        metadata={"help": "Save every N steps (rollout iterations for on-policy KD). -1 disables periodic saving."}
    )
    max_ckpt_num: int = field(
        default=-1,
        metadata={"help": "Maximum checkpoints to retain; -1 keeps all."}
    )

    def __post_init__(self):
        if not self.save_path.strip():
            raise ValueError("save_path must not be empty.")
        if self.resume_from is not None:
            self.resume_training = True
            if Path(self.resume_from).resolve().parent != Path(self.save_path).resolve():
                raise ValueError("--resume_from must point to a checkpoint directly inside --save_path.")
        if self.save_steps == -1:
            self.save_steps = float("inf")
        elif self.save_steps <= 0:
            raise ValueError("Set --save_steps to a positive integer, or -1 to disable periodic saving.")
        if self.max_ckpt_num != -1 and self.max_ckpt_num <= 0:
            raise ValueError("max_ckpt_num must be -1 (keep all) or a positive integer.")
