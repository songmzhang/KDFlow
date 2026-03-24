import importlib
import os
import glob
from functools import partial

LOSS_DICT = {}

def register_loss(name):
    def decorator(fn):
        LOSS_DICT[name] = fn
        return fn
    return decorator

_current_dir = os.path.dirname(os.path.abspath(__file__))
_module_files = glob.glob(os.path.join(_current_dir, "*.py"))

for f in _module_files:
    if f.endswith("__init__.py"):
        continue
    module_name = os.path.basename(f)[:-3]
    importlib.import_module(f"kdflow.loss.{module_name}")


def build_loss_fn(name, args):
    """Build a loss function with args-specific hyperparams pre-bound via functools.partial."""
    fn = LOSS_DICT[name]
    kd = args.kd

    common = {"temperature": kd.kd_temperature}

    extra_params = {
        "jsd":         {"jsd_beta": kd.jsd_beta},
        "skl":         {"skew_lambda": kd.skew_lambda},
        "srkl":        {"skew_lambda": kd.skew_lambda},
        "akl":         {"adaptive_kl_alpha": kd.adaptive_alpha},
        "hrl":         {"hrl_topk": kd.hrl_topk},
    }

    return partial(fn, **common, **extra_params.get(name, {}))
