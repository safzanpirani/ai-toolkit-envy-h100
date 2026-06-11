"""TRUE FP8-compute training support (torchao float8, dynamic scaling).

This is fundamentally different from ``model.quantize: true`` (optimum.quanto)
and from the fp8 checkpoints shipped by ideogram-ai: those are *storage-only*
quantization -- weights are dequantized to bf16 before every matmul, so the
GPU's FP8 tensor cores are never used. Here we swap eligible ``nn.Linear``
modules for ``torchao.float8.Float8Linear`` with dynamic (per-tensor, computed
on the fly) scaling, so the actual GEMMs run in float8 via
``torch._scaled_mm`` on Hopper/Ada tensor cores. On an H100 this typically
yields a 1.2-1.5x end-to-end training speedup for large transformers.

Composability with this toolkit's LoRA (``toolkit.lora_special``):

* LoRA here does NOT replace modules; it monkey-patches ``module.forward``
  (see ``LoRAModule.apply_to``) and computes ``org_forward(x) + lora_up(
  lora_down(x)) * scale``. As long as the float8 conversion happens BEFORE
  the network is applied (i.e. inside ``model.load_model()``, which
  ``BaseSDTrainProcess`` calls before constructing ``LoRASpecialNetwork``),
  the captured ``org_forward`` is the Float8Linear forward. Result: the BASE
  weight matmul runs in fp8, while the LoRA A/B matmuls stay in bf16 plain
  ``nn.Linear`` -- exactly what we want for training stability.
* ``Float8Linear`` is registered in ``LINEAR_MODULES`` in
  ``toolkit/lora_special.py`` so converted layers are still matched as LoRA
  targets.

Do NOT combine with ``model.quantize: true``; quanto's ``QLinear`` and
torchao's ``Float8Linear`` are mutually exclusive ways of handling the base
weights.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from toolkit.print import print_acc

# torchao.float8 dynamic-scaling training is stable as of torchao>=0.7;
# this repo pins torchao==0.10.0 in requirements_base.txt which is fine.
MIN_TORCHAO = (0, 7, 0)


def _torchao_version() -> tuple:
    import torchao

    return tuple(int(x) for x in re.findall(r"\d+", torchao.__version__)[:3])


def fp8_compute_available(raise_on_unavailable: bool = False) -> bool:
    """True if this machine can run torchao float8 training (H100/Ada+)."""
    err = None
    if not torch.cuda.is_available():
        err = "fp8_compute requires CUDA; no CUDA device available."
    else:
        major, minor = torch.cuda.get_device_capability()
        # float8 _scaled_mm requires sm89 (Ada) or sm90+ (Hopper: H100/H200).
        if (major, minor) < (8, 9):
            err = (
                f"fp8_compute requires GPU compute capability >= 8.9 "
                f"(Ada/Hopper, e.g. H100); found sm{major}{minor}."
            )
    if err is None:
        try:
            from torchao.float8 import convert_to_float8_training  # noqa: F401
        except ImportError:
            err = (
                "fp8_compute requires torchao with float8 training "
                f"(pip install 'torchao>=0.7.0'; this repo pins 0.10.0)."
            )
        else:
            if _torchao_version() < MIN_TORCHAO:
                err = (
                    "fp8_compute requires torchao>="
                    + ".".join(map(str, MIN_TORCHAO))
                    + " for stable float8 training."
                )
    if err and raise_on_unavailable:
        raise RuntimeError(err)
    if err:
        print_acc(f"fp8_compute unavailable: {err}")
    return err is None


def enable_h100_fast_math():
    """Enable TF32 for the remaining bf16/fp32 matmuls (norm-free win on H100).

    Note: a further easy win on H100 is ``torch.compile`` over the transformer
    (torchao Float8Linear is compile-friendly and gains the most from fused
    scaled-mm epilogues), but it is intentionally NOT enabled here -- wire it
    behind its own flag once the fp8 path is validated.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def convert_to_fp8_compute(
    module: nn.Module,
    include_patterns: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> nn.Module:
    """Swap eligible nn.Linear layers in ``module`` for torchao Float8Linear.

    Eligibility (enforced in ``module_filter_fn``):
      * plain ``nn.Linear`` only (Float8Linear subclasses Linear, but quanto
        QLinear / embeddings / norms are never Linear so are naturally skipped),
      * fully-qualified name matches ``include_patterns`` (if given) and none
        of ``exclude_patterns`` -- callers use this to keep embeddings, final
        projections and other numerically sensitive layers in bf16,
      * in_features and out_features both divisible by 16 (FP8 GEMM alignment
        requirement of ``torch._scaled_mm``).

    Must be called BEFORE LoRA network injection (see module docstring).
    """
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    include_res = [re.compile(p) for p in (include_patterns or [])]
    exclude_res = [re.compile(p) for p in (exclude_patterns or [])]
    converted: List[str] = []
    skipped: List[str] = []

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        if type(mod) is not nn.Linear:
            return False
        if include_res and not any(p.search(fqn) for p in include_res):
            skipped.append(fqn)
            return False
        if any(p.search(fqn) for p in exclude_res):
            skipped.append(fqn)
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            skipped.append(f"{fqn} (dims {mod.in_features}x{mod.out_features} not /16)")
            return False
        converted.append(fqn)
        return True

    # Dynamic scaling for input / weight / grad_output is the torchao default
    # and the most numerically robust recipe (no delayed-scaling history state).
    config = Float8LinearConfig()  # all-dynamic
    convert_to_float8_training(module, config=config, module_filter_fn=module_filter_fn)

    if verbose:
        print_acc(
            f"fp8_compute: converted {len(converted)} nn.Linear layers to "
            f"Float8Linear (dynamic scaling); {len(skipped)} eligible-by-type "
            f"layers kept in bf16"
        )
        if converted:
            print_acc(f"  first/last converted: {converted[0]} ... {converted[-1]}")
    return module
