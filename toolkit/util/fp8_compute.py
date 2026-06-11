"""TRUE low-precision-compute training support (torchao float8 / MX formats).

This is fundamentally different from ``model.quantize: true`` (optimum.quanto)
and from the fp8 checkpoints shipped by ideogram-ai: those are *storage-only*
quantization -- weights are dequantized to bf16 before every matmul, so the
GPU's FP8/FP4 tensor cores are never used. Here we swap eligible ``nn.Linear``
modules for torchao training linears so the actual GEMMs (forward AND both
backward GEMMs) run in low precision on tensor cores.

Recipes (``model_kwargs.fp8_compute_recipe``):

  * ``tensorwise`` (default) -- torchao ``Float8Linear`` with dynamic
    per-tensor scaling. Requires sm89+ (Ada / H100 / B200). ~1.2-1.5x on H100.
  * ``rowwise`` -- ``Float8Linear`` with per-row scaling: slightly slower cast,
    better numerics. sm89+.
  * ``mxfp8`` -- MX block-32 e4m3 via ``MXLinear`` (cuBLAS mx gemm). Requires
    sm100 (Blackwell, e.g. B200). Better numerics than per-tensor fp8 at
    full fp8 speed; the preferred recipe on B200.
  * ``mxfp4`` -- MX block-32 fp4 (e2m1) via ``MXLinear`` (CUTLASS gemm),
    sm100 only. EXPERIMENTAL: ~2x fp8 GEMM throughput on paper, but fp4
    training of a model not trained for it can diverge -- validate loss
    against a bf16 baseline before trusting runs. NOTE: this is *MX* fp4
    (block-32, e8m0 scales). "NVFP4" proper (block-16 e4m3 scales + global
    fp32 scale, the format of ComfyUI nvfp4 checkpoints) only exists in
    torchao as *inference* configs -- there is no NVFP4 training swap in any
    torchao release as of 0.17 -- so mxfp4 is the closest trainable fp4.

Composability with this toolkit's LoRA (``toolkit.lora_special``):

* LoRA here does NOT replace modules; it monkey-patches ``module.forward``
  (see ``LoRAModule.apply_to``) and computes ``org_forward(x) + lora_up(
  lora_down(x)) * scale``. As long as the conversion happens BEFORE the
  network is applied (i.e. inside ``model.load_model()``, which
  ``BaseSDTrainProcess`` calls before constructing ``LoRASpecialNetwork``),
  the captured ``org_forward`` is the Float8Linear/MXLinear forward. Result:
  the BASE weight matmul runs in fp8/fp4, while the LoRA A/B matmuls stay in
  bf16 plain ``nn.Linear`` -- exactly what we want for training stability.
* ``Float8Linear`` and ``MXLinear`` are registered in ``LINEAR_MODULES`` in
  ``toolkit/lora_special.py`` so converted layers are still matched as LoRA
  targets (matching is by class-name string).

Do NOT combine with ``model.quantize: true``; quanto's ``QLinear`` and the
torchao training linears are mutually exclusive ways of handling base weights.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from toolkit.print import print_acc

# torchao.float8 dynamic-scaling training is stable as of torchao>=0.7. The MX
# recipes use the torchao 0.10 prototype API (quantize_ + MXLinearConfig);
# this repo pins torchao==0.10.0 in requirements_base.txt, where both work.
MIN_TORCHAO = (0, 7, 0)

FP8_RECIPES = ("tensorwise", "rowwise")
MX_RECIPES = ("mxfp8", "mxfp4")
ALL_RECIPES = FP8_RECIPES + MX_RECIPES


def _torchao_version() -> tuple:
    import torchao

    return tuple(int(x) for x in re.findall(r"\d+", torchao.__version__)[:3])


def fp8_compute_available(
    raise_on_unavailable: bool = False, recipe: str = "tensorwise"
) -> bool:
    """True if this machine can run the requested torchao low-precision recipe."""
    err = None
    if recipe not in ALL_RECIPES:
        err = f"unknown fp8_compute_recipe '{recipe}'; choose one of {ALL_RECIPES}."
    elif not torch.cuda.is_available():
        err = "fp8_compute requires CUDA; no CUDA device available."
    else:
        cap = torch.cuda.get_device_capability()
        # float8 _scaled_mm requires sm89 (Ada) or sm90+ (Hopper: H100/H200).
        if recipe in FP8_RECIPES and cap < (8, 9):
            err = (
                f"fp8_compute recipe '{recipe}' requires compute capability >= 8.9 "
                f"(Ada/Hopper/Blackwell, e.g. H100/B200); found sm{cap[0]}{cap[1]}."
            )
        # MX (block-scaled) tensor-core gemms require sm100 (Blackwell: B200).
        elif recipe in MX_RECIPES and cap < (10, 0):
            err = (
                f"fp8_compute recipe '{recipe}' requires compute capability >= 10.0 "
                f"(Blackwell, e.g. B200); found sm{cap[0]}{cap[1]}. "
                f"Use 'tensorwise' or 'rowwise' on H100."
            )
    if err is None:
        try:
            if recipe in FP8_RECIPES:
                from torchao.float8 import convert_to_float8_training  # noqa: F401
            else:
                from torchao.prototype.mx_formats.config import (  # noqa: F401
                    MXLinearConfig,
                )
        except ImportError:
            err = (
                f"fp8_compute recipe '{recipe}' requires torchao with the matching "
                "training API (this repo pins torchao==0.10.0, which has both the "
                "float8 and MX training swaps; newer torchao moved/removed the MX "
                "module-swap API, so keep the pin if using mxfp8/mxfp4)."
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
    """Enable TF32 for the remaining bf16/fp32 matmuls (no-risk win on H100/B200).

    Note: a further easy win is ``torch.compile`` over the transformer
    (torchao Float8Linear/MXLinear are compile-friendly and gain the most from
    fused scaled-mm epilogues + cast fusion), but it is intentionally NOT
    enabled here -- wire it behind its own flag once this path is validated.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def convert_to_fp8_compute(
    module: nn.Module,
    include_patterns: Optional[Iterable[str]] = None,
    exclude_patterns: Optional[Iterable[str]] = None,
    recipe: str = "tensorwise",
    verbose: bool = True,
) -> nn.Module:
    """Swap eligible nn.Linear layers in ``module`` for torchao training linears.

    Eligibility (enforced in the filter fn):
      * plain ``nn.Linear`` only (quanto QLinear / embeddings / norms are never
        plain Linear so are naturally skipped),
      * fully-qualified name matches ``include_patterns`` (if given) and none
        of ``exclude_patterns`` -- callers use this to keep embeddings, final
        projections and other numerically sensitive layers in bf16,
      * fp8 recipes: in/out features divisible by 16 (``torch._scaled_mm``
        alignment); mx recipes: in/out features divisible by the MX block size
        (32), and additionally K%128 == 0 for the mxfp4 CUTLASS gemm.

    Must be called BEFORE LoRA network injection (see module docstring).
    """
    if recipe not in ALL_RECIPES:
        raise ValueError(f"unknown recipe '{recipe}'; choose one of {ALL_RECIPES}")

    include_res = [re.compile(p) for p in (include_patterns or [])]
    exclude_res = [re.compile(p) for p in (exclude_patterns or [])]
    converted: List[str] = []
    skipped: List[str] = []

    # FP8 _scaled_mm needs K and N % 16; MX gemms use block-32 scales so both
    # dims must be % 32 (and the fp4 CUTLASS kernel wants K % 128).
    dim_div = 16 if recipe in FP8_RECIPES else 32

    def module_filter_fn(mod: nn.Module, fqn: str) -> bool:
        if type(mod) is not nn.Linear:
            return False
        if include_res and not any(p.search(fqn) for p in include_res):
            skipped.append(fqn)
            return False
        if any(p.search(fqn) for p in exclude_res):
            skipped.append(fqn)
            return False
        if mod.in_features % dim_div != 0 or mod.out_features % dim_div != 0:
            skipped.append(
                f"{fqn} (dims {mod.in_features}x{mod.out_features} not /{dim_div})"
            )
            return False
        if recipe == "mxfp4" and mod.in_features % 128 != 0:
            skipped.append(f"{fqn} (K={mod.in_features} not /128 for mxfp4 cutlass)")
            return False
        converted.append(fqn)
        return True

    if recipe in FP8_RECIPES:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        # Dynamic scaling for input / weight / grad_output (torchao default for
        # tensorwise; rowwise per-row scales) -- no delayed-scaling history state.
        config = (
            Float8LinearConfig()
            if recipe == "tensorwise"
            else Float8LinearConfig.from_recipe_name("rowwise")
        )
        convert_to_float8_training(
            module, config=config, module_filter_fn=module_filter_fn
        )
        new_cls = "Float8Linear"
    else:
        from torchao.prototype.mx_formats.config import (
            MXLinearConfig,
            MXLinearRecipeName,
        )
        from torchao.quantization import quantize_

        mx_recipe = (
            MXLinearRecipeName.MXFP8_CUBLAS
            if recipe == "mxfp8"
            else MXLinearRecipeName.MXFP4_CUTLASS
        )
        config = MXLinearConfig.from_recipe_name(mx_recipe)
        # quantize_ with an MXLinearConfig performs the *training* swap to
        # MXLinear (autograd mx_mm: fwd + grad_input + grad_weight all in MX).
        quantize_(module, config, filter_fn=module_filter_fn)
        new_cls = "MXLinear"

    if verbose:
        print_acc(
            f"fp8_compute[{recipe}]: converted {len(converted)} nn.Linear layers "
            f"to {new_cls}; {len(skipped)} eligible-by-type layers kept in bf16"
        )
        if converted:
            print_acc(f"  first/last converted: {converted[0]} ... {converted[-1]}")
        if recipe == "mxfp4":
            print_acc(
                "  WARNING: mxfp4 training is experimental -- compare loss curves "
                "against a bf16/fp8 baseline before trusting long runs."
            )
    return module
