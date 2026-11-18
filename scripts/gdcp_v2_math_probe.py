"""Identify CPU/CUDA rounding differences in the real master AdamW step."""

import json
import math
import numpy as np
import torch
from tests.support.gdcp_v2_models import pattern


def values(device):
    name = "model.embed_tokens.weight"
    return dict(
        master=pattern(name + ".master_param", 2144, 0, device) + 2**-19,
        avg=pattern(name + ".exp_avg", 2144, 0, device),
        square=pattern(name + ".exp_avg_sq", 2144, 0, device).abs() + 0.25,
        gradient=pattern(name + ".gradient", 2144, 0, device, step=4),
    )


def report(name, a, b):
    x, y = a.detach().cpu(), b.detach().cpu()
    count = int((x.view(torch.int32) != y.view(torch.int32)).sum())
    print(
        json.dumps(
            dict(operation=name, different=count, max_abs=float((x - y).abs().max()))
        ),
        flush=True,
    )


def main():
    from torch.cuda.jiterator import _create_jit_fn

    sqrt_rn = _create_jit_fn(
        "template <typename T> T gdcp_sqrt_rn(T x) { return __fsqrt_rn(x); }"
    )
    div_rn = _create_jit_fn(
        "template <typename T> T gdcp_div_rn(T x, T y) { return __fdiv_rn(x,y); }"
    )
    a, b = values("cpu"), values("cuda:0")
    report("initial-master", a["master"], b["master"])
    for v in (a, b):
        v["master"].mul_(1 - 0.03125 * 0.0625)
        v["avg"].mul_(0.5).add_(v["gradient"] * 0.5)
        v["square"].mul_(0.75).add_(v["gradient"].square() * 0.25)
        v["sqrt"] = v["square"].sqrt()
        v["residual"] = torch.addcmul(v["square"], v["sqrt"], v["sqrt"], value=-1)
        v["corrected"] = v["sqrt"] + v["residual"] / (2 * v["sqrt"])
        v["corrected_denom"] = v["corrected"] * (1 / math.sqrt(1 - 0.75**4)) + 0.125
        v["corrected_master"] = v["master"] + (v["avg"] / v["corrected_denom"]) * (
            -0.03125 / (1 - 0.5**4)
        )
        v["div"] = v["sqrt"] / math.sqrt(1 - 0.75**4)
        v["mul"] = v["sqrt"] * (1 / math.sqrt(1 - 0.75**4))
        v["div_ratio"] = v["avg"] / (v["div"] + 0.125)
        v["mul_ratio"] = v["avg"] / (v["mul"] + 0.125)
        v["div_master"] = v["master"] + v["div_ratio"] * (-0.03125 / (1 - 0.5**4))
        v["mul_master"] = v["master"] + v["mul_ratio"] * (-0.03125 / (1 - 0.5**4))
    a["sqrt_rn"] = a["square"].sqrt()
    b["sqrt_rn"] = sqrt_rn(b["square"])
    a["numpy_sqrt"] = torch.from_numpy(np.sqrt(a["square"].numpy()))
    report("numpyCPU-vs-CUDA-rn", a["numpy_sqrt"], b["sqrt_rn"])
    a["numpy_denom"] = a["numpy_sqrt"] * (1 / math.sqrt(1 - 0.75**4)) + 0.125
    a["numpy_master"] = a["master"] + (a["avg"] / a["numpy_denom"]) * (
        -0.03125 / (1 - 0.5**4)
    )
    a["oracle_sqrt"] = a["square"].double().sqrt().float()
    report("CPU-vs-f64sqrt-oracle", a["sqrt_rn"], a["oracle_sqrt"])
    for v in (a, b):
        v["rn_denom"] = v["sqrt_rn"] * (1 / math.sqrt(1 - 0.75**4)) + 0.125
        v["rn_ratio"] = (
            div_rn(v["avg"], v["rn_denom"])
            if v["avg"].is_cuda
            else v["avg"] / v["rn_denom"]
        )
        v["rn_master"] = v["master"] + v["rn_ratio"] * (-0.03125 / (1 - 0.5**4))
    report("numpy-master-vs-CUDA-rn", a["numpy_master"], b["rn_master"])
    for name in (
        "sqrt_rn",
        "rn_ratio",
        "rn_master",
        "master",
        "avg",
        "square",
        "sqrt",
        "residual",
        "corrected",
        "corrected_master",
        "div",
        "mul",
        "div_ratio",
        "mul_ratio",
        "div_master",
        "mul_master",
    ):
        report(name, a[name], b[name])


if __name__ == "__main__":
    main()
