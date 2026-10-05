"""DistilForge 端到端演示：多数据集 × 多 seed 基准 + 确定性校验 + 落盘。

直接运行：python examples/run_demo.py
返回 (benchmark_dict, determinism_dict)，并把结果写入 benchmark.json。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from core.config import DistilForgeConfig  # noqa: E402
from core.seed import set_all  # noqa: E402
from core.types import BenchmarkRecord, DatasetSpec  # noqa: E402
from distill.teachers import AVAILABLE_SKLEARN_TEACHER  # noqa: E402
from pipeline.distill_pipeline import DistillPipeline  # noqa: E402

try:
    from scipy import stats as _stats

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def dataset_presets():
    return [
        DatasetSpec(
            name="synth_easy",
            n_classes=4,
            n_features=12,
            n_train=1200,
            n_test=600,
            difficulty=1.0,
            clusters_per_class=4,
            class_sep=3.0,
            seed=20261005,
            source="synthetic",
        ),
        DatasetSpec(
            name="synth_mid",
            n_classes=5,
            n_features=16,
            n_train=1500,
            n_test=700,
            difficulty=0.9,
            clusters_per_class=4,
            class_sep=3.5,
            seed=20261005,
            source="synthetic",
        ),
        DatasetSpec(
            name="synth_hard",
            n_classes=6,
            n_features=20,
            n_train=1800,
            n_test=900,
            difficulty=0.6,
            clusters_per_class=3,
            class_sep=3.5,
            seed=20261005,
            source="synthetic",
        ),
        DatasetSpec(
            name="digits",
            n_classes=10,
            n_features=64,
            n_train=0,
            n_test=0,
            seed=20261005,
            source="sklearn",
        ),
    ]


def _print_table(name: str, agg):
    print(f"\n=== 数据集：{name} ===")
    print(f"{'方法':<10}{'acc_mean':>10}{'acc_std':>9}{'err_mean':>10}{'rel_drop':>12}")
    for r in agg:
        rd = f"{r.mean_rel_drop * 100:+.2f}%" if r.mean_rel_drop is not None else "—"
        print(
            f"{r.method:<10}{r.mean_accuracy:>10.4f}{r.std_accuracy:>9.4f}"
            f"{r.mean_error:>10.4f}{rd:>12}"
        )


def _paired_significance(records: list[BenchmarkRecord]):
    """对 distill vs hard 做逐 seed 配对检验，返回 (p_value, wins, n_seeds)。"""
    by_seed: dict[int, dict[str, float]] = {}
    for r in records:
        if r.method in ("hard", "distill"):
            by_seed.setdefault(r.seed, {})[r.method] = r.accuracy
    diffs = [d["distill"] - d["hard"] for d in by_seed.values() if "hard" in d and "distill" in d]
    n = len(diffs)
    wins = sum(1 for d in diffs if d > 0)
    p = None
    if n >= 2 and _HAVE_SCIPY:
        if np.std(diffs) == 0:
            p = 0.0 if wins == n else 1.0
        else:
            p = float(_stats.ttest_1samp(diffs, 0.0).pvalue)
    return p, wins, n


def _verify_determinism(
    pipe: DistillPipeline, spec: DatasetSpec, seed: int, cfg: DistilForgeConfig
):
    r1 = {r.method: r for r in pipe.run_once(spec, seed, cfg)}
    r2 = {r.method: r for r in pipe.run_once(spec, seed, cfg)}
    max_delta = 0.0
    for m in ("hard", "pseudo", "distill"):
        d = abs(r1[m].accuracy - r2[m].accuracy)
        max_delta = max(max_delta, d)
    return {
        "pass": max_delta == 0.0,
        "max_abs_delta": max_delta,
        "seed": seed,
        "dataset": spec.name,
    }


def run_demo(
    config: DistilForgeConfig = None,
    seeds=(20261005, 20261006, 20261007),
    out_path: str = "benchmark.json",
):
    cfg = config or DistilForgeConfig()
    cfg.validate()
    set_all(cfg.seed)

    if not AVAILABLE_SKLEARN_TEACHER:
        raise RuntimeError("scikit-learn 不可用，无法运行 demo")

    pipe = DistillPipeline(cfg)
    datasets = dataset_presets()
    overall: list[dict] = []
    all_records = []
    synth_drops: list[float] = []

    print("== DistilForge 端到端基准（作者：晨星）==")
    print(
        f"配置：T={cfg.temperature} α={cfg.alpha} lr={cfg.lr} epochs={cfg.n_epochs} "
        f"student={cfg.student_type} teacher={cfg.teacher_type}"
    )
    print(f"seeds：{list(seeds)}")

    for spec in datasets:
        recs = pipe.benchmark(spec, list(seeds), cfg)
        all_records.extend(recs)
        agg = DistillPipeline.aggregate(recs)
        _print_table(spec.name, agg)
        p, wins, n = _paired_significance(recs)
        sig = (
            "显著"
            if (p is not None and p < cfg.significance)
            else ("不显著" if p is not None else "未检验")
        )
        print(f"  → distill vs hard 配对检验：wins={wins}/{n}，p={p:.4f}（{sig}）")
        drop_agg = next((a.mean_rel_drop for a in agg if a.method == "distill"), None)
        overall.append(
            {
                "dataset": spec.name,
                "aggregated": [
                    {
                        "method": a.method,
                        "mean_accuracy": a.mean_accuracy,
                        "std_accuracy": a.std_accuracy,
                        "mean_error": a.mean_error,
                        "std_error": a.std_error,
                        "mean_rel_drop": a.mean_rel_drop,
                        "std_rel_drop": a.std_rel_drop,
                    }
                    for a in agg
                ],
                "paired_p_value": p,
                "wins_vs_hard": wins,
                "n_seeds": n,
            }
        )
        if spec.source == "synthetic" and drop_agg is not None:
            synth_drops.append(drop_agg)

    # 头条 KPI：合成光谱（KD 有效区间）平均相对错误率下降
    kpi_mean = float(np.mean(synth_drops)) if synth_drops else None
    kpi_std = float(np.std(synth_drops)) if synth_drops else None
    kpi_pass = kpi_mean is not None and kpi_mean >= 0.05
    print("\n== 头条 KPI（合成光谱平均相对错误率下降，门槛 ≥5%）==")
    print(
        f"  mean={kpi_mean * 100:+.2f}%  std={kpi_std * 100:.2f}%  → "
        f"{'✅ 达标' if kpi_pass else '⚠️ 未达标'}"
    )

    det = _verify_determinism(pipe, datasets[0], seeds[0], cfg)

    out = {
        "system": "DistilForge",
        "author": "晨星",
        "config": {
            "temperature": cfg.temperature,
            "alpha": cfg.alpha,
            "lr": cfg.lr,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "student_type": cfg.student_type,
            "teacher_type": cfg.teacher_type,
            "n_hidden": cfg.n_hidden,
        },
        "seeds": list(seeds),
        "headline_kpi": {
            "metric": "synthetic_spectrum_mean_rel_error_drop",
            "mean": kpi_mean,
            "std": kpi_std,
            "threshold": 0.05,
            "pass": kpi_pass,
        },
        "datasets": overall,
        "records": [
            {
                "dataset": r.dataset,
                "method": r.method,
                "seed": r.seed,
                "accuracy": r.accuracy,
                "error_rate": r.error_rate,
                "rel_error_drop_vs_hard": r.rel_error_drop_vs_hard,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in all_records
        ],
        "determinism": det,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已落盘 {out_path}（记录 {len(all_records)} 条）")
    print(
        f"确定性校验：{'✅ PASS' if det['pass'] else '⚠️ FAIL'} (max|Δ|={det['max_abs_delta']:.2e})"
    )
    return out, det


if __name__ == "__main__":
    run_demo()
