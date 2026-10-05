"""DistilForge CLI 入口。

用法：
  python cli.py run [--seeds 20261005,20261006,20261007] [--out benchmark.json]
  python cli.py ablation [--dataset synth_mid]
  python cli.py doctor
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 仓库根加入 sys.path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from core.config import DistilForgeConfig  # noqa: E402
from core.seed import set_all  # noqa: E402
from core.types import DatasetSpec  # noqa: E402
from distill.teachers import AVAILABLE_SKLEARN_TEACHER  # noqa: E402
from examples.run_demo import run_demo  # noqa: E402
from pipeline.distill_pipeline import DistillPipeline  # noqa: E402


def _dataset_presets():
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


def _cmd_run(args):
    seeds = (
        [int(s) for s in args.seeds.split(",")] if args.seeds else [20261005, 20261006, 20261007]
    )
    cfg = DistilForgeConfig.from_env()
    cfg.validate()
    set_all(cfg.seed)
    out, det = run_demo(cfg, seeds=seeds, out_path=args.out)
    print(
        "\n确定性校验（同 seed 两次运行逐位一致）：",
        "✅ PASS" if det["pass"] else "⚠️ FAIL",
        f"(max|Δ|={det['max_abs_delta']:.2e})",
    )


def _cmd_ablation(args):
    cfg = DistilForgeConfig.from_env()
    cfg.validate()
    set_all(cfg.seed)
    spec = next((d for d in _dataset_presets() if d.name == args.dataset), None)
    if spec is None:
        spec = DatasetSpec(
            name=args.dataset,
            n_classes=5,
            n_features=16,
            n_train=1500,
            n_test=700,
            difficulty=0.9,
            seed=20261005,
            source="synthetic",
        )
    pipe = DistillPipeline(cfg)
    res = pipe.ablation(spec, seed=20261005, config=cfg)
    print(f"\n消融（数据集={spec.name}，student={cfg.student_type}，teacher={cfg.teacher_type}）：")
    print(f"{'T':>4} {'alpha':>6} {'accuracy':>10}")
    for r in res:
        print(f"{r['temperature']:>4} {r['alpha']:>6.2f} {r['accuracy']:>10.4f}")


def _cmd_doctor(args):
    cfg = DistilForgeConfig.from_env()
    print("== DistilForge doctor ==")
    print(f"sklearn teacher 可用 : {'✅' if AVAILABLE_SKLEARN_TEACHER else '⚠️（缺失）'}")
    print(
        f"默认配置            : T={cfg.temperature} alpha={cfg.alpha} lr={cfg.lr} "
        f"epochs={cfg.n_epochs} student={cfg.student_type} teacher={cfg.teacher_type}"
    )
    cfg.validate()
    print("配置校验            : ✅ 通过")


def main():
    p = argparse.ArgumentParser(
        prog="distilforge", description="知识蒸馏通用分类系统（作者：晨星）"
    )
    sub = p.add_subparsers(dest="cmd")
    pr = sub.add_parser("run", help="跑端到端基准并落盘 benchmark.json")
    pr.add_argument("--seeds", default="20261005,20261006,20261007")
    pr.add_argument("--out", default="benchmark.json")
    pa = sub.add_parser("ablation", help="扫描 (T, alpha) 消融")
    pa.add_argument("--dataset", default="synth_mid")
    sub.add_parser("doctor", help="环境自检")
    args = p.parse_args()
    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "ablation":
        _cmd_ablation(args)
    elif args.cmd == "doctor":
        _cmd_doctor(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
