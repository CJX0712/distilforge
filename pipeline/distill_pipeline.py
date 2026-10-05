"""DistillPipeline —— 端到端编排 + 多 seed 基准 + 消融。

防泄漏要点：
- StandardScaler 仅在训练集 fit，再 transform train/test
- teacher 仅在训练集训练
- 每个 student（hard/pseudo/distill）使用 *独立但同 seed* 的 RandomState，
  保证三者起点一致、差异仅来自训练目标（公平对照）

指标口径：accuracy 越大越好；rel_error_drop = (err_hard - err_x)/err_hard。
"""

import time

import numpy as np

from core.config import DistilForgeConfig
from core.errors import DataError, DependencyError, PipelineError
from core.types import AggregatedResult, BenchmarkRecord, DatasetSpec
from data.loaders import AVAILABLE_SKLEARN, load_csv, load_sklearn
from data.synthetic import make_synthetic
from distill.baselines import train_hard, train_pseudo
from distill.distiller import DistilFuse
from distill.students import build_student
from distill.teachers import AVAILABLE_SKLEARN_TEACHER, SklearnTeacher
from eval.metrics import accuracy, error_rate, rel_error_drop
from preprocess.scaler import StandardScaler


class DistillPipeline:
    def __init__(self, config: DistilForgeConfig = None):
        self.config = config or DistilForgeConfig()

    # ----- 数据 -----
    def _load(self, spec: DatasetSpec):
        if spec.source == "synthetic":
            return make_synthetic(spec)
        if spec.source == "sklearn":
            if not AVAILABLE_SKLEARN:
                raise DependencyError("scikit-learn 不可用，无法载入内置数据集")
            X, y = load_sklearn(spec.name, seed=spec.seed)
            from sklearn.model_selection import train_test_split

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=spec.seed, stratify=y
            )
            return Xtr, ytr, Xte, yte
        if spec.source == "csv":
            if not hasattr(spec, "path") or not spec.path:  # type: ignore[attr-defined]
                raise DataError("csv 数据源需要 spec.path")
            X, y = load_csv(spec.path)  # type: ignore[attr-defined]
            from sklearn.model_selection import train_test_split

            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=spec.seed, stratify=y
            )
            return Xtr, ytr, Xte, yte
        raise DataError(f"未知数据源: {spec.source}")

    # ----- 单 student 训练（独立同 seed rng，公平对照）-----
    def _train_student(self, kind, Xtr, ytr, teacher, teacher_logits, cfg, seed, n_f, n_c):
        rng = np.random.RandomState(seed)
        student = build_student(n_f, n_c, cfg)
        if kind == "hard":
            train_hard(student, Xtr, ytr, cfg, rng)
        elif kind == "pseudo":
            train_pseudo(student, Xtr, teacher, ytr, cfg, rng)
        elif kind == "distill":
            DistilFuse(cfg).distill(student, Xtr, teacher_logits, ytr, rng)
        else:
            raise PipelineError(f"未知 student 类型: {kind}")
        return student

    # ----- 单 seed 单配置 -----
    def run_once(self, spec: DatasetSpec, seed: int, config: DistilForgeConfig = None):
        cfg = config or self.config
        cfg.validate()
        if not AVAILABLE_SKLEARN_TEACHER:
            raise DependencyError("scikit-learn 不可用，无法构建 teacher")
        Xtr, ytr, Xte, yte = self._load(spec)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        n_f = Xtr_s.shape[1]
        n_c = int(max(int(ytr.max()), int(yte.max())) + 1)

        # teacher
        teacher = SklearnTeacher(cfg.teacher_type, seed=seed)
        teacher.fit(Xtr_s, ytr)
        teacher_logits = teacher.logits(Xtr_s)
        t0 = time.perf_counter()
        t_acc = accuracy(yte, teacher.predict(Xte_s))
        t_elapsed = time.perf_counter() - t0

        records: dict[str, BenchmarkRecord] = {}
        records["teacher"] = BenchmarkRecord(
            dataset=spec.name,
            method="teacher",
            seed=seed,
            accuracy=t_acc,
            error_rate=error_rate(yte, teacher.predict(Xte_s)),
            elapsed_sec=t_elapsed,
        )

        hard_student = self._train_student(
            "hard", Xtr_s, ytr, teacher, teacher_logits, cfg, seed, n_f, n_c
        )
        pseudo_student = self._train_student(
            "pseudo", Xtr_s, ytr, teacher, teacher_logits, cfg, seed, n_f, n_c
        )
        distill_student = self._train_student(
            "distill", Xtr_s, ytr, teacher, teacher_logits, cfg, seed, n_f, n_c
        )

        def _rec(kind, student):
            pred = student.predict(Xte_s)
            acc = accuracy(yte, pred)
            return BenchmarkRecord(
                dataset=spec.name,
                method=kind,
                seed=seed,
                accuracy=acc,
                error_rate=error_rate(yte, pred),
                elapsed_sec=0.0,
            )

        rec_hard = _rec("hard", hard_student)
        rec_pseudo = _rec("pseudo", pseudo_student)
        rec_distill = _rec("distill", distill_student)

        err_hard = rec_hard.error_rate
        rec_pseudo.rel_error_drop_vs_hard = rel_error_drop(err_hard, rec_pseudo.error_rate)
        rec_distill.rel_error_drop_vs_hard = rel_error_drop(err_hard, rec_distill.error_rate)

        return [rec_hard, rec_pseudo, rec_distill, records["teacher"]]

    # ----- 多 seed 聚合 -----
    def benchmark(self, spec: DatasetSpec, seeds, config: DistilForgeConfig = None):
        cfg = config or self.config
        records: list[BenchmarkRecord] = []
        for seed in seeds:
            records.extend(self.run_once(spec, seed, cfg))
        return records

    @staticmethod
    def aggregate(records: list[BenchmarkRecord]):
        by_method: dict[str, list[BenchmarkRecord]] = {}
        for r in records:
            by_method.setdefault(r.method, []).append(r)
        out: list[AggregatedResult] = []
        for method, recs in by_method.items():
            acc = np.array([r.accuracy for r in recs])
            err = np.array([r.error_rate for r in recs])
            drops = np.array(
                [r.rel_error_drop_vs_hard for r in recs if r.rel_error_drop_vs_hard is not None]
            )
            out.append(
                AggregatedResult(
                    dataset=recs[0].dataset,
                    method=method,
                    mean_accuracy=float(acc.mean()),
                    std_accuracy=float(acc.std()),
                    mean_error=float(err.mean()),
                    std_error=float(err.std()),
                    mean_rel_drop=(float(drops.mean()) if drops.size else None),
                    std_rel_drop=(float(drops.std()) if drops.size else None),
                )
            )
        # 稳定排序：teacher, hard, pseudo, distill
        order = {"teacher": 0, "hard": 1, "pseudo": 2, "distill": 3}
        out.sort(key=lambda x: order.get(x.method, 9))
        return out

    # ----- 消融：扫描 (T, alpha) -----
    def ablation(
        self,
        spec: DatasetSpec,
        seed: int,
        config: DistilForgeConfig = None,
        temperatures=(1, 2, 4, 8),
        alphas=(0.3, 0.5, 0.7),
    ):
        cfg_base = config or self.config
        Xtr, ytr, Xte, yte = self._load(spec)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        n_f = Xtr_s.shape[1]
        n_c = int(max(int(ytr.max()), int(yte.max())) + 1)
        teacher = SklearnTeacher(cfg_base.teacher_type, seed=seed)
        teacher.fit(Xtr_s, ytr)
        teacher_logits = teacher.logits(Xtr_s)
        student_ref = build_student(n_f, n_c, cfg_base)  # 仅用于尺寸
        del student_ref
        results = []
        for T in temperatures:
            for alpha in alphas:
                c = DistilForgeConfig(
                    temperature=T,
                    alpha=alpha,
                    lr=cfg_base.lr,
                    n_epochs=cfg_base.n_epochs,
                    batch_size=cfg_base.batch_size,
                    weight_decay=cfg_base.weight_decay,
                    student_type=cfg_base.student_type,
                    teacher_type=cfg_base.teacher_type,
                    n_hidden=cfg_base.n_hidden,
                    seed=cfg_base.seed,
                )
                rng_i = np.random.RandomState(seed)
                student = build_student(n_f, n_c, c)
                DistilFuse(c).distill(student, Xtr_s, teacher_logits, ytr, rng_i)
                acc = accuracy(yte, student.predict(Xte_s))
                results.append({"temperature": T, "alpha": alpha, "accuracy": acc})
        return results
