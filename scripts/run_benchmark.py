"""Run the controlled benchmark from configs/experiments/benchmark.yaml.

tune   for each model, `trials` random-search trials in the tuning target mode: learning rate
       and weight decay (log-uniform; trial i uses the same draw for every model) and the
       model's width/depth knob; validation rows only.
select pick each model's trial with the lowest validation Ox RMSE (results/benchmark_selection.json).
final  the selected config for every target mode and seed; validation and test rows.

Between runs only the model, its tuned settings, the target mode and the seed change; the
epochs, patience, batch size, loss, inputs and splits come from the benchmark config. A run
is skipped when results.csv already has its rows for the same config hash, so the script can
be restarted, and --model runs one model at a time.

    uv run python scripts/run_benchmark.py                          # tune, select, final
    uv run python scripts/run_benchmark.py --stage tune --model unet
    uv run python scripts/run_benchmark.py --dry-run
"""
import argparse
import copy
import csv
import fcntl
import gc
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from aerocast.config import config_hash, load_config
from aerocast.evaluate import evaluate_run
from aerocast.models import build_model, count_parameters
from aerocast.splits import days_to_load
from aerocast.train import train


def knob_settings(tuning):
    keys = tuning["knob"] if isinstance(tuning["knob"], list) else [tuning["knob"]]
    return [dict(zip(keys, option if isinstance(tuning["knob"], list) else [option])) for option in tuning["options"]]


def trial_draws(tuning):
    """(lr, weight_decay) per trial, shared by all models."""
    rng = np.random.default_rng(tuning["seed"])
    n = tuning["trials"]
    lr = np.exp(rng.uniform(*np.log(tuning["lr"]), size=n))
    wd = np.exp(rng.uniform(*np.log(tuning["weight_decay"]), size=n))
    return [(float(a), float(b)) for a, b in zip(lr, wd)]


def knob_draws(model, tuning, n_options):
    """Knob option index per trial, from a model-specific random stream."""
    rng = np.random.default_rng([tuning["seed"], zlib.crc32(model.encode())])
    return rng.integers(0, n_options, size=tuning["trials"]).tolist()


class Benchmark:
    def __init__(self, path, overrides=()):
        self.path, self.overrides = path, list(overrides)
        cfg = load_config(path, self.overrides)
        self.bench = cfg["benchmark"]
        self.budget = (cfg["train"]["epochs"], cfg["train"]["patience"])
        self.tuning_seed = cfg["seed"]  # tuning trials all train with the base seed
        if cfg["split"]["mode"] != "dates":
            raise SystemExit(f"The benchmark needs split.mode: dates (got {cfg['split']['mode']!r}); "
                             "smoke-test runs never feed benchmark tables.")
        self.results_csv = Path(cfg["output"]["results_csv"])
        # Namespaced by config stem so configs that share one results_csv (e.g. the CV
        # fold configs, all pointed at the same results/results.csv) get separate selection
        # files instead of clobbering each other's entries.
        stem = Path(path).stem
        name = "benchmark_selection.json" if stem == "benchmark" else f"benchmark_selection_{stem}.json"
        self.selection_path = self.results_csv.parent / name

    def config(self, model, name, target_mode, seed, lr, weight_decay, setting, note):
        cfg = load_config(self.path, self.overrides, model=model)
        cfg.pop("benchmark")
        # Prefixed by the experiment config's own stem (unless it's the plain default) so
        # run_id/EXPERIMENTS.md keep saying which config produced a run - e.g. which CV fold,
        # for configs/experiments/benchmark_equates_fold*.yaml.
        stem = Path(self.path).stem
        cfg["name"] = name if stem == "benchmark" else f"{stem}-{name}"
        cfg["target_mode"], cfg["seed"] = target_mode, seed
        cfg["train"]["lr"], cfg["train"]["weight_decay"] = lr, weight_decay
        cfg["train"]["epochs"], cfg["train"]["patience"] = self.budget  # model files cannot change the budget
        cfg["model"].update(setting)
        cfg["model"].update((self.bench.get("model_overrides") or {}).get(model, {}))
        cfg["output"]["note"] = note
        cfg["data"]["dates"] = [str(d) for d in days_to_load(cfg["data"], cfg["split"])]  # as train() records them
        return cfg

    def tuning_runs(self, model):
        tuning = self.bench["tuning"]
        settings = knob_settings(load_config(self.path, self.overrides, model=model)["tuning"])
        runs = []
        for trial, ((lr, wd), knob) in enumerate(zip(trial_draws(tuning), knob_draws(model, tuning, len(settings)))):
            cfg = self.config(model, f"tune-{model}-t{trial}", tuning["target_mode"], self.tuning_seed,
                              lr, wd, settings[knob], f"tuning trial {trial}")
            runs.append({"trial": trial, "lr": lr, "weight_decay": wd, "setting": settings[knob], "cfg": cfg})
        return runs

    def results(self):
        if not self.results_csv.exists():
            return []
        with open(self.results_csv, newline="") as f:
            return list(csv.DictReader(f))

    def done(self, cfg, split, rows):
        h = config_hash(cfg)
        return any(r["config_hash"] == h and r["split"] == split and r["model"] == cfg["model"]["name"] for r in rows)

    def check_budget(self, cfg):
        params = count_parameters(build_model(cfg))
        low, high = self.bench["parameter_budget"]
        if not low <= params <= high:
            raise SystemExit(f"{cfg['name']}: {params / 1e6:.2f}M parameters is outside the budget {low / 1e6:.1f}-{high / 1e6:.1f}M")
        return params

    def execute(self, cfg, labels, dry_run):
        self.check_budget(cfg)
        if dry_run:
            print(f"  would run {cfg['name']} ({config_hash(cfg)})")
            return
        run_dir = train(cfg)
        evaluate_run(run_dir, labels=labels)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tune(self, models, dry_run):
        for model in models:
            rows = self.results()
            for run in self.tuning_runs(model):
                if self.done(run["cfg"], "val", rows):
                    print(f"  skip {run['cfg']['name']}: validation rows exist")
                    continue
                print(f"== tuning {model} trial {run['trial']}: lr {run['lr']:.2e}, weight decay {run['weight_decay']:.2e}, {run['setting']}")
                self.execute(run["cfg"], ["val"], dry_run)

    def select(self, models):
        rows = self.results()
        selection = {}
        for model in models:
            scored = []
            for run in self.tuning_runs(model):
                h = config_hash(run["cfg"])
                rmse = [float(r["value"]) for r in rows if r["config_hash"] == h and r["split"] == "val"
                        and r["model"] == model and r["metric"] == "rmse" and r["lead_hour"] == "all"]
                if not rmse:
                    raise SystemExit(f"{model}: tuning trial {run['trial']} has no validation rows; run --stage tune first")
                scored.append((rmse[-1], run))
            rmse, best = min(scored, key=lambda item: item[0])
            selection[model] = {"trial": best["trial"], "lr": best["lr"], "weight_decay": best["weight_decay"],
                                "setting": best["setting"], "val_ox_rmse": rmse, "config_hash": config_hash(best["cfg"]),
                                "trials": {str(run["trial"]): value for value, run in scored}}
            print(f"== {model}: trial {best['trial']} selected (val Ox RMSE {rmse:.3f}; {best['setting']})")
        # Read-modify-write under an exclusive lock: several models (or, across fold configs,
        # several Benchmark instances sharing one results_csv) can call select() concurrently,
        # each computing only its own models' entries but merging into the same file.
        self.selection_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.selection_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            existing = json.loads(f.read() or "{}")
            existing.update(selection)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(existing, indent=2) + "\n")
        return existing

    def final(self, models, dry_run):
        selection = self.select(models)
        for model in models:
            chosen = selection[model]
            for mode in self.bench["target_modes"]:
                for seed in self.bench["seeds"]:
                    cfg = self.config(model, f"final-{model}-{mode}-s{seed}", mode, seed, chosen["lr"],
                                      chosen["weight_decay"], chosen["setting"], f"final, tuning trial {chosen['trial']}")
                    if self.done(cfg, "test", self.results()):
                        print(f"  skip {cfg['name']}: test rows exist")
                        continue
                    print(f"== final {model} {mode} seed {seed}")
                    self.execute(cfg, None, dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the controlled model benchmark (resumable).")
    parser.add_argument("--config", default="configs/experiments/benchmark.yaml")
    parser.add_argument("--stage", choices=["tune", "final", "all"], default="all")
    parser.add_argument("--model", action="append", default=None, help="only this model (repeatable)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would be executed")
    args = parser.parse_args(argv)
    bench = Benchmark(args.config, args.overrides)
    models = args.model or bench.bench["models"]
    unknown = set(models) - set(bench.bench["models"])
    if unknown:
        raise SystemExit(f"Not in the benchmark: {sorted(unknown)}")
    if args.stage in ("tune", "all"):
        bench.tune(models, args.dry_run)
    if args.stage in ("final", "all") and not args.dry_run:
        bench.final(models, args.dry_run)


if __name__ == "__main__":
    main()
