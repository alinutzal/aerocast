"""Stage an OSC job's data and archive its outputs. Invoked by the sbatch scripts."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess

from aerocast.config import load_config


def sync(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", str(source) + "/", str(destination) + "/"], check=True)


def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} before submitting (see README, OSC job storage).")
    return value


def size(path):
    # Apparent size accounts for sparse files as rsync copies them normally.
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if Path(args.scope).name != args.scope or args.scope in (".", ".."):
        parser.error("scope must be a single directory name")
    # Honor benchmark --set data.* overrides when identifying the source dataset.
    overrides = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--set"]
    cfg = load_config(args.config, overrides)
    source = Path(cfg["data"]["data_dir"]).resolve()
    if not source.is_dir():
        raise SystemExit(f"Missing source data: {source}")
    scratch = Path(required_env("AEROCAST_SCRATCH_ROOT"))
    project = Path(required_env("AEROCAST_PROJECT_ROOT"))
    dataset = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    working = scratch / "data" / dataset
    working.parent.mkdir(parents=True, exist_ok=True)
    with open(working.parent / (dataset + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sync(source, working)
        static = cfg["data"].get("static_file")
        if static and Path(static).is_absolute():
            shutil.copy2(static, working / Path(static).name)
    data = working
    cache = scratch / "cache" / dataset / args.scope
    tmp = os.environ.get("TMPDIR")
    if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) == 1 and tmp:
        # Reserve headroom for preprocessing and other users of the node-local disk.
        required = 2 * size(working) + (size(cache) if cache.exists() else 0)
        if shutil.disk_usage(tmp).free > required + 2 * 1024**3:
            local = Path(tmp) / "aerocast"
            data = local / "data"
            sync(working, data)
            local_cache = local / "cache"
            if cache.exists():
                sync(cache, local_cache)
            cache = local_cache
    output = scratch / "outputs" / args.scope
    archive = project / "results" / args.scope
    output.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate submissions for the same fold/model from racing.
    with open(output / ".job.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if archive.exists():
            sync(archive, output)
        # Seed resumability with the existing repository's metrics on first use.
        results = output / "results/results.csv"
        previous = Path(cfg["output"]["results_csv"])
        if not results.exists() and previous.exists():
            results.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, results)
        env = dict(os.environ, AEROCAST_DATA_DIR=str(data), AEROCAST_CACHE_DIR=str(cache),
                   AEROCAST_OUTPUT_DIR=str(output), WANDB_DIR=str(output),
                   UV_CACHE_DIR=str(scratch / "uv-cache"))
        print(f"Data: {data}\nCache: {cache}\nOutputs: {output}\nArchive: {archive}", flush=True)
        child = subprocess.Popen(command, env=env)
        def forward(signum, frame):
            child.send_signal(signum)
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        try:
            code = child.wait()
        finally:
            # Archive successful and partially completed runs; scratch is retained.
            # Exclude the active lock inode from copying/restoring.
            archive.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--exclude=.job.lock", str(output) + "/", str(archive) + "/"], check=True)
        raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
