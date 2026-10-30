"""Command-line entry point.

Examples
--------
Run everything (from the repository root, the folder that contains ``config.yaml``)::

    python -m poroseg.run --stages all

Fast smoke test (fewer bootstrap replicates / repetitions; written to ``outputs_quick/``)::

    python -m poroseg.run --stages all --quick

Only some stages (they read the outputs of earlier stages from disk)::

    python -m poroseg.run --stages spatial,inference
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
import traceback
from pathlib import Path

from .config import load_config
from .context import Context, _json_default

STAGES = ["segmentation", "descriptive", "spatial", "sensitivity", "perturbation",
          "morphology", "synthetic", "ml_baseline", "inference", "legacy", "report"]


def _setup_logging(logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("poroseg")
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = logging.FileHandler(logdir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def run(config: str, stages: list[str], quick: bool = False, samples: list[str] | None = None,
        keep_going: bool = False) -> dict:
    cfg = load_config(config)
    if quick:
        cfg["_quick"] = True
        out = Path(cfg["paths"]["output_dir"])
        cfg["paths"]["output_dir"] = str(out.with_name(out.name + "_quick"))   # never overwrite the full run
    _setup_logging(Path(cfg["paths"]["output_dir"]) / "logs")
    log = logging.getLogger("poroseg")
    ctx = Context(cfg, quick=quick, samples=samples, stage="init")
    status = {}
    for st in stages:
        try:
            mod = importlib.import_module(f"poroseg.stages.{st}")
        except ModuleNotFoundError as e:
            if e.name == f"poroseg.stages.{st}":
                log.warning("stage '%s' not implemented - skipped", st)
                status[st] = "missing"
                continue
            raise
        ctx.stage = st
        ctx._catalog, ctx._keys = [], {}
        t0 = time.time()
        log.info("=== stage %s ===", st)
        try:
            mod.run(ctx)
            ctx.finalize()
            status[st] = f"ok ({time.time() - t0:.1f}s)"
        except Exception:
            status[st] = "FAILED"
            log.error("stage %s failed:\n%s", st, traceback.format_exc())
            if not keep_going:
                raise
    prov = ctx.provenance(stages)
    prov["status"] = status
    # the outputs of a stage come from the last run that included it: keep that record for every stage,
    # so that results built over several runs remain traceable
    prov_path = ctx.dirs["logs"] / "provenance.json"
    history = {}
    if prov_path.exists():
        try:
            history = json.loads(prov_path.read_text(encoding="utf-8")).get("stage_runs", {})
        except (ValueError, OSError):
            history = {}
    for st, s in status.items():
        if s.startswith("ok"):
            history[st] = {k: prov[k] for k in ("timestamp", "python", "platform", "packages", "quick",
                                                "segmentation_signature", "input_sha256")}
            history[st]["samples"] = samples or "all"
    prov["stage_runs"] = history
    prov_path.write_text(json.dumps(prov, indent=1, default=_json_default), encoding="utf-8")
    log.info("status: %s", status)
    return status


def main(argv=None) -> None:
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Segmentation-uncertainty analysis of uCT porosity")
    ap.add_argument("--config", default=str(here / "config.yaml"))
    ap.add_argument("--stages", default="all", help="comma-separated list or 'all' (" + ",".join(STAGES) + ")")
    ap.add_argument("--quick", action="store_true", help="reduced replicates for a fast smoke test")
    ap.add_argument("--samples", default=None, help="comma-separated subset of sample_id")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing stage")
    a = ap.parse_args(argv)
    stages = STAGES if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stages {bad}")
    run(a.config, stages, a.quick, a.samples.split(",") if a.samples else None, a.keep_going)


if __name__ == "__main__":
    main()
