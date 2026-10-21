"""Command-line entry point: `dp-ecg <command>` (or `python -m dp_ecg.cli <command>`)."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dp-ecg", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env", help="print library versions and hardware")
    sub.add_parser("download", help="download PTB-XL 100 Hz records (SHA-256 verified)")
    sub.add_parser("prepare", help="build waveform cache, manifest, features and data report")
    sub.add_parser("verify-mechanisms", help="exhaustive toy-universe checks -> artifacts/reports")
    sub.add_parser("summary-stats", help="DP summary statistics under one budget + error diagnostic")
    tb = sub.add_parser("train-baseline", help="non-private CNN")
    tb.add_argument("--epochs", type=int)
    td = sub.add_parser("train-dp", help="DP-SGD sweep over configs/dp_sgd.yaml target epsilons")
    td.add_argument("--epochs", type=int)
    sub.add_parser("attack", help="membership inference on every completed run")
    sub.add_parser("report", help="figures and Markdown summary from saved artifacts")
    sub.add_parser("smoke", help="whole pipeline on a tiny SYNTHETIC dataset (not PTB-XL results)")
    sv = sub.add_parser("serve", help="FastAPI + Gradio dashboard")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    if args.cmd == "env":
        from .reproducibility import environment_info
        print(json.dumps(environment_info(), indent=2))
    elif args.cmd == "download":
        from .config import load_config, resolve
        from .data.download import download
        cfg = load_config("data")
        download(resolve(cfg["dataset_root"]), cfg["mirror_url"])
    elif args.cmd == "prepare":
        from .config import load_config
        from .data.prepare import prepare_features, prepare_waveforms
        cfg = load_config("data")
        print(json.dumps(prepare_waveforms(cfg), indent=2))
        prepare_features(cfg)
    elif args.cmd == "verify-mechanisms":
        from .evaluation.reports import mechanism_verification_report
        r = mechanism_verification_report()
        print("all checks passed" if r["all_passed"] else "SOME CHECKS FAILED")
        if not r["all_passed"]:
            raise SystemExit(1)
    elif args.cmd == "summary-stats":
        from .evaluation.summary_stats import error_diagnostic, release_summary_statistics
        r = release_summary_statistics()
        print(json.dumps(r["budget"], indent=2))
        print(error_diagnostic().to_string(index=False))
    elif args.cmd in ("train-baseline", "train-dp"):
        from .experiments import run_training
        recs = run_training(only="baseline" if args.cmd == "train-baseline" else "dp", epochs=args.epochs)
        for r in recs:
            print(r["run_id"], r["status"], round(r["test"]["auroc"], 4))
    elif args.cmd == "attack":
        from .experiments import run_membership_inference
        out = run_membership_inference()
        for rid, r in out["results"].items():
            print(rid, round(r["attacks"]["loss"]["auroc"], 4))
    elif args.cmd == "report":
        from .evaluation.reports import generate_all
        print(generate_all())
    elif args.cmd == "smoke":
        from .experiments import run_membership_inference, run_training
        recs = run_training(synthetic=True, epochs=1)
        assert all(r["status"] == "completed" for r in recs), "smoke run failed"
        run_membership_inference(synthetic=True)
        print("smoke test passed (synthetic data; not PTB-XL results)")
    elif args.cmd == "serve":
        import uvicorn

        from .app.api import create_app
        uvicorn.run(create_app(with_ui=True), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
