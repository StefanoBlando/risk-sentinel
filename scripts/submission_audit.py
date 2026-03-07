#!/usr/bin/env python3
"""Lightweight submission readiness audit for RiskSentinel."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _latest_bundle() -> str:
    artifacts = ROOT / "artifacts"
    if not artifacts.exists():
        return ""
    zips = sorted(artifacts.glob("submission_bundle_*.zip"))
    return str(zips[-1].relative_to(ROOT)) if zips else ""


def _site_link_status() -> dict[str, bool]:
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    live_ok = bool(
        re.search(
            r"<h3>\s*Live Demo URL\s*</h3>[\s\S]*?<a[^>]+href=\"(?!#)[^\"]+\"",
            html,
            re.IGNORECASE,
        )
    )
    video_ok = bool(
        re.search(
            r"<h3>\s*Video URL\s*</h3>[\s\S]*?<a[^>]+href=\"(?!#)[^\"]+\"",
            html,
            re.IGNORECASE,
        )
    )
    return {"live_demo_link_set": live_ok, "video_link_set": video_ok}


def build_audit() -> dict:
    docs_required = [
        "docs/pitch.md",
        "docs/demo_script.md",
        "docs/architecture_diagram.md",
    ]
    docs_present = {p: (ROOT / p).exists() for p in docs_required}

    git_status = _run(["git", "status", "--short"])
    dirty_lines = [ln for ln in git_status.splitlines() if ln.strip()]
    non_local_dirty = [ln for ln in dirty_lines if " TODO.md" not in ln]

    site_links = _site_link_status()
    latest_bundle = _latest_bundle()
    demo_check_exists = (ROOT / "artifacts" / "demo_check_latest.json").exists()

    blockers: list[str] = []
    if not all(docs_present.values()):
        blockers.append("Missing one or more required docs (pitch/demo_script/architecture).")
    if not demo_check_exists:
        blockers.append("Missing artifacts/demo_check_latest.json.")
    if not latest_bundle:
        blockers.append("No submission_bundle_*.zip found in artifacts/.")
    if not site_links["live_demo_link_set"]:
        blockers.append("Site is missing final Live Demo URL.")
    if not site_links["video_link_set"]:
        blockers.append("Site is missing final Video URL.")
    if non_local_dirty:
        blockers.append("Git working tree has uncommitted changes (excluding TODO.md).")

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ok": len(blockers) == 0,
        "blockers": blockers,
        "docs_present": docs_present,
        "artifacts": {
            "demo_check_latest_json": demo_check_exists,
            "latest_submission_bundle": latest_bundle,
        },
        "site_links": site_links,
        "git": {
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _run(["git", "rev-parse", "--short", "HEAD"]),
            "dirty_entries": dirty_lines,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Submission readiness audit.")
    parser.add_argument(
        "--output",
        default="artifacts/submission_audit_latest.json",
        help="Path to JSON output (default: artifacts/submission_audit_latest.json)",
    )
    args = parser.parse_args()

    payload = build_audit()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote audit: {out_path.relative_to(ROOT)}")
    if payload["ok"]:
        print("Submission audit: PASS")
        return 0

    print("Submission audit: FAIL")
    for blocker in payload["blockers"]:
        print(f"- {blocker}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
