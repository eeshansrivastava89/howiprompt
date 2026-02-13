#!/usr/bin/env python3
"""
How I Prompt - Build System

Orchestrator that coordinates the analytics pipeline:
1. Sync local data sources
2. Parse Claude Code + Codex conversations into SQLite
3. Compute all metrics
4. Copy pre-built Astro frontend + metrics.json to output

Usage:
    python build.py                    # Full build
    python build.py --metrics-only     # Only compute metrics.json
    python build.py --skip-copy-claude-code # Skip Claude Code auto-sync
    python build.py --skip-copy-codex  # Skip Codex auto-sync
    python build.py --no-open          # Build without opening browser
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, load_branding, load_config
from src.db import init_db, insert_messages
from src.metrics import compute_source_views
from src.nlp import enrich_nlp
from src.parsers import parse_claude_code, parse_codex_history, parse_codex_session_metadata
from src.sync import copy_claude_code_data, copy_codex_data


def main():
    parser = argparse.ArgumentParser(
        description="How I Prompt Wrapped 2025 - Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                    # Full build
  python build.py --metrics-only     # Only metrics.json
  python build.py --no-open          # Build without opening dashboard in browser
  python build.py --skip-copy-codex  # Skip Codex auto-sync for this run
        """
    )
    parser.add_argument("--metrics-only", action="store_true", help="Only compute metrics, skip HTML")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--skip-copy-claude-code", action="store_true", help="Skip Claude Code auto-sync from ~/.claude/projects")
    parser.add_argument("--skip-copy-codex", action="store_true", help="Skip Codex auto-sync from ~/.codex/history.jsonl")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open dashboard HTML in browser")
    args = parser.parse_args()

    config = load_config()
    output_dir = args.output or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Sync local source data by default.
    print("\n[0/3] Syncing local data sources...")
    if not args.skip_copy_claude_code:
        print("  Claude Code...")
        if not copy_claude_code_data(config.claude_code_path):
            print("  Warning: Claude Code sync failed; continuing with existing data.")
    else:
        print("  Skipped Claude Code sync")

    if not args.skip_copy_codex:
        print("  Codex...")
        if not copy_codex_data(config.codex_history_path):
            print("  Warning: Codex sync failed; continuing with existing data.")
    else:
        print("  Skipped Codex sync")

    print("=" * 50)
    print("How I Prompt Wrapped 2025 - Build")
    print("=" * 50)

    # Step 1: Parse data into SQLite
    print("\n[1/3] Parsing data sources...")
    conn = init_db()
    msg_count = 0
    msg_count += insert_messages(conn, parse_claude_code(config.claude_code_path))
    codex_session_models = parse_codex_session_metadata(config.codex_sessions_path)
    msg_count += insert_messages(conn, parse_codex_history(config.codex_history_path, codex_session_models))
    print(f"  Total: {msg_count} messages")

    if msg_count == 0:
        print("\nError: No messages found. Check your data paths.")
        sys.exit(1)

    # Step 1b: NLP enrichment (classifiers run once, aggregation per-view)
    enrich_nlp(conn)

    # Step 2: Compute metrics
    print("\n[2/3] Computing metrics...")
    source_views, source_defaults = compute_source_views(conn, config)
    dashboard_default_view = source_defaults["default_view"]
    wrapped_base_view = "claude_code" if source_views.get("claude_code") is not None else dashboard_default_view

    metrics = dict(source_views[wrapped_base_view])
    metrics["source_views"] = source_views
    metrics["default_view"] = dashboard_default_view

    branding = load_branding()
    metrics["branding"] = branding or {}

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")

    if args.metrics_only:
        print("\n[Done] Metrics-only build complete.")
        return

    # Step 3: Copy pre-built Astro frontend to output/
    print("\n[3/3] Generating HTML...")
    if branding:
        print(f"  Branding: {branding['site_name']}")

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if not frontend_dist.exists() or not (frontend_dist / "index.html").exists():
        print("  Error: Astro frontend not built. Run: cd frontend && npx astro build")
        sys.exit(1)

    print("  Using pre-built Astro frontend from frontend/dist/")

    # Dashboard: frontend/dist/index.html → output/dashboard.html
    dashboard_path = output_dir / "dashboard.html"
    shutil.copy2(frontend_dist / "index.html", dashboard_path)
    print(f"  Saved: {dashboard_path}")

    # Wrapped: frontend/dist/wrapped/index.html → output/index.html
    html_path = output_dir / "index.html"
    wrapped_src = frontend_dist / "wrapped" / "index.html"
    if wrapped_src.exists():
        shutil.copy2(wrapped_src, html_path)
        print(f"  Saved: {html_path}")
    else:
        print(f"  Warning: {wrapped_src} not found, skipping wrapped page")

    # Copy JS/CSS assets
    astro_assets = frontend_dist / "_astro"
    if astro_assets.exists():
        output_assets = output_dir / "_astro"
        if output_assets.exists():
            shutil.rmtree(output_assets)
        shutil.copytree(astro_assets, output_assets)
        print(f"  Saved: {output_assets}/ ({len(list(output_assets.iterdir()))} assets)")

    # Summary
    print("\n" + "=" * 50)
    print("BUILD COMPLETE")
    print("=" * 50)
    print(f"\nPersona: {metrics['persona']['name']}")
    print(f"  {metrics['persona']['description']}")
    print(f"\nQuadrant scores:")
    print(f"  Engagement: {metrics['persona']['quadrant']['engagement_score']} ({'High' if metrics['persona']['quadrant']['high_engagement'] else 'Low'})")
    print(f"  Politeness: {metrics['persona']['quadrant']['politeness_score']} ({'High' if metrics['persona']['quadrant']['high_politeness'] else 'Low'})")
    print(f"\nOutput:")
    print(f"  {metrics_path}")
    print(f"  {output_dir / 'index.html'}")
    print(f"  {output_dir / 'dashboard.html'}")

    # Copy to docs/ for GitHub Pages
    docs_dir = PROJECT_ROOT / "docs"
    wrapped_dir = docs_dir / "wrapped"
    wrapped_dir.mkdir(parents=True, exist_ok=True)

    if docs_dir.exists():
        # Dashboard becomes the main index
        if dashboard_path.exists():
            shutil.copy2(dashboard_path, docs_dir / "index.html")
            print(f"  {docs_dir / 'index.html'} (Dashboard → howiprompt.eeshans.com)")

        # Full experience goes to /wrapped
        if html_path.exists():
            shutil.copy2(html_path, wrapped_dir / "index.html")
            print(f"  {wrapped_dir / 'index.html'} (Full → /wrapped)")

        # Copy metrics.json to docs/ for Astro frontend fetch()
        shutil.copy2(metrics_path, docs_dir / "metrics.json")
        print(f"  {docs_dir / 'metrics.json'} (Runtime data)")
        shutil.copy2(metrics_path, wrapped_dir / "metrics.json")
        print(f"  {wrapped_dir / 'metrics.json'} (Runtime data for /wrapped)")

        # Copy JS/CSS assets to docs/
        if astro_assets.exists():
            docs_assets = docs_dir / "_astro"
            if docs_assets.exists():
                shutil.rmtree(docs_assets)
            shutil.copytree(astro_assets, docs_assets)
            print(f"  {docs_assets}/ (Frontend assets)")
            wrapped_assets = wrapped_dir / "_astro"
            if wrapped_assets.exists():
                shutil.rmtree(wrapped_assets)
            shutil.copytree(astro_assets, wrapped_assets)
            print(f"  {wrapped_assets}/ (Frontend assets for /wrapped)")

    print("\n[4/4] Astro frontend requires a local server to preview.")
    print("  Run: npx serve docs")


if __name__ == "__main__":
    main()
