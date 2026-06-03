"""SurgSync CLI — Hydra-driven entry points for the packer + reader.

Subcommands:
    build    — pack raw clips into the SurgSync dataset format.
    index    — rebuild meta/episodes.parquet, index.parquet, stats.parquet, manifest.json.
    validate — run raw-clip / episode / dataset validators.
    selftest — codec round-trip checks.
    unpack   — decompose a packed dataset back into the raw + preprocess
               on-disk tree (the inverse of `build`).
    release  — write README.md + CHANGELOG.md into the dataset root;
               optionally bump meta/dataset.json:data_version.

Invocation from a clone:

    python -m dvrk_data_processing.surgsync.cli build \
        --config-name=build \
        path_config=jack_local \
        clips.source=dataset clips.dataset_name=online_data

`pyproject.toml` registers `surgsync = "dvrk_data_processing.surgsync.cli:main"`
so a direct `surgsync build ...` works after `pip install -e .`.
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger("surgsync.cli")


# Resolve `<repo>/config/surgsync` regardless of where the CLI is invoked from.
_CONFIG_DIR = (Path(__file__).resolve().parents[3] / "config" / "surgsync").as_posix()


# ---------------------------------------------------------------------------
# build — the main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="build")
def _build_main(cfg: DictConfig) -> None:
    """Hydra entry that calls into the per-release orchestrator."""
    # Defer the heavy imports until this entry runs so `--help` is cheap.
    from dvrk_data_processing.surgsync.pipeline.per_release import build_release

    log.info("surgsync build — resolved config:\n%s", OmegaConf.to_yaml(cfg))
    build_release(cfg)


# ---------------------------------------------------------------------------
# selftest — quick smoke check on the codec layer
# ---------------------------------------------------------------------------

def _selftest_main() -> int:
    from dvrk_data_processing.surgsync.encode.codec import roundtrip_selftest
    try:
        roundtrip_selftest()
    except Exception as e:
        print(f"selftest FAILED: {e}", file=sys.stderr)
        return 1
    print("selftest OK")
    return 0


# ---------------------------------------------------------------------------
# index — rebuild all four meta/ indexes against an existing dataset
# ---------------------------------------------------------------------------

def _index_main(argv: Optional[list[str]] = None) -> int:
    """Rebuild `meta/{episodes.parquet, index.parquet, stats.parquet, manifest.json}`
    against an already-built dataset root."""
    import argparse
    from dvrk_data_processing.surgsync.index import (
        build_episodes_index, build_frames_index, build_stats, build_manifest,
    )

    p = argparse.ArgumentParser(prog="surgsync index", description=_index_main.__doc__)
    p.add_argument("dataset_root", help="Path to the dataset root")
    p.add_argument("--data-version", default="1.0",
                   help="data_version string written into manifest.json (default: '1.0')")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 1

    log.info("rebuilding indexes under %s", root)
    ep = build_episodes_index(root)
    fr = build_frames_index(root)
    st = build_stats(root)
    mn = build_manifest(root, data_version=args.data_version)
    print(
        f"index OK: episodes={ep.get('n_episodes', 0)} "
        f"frames={fr.get('n_frames', 0)} stats_cols={st.get('n_columns', 0)} "
        f"manifest_files={mn.get('n_files', 0)}"
    )
    return 0


# ---------------------------------------------------------------------------
# validate — three-layer validators
# ---------------------------------------------------------------------------

def _validate_main(argv: Optional[list[str]] = None) -> int:
    """Run validators against a clip / episode / dataset."""
    import argparse
    from dvrk_data_processing.surgsync.validate import (
        validate_raw_clip, validate_episode, validate_dataset,
    )

    p = argparse.ArgumentParser(prog="surgsync validate", description=_validate_main.__doc__)
    p.add_argument("--layer", choices=["raw_clip", "episode", "dataset", "all"],
                   default="all")
    p.add_argument("--raw-clip", default=None,
                   help="Path to one raw clip dir (for layer=raw_clip|all)")
    p.add_argument("--episode", default=None,
                   help="Path to one finalized episode dir (for layer=episode|all)")
    p.add_argument("--dataset-root", default=None,
                   help="Path to the dataset root (for layer=dataset|all)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    issues_total = 0
    layers_run: list[str] = []

    if args.layer in ("raw_clip", "all") and args.raw_clip:
        issues = validate_raw_clip(Path(args.raw_clip))
        layers_run.append("raw_clip")
        for it in issues:
            print(f"[raw_clip] {it.severity}: {it.message}")
            if it.severity == "ERROR":
                issues_total += 1

    if args.layer in ("episode", "all") and args.episode:
        issues = validate_episode(Path(args.episode))
        layers_run.append("episode")
        for it in issues:
            print(f"[episode] {it.severity}: {it.message}")
            if it.severity == "ERROR":
                issues_total += 1

    if args.layer in ("dataset", "all") and args.dataset_root:
        issues = validate_dataset(Path(args.dataset_root))
        layers_run.append("dataset")
        for it in issues:
            print(f"[dataset] {it.severity}: {it.message}")
            if it.severity == "ERROR":
                issues_total += 1

    if not layers_run:
        print("validate: nothing to do — pass --raw-clip / --episode / --dataset-root",
              file=sys.stderr)
        return 1

    print(f"validate: {issues_total} ERROR(s) across {', '.join(layers_run)}")
    return 0 if issues_total == 0 else 2


# ---------------------------------------------------------------------------
# unpack — decompose a built dataset into the pre-pack raw + preprocess tree
# ---------------------------------------------------------------------------

def _unpack_main(argv: Optional[list[str]] = None) -> int:
    """Decompose a packed SurgSync dataset back to the raw + preprocess
    layout (the inverse of `surgsync build`)."""
    import argparse
    from dvrk_data_processing.surgsync.decompose import decompose

    p = argparse.ArgumentParser(prog="surgsync unpack", description=_unpack_main.__doc__)
    p.add_argument("dataset_root", help="Path to the packed dataset root")
    p.add_argument("--out", required=True, help="Output root for the decomposed tree")
    p.add_argument("--episode-id", action="append", default=None,
                   help="Filter to one or more episode_ids (repeatable)")
    p.add_argument("--clip", action="append", default=None,
                   help="Filter to one or more <dataset>/<clip_idx> pairs (repeatable)")
    p.add_argument("--task", action="append", default=None,
                   help="Filter to one or more task names (repeatable)")
    p.add_argument("--dataset-name", action="append", default=None,
                   help="Filter to one or more top-level dataset names "
                        "(e.g. online_data) (repeatable)")
    p.add_argument("--streams", default="raw,preprocess",
                   help="Comma-separated subset of {raw,preprocess}. "
                        "Default writes both.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite already-populated output clip dirs.")
    p.add_argument("--parallelism", type=int, default=1,
                   help="Pack N clips concurrently via ProcessPoolExecutor.")
    p.add_argument("--workers-per-clip", type=int, default=4,
                   help="PNG-writer threads per clip.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 1

    streams = tuple(s.strip() for s in args.streams.split(",") if s.strip())
    unknown = set(streams) - {"raw", "preprocess"}
    if unknown:
        print(f"unknown --streams entries: {sorted(unknown)}; allowed: raw, preprocess",
              file=sys.stderr)
        return 1

    report = decompose(
        dataset_root=root,
        out_root=Path(args.out),
        episode_ids=args.episode_id,
        clips=args.clip,
        tasks=args.task,
        dataset_names=args.dataset_name,
        streams=streams,
        force=args.force,
        parallelism=args.parallelism,
        workers_per_clip=args.workers_per_clip,
    )
    print(
        f"unpack OK: {report.n_episodes_ok}/{report.n_episodes_seen} "
        f"episodes; {report.n_episodes_fail} failed. "
        f"Report: {Path(args.out) / 'decompose_report.json'}"
    )
    return 0 if report.n_episodes_fail == 0 else 2


# ---------------------------------------------------------------------------
# release — README + CHANGELOG generator
# ---------------------------------------------------------------------------

def _release_main(argv: Optional[list[str]] = None) -> int:
    """Write README.md + CHANGELOG.md into the dataset root; optionally
    bump `meta/dataset.json:data_version` per semver."""
    import argparse
    from dvrk_data_processing.surgsync.pipeline.release import run_release

    p = argparse.ArgumentParser(prog="surgsync release", description=_release_main.__doc__)
    p.add_argument("dataset_root", help="Path to the packed dataset root")
    p.add_argument("--bump-version", choices=["patch", "minor", "major"], default=None,
                   help="Bump meta/dataset.json:data_version per semver before writing docs.")
    p.add_argument("--notes", default=None,
                   help="CHANGELOG entry body. Omit to leave a TODO placeholder "
                        "for the next manual diff.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"dataset root does not exist: {root}", file=sys.stderr)
        return 1

    summary = run_release(root, bump=args.bump_version, notes=args.notes)
    print(
        f"release OK: data_version={summary['data_version']}"
        + (f" (bumped {summary['bumped_from']} → {summary['bumped_to']})"
           if summary.get('bumped_to') else "")
        + f"; episodes={summary['n_episodes']}; tasks={summary['n_tasks']}; "
          f"frames={summary['total_frames']}\n"
        f"  README:    {summary['readme_path']}\n"
        f"  CHANGELOG: {summary['changelog_path']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """Multiplex `build` / `selftest` / `validate` / `index` subcommands.

    Hydra's normal `@hydra.main` swallows argv, so we manually inspect
    the first non-flag argument to pick the subcommand, then let Hydra
    handle the rest.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0

    subcmd = argv[0]
    sys.argv = [sys.argv[0], *argv[1:]]   # rewrite for hydra

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if subcmd == "build":
        _build_main()  # hydra wraps; returns None on success, raises on failure
        return 0
    if subcmd == "selftest":
        return _selftest_main()
    if subcmd == "validate":
        return _validate_main()
    if subcmd == "index":
        return _index_main()
    if subcmd == "unpack":
        return _unpack_main()
    if subcmd == "release":
        return _release_main()

    print(f"Unknown subcommand: {subcmd!r}\n", file=sys.stderr)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
