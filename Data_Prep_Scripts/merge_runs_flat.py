#!/usr/bin/env python3
import argparse, glob, os, shutil, time
from pathlib import Path

MANIFEST_DIR = "manifests_ipa"
FILES = ["train_list.txt", "val_list.txt", "OOD_texts.txt"]

def merge_flat(src_dir: Path, dst_dir: Path, exts: tuple[str, ...] | None, run_prefix: str,
               return_mapping: bool = False) -> dict[str, str]:
    """
    Flatten-copy files from src_dir (recursively) into dst_dir.
    If exts is not None, only copy files whose suffix matches exts.
    On name collision, prefix with '<run_prefix>__'.
    If return_mapping=True, returns {abs_src: abs_dst} for copied files.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    if not src_dir.exists():
        return mapping
    for root, _, files in os.walk(src_dir):
        root_p = Path(root)
        for f in files:
            if exts and Path(f).suffix.lower() not in exts:
                continue
            src = root_p / f
            out = dst_dir / f
            if out.exists():
                out = dst_dir / f"{run_prefix}__{f}"
            shutil.copy2(src, out)
            if return_mapping:
                mapping[str(src.resolve())] = str(out.resolve())
    return mapping

def rewrite_line(line: str, mapping: dict[str, str], merged_cuts: Path) -> str:
    """
    Expects pipe format: PATH|TEXT|SPK
    Rewrites PATH using mapping; if not found, tries basename match inside merged_cuts.
    Leaves non-pipe lines unchanged (just in case).
    """
    s = line.rstrip("\n")
    if "|" not in s:
        return line  # keep as-is
    parts = s.split("|", 2)
    if not parts or len(parts[0].strip()) == 0:
        return line

    orig_path = parts[0].strip()
    # exact match first
    new_path = mapping.get(orig_path)
    if not new_path:
        # try normalized absolute
        try:
            orig_abs = str(Path(orig_path).resolve())
            new_path = mapping.get(orig_abs)
        except Exception:
            new_path = None

    if not new_path:
        # fallback: basename lookup in merged cuts
        base = Path(orig_path).name
        candidates = list(merged_cuts.glob(base))
        if len(candidates) == 1:
            new_path = str(candidates[0].resolve())
        else:
            # try any prefixed versions if collision occurred
            candidates = list(merged_cuts.glob(f"*__{base}"))
            if len(candidates) == 1:
                new_path = str(candidates[0].resolve())

    if new_path:
        parts[0] = new_path
        return "|".join(parts) + "\n"
    else:
        # No mapping found; leave line unchanged but keep format
        return line if line.endswith("\n") else (line + "\n")

def main():
    ap = argparse.ArgumentParser(description="Merge runs (flat) and rewrite manifests to merged cuts/")
    ap.add_argument("base_dir", nargs="?", default=".", help="Base directory containing run folders")
    ap.add_argument("--pattern", required=True, help='Glob for runs, e.g. "Sep_*"')
    ap.add_argument("--out", default=None, help="Merged folder name (default: <pattern>_merged_<timestamp>)")
    ap.add_argument("--dry_run", action="store_true", help="List actions only")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    runs = sorted([Path(p) for p in glob.glob(str(base / args.pattern)) if Path(p).is_dir()])
    if not runs:
        print(f"[!] No runs matched {base}/{args.pattern}")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    merged_name = args.out or f"{args.pattern.replace('*','STAR')}_merged_{stamp}"
    merged = base / merged_name
    cuts_merged = merged / "cuts"
    mel_merged  = merged / "mel_cache"
    mani_merged = merged / MANIFEST_DIR

    print(f"[i] Base:    {base}")
    print(f"[i] Pattern: {args.pattern}")
    print(f"[i] Runs:    {', '.join(r.name for r in runs)}")
    print(f"[i] Output:  {merged}")

    if args.dry_run:
        return

    merged.mkdir(parents=True, exist_ok=True)
    mani_merged.mkdir(parents=True, exist_ok=True)

    # 1) Merge cuts (flat) and build mapping orig_abs -> merged_abs
    path_mapping = {}
    wav_exts = (".wav",)
    total_wavs = 0
    for r in runs:
        m = merge_flat(r / "cuts", cuts_merged, wav_exts, run_prefix=r.name, return_mapping=True)
        path_mapping.update(m)
        total_wavs += len(m)

    # 2) Merge mel_cache (flat, all files)
    for r in runs:
        merge_flat(r / "mel_cache", mel_merged, None, run_prefix=r.name, return_mapping=False)

    # 3) Concatenate & rewrite manifests from manifests_ipa only
    for fname in FILES:
        out_fp = mani_merged / fname
        with out_fp.open("w", encoding="utf-8") as out_f:
            for r in runs:
                src_fp = r / MANIFEST_DIR / fname
                if not src_fp.exists():
                    continue
                with src_fp.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        # Rewrite WAV path (field #1) to merged cuts
                        out_f.write(rewrite_line(line, path_mapping, cuts_merged))

    # 4) Summary
    print("\n=== Summary ===")
    print(f"[=] WAVs merged:       {total_wavs}")
    print(f"[=] cuts/:             {cuts_merged}")
    print(f"[=] mel_cache/:        {mel_merged}")
    print(f"[=] manifests_ipa/:    {mani_merged}")
    for f in FILES:
        print(f"    - {f}: {'ok' if (mani_merged/f).exists() else 'missing'}")

if __name__ == "__main__":
    main()
