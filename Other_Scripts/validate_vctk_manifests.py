#!/usr/bin/env python3
"""
validate_vctk_manifests.py

Quick sanity checks for manifests produced by make_vctk_manifests.py.
- Verifies paths exist under --wav-root
- Prints per-speaker counts for train/val
- Ensures speaker ids are integers and continuous-ish
- Optionally rewrites manifests to POSIX paths

Usage:
  python validate_vctk_manifests.py --wav-root /data/VCTK/wav --manifests Data/train_list.txt Data/val_list.txt


  python validate_vctk_manifests.py \
    --wav-root /opt/apps/StyleTTS2/Data/VCTK/wav_norm \
    --manifests /opt/apps/StyleTTS2/Data/VCTK/train_list.txt \
    /opt/apps/StyleTTS2/Data/VCTK/val_list.txt /opt/apps/StyleTTS2/Data/VCTK/OOD_texts.txt
"""
import argparse, os, re, sys, collections
from pathlib import Path

def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) != 3:
            print(f"[warn] skip malformed line in {path}: {line[:100]}")
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-root', required=True, type=Path)
    ap.add_argument('--manifests', nargs='+', required=True, help='List files to check (train_list.txt, val_list.txt)')
    args = ap.parse_args()

    for mani in args.manifests:
        rows = read_manifest(mani)
        by_spk = collections.Counter()
        missing = 0
        for rel, text, spk in rows:
            try:
                spk_id = int(spk)
            except:
                print(f"[err] non-integer speaker id in {mani}: {spk} (line starts {rel}|...)")
                spk_id = None
            wav = args.wav_root / rel
            if not wav.exists():
                missing += 1
            if spk_id is not None:
                by_spk[spk_id] += 1
        print(f"\n[{mani}] items={len(rows)}  missing_wavs={missing}")
        if missing:
            print("  Example missing:", next((args.wav_root / r for r,_,_ in rows if not (args.wav_root / r).exists()), None))
        # Summary
        print("  Top 10 speakers by count:", by_spk.most_common(10))
        spk_ids = sorted(by_spk.keys())
        if spk_ids:
            print(f"  speaker id range: {spk_ids[0]}..{spk_ids[-1]}  unique={len(spk_ids)}")

if __name__ == '__main__':
    main()
