#!/usr/bin/env python3
"""
Precompute mel-spectrogram caches for StyleTTS2 (offline), using the SAME
preprocess parameters as meldataset.py (n_mels=80, n_fft=2048, win=1200, hop=300)
and log-normalisation (mean=-4, std=4).

Defaults are read from the training config (Configs/config.yml by default):
- data_params.root_path
- data_params.train_data
- data_params.val_data
- data_params.mel_cache_dir   (NEW: add this in your config)
You can override any of these via CLI flags.

Usage (zero-arg, uses config defaults):
    python precompute_mels.py

Usage (custom config or paths):
    python precompute_mels.py --config path/to/config.yml
    python precompute_mels.py --root /data/wavs --train Data/train.txt --val Data/val.txt --out /opt/data/mel_cache
"""
import argparse
import os
import os.path as osp
import sys
import yaml
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

# ---------- Defaults (MUST match meldataset.py) ----------
SR_DEFAULT = 24000
N_MELS = 80
N_FFT = 2048
WIN = 1200
HOP = 300
MEAN = -4.0
STD  =  4.0

def load_cfg(cfg_path_candidates: List[str]) -> dict:
    for p in cfg_path_candidates:
        if p and osp.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Could not find config in any of: {cfg_path_candidates}")

def infer_defaults_from_cfg(cfg: dict) -> Tuple[str, str, str, str, int]:
    dp = (cfg or {}).get("data_params", {}) or {}
    pp = (cfg or {}).get("preprocess_params", {}) or {}
    spect = (pp or {}).get("spect_params", {}) or {}

    root = dp.get("root_path", "Data")
    train = dp.get("train_data", "Data/train_list.txt")
    val = dp.get("val_data", "Data/val_list.txt")
    out = dp.get("mel_cache_dir", "/opt/data/mel_cache")  # NEW: add to your config
    sr  = pp.get("sr", SR_DEFAULT)
    return root, train, val, out, int(sr)

def read_manifest_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # each line is "path|text|speaker" or "path|text"
    rels = [ln.split("|", 1)[0] for ln in lines]
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for r in rels:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq

def build_mel_transform(sr: int):
    to_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=N_MELS, n_fft=N_FFT, win_length=WIN, hop_length=HOP
    )
    return to_mel

def wav_to_mel(wave: np.ndarray, to_mel, sr: int) -> torch.Tensor:
    # mono-ise (meldataset uses first channel if stereo)
    if wave.ndim == 2:
        if wave.shape[0] < wave.shape[1]:  # soundfile returns (frames, channels)
            wave = wave[:, 0]
        else:
            wave = wave[0, :]
    x = torch.from_numpy(wave).float()
    mel = to_mel(x)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEAN) / STD
    return mel.squeeze(0).contiguous().float()  # [80, T], float32

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def process_split(root: str, rel_list: List[str], out_dir: str, to_mel, sr: int) -> Tuple[int, int]:
    done, skipped = 0, 0
    for rel in rel_list:
        wav_path = Path(root) / rel
        out_path = Path(out_dir) / Path(rel).with_suffix(".pt")
        ensure_parent(out_path)

        if out_path.exists():
            skipped += 1
            continue
        try:
            wave, in_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if int(in_sr) != int(sr):
                wave = librosa.resample(wave, orig_sr=int(in_sr), target_sr=int(sr))
            mel = wav_to_mel(np.asarray(wave), to_mel, sr)
            torch.save(mel.cpu(), str(out_path))
            done += 1
        except Exception as e:
            print(f"[warn] failed: {rel} ({e})", file=sys.stderr)
    return done, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("CFG", "Configs/config.yml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sr", type=int, default=None)
    args = ap.parse_args()

    # Load config (try a few fallbacks)
    cfg = load_cfg([args.config, "Configs/config.yml", "config.yml"])
    root_d, train_d, val_d, out_d, sr_d = infer_defaults_from_cfg(cfg)

    root = args.root or root_d
    train = args.train or train_d
    val = args.val or val_d
    out = args.out or out_d
    sr = int(args.sr or sr_d or SR_DEFAULT)

    print(f"[precompute] root={root}")
    print(f"[precompute] train={train}")
    print(f"[precompute] val={val}")
    print(f"[precompute] out={out}")
    print(f"[precompute] sr={sr}")

    Path(out).mkdir(parents=True, exist_ok=True)
    to_mel = build_mel_transform(sr)

    total_done = 0
    total_skipped = 0
    for mani in [train, val]:
        if not mani:
            continue
        rels = read_manifest_list(mani)
        print(f"[precompute] {mani}: {len(rels)} items")
        done, skipped = process_split(root, rels, out, to_mel, sr)
        print(f"[precompute] {mani}: wrote {done}, skipped {skipped}")
        total_done += done
        total_skipped += skipped

    print(f"[precompute] complete: wrote={total_done}, skipped={total_skipped}, out={out}")

if __name__ == "__main__":
    main()
