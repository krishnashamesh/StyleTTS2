#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 — Final Cutting (audio is truth)

Inputs:
  - proposals.json from Phase 2 (contains utt_id, speaker, guarded times, sample indices, etc.)
  - clip_full_24k.wav (24 kHz speech master from Phase 0)

What this does:
  - Renders 1 WAV per proposal (no drops)
  - Applies small fade-in/out (default 15 ms) and pad (default 20 ms) for TTS stability
  - Clamps to audio bounds, heals degenerate spans (guarantees >= 1 sample)
  - Writes PCM16 WAVs deterministically
  - Emits a breadcrumbs JSONL (immutable provenance + stats)

Outputs:
  - {out_wavs_dir}/utt_*.wav
  - {breadcrumbs} (JSONL)
  - A summary line with counts and edge-case notes

Notes:
  - We trust proposals.json; Phase 3 never re-segments or reorders.
  - If proposals["sr"] != audio SR, we recompute sample indices from guarded times.
"""

import argparse, json, os, sys, math
from dataclasses import asdict
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf

def _load_audio(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:,0]
    return audio, sr

def _safe_idx(x: int, n: int) -> int:
    return min(max(0, x), n)

def _apply_fades(chunk: np.ndarray, sr: int, fade_ms: int) -> Tuple[np.ndarray, bool]:
    if fade_ms <= 0: 
        return chunk, False
    fade_samps = int(round(sr * fade_ms / 1000.0))
    if fade_samps <= 0 or chunk.size == 0:
        return chunk, False
    # If clip shorter than 2*fade, shrink fades proportionally
    shrink = False
    if chunk.size < 2 * fade_samps:
        fade_samps = max(1, chunk.size // 4)
        shrink = True
    # linear ramps
    ramp_in  = np.linspace(0.0, 1.0, fade_samps, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, fade_samps, dtype=np.float32)
    out = chunk.copy()
    out[:fade_samps] *= ramp_in
    out[-fade_samps:] *= ramp_out
    return out, shrink

def _apply_pad(chunk: np.ndarray, sr: int, pad_ms: int) -> np.ndarray:
    if pad_ms <= 0:
        return chunk
    pad_samps = int(round(sr * pad_ms / 1000.0))
    if pad_samps <= 0:
        return chunk
    pre = np.zeros(pad_samps, dtype=np.float32)
    post = np.zeros(pad_samps, dtype=np.float32)
    return np.concatenate([pre, chunk, post], axis=0)

def _pcm16(x: np.ndarray) -> np.ndarray:
    # Soft clip to prevent wrap; keep headroom
    x = np.clip(x, -0.999, 0.999)
    return (x * 32767.0).astype(np.int16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio24k", required=True, help="24 kHz master (clip_full_24k.wav)")
    ap.add_argument("--proposals_json", required=True, help="Phase-2 proposals.json path")
    ap.add_argument("--out_wavs_dir", required=True, help="Directory to write cuts/utt_*.wav")
    ap.add_argument("--breadcrumbs", required=True, help="Path to breadcrumbs JSONL")
    ap.add_argument("--fade_ms", type=int, default=15, help="Fade-in/out ms (default 15)")
    ap.add_argument("--pad_ms", type=int, default=20, help="Pad ms on both sides (default 20)")
    args = ap.parse_args()

    os.makedirs(args.out_wavs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.breadcrumbs), exist_ok=True)

    # Load source audio
    audio, sr = _load_audio(args.audio24k)
    n = audio.size

    # Load proposals
    P = json.load(open(args.proposals_json, "r", encoding="utf-8"))
    items = P.get("items", [])
    prop_sr = int(P.get("sr", sr))

    if sr != prop_sr:
        print(f"[WARN] proposals SR={prop_sr} ≠ audio SR={sr}. Using times to compute indices.")

    # Render loop
    wrote = 0
    healed = 0
    fade_shrunk = 0
    clipped = 0

    with open(args.breadcrumbs, "w", encoding="utf-8") as logf:
        for it in items:
            utt_id  = it["utt_id"]
            spk     = it["speaker"]
            purity  = float(it.get("purity", 1.0))
            # Prefer guarded times
            tg0 = float(it.get("t_start_guarded", it.get("t_start")))
            tg1 = float(it.get("t_end_guarded",   it.get("t_end")))
            if sr == prop_sr and "s_start_24k" in it and "s_end_24k" in it:
                s0 = int(it["s_start_24k"])
                s1 = int(it["s_end_24k"])
            else:
                s0 = int(round(tg0 * sr))
                s1 = int(round(tg1 * sr))

            # Clamp to bounds
            s0 = _safe_idx(s0, n)
            s1 = _safe_idx(s1, n)

            # Heal degenerate spans (never drop)
            fix_short = False
            if s1 <= s0:
                want = max(1, int(round(0.01 * sr)))  # +10 ms
                s1 = _safe_idx(s0 + want, n)
                if s1 <= s0:  # if at tail, expand left
                    s0 = _safe_idx(max(0, s1 - want), n)
                fix_short = True
                healed += 1

            chunk = audio[s0:s1]
            # Fades
            chunk, shr = _apply_fades(chunk, sr, args.fade_ms)
            if shr: fade_shrunk += 1
            # Pads
            chunk = _apply_pad(chunk, sr, args.pad_ms)

            # Peak check
            peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            was_clipped = peak >= 1.0 - 1e-6
            if was_clipped: clipped += 1

            # Write WAV
            out_path = os.path.join(args.out_wavs_dir, f"{utt_id}.wav")
            sf.write(out_path, _pcm16(chunk), sr, subtype="PCM_16")

            # Breadcrumbs (JSONL): immutable provenance + ops
            rec = {
                "utt_id": utt_id,
                "out_path": out_path,
                "speaker": spk,
                "purity": round(purity, 3),
                "src_audio": os.path.abspath(args.audio24k),
                "sr": sr,
                "proposal_sr": prop_sr,
                "proposal_times": {
                    "t_start": float(it.get("t_start")),
                    "t_end": float(it.get("t_end")),
                    "t_start_guarded": tg0,
                    "t_end_guarded": tg1
                },
                "sample_indices": {
                    "s_start": s0,
                    "s_end": s1,
                    "len_samples": int(chunk.size)
                },
                "ops": {
                    "fade_ms": args.fade_ms,
                    "pad_ms": args.pad_ms,
                    "fade_shrunk": bool(shr),
                    "fix_short_applied": bool(fix_short),
                },
                "stats": {
                    "peak_abs": round(peak, 6),
                    "dur_seconds": round(float(chunk.size)/sr, 6)
                },
                "provenance": {
                    "dia_idx": it.get("dia_idx"),
                    "intra_idx": it.get("intra_idx")
                }
            }
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[Phase3] Wrote {wrote} cuts to {args.out_wavs_dir}")
    print(f"[Phase3] Healed degenerate spans: {healed}  |  Fade windows shrunk: {fade_shrunk}  |  Peak>=1.0 before PCM: {clipped}")
    print(f"[Phase3] Breadcrumbs: {args.breadcrumbs}")
    print("[Phase3] 1:1 invariant preserved (no drops).")

if __name__ == "__main__":
    main()
