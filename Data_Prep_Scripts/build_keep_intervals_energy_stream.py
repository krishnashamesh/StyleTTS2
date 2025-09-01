#!/usr/bin/env python3
# build_keep_intervals_energy_stream.py

"""
Usage

python build_keep_intervals_energy_stream.py \
/opt/apps/bandit/workspace/clip_full_16k.wav \
/opt/apps/bandit/workspace/keep_intervals.json \
--top_db 35 --min_silence_sec 2.0 --block_sec 30

"""


import argparse, json
import numpy as np
import soundfile as sf
from math import floor

def scan_peak_and_duration(path, block_sec=30.0):
    with sf.SoundFile(path, 'r') as f:
        sr = f.samplerate
        block = int(sr * block_sec)
        peak = 0.0
        total = 0
        while True:
            x = f.read(block, dtype='float32', always_2d=False)
            if x.size == 0: break
            peak = max(peak, float(np.max(np.abs(x))))
            total += x.size
    return peak, total / sr, sr

def stream_rms_frames(path, sr, frame_ms=25.0, hop_ms=10.0, block_sec=30.0):
    """Yield (frame_start_sec, frame_end_sec, rms) streaming without loading whole file."""
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len   = int(sr * hop_ms   / 1000.0)
    assert frame_len > 0 and hop_len > 0
    keep_tail = frame_len - hop_len  # samples to keep between blocks
    block = int(sr * block_sec)

    buf = np.zeros(0, dtype=np.float32)
    pos0 = 0  # absolute sample index of buf[0]
    with sf.SoundFile(path, 'r') as f:
        while True:
            x = f.read(block, dtype='float32', always_2d=False)
            if x.size == 0 and buf.size < frame_len:
                break
            buf = np.concatenate([buf, x]) if buf.size else x
            # how many frames we can compute now?
            if buf.size >= frame_len:
                n_frames = 1 + (buf.size - frame_len) // hop_len
                for i in range(int(n_frames)):
                    s = i * hop_len
                    e = s + frame_len
                    frame = buf[s:e]
                    rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                    fs = (pos0 + s) / sr
                    fe = (pos0 + e) / sr
                    yield fs, fe, rms
                # drop consumed, keep tail
                drop = int(n_frames * hop_len)
                pos0 += drop
                buf = buf[drop:]
            if x.size == 0:
                break

def build_keep_intervals(path16,
                         top_db=35.0,
                         min_silence_sec=2.0,
                         frame_ms=25.0,
                         hop_ms=10.0,
                         block_sec=30.0):
    peak, duration, sr = scan_peak_and_duration(path16, block_sec)
    # If the file is entirely silent, keep nothing (or keep tiny?), here: keep nothing → return empty keep
    if peak <= 1e-9:
        return {"duration": duration, "keep": [], "params":{
            "top_db": top_db, "min_silence_sec": min_silence_sec,
            "frame_ms": frame_ms, "hop_ms": hop_ms, "block_sec": block_sec,
            "ref": "db below global peak"
        }}

    thr = peak * (10.0 ** (-top_db / 20.0))

    silent_runs = []
    in_sil = False
    sil_start = None
    last_frame_end = 0.0

    for fs, fe, rms in stream_rms_frames(path16, sr, frame_ms, hop_ms, block_sec):
        last_frame_end = fe
        is_sil = (rms < thr)
        if is_sil and not in_sil:
            in_sil = True
            sil_start = fs
        elif (not is_sil) and in_sil:
            # silence ended at previous frame end
            if (fe - sil_start) >= min_silence_sec:
                silent_runs.append((sil_start, fe))
            in_sil = False
            sil_start = None

    # trailing silence
    if in_sil:
        if (duration - sil_start) >= min_silence_sec:
            silent_runs.append((sil_start, duration))

    # Complement → keep intervals
    keep = []
    cursor = 0.0
    for s, e in silent_runs:
        if s > cursor:
            keep.append([cursor, s])
        cursor = e
    if cursor < duration:
        keep.append([cursor, duration])

    # If nothing qualified as long silence, keep whole file
    if not keep:
        keep = [[0.0, duration]]

    return {
        "duration": duration,
        "keep": keep,
        "params": {
            "top_db": top_db,
            "min_silence_sec": min_silence_sec,
            "frame_ms": frame_ms,
            "hop_ms": hop_ms,
            "block_sec": block_sec,
            "ref": "db below global peak"
        }
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_wav", help="16 kHz mono proxy (full timeline)")
    ap.add_argument("out_json", help="Output keep_intervals.json")
    ap.add_argument("--top_db", type=float, default=35.0,
                    help="Frames with RMS < peak*10^(-top_db/20) are 'silence'.")
    ap.add_argument("--min_silence_sec", type=float, default=2.0,
                    help="Only remove silent spans >= this length.")
    ap.add_argument("--frame_ms", type=float, default=25.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--block_sec", type=float, default=30.0,
                    help="Streaming read size (seconds).")
    args = ap.parse_args()

    meta = build_keep_intervals(
        args.in_wav, args.top_db, args.min_silence_sec,
        args.frame_ms, args.hop_ms, args.block_sec
    )
    with open(args.out_json, "w") as f:
        json.dump(meta, f, indent=2)
    kept = sum(e - s for s, e in meta["keep"])
    print(f"[keep] spans={len(meta['keep'])}, kept≈{kept:.2f}s / dur≈{meta['duration']:.2f}s")
