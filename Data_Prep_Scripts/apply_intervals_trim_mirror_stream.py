#!/usr/bin/env python3
# apply_intervals_trim_mirror_stream.py

"""

python apply_intervals_trim_mirror_stream.py \
/opt/apps/bandit/workspace/clip_full_16k.wav \
/opt/apps/bandit/workspace/clip_full_24k.wav \
/opt/apps/bandit/workspace/keep_intervals.json \
/opt/apps/bandit/workspace/clip_trim_16k.wav \
/opt/apps/bandit/workspace/clip_trim_24k.wav \
--pad_sec 0.05 --fade_ms 5 --block_sec 5

"""


import argparse, json
import numpy as np
import soundfile as sf

def write_intervals_stream(in_path, out_path, sr_expect, keep, pad_sec=0.05,
                           fade_ms=5.0, block_sec=5.0):
    with sf.SoundFile(in_path, 'r') as fin:
        sr = fin.samplerate
        assert sr == sr_expect, f"Expected {sr_expect} Hz, got {sr}"
        n_total = len(fin)
        dur = n_total / sr
        block = int(sr * block_sec)
        fade_n = int(sr * (fade_ms / 1000.0))
        with sf.SoundFile(out_path, 'w', samplerate=sr, channels=1, subtype='PCM_16') as fout:
            for (s, e) in keep:
                # expand and clamp
                s_ = max(0.0, s - pad_sec)
                e_ = min(dur, e + pad_sec)
                start = int(round(s_ * sr))
                end   = int(round(e_ * sr))
                fin.seek(start)
                first = True
                pos = start
                while pos < end:
                    to_read = min(block, end - pos)
                    x = fin.read(to_read, dtype='float32', always_2d=False)
                    if x.size == 0:
                        break
                    if first and fade_n > 0:
                        n = min(fade_n, x.size)
                        x[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32, endpoint=False)
                        first = False
                    fout.write(x)
                    pos += x.size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in16", help="Full-timeline 16 kHz mono WAV")
    ap.add_argument("in24", help="Full-timeline 24 kHz mono WAV")
    ap.add_argument("keep_json", help="keep_intervals.json produced earlier")
    ap.add_argument("out16", help="Output trimmed 16 kHz WAV")
    ap.add_argument("out24", help="Output trimmed 24 kHz WAV")
    ap.add_argument("--pad_sec", type=float, default=0.05, help="Safety pad around keeps (sec)")
    ap.add_argument("--fade_ms", type=float, default=5.0, help="Fade-in at each join (ms)")
    ap.add_argument("--block_sec", type=float, default=5.0, help="Streaming read size (sec)")
    args = ap.parse_args()

    meta = json.load(open(args.keep_json))
    keep = meta["keep"]

    write_intervals_stream(args.in16, args.out16, 16000, keep,
                           pad_sec=args.pad_sec, fade_ms=args.fade_ms, block_sec=args.block_sec)
    write_intervals_stream(args.in24, args.out24, 24000, keep,
                           pad_sec=args.pad_sec, fade_ms=args.fade_ms, block_sec=args.block_sec)

    # quick sanity
    with sf.SoundFile(args.out16) as f1, sf.SoundFile(args.out24) as f2:
        print(f"[out] 16k {len(f1)/f1.samplerate:.2f}s, 24k {len(f2)/f2.samplerate:.2f}s (should match)")

if __name__ == "__main__":
    main()
