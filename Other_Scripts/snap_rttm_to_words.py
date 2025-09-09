#!/usr/bin/env python3
"""
Snap a diarization RTTM to word edges using either:
  (A) NFA word CTM  (--ctm path/to/words.ctm), or
  (B) aligned JSON  (--aligned_json path/to/aligned.json)
      Supported JSON shapes:
        - {"utterances":[{"words":[{"start":..,"end":..,"word":".."}, ...]}, ...]}
        - {"segments":[{"words":[...]}]}   # WhisperX-compatible
        - {"words":[...]}                  # flat words list

Output: a snapped (and lightly gap-filled) RTTM.

Default behavior:
- Snap each diar turn boundary to the nearest word edge if within --snap_ms (default 150 ms).
- Pad a tiny margin around each word (--word_pad_ms, default 80 ms) to account for breath/consonant tails.
- If a diar gap is small (<= --fill_gap_ms, default 350 ms) and words overlap the gap, bridge it (eliminate the gap).

Usage:
  python snap_rttm_to_words.py \
    --rttm_in  clip_trim_16k.rttm \
    --ctm      /opt/apps/NeMo/nfa_out_parakeetctc/ctm/words/clip_trim_16k.ctm \
    --rttm_out clip_trim_16k.snapped.rttm

  # or with JSON
  python snap_rttm_to_words.py \
    --rttm_in  clip_trim_16k.rttm \
    --aligned_json /opt/apps/whisperx_compat/aligned_clean.json \
    --rttm_out clip_trim_16k.snapped.rttm
"""

import json
import argparse
from pathlib import Path

def merge_intervals(ints):
    """Merge overlapping intervals."""
    ints = sorted(ints)
    out = []
    for s, e in ints:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def load_words_from_ctm(ctm_path, pad=0.0):
    """Parse NFA word CTM into padded intervals [(s,e), ...]."""
    edges = []
    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                s = float(p[2]); d = float(p[3]); e = s + d
            except Exception:
                continue
            # skip blanks
            w = p[4]
            if w == "<eps>":
                continue
            edges.append((max(0.0, s - pad), e + pad))
    return merge_intervals(edges)

def load_words_from_json(json_path, pad=0.0):
    """Accept clean JSON, WhisperX-compatible, or flat words list."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    words = []

    # clean schema: {"utterances":[{"words":[...]}]}
    if isinstance(data, dict) and "utterances" in data:
        for u in data["utterances"]:
            for w in u.get("words", []):
                s = float(w["start"]); e = float(w["end"])
                words.append((max(0.0, s - pad), e + pad))

    # whisperx-compatible: {"segments":[{"words":[...]}]}
    elif isinstance(data, dict) and "segments" in data:
        for seg in data["segments"]:
            for w in seg.get("words", []):
                s = float(w["start"]); e = float(w["end"])
                words.append((max(0.0, s - pad), e + pad))

    # flat list: {"words":[...]}
    elif isinstance(data, dict) and "words" in data:
        for w in data["words"]:
            s = float(w["start"]); e = float(w["end"])
            words.append((max(0.0, s - pad), e + pad))

    else:
        raise ValueError("Unsupported JSON shape for aligned words.")

    return merge_intervals(words)

def parse_rttm(rttm_path):
    """Return list of [start, end, speaker, fileid] from RTTM."""
    turns = []
    for line in Path(rttm_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or not line.startswith("SPEAKER"):
            continue
        p = line.split()
        fileid = p[1]
        start = float(p[3])
        dur = float(p[4])
        end = start + dur
        spk = p[7]
        turns.append([start, end, spk, fileid])
    turns.sort()
    return turns

def snap_rttm_to_word_edges(rttm_turns, word_blocks, snap_tol=0.15, fill_gap=0.35):
    """Snap boundaries to nearest word edge within snap_tol and bridge tiny gaps if words exist in the gap."""
    # Precompute candidate edges (starts/ends of word blocks)
    edges = [t for se in word_blocks for t in se]

    def nearest_edge(x):
        if not edges:
            return x
        t = min(edges, key=lambda t0: abs(t0 - x))
        return t if abs(t - x) <= snap_tol else x

    # Snap each turn
    snapped = []
    for s, e, spk, fid in rttm_turns:
        s2 = nearest_edge(s)
        e2 = nearest_edge(e)
        # prevent inversion; if snapping collapses, keep original
        if e2 <= s2:
            s2, e2 = s, e
        # enforce tiny minimum turn length
        if e2 - s2 < 0.05:
            s2, e2 = s, e
        snapped.append([s2, e2, spk, fid])

    snapped.sort()

    # Bridge tiny gaps if words cover the gap
    filled = []
    for s, e, spk, fid in snapped:
        if filled:
            prev_s, prev_e, prev_spk, prev_fid = filled[-1]
            gap = s - prev_e
            if gap > 0.0 and gap <= fill_gap:
                # any word block overlapping [prev_e, s] ?
                has_asr = any(not (wb[1] <= prev_e or wb[0] >= s) for wb in word_blocks)
                if has_asr:
                    # extend previous segment to s (eliminate gap)
                    filled[-1][1] = s
        filled.append([s, e, spk, fid])

    # Optional: if ASR extends a bit beyond the last diar turn, extend last turn modestly
    if word_blocks and filled:
        last_asr_end = word_blocks[-1][1]
        if last_asr_end - filled[-1][1] <= 2.5 and last_asr_end > filled[-1][1]:
            filled[-1][1] = last_asr_end

    return filled

def write_rttm(turns, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for s, e, spk, fid in turns:
            f.write(f"SPEAKER {fid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rttm_in", required=True, help="Input RTTM from NeMo diarizer")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--ctm", help="NFA word-level CTM (ctm/words/*.ctm)")
    group.add_argument("--aligned_json", help="Aligned words JSON (clean or WhisperX-compatible)")
    ap.add_argument("--rttm_out", required=True, help="Output snapped RTTM path")
    ap.add_argument("--word_pad_ms", type=float, default=80.0, help="Pad around each word edge (ms)")
    ap.add_argument("--snap_ms", type=float, default=150.0, help="Max distance to snap boundary to nearest word edge (ms)")
    ap.add_argument("--fill_gap_ms", type=float, default=350.0, help="Max diar gap to bridge if words overlap (ms)")
    args = ap.parse_args()

    turns = parse_rttm(args.rttm_in)

    pad = args.word_pad_ms / 1000.0
    snap_tol = args.snap_ms / 1000.0
    fill_gap = args.fill_gap_ms / 1000.0

    if args.ctm:
        word_blocks = load_words_from_ctm(args.ctm, pad=pad)
    else:
        word_blocks = load_words_from_json(args.aligned_json, pad=pad)

    snapped = snap_rttm_to_word_edges(turns, word_blocks, snap_tol=snap_tol, fill_gap=fill_gap)
    write_rttm(snapped, args.rttm_out)
    print(f"Wrote snapped RTTM: {args.rttm_out}")

if __name__ == "__main__":
    main()
