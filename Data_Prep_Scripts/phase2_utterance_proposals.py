#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — Utterance Proposals (diar-guided, silence-aware, bucketed)
Inputs:
  - blocks.cleaned.rttm (and/or blocks.cleaned.json) from Phase 1
  - 24 kHz speech master (clip_full_24k.wav)

Outputs:
  - proposals/proposals.json      # full machine-readable plan
  - proposals/proposals.tsv       # quick human scan (tab-separated)
  - proposals/stats.json          # summary metrics

Contract:
  - Never cross speaker boundaries (operate inside each diar block).
  - Split on silence gaps >= min_silence_sec.
  - Bucket to [bucket_min_sec, bucket_max_sec], target ~bucket_target_sec.
  - Enforce min_utt_sec; heal too-short singletons by attaching to neighbors.
  - Add join guards on both sides; clamp to diar block.
  - Provenance-first naming: utt_{diaidx:06d}_{split:02d}_spk{N}.
"""

import argparse, json, os, math, sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf

# ----------------- IO helpers -----------------

@dataclass
class Block:
    dia_idx: int
    speaker: str
    start: float
    end: float
    purity: float = 1.0  # defaults to 1.0 if not provided

@dataclass
class Utterance:
    utt_id: str
    dia_idx: int
    intra_idx: int
    speaker: str
    purity: float
    t_start: float
    t_end: float
    t_start_guarded: float
    t_end_guarded: float
    s_start_24k: int
    s_end_24k: int
    dur: float

def _load_blocks(rttm_path: str, json_path: Optional[str]) -> List[Block]:
    blocks: List[Block] = []
    purity_map = {}
    if json_path and os.path.exists(json_path):
        try:
            data = json.load(open(json_path, "r", encoding="utf-8"))
            for i, b in enumerate(data.get("blocks", [])):
                purity_map[(round(b["start"],3), round(b["end"],3), b["speaker"])] = float(b.get("purity", 1.0))
        except Exception:
            pass

    with open(rttm_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # RTTM: SPEAKER <rec> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
    raw = []
    for ln in lines:
        parts = ln.split()
        if parts[0] != "SPEAKER":
            continue
        start = float(parts[3]); dur = float(parts[4]); spk = parts[7]
        raw.append((start, start + dur, spk))
    raw.sort(key=lambda x: (x[0], x[1]))

    for i, (s, e, spk) in enumerate(raw):
        pur = purity_map.get((round(s,3), round(e,3), spk), 1.0)
        blocks.append(Block(dia_idx=i, speaker=spk, start=s, end=e, purity=pur))
    return blocks

# ----------------- Silence finding -----------------

def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        pad = np.zeros(win - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    n = 1 + (len(x) - win) // hop
    # stride trick
    shape = (n, win)
    strides = (x.strides[0]*hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1) + 1e-12)
    return rms

def find_silences(x: np.ndarray, sr: int, min_silence_sec: float,
                  thr_method: str = "percentile", thr_value: float = 20.0,
                  win_ms: int = 25, hop_ms: int = 10) -> List[Tuple[float, float]]:
    """
    Return list of [t0, t1] where signal is 'silent' for >= min_silence_sec.
    thr_method:
      - 'percentile': rms < percentile(thr_value) within the diar block
      - 'db': rms_db < thr_value (e.g., -45)
    """
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    rms = frame_rms(x, win, hop)
    if thr_method == "percentile":
        thr = np.percentile(rms, max(1.0, min(99.0, thr_value)))
        mask = rms < thr
    else:  # 'db'
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
        mask = rms_db < float(thr_value)
    # group contiguous regions
    silences = []
    i = 0
    min_frames = math.ceil((min_silence_sec * sr - win) / hop) + 1
    while i < len(mask):
        if mask[i]:
            j = i + 1
            while j < len(mask) and mask[j]:
                j += 1
            # [i, j) is silent
            dur_frames = j - i
            if dur_frames >= min_frames:
                t0 = i * hop / sr
                t1 = (j * hop + win) / sr
                silences.append((t0, t1))
            i = j
        else:
            i += 1
    return silences

# ----------------- Proposal logic -----------------

def split_block_on_silence(block: Block, audio: np.ndarray, sr: int,
                           min_silence_sec: float,
                           thr_method: str, thr_value: float,
                           join_guard_ms: int) -> List[Tuple[float, float]]:
    """
    Returns initial segments [t0, t1] inside [block.start, block.end], split at midpoints of silences.
    """
    s0 = int(round(block.start * sr))
    s1 = int(round(block.end * sr))
    x = audio[s0:s1]

    silences = find_silences(x, sr, min_silence_sec, thr_method, thr_value)
    # derive candidate cut points (midpoints of silences)
    cuts = []
    for (a, b) in silences:
        mid = 0.5 * (a + b)
        # avoid guards near the edges of the block
        if mid - (join_guard_ms/1000.0) <= 0.0: 
            continue
        if mid + (join_guard_ms/1000.0) >= (block.end - block.start):
            continue
        cuts.append(mid)

    cuts = sorted(set(cuts))
    # build segments from cuts
    segs = []
    last = 0.0
    for c in cuts:
        segs.append((last, c))
        last = c
    segs.append((last, block.end - block.start))

    # shift to absolute time
    abs_segs = [(block.start + a, block.start + b) for (a, b) in segs]
    return abs_segs

def heal_short_segments(segs: List[Tuple[float,float]], min_utt_sec: float) -> List[Tuple[float,float]]:
    """Attach too-short segments to neighbors to ensure every piece >= min_utt_sec."""
    if not segs:
        return []
    segs = segs[:]
    i = 0
    while i < len(segs):
        a, b = segs[i]
        if (b - a) + 1e-6 < min_utt_sec:
            if i == 0 and len(segs) > 1:
                # attach forward
                na, nb = segs[i+1]
                segs[i] = (a, nb)
                segs.pop(i+1)
                continue
            elif i > 0:
                # attach backward
                pa, pb = segs[i-1]
                segs[i-1] = (pa, b)
                segs.pop(i)
                i -= 1
                continue
        i += 1
    return segs

def bucket_segments(segs: List[Tuple[float,float]],
                    bucket_min: float, bucket_target: float, bucket_max: float) -> List[Tuple[float,float]]:
    """
    Greedy merge to hit target; never exceed max unless necessary at block tail.
    """
    if not segs:
        return []
    out = []
    cur_a, cur_b = segs[0]
    for a, b in segs[1:]:
        cur_dur = cur_b - cur_a
        next_dur = b - a
        # if current below min, keep merging
        if cur_dur < bucket_min:
            cur_b = b
            continue
        # if current near/above target and adding next would exceed max, close current
        if cur_dur >= bucket_target and (cur_dur + next_dur) > bucket_max:
            out.append((cur_a, cur_b))
            cur_a, cur_b = a, b
        else:
            # merge (stay flexible to reach target)
            cur_b = b
    out.append((cur_a, cur_b))
    # post-pass: if last < min, attach to previous when possible
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < bucket_min:
        a1, b1 = out[-2]
        a2, b2 = out[-1]
        out[-2] = (a1, b2)
        out.pop()
    return out

def _rms_envelope(x: np.ndarray, sr: int, win_ms: int, hop_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rms, times_sec) for the short-time RMS."""
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    rms = frame_rms(x, win, hop)
    # center each frame at its midpoint
    t = (np.arange(len(rms)) * hop + win / 2.0) / float(sr)
    return rms, t

def enforce_hard_cap(
    segs: List[Tuple[float,float]], block: Block, audio: np.ndarray, sr: int, *,
    hard_max_sec: float, bucket_target_sec: float, min_utt_sec: float,
    micro_gap_ms: int, micro_thr_percentile: float, micro_win_ms: int, micro_hop_ms: int
) -> List[Tuple[float,float]]:
    """
    Split any segment > hard_max_sec into smaller pieces.
    Strategy:
      - Aim for ~bucket_target_sec pieces.
      - For each desired cut time, look +/- micro_gap_ms for the local RMS minimum.
      - Respect min_utt_sec. If no good micro-dip, fall back to uniform splits.
    """
    if hard_max_sec is None or hard_max_sec <= 0:
        return segs
    out: List[Tuple[float,float]] = []
    for (a, b) in segs:
        dur = b - a
        if dur <= hard_max_sec + 1e-6:
            out.append((a, b))
            continue
        # Number of pieces: ceil(dur / hard_max_sec)
        pieces = int(math.ceil(dur / float(hard_max_sec)))
        pieces = max(2, pieces)
        # Desired cut times (absolute) roughly evenly spaced
        step = dur / pieces
        desired = [a + step * k for k in range(1, pieces)]
        # Prepare micro-RMS in the segment
        s0 = max(0, int(round(a * sr)))
        s1 = min(len(audio), int(round(b * sr)))
        x = audio[s0:s1]
        rms, t_rel = _rms_envelope(x, sr, micro_win_ms, micro_hop_ms)  # times relative to (a)
        if rms.size:
            thr = np.percentile(rms, max(1.0, min(99.0, micro_thr_percentile)))
        cuts_abs: List[float] = []
        for t_des in desired:
            # Search +/- micro_gap around desired cut
            w = micro_gap_ms / 1000.0
            lo = max(0.0, (t_des - a) - w)
            hi = min(t_rel[-1] if t_rel.size else (b - a), (t_des - a) + w)
            if rms.size:
                mask = (t_rel >= lo) & (t_rel <= hi)
                if mask.any():
                    # pick argmin RMS (prefer deep dips), else percentile threshold
                    idxs = np.where(mask)[0]
                    k = int(idxs[np.argmin(rms[idxs])])
                    t_cut = a + float(t_rel[k])
                else:
                    t_cut = t_des
            else:
                t_cut = t_des
            cuts_abs.append(t_cut)
        # Validate cuts against min_utt_sec; adjust/skip if needed
        cuts_abs = sorted([t for t in cuts_abs if a + min_utt_sec <= t <= b - min_utt_sec])
        # If everything invalid, fall back to uniform safe splits
        if not cuts_abs:
            parts = []
            start = a
            while start + hard_max_sec < b - 1e-6:
                t_cut = min(start + bucket_target_sec, b - min_utt_sec)
                if t_cut - start < min_utt_sec:
                    break
                parts.append((start, t_cut))
                start = t_cut
            if b - start >= min_utt_sec:
                parts.append((start, b))
            else:
                # if tail too short, merge back to previous
                if parts:
                    pa, pb = parts[-1]
                    parts[-1] = (pa, b)
                else:
                    parts = [(a, b)]
            out.extend(parts)
            continue
        # Build final pieces from valid cuts
        last = a
        for t_cut in cuts_abs:
            if t_cut - last >= min_utt_sec:
                out.append((last, t_cut))
                last = t_cut
        if b - last >= min_utt_sec:
            out.append((last, b))
        else:
            # merge tail into previous if too short
            if out:
                pa, pb = out[-1]
                out[-1] = (pa, b)
            else:
                out.append((a, b))
    return out

def apply_join_guards(
    seg: Tuple[float,float], block: Block,
    guard_ms: int, edge_window_ms: int, edge_scale: float
) -> Tuple[float,float]:
    """Directional guards: reduce guard near diar edges to avoid inhaling neighbour back-channels."""
    a, b = seg
    distL_ms = (a - block.start) * 1000.0
    distR_ms = (block.end - b) * 1000.0
    gL = guard_ms
    gR = guard_ms
    if distL_ms <= edge_window_ms:
        gL = max(0, int(round(gL * edge_scale)))
    if distR_ms <= edge_window_ms:
        gR = max(0, int(round(gR * edge_scale)))
    a2 = max(block.start, a - gL/1000.0)
    b2 = min(block.end, b + gR/1000.0)
    if (b2 - a2) < (b - a):  # ensure we never shrink
        a2, b2 = a, b
    return (a2, b2)

# ----------------- Orchestrate -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio24k", required=True, help="Phase-0 24 kHz speech master (clip_full_24k.wav)")
    ap.add_argument("--blocks_rttm", required=True, help="Phase-1 cleaned RTTM (blocks.cleaned.rttm)")
    ap.add_argument("--blocks_json", default="", help="Optional Phase-1 JSON (blocks.cleaned.json) for purity")
    ap.add_argument("--out_dir", required=True, help="Directory to write proposals/*")
    ap.add_argument("--prefix", default="", help="Optional prefix (e.g., video name). Whitespaces will be replaced with '_' and prepended to every utt_id.")

    # Silence detection knobs
    ap.add_argument("--min_silence_sec", type=float, default=0.9)
    ap.add_argument("--silence_thr_method", choices=["percentile","db"], default="percentile")
    ap.add_argument("--silence_thr_value", type=float, default=20.0, help="percentile (1-99) or dB (e.g., -45)")

    # Durations
    ap.add_argument("--min_utt_sec", type=float, default=1.0)
    ap.add_argument("--bucket_min_sec", type=float, default=6.0)
    ap.add_argument("--bucket_target_sec", type=float, default=8.0)
    ap.add_argument("--bucket_max_sec", type=float, default=10.0)

    # Guards
    ap.add_argument("--join_guard_ms", type=int, default=120)

    ap.add_argument("--edge_guard_window_ms", type=int, default=250,
                    help="Distance from diar edge where guards are reduced.")
    ap.add_argument("--edge_guard_scale", type=float, default=0.5,
                    help="Scale join_guard when within edge window (e.g., 0.5 = half).")

    # Hard cap (optional): split very long speaker-pure segments even if no long silence exists
    ap.add_argument("--hard_max_sec", type=float, default=0.0,
                    help="If >0, split any proposal longer than this (e.g., 12.0)")
    ap.add_argument("--micro_gap_ms", type=int, default=120,
                    help="± window around desired cut to search for a micro-silence")
    ap.add_argument("--micro_thr_percentile", type=float, default=35.0,
                    help="Percentile for micro RMS sensitivity (higher = more sensitive)")
    ap.add_argument("--micro_win_ms", type=int, default=25)
    ap.add_argument("--micro_hop_ms", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prop_dir = os.path.join(args.out_dir, "proposals")
    os.makedirs(prop_dir, exist_ok=True)

    # sanitize prefix: collapse any whitespace runs to single underscore
    def _sanitize_ws(s: str) -> str:
        return "_".join(s.split()) if s else ""
    prefix = _sanitize_ws(args.prefix)

    # Load audio
    audio, sr = sf.read(args.audio24k, dtype="float32", always_2d=False)
    if sr != 24000:
        print(f"[WARN] audio SR={sr} (not 24000). Continuing; sample indices will reflect {sr} Hz.")
    # Load blocks
    blocks = _load_blocks(args.blocks_rttm, args.blocks_json if args.blocks_json else None)
    if not blocks:
        print("[ERROR] No blocks loaded from RTTM.", file=sys.stderr)
        sys.exit(2)

    proposals: List[Utterance] = []
    for blk in blocks:
        # 1) split on silence midpoints
        raw_segs = split_block_on_silence(
            blk, audio, sr,
            min_silence_sec=args.min_silence_sec,
            thr_method=args.silence_thr_method,
            thr_value=args.silence_thr_value,
            join_guard_ms=args.join_guard_ms
        )
        # 2) heal short
        healed = heal_short_segments(raw_segs, args.min_utt_sec)
        # 3) bucket
        bucketed = bucket_segments(healed, args.bucket_min_sec, args.bucket_target_sec, args.bucket_max_sec)

        bucketed = enforce_hard_cap(
            bucketed, blk, audio, sr,
            hard_max_sec=args.hard_max_sec,
            bucket_target_sec=args.bucket_target_sec,
            min_utt_sec=args.min_utt_sec,
            micro_gap_ms=args.micro_gap_ms,
            micro_thr_percentile=args.micro_thr_percentile,
            micro_win_ms=args.micro_win_ms,
            micro_hop_ms=args.micro_hop_ms
        )

        # 4) guards + emit
        for intra_idx, (a, b) in enumerate(bucketed):

            ag, bg = apply_join_guards(
                (a,b), blk, args.join_guard_ms,
                args.edge_guard_window_ms, args.edge_guard_scale
            )

            s0 = int(round(ag * sr))
            s1 = int(round(bg * sr))
            dur = bg - ag
            base_utt = f"utt_{blk.dia_idx:06d}_{intra_idx:02d}_spk{blk.speaker.replace('spk', '').replace('speaker','').strip() or blk.speaker}"
            utt_id = f"{prefix}_{base_utt}" if prefix else base_utt
            proposals.append(Utterance(
                utt_id=utt_id,
                dia_idx=blk.dia_idx,
                intra_idx=intra_idx,
                speaker=blk.speaker,
                purity=float(blk.purity),
                t_start=round(a, 3),
                t_end=round(b, 3),
                t_start_guarded=round(ag, 3),
                t_end_guarded=round(bg, 3),
                s_start_24k=s0, s_end_24k=s1,
                dur=round(dur, 3)
            ))

    # Chronological order (already per-block sequential)
    proposals.sort(key=lambda u: (u.dia_idx, u.intra_idx))

    # Write JSON
    json_path = os.path.join(prop_dir, "proposals.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({
            "audio": args.audio24k,
            "sr": sr,
            "params": {
                "prefix": prefix,
                "min_silence_sec": args.min_silence_sec,
                "silence_thr_method": args.silence_thr_method,
                "silence_thr_value": args.silence_thr_value,
                "min_utt_sec": args.min_utt_sec,
                "bucket_min_sec": args.bucket_min_sec,
                "bucket_target_sec": args.bucket_target_sec,
                "bucket_max_sec": args.bucket_max_sec,
                "join_guard_ms": args.join_guard_ms,
                "edge_guard_window_ms": args.edge_guard_window_ms,
                "edge_guard_scale": args.edge_guard_scale,
                "hard_max_sec": args.hard_max_sec,
                "micro_gap_ms": args.micro_gap_ms,
                "micro_thr_percentile": args.micro_thr_percentile,
                "micro_win_ms": args.micro_win_ms,
                "micro_hop_ms": args.micro_hop_ms
            },
            "count": len(proposals),
            "items": [asdict(p) for p in proposals]
        }, jf, indent=2)

    # Write TSV (for a quick scan)
    tsv_path = os.path.join(prop_dir, "proposals.tsv")
    with open(tsv_path, "w", encoding="utf-8") as tf:
        tf.write("utt_id\tdia_idx\tintra_idx\tspeaker\tpurity\tt_start\tt_end\tt_start_guarded\tt_end_guarded\tdur\n")
        for p in proposals:
            tf.write(f"{p.utt_id}\t{p.dia_idx}\t{p.intra_idx}\t{p.speaker}\t{p.purity:.3f}\t"
                     f"{p.t_start:.3f}\t{p.t_end:.3f}\t{p.t_start_guarded:.3f}\t{p.t_end_guarded:.3f}\t{p.dur:.3f}\n")

    # Stats
    durs = np.array([p.dur for p in proposals], dtype=np.float64) if proposals else np.array([0.0])
    stats = {
        "count": len(proposals),
        "total_hours": float(durs.sum()/3600.0),
        "median_dur": float(np.median(durs)),
        "p05_dur": float(np.percentile(durs, 5)) if len(durs)>1 else float(durs[0]),
        "p95_dur": float(np.percentile(durs, 95)) if len(durs)>1 else float(durs[0]),
        "min_dur": float(durs.min()),
        "max_dur": float(durs.max())
    }
    with open(os.path.join(prop_dir, "stats.json"), "w", encoding="utf-8") as sfp:
        json.dump(stats, sfp, indent=2)

    print(f"[Phase2] proposals={stats['count']}  median={stats['median_dur']:.2f}s  "
          f"range={stats['min_dur']:.2f}–{stats['max_dur']:.2f}s  total={stats['total_hours']:.2f}h")
    print(f"[Phase2] wrote:\n  {json_path}\n  {tsv_path}\n  {os.path.join(prop_dir, 'stats.json')}")
    print("[Phase2] OK — feed proposals.json to Phase 3 (render cuts).")

if __name__ == "__main__":
    main()
