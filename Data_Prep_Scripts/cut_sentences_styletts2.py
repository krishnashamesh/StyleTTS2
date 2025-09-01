#!/usr/bin/env python3
"""
Cut sentence-level 24 kHz WAVs for StyleTTS2 from:
  - WhisperX aligned words (segments_aligned.json)
  - NeMo diarization (RTTM)
  - 24 kHz mirror of trimmed proxy (Mode B)

Policies (default):
  - Min duration >= 1.0 s
  - Sentence boundary: punctuation [.?!] OR inter-word gap >= 1.0 s
  - Mean VAD >= 0.80 (if vad.npy provided). Otherwise, use speech-coverage proxy >= 0.80
  - Overlap policy: "drop" (drop multi-speaker overlaps)

Outputs:
  - cuts_dir/*.wav (PCM16 @ 24k)
  - manifest.txt  (lines: relpath.wav | IPA_or_text | speaker_id_int)
  - spk2id.json
  - qc.csv (optional quick stats)
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import soundfile as sf
import re

import logging

# module logger (user can control level externally; defaults to WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# one-time status latch for phonemizer logging
_PHONEMIZER_STATUS = "init"

# -----------------------
# Helpers
# -----------------------

def load_segments_aligned(path: str | Path) -> List[Dict[str, Any]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    # expected: {"segments":[{"start","end","text","words":[{"word","start","end"},...]},...]}
    return data.get("segments", [])

def parse_rttm(path: str | Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns {speaker_name: [(start, end), ...]} from RTTM.
    Lines like: SPEAKER <file> <..> <start> <dur> <..> <spk_id>
    """
    spk_intvs: Dict[str, List[Tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if parts[0] != "SPEAKER": continue
            start = float(parts[3]); dur = float(parts[4]); end = start + dur
            spk = parts[7]
            spk_intvs.setdefault(spk, []).append((start, end))
    # sort & merge within each speaker for fast coverage calc
    for spk, ivs in spk_intvs.items():
        spk_intvs[spk] = merge_intervals(sorted(ivs))
    return spk_intvs

def merge_intervals(iv: List[Tuple[float,float]], tol: float=1e-6) -> List[Tuple[float,float]]:
    if not iv: return []
    iv.sort()
    out=[iv[0]]
    for s,e in iv[1:]:
        ps,pe = out[-1]
        if s <= pe + tol:
            out[-1]=(ps, max(pe,e))
        else:
            out.append((s,e))
    return out

def interval_length(iv: List[Tuple[float,float]]) -> float:
    return sum(e-s for s,e in iv)

def intersect_len(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    s=max(a[0], b[0]); e=min(a[1], b[1])
    return max(0.0, e-s)

def union_coverage_len(target: Tuple[float,float], ivs: List[Tuple[float,float]]) -> float:
    """Coverage of 'target' that is overlapped by 'ivs' (as union)."""
    if not ivs: return 0.0
    # clip each to target window, merge, sum
    clipped = [(max(target[0],s), min(target[1],e)) for s,e in ivs if e>target[0] and s<target[1]]
    return interval_length(merge_intervals(sorted(clipped)))

def dominant_speaker_and_overlap(
    win: Tuple[float,float],
    spk_intvs: Dict[str, List[Tuple[float,float]]],
    cover_thres: float = 0.05
) -> Tuple[Optional[str], bool, Dict[str,float]]:
    """
    Return (dominant_spk, has_overlap, cover_map).
    dominant_spk chosen by max coverage within 'win'.
    has_overlap = True if 2+ speakers each cover >= cover_thres of 'win'.
    """
    dur = max(1e-9, win[1]-win[0])
    cover_map={}
    for spk, ivs in spk_intvs.items():
        cov = union_coverage_len(win, ivs)
        if cov > 0: cover_map[spk]=cov
    if not cover_map:
        return None, False, {}
    # check overlap
    big = [spk for spk,cov in cover_map.items() if cov/dur >= cover_thres]
    has_overlap = len(big) >= 2
    # dominant
    dom = max(cover_map.items(), key=lambda kv: kv[1])[0]
    return dom, has_overlap, cover_map

def load_vad_probs(path_npy: Optional[str], hop_s: Optional[float]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not path_npy: return None, None
    arr = np.load(path_npy)  # shape [T] or [T, 2]? assume speech prob vector
    if arr.ndim==2 and arr.shape[1] >= 1:
        arr = arr[:, 1] if arr.shape[1] > 1 else arr[:, 0]
    return arr.astype(np.float32), hop_s

def mean_vad_in_window(vad: Optional[np.ndarray], hop: Optional[float], win: Tuple[float,float]) -> Optional[float]:
    if vad is None or hop is None: return None
    s_idx = max(0, int(math.floor(win[0] / hop)))
    e_idx = min(len(vad), int(math.ceil(win[1] / hop)))
    if e_idx <= s_idx: return 0.0
    return float(np.mean(vad[s_idx:e_idx]))

FILLER_RE = re.compile(
    r"^(uh|um|erm|hmm+|mm+|mhm+|uh\-huh|uhh+|ah+|oh+|eh+|huh+|hmmm+)$",
    re.IGNORECASE,
)

def _is_filler(tok: str) -> bool:
    t = tok.strip().strip(".,!?;:…'\"-—").lower()
    return bool(FILLER_RE.match(t))

def smart_sentence_chunks_from_words(
    words: List[Dict[str,Any]],
    gap_soft: float = 1.0,        # break if inter-word gap >= this
    gap_period: float = 0.8,      # require at least this gap after '.' to break
    gap_ellipsis: float = 1.2,    # require larger gap after '...'/'…' to break
    hard_punct: str = "?!",       # always break on these
    max_chars: Optional[int] = None,
    min_dur_singleton: float = 0.30,  # keep tiny filler-only chunks down to this
) -> List[Tuple[float,float,str]]:
    """
    Smarter segmentation:
      - Hard breaks on '?/!' regardless of gap.
      - '.' is a soft break: only split if gap >= gap_period.
      - '...' or '…' is even softer: split only if gap >= gap_ellipsis.
      - Always split on a big acoustic pause: gap >= gap_soft.
      - Preserve filler-only singletons down to min_dur_singleton.
    """
    if not words:
        return []

    chunks: List[Tuple[float,float,str]] = []
    buf: List[Dict[str,Any]] = [words[0]]
    cur_start = words[0].get("start", 0.0)
    last_end  = words[0].get("end", cur_start)

    def buf_text() -> str:
        return " ".join(w.get("word","") for w in buf).strip()
    def buf_dur() -> float:
        return (buf[-1].get("end", last_end) - buf[0].get("start", cur_start)) if buf else 0.0
    def buf_all_fillers() -> bool:
        return bool(buf) and all(_is_filler(w.get("word","")) for w in buf)
    def flush(force=False):
        nonlocal buf, cur_start, last_end
        if not buf:
            return
        d = buf_dur()
        txt = buf_text()
        if not force and d < 1.0 and not buf_all_fillers():
            return
        chunks.append((buf[0]["start"], buf[-1]["end"], txt))
        buf = []

    for i in range(1, len(words)):
        prev, cur = words[i-1], words[i]
        gap = (cur.get("start", last_end) - prev.get("end", cur_start))
        last_token = prev.get("word","").strip()
        last_char  = last_token[-1:] if last_token else ""
        is_ellipsis_tok = (last_token in {"...", "…"} or last_token.endswith("..."))
        is_period  = (last_char == ".") and not is_ellipsis_tok
        is_hard    = last_char in hard_punct

        boundary = False
        if gap >= gap_soft:
            boundary = True
        elif is_hard:
            boundary = True
        elif is_ellipsis_tok and gap >= gap_ellipsis:
            boundary = True
        elif is_period and gap >= gap_period:
            boundary = True

        if boundary:
            if buf and buf_dur() < 1.0 and not buf_all_fillers():
                # too short to flush; keep accumulating
                pass
            else:
                flush(force=True)
                buf = [cur]
                cur_start = cur.get("start", prev.get("end", cur_start))
                last_end  = cur.get("end", cur_start)
                continue

        buf.append(cur)
        last_end = cur.get("end", last_end)

        if max_chars is not None and len(buf_text()) >= max_chars:
            flush(force=True)
            buf = []

    if buf:
        if buf_dur() < 1.0 and not buf_all_fillers():
            if chunks:
                s, e, t = chunks[-1]
                chunks[-1] = (s, buf[-1].get("end", e), (t + " " + buf_text()).strip())
            else:
                chunks.append((buf[0].get("start", 0.0), buf[-1].get("end", 0.0), buf_text()))
        else:
            chunks.append((buf[0].get("start", 0.0), buf[-1].get("end", 0.0), buf_text()))

    out: List[Tuple[float,float,str]] = []
    for (s,e,txt) in chunks:
        d = e - s
        if d < 1.0 and d >= min_dur_singleton and all(_is_filler(w) for w in txt.split()):
            out.append((s,e,txt))
        elif d >= 1.0:
            out.append((s,e,txt))
    return out


def _strong_end(txt: str) -> bool:
    return len(txt) > 0 and txt.rstrip().endswith(('.', '!', '?'))

def _words_text_in_range(all_words, start_s, end_s):
    """Rebuild text from words whose midpoint lies inside [start_s, end_s]."""
    mid = lambda w: 0.5 * (w.get("start", 0.0) + w.get("end", 0.0))
    toks = [w.get("word","") for w in all_words if start_s <= mid(w) <= end_s]
    txt = " ".join(toks).strip()
    return txt

def _stitch_extend_global(windows, stitch_gap_ms, extend_tail_ms, audio_duration_seconds):
    """windows: list of (s,e,txt) across the whole file, time-sorted."""
    if not windows:
        return windows
    max_gap_s = stitch_gap_ms / 1000.0
    extend_s  = extend_tail_ms / 1000.0
    # 1) stitch tiny gaps if no strong punctuation
    stitched = []
    for s, e, t in windows:
        if stitched:
            ps, pe, pt = stitched[-1]
            gap = s - pe
            if gap <= max_gap_s and not _strong_end(pt):
                stitched[-1] = (ps, max(pe, e), (pt + " " + t).strip())
                continue
        stitched.append((s, e, t))
    # 2) extend tails (leave 10 ms guard before next)
    out = []
    for i, (s, e, t) in enumerate(stitched):
        next_start = stitched[i+1][0] if i+1 < len(stitched) else None
        if next_start is None:
            e2 = min(e + extend_s, audio_duration_seconds)
        else:
            e2 = min(e + extend_s, max(s, next_start - 0.010))
        out.append((s, e2, t))
    return out


def phonemize_ipa(text: str, lang: str="en") -> str:
    """
    Optional IPA. If 'phonemizer' not installed, fallback to raw text.
    Install: pip install phonemizer && apt-get install espeak-ng
    """
    global _PHONEMIZER_STATUS
    # normalise common tags for espeak-ng
    lang_key = lang.lower().replace("_", "-")
    if lang_key == "en":
        lang_key = "en-us"
    try:
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend(language=lang_key)
        ipa = backend.phonemize([text], strip=True, njobs=1)[0]
        if _PHONEMIZER_STATUS != "ok":
            logger.info("Phonemizer: using EspeakBackend for '%s' (IPA enabled).", lang_key)
            _PHONEMIZER_STATUS = "ok"
        return ipa
    except Exception as e:
        if _PHONEMIZER_STATUS != "fail":
            logger.warning("Phonemizer unavailable or failed (%s); falling back to plain text.", e)
            _PHONEMIZER_STATUS = "fail"
        return text

# -----------------------
# Main cutter
# -----------------------

def cut_sentences(
    audio24k_path: str | Path,
    aligned_json: str | Path,
    rttm_path: str | Path,
    out_wavs_dir: str | Path,
    manifest_path: str | Path,
    spk2id_path: str | Path,
    vad_npy: Optional[str]=None,
    vad_hop: Optional[float]=None,    # seconds, e.g., 0.02
    min_dur: float=1.0,
    gap_thres: float=1.0,
    period_gap: float=0.8,
    ellipsis_gap: float=1.2,
    max_chars: Optional[int]=None,
    min_dur_singleton: float=0.30,
    pad_ms: int=12,
    mean_vad_thres: float=0.80,
    overlap_policy: str="drop",       # "drop" | "dominant"
    fade_ms: int=8,
    lang_for_ipa: str="en",
    qc_csv: Optional[str]=None,
    stitch_gap_ms: int = 180,
    extend_tail_ms: int = 120,
):
    """
    Cuts PCM16 WAVs from 24k master by sentence windows and writes StyleTTS2 manifest.
    """
    audio24k_path = Path(audio24k_path)
    aligned = load_segments_aligned(aligned_json)
    all_words = [w for seg in aligned for w in seg.get("words", [])]
    spk_intvs = parse_rttm(rttm_path)
    vad, hop = load_vad_probs(vad_npy, vad_hop)
    sr24 = 24000
    _outdir = Path(out_wavs_dir); _outdir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    man_f = open(manifest_path, "w", encoding="utf-8")
    spk2id: Dict[str,int] = {}
    qc_rows = []

    # open once; slice reads per sentence (no full-file load)
    sf_in = sf.SoundFile(str(audio24k_path), mode="r")
    assert sf_in.samplerate == sr24, f"Expected 24k master, got {sf_in.samplerate}"
    total_frames = len(sf_in)
    audio_duration_seconds = total_frames / float(sr24)

    def _strong_end(txt: str) -> bool:
        return len(txt) > 0 and txt.rstrip().endswith(('.', '!', '?'))

    def _apply_stitch_and_extend(windows: List[Tuple[float,float,str]]) -> List[Tuple[float,float,str]]:
        """Merge tiny gaps (no strong end punct), then extend tail safely."""
        if not windows:
            return windows
        max_gap_s = stitch_gap_ms / 1000.0
        extend_s  = extend_tail_ms / 1000.0
        # 1) stitch
        stitched: List[Tuple[float,float,str]] = []
        for s, e, t in windows:
            if stitched:
                ps, pe, pt = stitched[-1]
                gap = s - pe
                if gap <= max_gap_s and not _strong_end(pt):
                    # merge
                    stitched[-1] = (ps, max(pe, e), (pt + " " + t).strip())
                    continue
            stitched.append((s, e, t))
        # 2) extend tails (leave tiny 10ms guard before next start)
        out: List[Tuple[float,float,str]] = []
        for i, (s, e, t) in enumerate(stitched):
            next_start = stitched[i+1][0] if i+1 < len(stitched) else None
            if next_start is None:
                e2 = min(e + extend_s, audio_duration_seconds)
            else:
                e2 = min(e + extend_s, max(s, next_start - 0.010))
            out.append((s, e2, t))
        return out

    # Collect all windows across the whole file first
    pending = []
    for seg in aligned:
        words = seg.get("words", [])
        if not words:
            continue
        seg_windows = smart_sentence_chunks_from_words(
            words,
            gap_soft=gap_thres,
            gap_period=period_gap,
            gap_ellipsis=ellipsis_gap,
            max_chars=max_chars,
            min_dur_singleton=min_dur_singleton,
        )
        pending.extend(seg_windows)

    # Sort by time and apply global stitch/extend
    pending.sort(key=lambda t: t[0])
    audio_duration_seconds = len(sf_in) / float(sr24)
    windows = _stitch_extend_global(pending, stitch_gap_ms, extend_tail_ms, audio_duration_seconds)

    seg_idx = 0
    for (s, e, _txt_premerge) in windows:
        # rebuild text from ALL words (post-merge), not just the segment-local buffer
        sent_text = _words_text_in_range(all_words, s, e) or _txt_premerge
        dur = e - s
        if dur < min_dur:
            continue

        # Speaker resolve & overlap detection
        dom_spk, has_ovl, cover_map = dominant_speaker_and_overlap((s, e), spk_intvs, cover_thres=0.05)

        if has_ovl and overlap_policy == "drop":
            continue
        if dom_spk is None:
            # no speaker coverage; likely VAD misfire or non-speech
            continue

        # VAD mean check: use posterior if available; else speech-coverage proxy
        if vad is not None and hop is not None:
            mean_vad = mean_vad_in_window(vad, hop, (s,e)) or 0.0
        else:
            cov = union_coverage_len((s,e), sum(spk_intvs.values(), []))  # union across speakers
            mean_vad = cov / max(1e-9, dur)

        if mean_vad < mean_vad_thres:
            continue

        # speaker id mapping (stable order)
        if dom_spk not in spk2id:
            spk2id[dom_spk] = len(spk2id)

        # Read slice and write WAV (with short fades to avoid clicks)

        pad = int(round((pad_ms/1000.0) * sr24)) if pad_ms > 0 else 0
        start_frame = max(0, int(round(s * sr24)) - pad)
        end_frame   = min(total_frames, int(round(e * sr24)) + pad)

        nframes = max(0, end_frame - start_frame)
        if nframes <= 0: 
            continue

        sf_in.seek(start_frame)
        audio = sf_in.read(frames=nframes, dtype="int16", always_2d=False)

        # apply tiny linear fade-in/out on float, then convert back
        if fade_ms > 0 and len(audio) > 0:
            fade_len = int(sr24 * fade_ms / 1000.0)
            fade_len = min(fade_len, len(audio)//2)
            if fade_len > 0:
                a = audio.astype(np.float32)
                ramp_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                ramp_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                a[:fade_len] *= ramp_in
                a[-fade_len:] *= ramp_out
                audio = np.clip(np.round(a), -32768, 32767).astype(np.int16)

        wav_name = f"utt_{seg_idx:06d}_spk{spk2id[dom_spk]}.wav"
        wav_path = _outdir / wav_name
        sf.write(str(wav_path), audio, sr24, subtype="PCM_16")

        # IPA (optional; fallback to text)
        ipa = phonemize_ipa(sent_text, lang_for_ipa)
        rel = wav_name  # relative to out_wavs_dir
        man_f.write(f"{rel} | {ipa} | {spk2id[dom_spk]}\n")

        # QC row — use *padded* times so CSV matches the WAV exactly
        start_s_padded = round(start_frame / sr24, 3)
        end_s_padded   = round(end_frame   / sr24, 3)
        qc_rows.append({
            "utt": wav_name,
            "spk_name": dom_spk,
            "spk_id": spk2id[dom_spk],
            "start": start_s_padded,
            "end": end_s_padded,
            "dur": round(end_s_padded - start_s_padded, 3),
            "mean_vad": round(mean_vad, 3),
            "overlap": int(has_ovl),
            "text": sent_text,
        })
        seg_idx += 1

    sf_in.close()
    man_f.close()

    # spk2id.json
    with open(spk2id_path, "w", encoding="utf-8") as f:
        json.dump(spk2id, f, indent=2)

    # qc.csv (optional)
    if qc_csv:
        os.makedirs(os.path.dirname(qc_csv), exist_ok=True)
        with open(qc_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()) if qc_rows else
                               ["utt","spk_name","spk_id","start","end","dur","mean_vad","overlap","text"])
            w.writeheader()
            for r in qc_rows: w.writerow(r)

    print(f"Done. Wrote {seg_idx} WAVs to {out_wavs_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"spk2id:   {spk2id_path}")
    if qc_csv: print(f"QC CSV:   {qc_csv}")

# -----------------------
# CLI
# -----------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Cut sentence WAVs for StyleTTS2.")
    p.add_argument("--audio24k", required=True, help="24 kHz mono WAV (trimmed mirror of proxy)")
    p.add_argument("--aligned_json", required=True, help="WhisperX segments_aligned.json")
    p.add_argument("--rttm", required=True, help="NeMo RTTM path")
    p.add_argument("--out_wavs_dir", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--spk2id_path", required=True)
    p.add_argument("--vad_npy", default=None, help="Optional VAD posterior .npy")
    p.add_argument("--vad_hop", type=float, default=None, help="VAD hop in seconds (e.g., 0.02)")
    p.add_argument("--min_dur", type=float, default=1.0)
    p.add_argument("--gap_thres", type=float, default=1.0)
    p.add_argument("--period_gap", type=float, default=0.8, help="Gap needed after '.' to split.")
    p.add_argument("--ellipsis_gap", type=float, default=1.2, help="Gap needed after '...'/'…' to split.")
    p.add_argument("--max_chars", type=int, default=None, help="Optional cap on characters per sentence.")
    p.add_argument("--min_dur_singleton", type=float, default=0.30, help="Keep filler-only utterances down to this.")
    p.add_argument("--pad_ms", type=int, default=12, help="Pad each cut on both sides (ms).")
    p.add_argument("--mean_vad_thres", type=float, default=0.80)
    p.add_argument("--overlap_policy", choices=["drop","dominant"], default="dominant")
    p.add_argument("--fade_ms", type=int, default=8)
    p.add_argument("--lang_for_ipa", default="en-us")
    p.add_argument("--stitch_gap_ms", type=int, default=180,
                   help="If gap between consecutive segments < this, merge them (no strong end punctuation).")
    p.add_argument("--extend_tail_ms", type=int, default=120,
                   help="Extend each segment tail by this much unless it overruns the next segment.")
    p.add_argument("--qc_csv", default=None)
    return p

def main():
    args = build_argparser().parse_args()
    cut_sentences(
        audio24k_path=args.audio24k,
        aligned_json=args.aligned_json,
        rttm_path=args.rttm,
        out_wavs_dir=args.out_wavs_dir,
        manifest_path=args.manifest_path,
        spk2id_path=args.spk2id_path,
        vad_npy=args.vad_npy,
        vad_hop=args.vad_hop,
        min_dur=args.min_dur,
        gap_thres=args.gap_thres,
        period_gap=args.period_gap,
        ellipsis_gap=args.ellipsis_gap,
        max_chars=args.max_chars,
        min_dur_singleton=args.min_dur_singleton,
        pad_ms=args.pad_ms,
        mean_vad_thres=args.mean_vad_thres,
        overlap_policy=args.overlap_policy,
        fade_ms=args.fade_ms,
        lang_for_ipa=args.lang_for_ipa,
        qc_csv=args.qc_csv,
        stitch_gap_ms=args.stitch_gap_ms,
        extend_tail_ms=args.extend_tail_ms,
    )

if __name__ == "__main__":
    main()
