#!/usr/bin/env python3
"""
Cut sentence-level 24 kHz WAVs for StyleTTS2 from:
  - An aligned word-level JSON (generic): either top-level "utterances" or "segments",
    each item containing a "words" list of {word, start, end}
  - NeMo diarization RTTM (speaker turns)

This script is WhisperX-independent. It only needs word-level timings + RTTM.

Policies (configurable via CLI):
  - Min duration >= 1.0 s (non-filler) or >= min_dur_singleton if fillers-only
  - Sentence boundary: punctuation [.?!] rules + acoustic pauses with tunable gap thresholds
  - Mean speech coverage >= mean_vad_thres (uses RTTM coverage as proxy if no VAD posterior)
  - Overlap policy: "drop" (discard overlapping multi-speaker) or "dominant" (keep dominant)
  - Optional stitching of tiny gaps and safe tail extension with guard before next cut

Outputs:
  - cuts_dir/*.wav (PCM16 @ 24k)
  - manifest.txt  (lines: relpath.wav | IPA_or_text | speaker_id_int)
  - spk2id.json
  - qc.csv (optional quick stats)
"""

from __future__ import annotations
import argparse, csv, json, math, os, re, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("cutter")
logging.basicConfig(level=logging.INFO)

_PHONEMIZER_STATUS = "init"

# -----------------------
# JSON loaders (robust)
# -----------------------

def _coerce_to_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts JSON with either:
      - 'utterances': [{start,end,text?,words:[{word,start,end},...]}...]
      - 'segments'  : [{start,end,text?,words:[{word,start,end},...]}...]
    Returns a list of segment dicts with at least 'words' present.
    """
    if isinstance(data, dict):
        if "utterances" in data and isinstance(data["utterances"], list):
            base = data["utterances"]
        elif "segments" in data and isinstance(data["segments"], list):
            base = data["segments"]
        else:
            raise ValueError("Aligned JSON must contain a list at key 'utterances' or 'segments'.")
    elif isinstance(data, list):
        base = data
    else:
        raise ValueError("Aligned JSON must be a dict or list.")

    out = []
    for item in base:
        words = item.get("words", [])
        # tolerate common alt fields
        if not words and "tokens" in item and isinstance(item["tokens"], list):
            words = [{"word": t.get("text",""), "start": t.get("start"), "end": t.get("end")} for t in item["tokens"]]
        # drop words missing timings
        w_clean = []
        for w in words:
            try:
                ws = float(w["start"]); we = float(w["end"])
                if we > ws:
                    w_clean.append({"word": str(w.get("word","")).strip(), "start": ws, "end": we})
            except Exception:
                continue
        if not w_clean:
            continue
        s = min(w["start"] for w in w_clean)
        e = max(w["end"]   for w in w_clean)
        out.append({"start": float(item.get("start", s)), "end": float(item.get("end", e)),
                    "text": item.get("text","").strip(), "words": w_clean})
    return out

def load_aligned_words(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = _coerce_to_segments(data)
    n_words = sum(len(s["words"]) for s in segs)
    logger.info("Aligned JSON loaded: %d segments, %d words total.", len(segs), n_words)
    if n_words == 0:
        raise RuntimeError("No words found in aligned JSON. Check the schema/keys.")
    return segs

# -----------------------
# RTTM & timing helpers
# -----------------------

def parse_rttm(path: str | Path) -> Dict[str, List[Tuple[float, float]]]:
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
    for spk, ivs in spk_intvs.items():
        spk_intvs[spk] = merge_intervals(sorted(ivs))
    logger.info("RTTM loaded: %d speakers.", len(spk_intvs))
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

def union_coverage_len(target: Tuple[float,float], ivs: List[Tuple[float,float]]) -> float:
    if not ivs: return 0.0
    s0,e0 = target
    clipped = [(max(s0,s), min(e0,e)) for s,e in ivs if e>s0 and s<e0]
    return interval_length(merge_intervals(sorted(clipped)))

def dominant_speaker_and_overlap(
    win: Tuple[float,float],
    spk_intvs: Dict[str, List[Tuple[float,float]]],
    cover_thres: float = 0.05
) -> Tuple[Optional[str], bool, Dict[str,float]]:
    dur = max(1e-9, win[1]-win[0])
    cover={}
    for spk, ivs in spk_intvs.items():
        cov = union_coverage_len(win, ivs)
        if cov>0: cover[spk]=cov
    if not cover: return None, False, {}
    big = [spk for spk,c in cover.items() if c/dur >= cover_thres]
    has_overlap = len(big) >= 2
    dom = max(cover.items(), key=lambda kv: kv[1])[0]
    return dom, has_overlap, cover

# -----------------------
# Segmentation from words
# -----------------------

FILLER_RE = re.compile(
    r"^(uh|um|erm|hmm+|mm+|mhm+|uh\-huh|uhh+|ah+|oh+|eh+|huh+|hmmm+)$",
    re.IGNORECASE,
)
def _is_filler(tok: str) -> bool:
    t = tok.strip().strip(".,!?;:…'\"-—").lower()
    return bool(FILLER_RE.match(t))

def smart_sentence_chunks_from_words(
    words: List[Dict[str,Any]],
    gap_soft: float = 1.0,
    gap_period: float = 0.8,
    gap_ellipsis: float = 1.2,
    hard_punct: str = "?!",
    max_chars: Optional[int] = None,
    min_dur_singleton: float = 0.30,
) -> List[Tuple[float,float,str]]:
    if not words: return []
    chunks: List[Tuple[float,float,str]] = []
    buf: List[Dict[str,Any]] = [words[0]]

    def buf_text() -> str: return " ".join(w.get("word","") for w in buf).strip()
    def buf_dur()  -> float: return (buf[-1]["end"] - buf[0]["start"]) if buf else 0.0
    def all_fillers()-> bool: return bool(buf) and all(_is_filler(w.get("word","")) for w in buf)

    CAPITAL_BREAK_MIN_GAP = 0.06  # s

    for i in range(1, len(words)):
        prev, cur = words[i-1], words[i]
        gap = cur["start"] - prev["end"]
        last_tok = prev.get("word","").strip()
        last_char= last_tok[-1:] if last_tok else ""
        is_ellipsis_tok = (last_tok in {"...", "…"} or last_tok.endswith("..."))
        is_period  = (last_char == ".") and not is_ellipsis_tok
        is_hard    = last_char in hard_punct

        cur_tok = cur.get("word","").strip()
        cur_last= cur_tok[-1:] if cur_tok else ""
        cur_is_hard = cur_last in hard_punct
        cur_first = cur_tok[:1]
        cur_is_capital = bool(cur_first and cur_first.isupper())
        cur_is_starter = cur_is_capital or cur_tok.lower() in {
            "who","what","why","when","where","how",
            "did","do","does","are","is","was","were",
            "can","could","will","would","should","shall","have","has","had","oh","well"
        }

        boundary = False
        if gap >= gap_soft: boundary = True
        elif is_hard:       boundary = True
        elif is_ellipsis_tok and gap >= gap_ellipsis: boundary = True
        elif is_period and gap >= gap_period:         boundary = True
        if not boundary and gap >= CAPITAL_BREAK_MIN_GAP and (is_ellipsis_tok or is_period) and cur_is_starter:
            boundary = True

        if boundary:
            if cur_is_hard and buf:
                buf.append(cur)
                chunks.append((buf[0]["start"], buf[-1]["end"], buf_text()))
                buf = []
                continue
            if buf and buf_dur() < 1.0 and not all_fillers():
                # keep accumulating until we reach 1s or punctuation
                pass
            else:
                chunks.append((buf[0]["start"], buf[-1]["end"], buf_text()))
                buf = [cur]
                continue
        buf.append(cur)

    if buf:
        if buf_dur() < 1.0 and chunks:
            s,e,t = chunks[-1]
            chunks[-1] = (s, buf[-1]["end"], (t + " " + " ".join(w["word"] for w in buf)).strip())
        else:
            chunks.append((buf[0]["start"], buf[-1]["end"], " ".join(w["word"] for w in buf)))

    out: List[Tuple[float,float,str]] = []
    for (s,e,txt) in chunks:
        d = e - s
        if d < 1.0 and d >= min_dur_singleton and all(_is_filler(w) for w in txt.split()):
            out.append((s,e,txt))
        elif d >= 1.0:
            out.append((s,e,txt))
    return out

def words_text_in_range(all_words: List[Dict[str,Any]], start_s: float, end_s: float, eps: float=0.005) -> str:
    toks=[]
    for w in all_words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end",   0.0))
        if (ws < (end_s - eps)) and (we > (start_s + eps)):
            toks.append(w.get("word",""))
    return " ".join(toks).strip()

# -----------------------
# IPA (optional)
# -----------------------

def phonemize_ipa(text: str, lang: str="en-us") -> str:
    global _PHONEMIZER_STATUS
    try:
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend(language=lang)
        ipa = backend.phonemize([text], strip=True, njobs=1)[0]
        if _PHONEMIZER_STATUS != "ok":
            logger.info("Phonemizer: using EspeakBackend for '%s'.", lang)
            _PHONEMIZER_STATUS = "ok"
        return ipa
    except Exception as e:
        if _PHONEMIZER_STATUS != "fail":
            logger.warning("Phonemizer unavailable (%s); falling back to plain text.", e)
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
    *,
    vad_npy: Optional[str]=None,
    vad_hop: Optional[float]=None,
    min_dur: float=1.0,
    gap_thres: float=1.2,
    period_gap: float=0.8,
    ellipsis_gap: float=1.4,
    max_chars: Optional[int]=None,
    min_dur_singleton: float=0.30,
    pad_ms: int=20,
    mean_vad_thres: float=0.80,
    overlap_policy: str="dominant",
    fade_ms: int=10,
    lang_for_ipa: str="en-us",
    qc_csv: Optional[str]=None,
    stitch_gap_ms: int = 0,
    extend_tail_ms: int = 0,
    join_guard_ms: int = 120,
):
    audio24k_path = Path(audio24k_path)
    segs = load_aligned_words(aligned_json)
    all_words = [w for s in segs for w in s["words"]]
    spk_intvs = parse_rttm(rttm_path)

    sr24 = 24000
    out_dir = Path(out_wavs_dir); out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(Path(manifest_path).parent, exist_ok=True)

    sf_in = sf.SoundFile(str(audio24k_path), mode="r")
    if sf_in.samplerate != sr24:
        raise ValueError(f"Expected 24k WAV, got {sf_in.samplerate} Hz")
    total_frames = len(sf_in)
    audio_dur = total_frames / float(sr24)

    # 1) Build window candidates from words (per segment), then global merging if requested
    pending = []
    for s in segs:
        words = s["words"]
        pending.extend(
            smart_sentence_chunks_from_words(
                words,
                gap_soft=gap_thres,
                gap_period=period_gap,
                gap_ellipsis=ellipsis_gap,
                max_chars=max_chars,
                min_dur_singleton=min_dur_singleton,
            )
        )
    pending.sort(key=lambda x: x[0])

    def _strong_end(txt: str) -> bool:
        if not txt: return False
        t = txt.rstrip()
        if t.endswith('!') or t.endswith('?'): return True
        if t.endswith('...') or t.endswith('…'): return False
        return t.endswith('.')

    # Stitch neighboring if desired
    windows: List[Tuple[float,float,str]]
    if stitch_gap_ms > 0 or extend_tail_ms > 0:
        max_gap_s = stitch_gap_ms/1000.0
        extend_s  = extend_tail_ms/1000.0
        stitched=[]
        for s,e,t in pending:
            if stitched:
                ps,pe,pt = stitched[-1]
                if (s - pe) <= max_gap_s and not _strong_end(pt):
                    stitched[-1] = (ps, max(pe, e), (pt + " " + t).strip())
                    continue
            stitched.append((s,e,t))
        out=[]
        guard_s = join_guard_ms/1000.0
        pad_s   = pad_ms/1000.0
        for i,(s,e,t) in enumerate(stitched):
            nxt = stitched[i+1][0] if i+1 < len(stitched) else None
            if nxt is None:
                e2 = min(e + extend_s, audio_dur)
            else:
                # cap so that padded slices still leave guard
                cap = max(e, nxt - (pad_s + guard_s))
                e2 = min(e + extend_s, cap)
            out.append((s,e2,t))
        windows = out
    else:
        windows = pending

    if not windows:
        logger.warning("No sentence windows built from aligned words. Check thresholds / input.")
    logger.info("Candidate windows: %d", len(windows))

    # Optional VAD posterior
    vad: Optional[np.ndarray] = None
    hop: Optional[float] = None
    if vad_npy:
        try:
            arr = np.load(vad_npy)
            if arr.ndim==2 and arr.shape[1]>1: arr = arr[:,1]
            vad = arr.astype(np.float32)
            hop = float(vad_hop) if vad_hop else None
        except Exception as e:
            logger.warning("Failed to load VAD npy: %s", e)

    def mean_vad_in_window(win: Tuple[float,float]) -> float:
        s,e = win
        dur = max(1e-9, e-s)
        if vad is None or hop is None:
            # proxy: speech coverage by union of all speakers
            all_union = []
            for ivs in spk_intvs.values():
                all_union.extend(ivs)
            cov = union_coverage_len((s,e), merge_intervals(sorted(all_union)))
            return cov/dur
        si = max(0, int(math.floor(s / hop)))
        ei = min(len(vad), int(math.ceil(e / hop)))
        if ei<=si: return 0.0
        return float(np.mean(vad[si:ei]))

    # 2) Filter, slice audio, write files + manifest + QC
    spk2id: Dict[str,int] = {}
    seg_idx = 0
    qc_rows = []
    with open(manifest_path, "w", encoding="utf-8") as man_f:
        for i,(s,e,txt_pre) in enumerate(windows):
            txt = words_text_in_range(all_words, s, e) or txt_pre
            dur = e - s
            if dur < min_dur:
                continue

            dom_spk, has_ovl, _ = dominant_speaker_and_overlap((s,e), spk_intvs, cover_thres=0.05)
            if has_ovl and overlap_policy == "drop":
                continue
            if dom_spk is None:
                continue

            mean_vad = mean_vad_in_window((s,e))
            if mean_vad < mean_vad_thres:
                continue

            if dom_spk not in spk2id:
                spk2id[dom_spk] = len(spk2id)

            # compute padded/capped frame range (no overlap into next)
            pad_frames   = int(round((pad_ms/1000.0)*sr24)) if pad_ms>0 else 0
            guard_frames = int(round((join_guard_ms/1000.0)*sr24)) if join_guard_ms>0 else 0
            start_frame  = max(0, int(round(s*sr24)) - pad_frames)
            next_start   = windows[i+1][0] if i+1 < len(windows) else None
            tail_plain   = int(round(e*sr24))
            tail_withpad = tail_plain + pad_frames
            if next_start is not None:
                next_frames = int(round(next_start*sr24))
                if next_frames > tail_plain:
                    end_cap = max(0, next_frames - guard_frames)
                    end_frame = min(total_frames, min(tail_withpad, end_cap))
                else:
                    end_frame = min(total_frames, tail_plain)
            else:
                end_frame = min(total_frames, tail_withpad)

            nframes = max(0, end_frame - start_frame)
            if nframes <= 0:
                continue

            sf_in.seek(start_frame)
            audio = sf_in.read(frames=nframes, dtype="int16", always_2d=False)

            # tiny fade to avoid clicks
            if fade_ms>0 and len(audio)>0:
                fade_len = int(sr24*fade_ms/1000.0)
                fade_len = min(fade_len, len(audio)//2)
                if fade_len>0:
                    a = audio.astype(np.float32)
                    a[:fade_len]  *= np.linspace(0.0,1.0,fade_len, dtype=np.float32)
                    a[-fade_len:] *= np.linspace(1.0,0.0,fade_len, dtype=np.float32)
                    audio = np.clip(np.round(a), -32768, 32767).astype(np.int16)

            wav_name = f"utt_{seg_idx:06d}_spk{spk2id[dom_spk]}.wav"
            wav_path = out_dir / wav_name
            sf.write(str(wav_path), audio, sr24, subtype="PCM_16")

            ipa_or_text = phonemize_ipa(txt, lang_for_ipa) if txt else ""
            man_f.write(f"{wav_name} | {ipa_or_text} | {spk2id[dom_spk]}\n")

            # QC uses padded/capped times for transparency
            s_pad = round(start_frame/sr24, 3)
            e_pad = round(end_frame  /sr24, 3)
            qc_rows.append({
                "utt": wav_name,
                "spk_name": dom_spk,
                "spk_id": spk2id[dom_spk],
                "start": s_pad,
                "end": e_pad,
                "dur": round(e_pad-s_pad, 3),
                "mean_vad": round(mean_vad,3),
                "overlap": int(has_ovl),
                "text": txt,
            })
            seg_idx += 1

    sf_in.close()

    with open(spk2id_path, "w", encoding="utf-8") as f:
        json.dump(spk2id, f, indent=2)

    if qc_csv:
        os.makedirs(Path(qc_csv).parent, exist_ok=True)
        with open(qc_csv, "w", newline="", encoding="utf-8") as f:
            cols = ["utt","spk_name","spk_id","start","end","dur","mean_vad","overlap","text"]
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in qc_rows: w.writerow(r)

    print(f"Done. Wrote {seg_idx} WAVs to {out_wavs_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"spk2id:   {spk2id_path}")
    if qc_csv: print(f"QC CSV:   {qc_csv}")

# -----------------------
# CLI
# -----------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Sentence cutter (no WhisperX deps).")
    p.add_argument("--audio24k", required=True, help="24 kHz mono WAV")
    p.add_argument("--aligned_json", required=True, help="Aligned JSON with top-level 'utterances' or 'segments'")
    p.add_argument("--rttm", required=True, help="NeMo diarization RTTM")
    p.add_argument("--out_wavs_dir", required=True)
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--spk2id_path", required=True)
    p.add_argument("--vad_npy", default=None)
    p.add_argument("--vad_hop", type=float, default=None)

    p.add_argument("--min_dur", type=float, default=1.0)
    p.add_argument("--gap_thres", type=float, default=1.2)
    p.add_argument("--period_gap", type=float, default=0.8)
    p.add_argument("--ellipsis_gap", type=float, default=1.4)
    p.add_argument("--max_chars", type=int, default=None)
    p.add_argument("--min_dur_singleton", type=float, default=0.30)

    p.add_argument("--pad_ms", type=int, default=20)
    p.add_argument("--mean_vad_thres", type=float, default=0.80)
    p.add_argument("--overlap_policy", choices=["drop","dominant"], default="dominant")
    p.add_argument("--fade_ms", type=int, default=10)
    p.add_argument("--lang_for_ipa", default="en-us")

    p.add_argument("--stitch_gap_ms", type=int, default=0)
    p.add_argument("--extend_tail_ms", type=int, default=0)
    p.add_argument("--join_guard_ms", type=int, default=120)

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
        join_guard_ms=args.join_guard_ms,
    )

if __name__ == "__main__":
    main()
