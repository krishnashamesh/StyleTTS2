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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger("cutter")
logging.basicConfig(level=logging.INFO)

_PHONEMIZER_STATUS = "init"

# -----------------------
# Punctuation graft helpers
# -----------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")

def _norm_token(t: str, keep_apostrophes: bool = True) -> str:
    t = t.strip()
    t = _ELLIPSIS_RE.sub("...", t)
    if keep_apostrophes:
        # keep inner apostrophes: gaffer’s -> gaffer’s, don't -> don't
        t = re.sub(r"(^[^\w']+|[^\w']+$)", "", t)
        t = re.sub(r"[^\w']+", " ", t)
    else:
        t = re.sub(r"[^\w]+", " ", t)
    return re.sub(r"\s+", " ", t.lower()).strip()

def _tokenize_src_text_with_punct(text: str):
    """
    Returns:
      tokens: list of dicts:
        - type: "word" | "punct"
        - text: original token text (preserve case/punct)
        - norm: normalized (for words only)
      word_to_token_idx: list mapping src_word_index -> token_index
      src_words_norm: list of normalized word tokens (sequence for alignment)
    """
    # split into words and punctuation tokens while preserving order
    tokens = []
    i = 0
    while i < len(text):
        m_ell = _ELLIPSIS_RE.match(text, i)
        if m_ell:
            tokens.append({"type":"punct","text":m_ell.group(0)})
            i = m_ell.end()
            continue
        m_w = _WORD_RE.match(text, i)
        if m_w:
            w = m_w.group(0)
            tokens.append({"type":"word","text":w,"norm":_norm_token(w)})
            i = m_w.end()
            continue
        # single char punctuation/space
        ch = text[i]
        if not ch.isspace():
            tokens.append({"type":"punct","text":ch})
        i += 1
    # build src word sequence + map to token indices
    word_to_token_idx = []
    src_words_norm = []
    for ti, tok in enumerate(tokens):
        if tok["type"] == "word":
            word_to_token_idx.append(ti)
            src_words_norm.append(tok["norm"])
    return tokens, word_to_token_idx, src_words_norm

def _load_punct_source_text(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        logger.warning("punct_source not found: %s", path)
        return None
    try:
        raw = p.read_text(encoding="utf-8").strip()
        # Try JSONL (first non-empty line)
        first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
        try:
            obj = json.loads(first_line)
            t = obj.get("text", "") if isinstance(obj, dict) else ""
            if t: return t
        except Exception:
            pass
        # Try pretty JSON (object or list)
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj.get("text", "") or ""
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict) and rec.get("text"):
                    return rec["text"]
        return None
    except Exception as e:
        logger.warning("Failed to read punct_source %s (%s)", path, e)
        return None

def _align_global(all_words_norm: List[str], src_words_norm: List[str]):
    """Build a mapping from global aligned-words index -> src word index via SequenceMatcher."""
    from difflib import SequenceMatcher
    sm = SequenceMatcher(a=all_words_norm, b=src_words_norm, autojunk=False)
    map_a_to_b = {}
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal":
            for i in range(a0, a1):
                map_a_to_b[i] = b0 + (i - a0)
        elif tag in ("replace",):
            # if lengths match, try 1:1 salvage
            if (a1 - a0) == (b1 - b0):
                for i in range(a0, a1):
                    map_a_to_b[i] = b0 + (i - a0)
            # else: leave unmapped; will fallback later
    return map_a_to_b

# ----- helpers for locality-aware graft + validation -----
def _norm_for_compare(s: str) -> str:
    """Lowercase, collapse spaces, strip non-word (keep apostrophes)."""
    t = s.lower()
    t = _ELLIPSIS_RE.sub("...", t)
    t = re.sub(r"[^\w'\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _seq_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(a=_norm_for_compare(a), b=_norm_for_compare(b)).ratio()

def _words_indices_in_range(all_words: List[Dict[str,Any]], s: float, e: float, eps: float=0.005) -> List[int]:
    out = []
    for w in all_words:
        ws = float(w.get("start",0.0)); we = float(w.get("end",0.0))
        if (ws < (e - eps)) and (we > (s + eps)):
            out.append(w["__idx"])
    return out


# Coverage-gated word selection (drop boundary slivers)
def _filter_idxs_by_coverage(
    idxs: List[int],
    all_words: List[Dict[str,Any]],
    s: float, e: float,
    min_cov: float = 0.50,
    min_ms: int = 80,
) -> Tuple[List[int], List[int]]:
    keep, drop = [], []
    min_s = max(0.0, float(min_ms) / 1000.0)
    for i in idxs:
        w = all_words[i]
        ws = w.get("start", None); we = w.get("end", None)
        if ws is None or we is None: continue
        inter = max(0.0, min(e, we) - max(s, ws))
        dur   = max(1e-6, we - ws)
        if (inter >= min_s) or (inter / dur >= min_cov):
            keep.append(i)
        else:
            drop.append(i)
    return keep, drop

def _graft_text_for_window(
    idxs: List[int],
    all_words: List[Dict[str,Any]],
    map_idx_to_src: Dict[int,int],
    src_tokens: List[Dict[str,Any]],
    src_word_to_tok: List[int],
    return_meta: bool = False,
) -> Optional[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Reconstruct cased+punctuated text covering mapped words; return meta if requested."""
    src_idxs = [map_idx_to_src.get(i) for i in idxs if i in map_idx_to_src]
    if not src_idxs:
        return None
    lo, hi = min(src_idxs), max(src_idxs)
    # map src word indices to token indices
    try:
        t0 = src_word_to_tok[lo]
        t1 = src_word_to_tok[hi]
    except Exception:
        return None
    # include all tokens between (inclusive)
    toks = src_tokens[t0 : t1 + 1]
    # join with no extra spaces around punctuation; keep original case
    out = []
    for i, tk in enumerate(toks):
        if tk["type"] == "punct":
            # attach punctuation to previous if exists
            if out:
                out[-1] = out[-1] + tk["text"]
            else:
                out.append(tk["text"])
        else:
            # word
            if out:
                out.append(" " + tk["text"])
            else:
                out.append(tk["text"])
    # return text + debug breadcrumbs about the chosen src word span

    text = "".join(out).strip()
    if return_meta:
        meta = {"mode":"global","src_lo":lo,"src_hi":hi,"t0":t0,"t1":t1}
        return text, meta
    return text


def _graft_text_for_window_local(
    idxs: List[int],
    all_words: List[Dict[str,Any]],
    src_tokens: List[Dict[str,Any]],
    src_word_to_tok: List[int],
    src_words_norm: List[str],
    radius_words: int = 80,
    return_meta: bool = False,
) -> Optional[Union[str, Tuple[str, Dict[str, Any]]]]:
    """Local re-alignment near mapped anchor; return alignment diagnostics if requested."""
    if not idxs:
        return None
    # window words (normalized sequence)
    win_words = [all_words[i]["__norm"] for i in idxs]
    if len(win_words) == 0:
        return None

    # estimate anchor via the first successfully mapped index (fallback to middle)
    mapped = [i for i in idxs if "__src_idx" in all_words[i]]
    if mapped:
        anchor_src = int(np.median([all_words[i]["__src_idx"] for i in mapped]))
    else:
        anchor_src = None

    # define local search slice in src_words_norm
    if anchor_src is None:
        lo = 0
        hi = len(src_words_norm)
    else:
        lo = max(0, anchor_src - radius_words)
        hi = min(len(src_words_norm), anchor_src + radius_words)
    if hi - lo < 5:
        return None

    from difflib import SequenceMatcher
    sm = SequenceMatcher(a=win_words, b=src_words_norm[lo:hi], autojunk=False)
    # find best matching contiguous block in the local window
    best = None
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag == "equal":
            span = (a1 - a0)
            if span >= 2:  # require at least a couple of words
                score = span
                if (best is None) or (score > best[0]):
                    best = (score, b0, b1)
    if best is None:
        # fall back to the whole local window via overall ratio
        # take the longest b-span suggested by matcher even if not 'equal'
        b0 = 0; b1 = 0
        maxspan = -1
        for tag, a0, a1, bb0, bb1 in sm.get_opcodes():
            span = bb1 - bb0
            if span > maxspan:
                maxspan = span; b0 = bb0; b1 = bb1
        if maxspan <= 0:
            return None
    else:
        _, b0, b1 = best
    # map back to token indices and reconstruct with punctuation
    try:
        t0 = src_word_to_tok[lo + b0]
        t1 = src_word_to_tok[min(lo + b1 - 1, len(src_word_to_tok)-1)]
    except Exception:
        return None
    toks = src_tokens[t0:t1+1]
    out = []
    for tk in toks:
        if tk["type"] == "punct":
            if out: out[-1] = out[-1] + tk["text"]
            else:   out.append(tk["text"])
        else:
            out.append(((" " if out else "") + tk["text"]))

    # return text + debug breadcrumbs: resolved local source word span and anchor
    meta = {"mode": "local", "src_lo": lo + b0, "src_hi": lo + b1 - 1, "anchor_src": (anchor_src if anchor_src is not None else -1)}
    text = "".join(out).strip()
    if return_meta:
        meta = {"mode":"local","anchor_src":(anchor_src if anchor_src is not None else -1),
                "src_lo": lo + b0, "src_hi": lo + b1 - 1, "lo":lo, "hi":hi, "b0":b0, "b1":b1, "t0":t0, "t1":t1}
        return text, meta
    return text

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
    split_on_speaker_change: bool = True,
) -> List[Tuple[float,float,str]]:
    if not words: return []
    chunks: List[Tuple[float,float,str]] = []
    buf: List[Dict[str,Any]] = [words[0]]
    cur_spk = words[0].get("spk", None)

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
        # hard break on speaker change
        if split_on_speaker_change and (cur.get("spk") != cur_spk):
            boundary = True
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
                cur_spk = cur.get("spk")
                continue
            if buf and buf_dur() < 1.0 and not all_fillers():
                # keep accumulating until we reach 1s or punctuation
                pass
            else:
                chunks.append((buf[0]["start"], buf[-1]["end"], buf_text()))
                buf = [cur]
                cur_spk = cur.get("spk")
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
# (optional) restore mode stub
# -----------------------
def _restore_punct_and_case(text: str) -> str:
    """
    Placeholder: hook up NeMo Punctuation+Capitalization here if desired.
    For now, return text unchanged (we rely on 'graft' to supply real punctuation).
    """
    return text

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
    punct_mode: str = "graft",
    punct_source: Optional[str] = None,
    manifest_field: str = "text",
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
    overlap_policy: str="drop",
    fade_ms: int=10,
    lang_for_ipa: str="en-us",
    speaker_purity_min: float = 0.90,
    breadcrumbs: Optional[str] = None,
    qc_csv: Optional[str]=None,
    stitch_gap_ms: int = 0,
    extend_tail_ms: int = 0,
    join_guard_ms: int = 120,
    word_min_cov: float=0.50,
    word_min_ms: int=80,
    graft_min_sim: float = 0.60,
    graft_local_radius: int = 80,
):
    audio24k_path = Path(audio24k_path)

    segs = load_aligned_words(aligned_json)
    # Breadcrumbs writer
    crumb_f = open(breadcrumbs, "w", encoding="utf-8") if breadcrumbs else None
    def _crumb(event: str, **kw):
        if crumb_f:
            rec = {"event": event, **kw}
            try:
                crumb_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    all_words = []
    for s in segs:
        for w in s["words"]:
            w = dict(w)
            w["__idx"] = len(all_words)
            w["__norm"] = _norm_token(w.get("word",""))
            all_words.append(w)

    spk_intvs = parse_rttm(rttm_path)
    _crumb("config",
           audio=str(audio24k_path), aligned_json=str(aligned_json), rttm=str(rttm_path),
           out_wavs_dir=str(out_wavs_dir), manifest=str(manifest_path),
           punct_mode=punct_mode, punct_source=punct_source, manifest_field=manifest_field,
           overlap_policy=overlap_policy, mean_vad_thres=mean_vad_thres,
           pad_ms=pad_ms, join_guard_ms=join_guard_ms,
           graft_min_sim=graft_min_sim, graft_local_radius=graft_local_radius,
           speaker_purity_min=speaker_purity_min,word_min_cov=word_min_cov, word_min_ms=word_min_ms)

    sr24 = 24000
    out_dir = Path(out_wavs_dir); out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(Path(manifest_path).parent, exist_ok=True)

    sf_in = sf.SoundFile(str(audio24k_path), mode="r")
    if sf_in.samplerate != sr24:
        raise ValueError(f"Expected 24k WAV, got {sf_in.samplerate} Hz")
    total_frames = len(sf_in)
    audio_dur = total_frames / float(sr24)


    # 1) Annotate words with per-word speaker (max overlap with RTTM)
    def _assign_spk(w):
        ws, we = w["start"], w["end"]
        best, best_cov = None, 0.0
        for spk, ivs in spk_intvs.items():
            cov = union_coverage_len((ws,we), ivs)
            if cov > best_cov:
                best, best_cov = spk, cov
        w["spk"] = best
        return w

    all_words = list(map(_assign_spk, all_words))

    # 2) Build window candidates from words (per segment), speaker-aware
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
                split_on_speaker_change=True,
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
    _crumb("windows_built", count=len(windows))


    # ---------- Optional time-bucket packing (min=1, target=8, max=10 s) ----------
    def _pack_time_buckets(win_list, target: float, bmin: float, bmax: float, prefer_sil_ms: int):
        """
        Greedy packer over (s,e,txt) windows:
          * accumulate until >= bmin
          * prefer endpoints preceded by >= prefer_sil_ms gap
          * cap at bmax (or nearest boundary)
        Returns list of merged (s,e,txt) windows.
        """
        out = []
        i = 0
        while i < len(win_list):
            j = i
            s = win_list[i][0]
            best_j = None
            best_over = 1e9
            while j < len(win_list):
                e = win_list[j][1]
                dur = e - s
                if dur > bmax:
                    if best_j is not None:
                        j = best_j
                    break
                # gap before this boundary (prefer a silence)
                nice_sil = False
                if j > i:
                    gap_ms = max(0.0, (win_list[j][0] - win_list[j-1][1])) * 1000.0
                    nice_sil = gap_ms >= prefer_sil_ms
                if dur >= bmin:
                    over = abs(dur - target)
                    if nice_sil and over <= best_over:
                        best_over = over; best_j = j
                    elif best_j is None and over < best_over:
                        best_over = over; best_j = j
                j += 1
            if best_j is None:
                best_j = max(i, min(j-1, len(win_list)-1))
            # merge text across [i..best_j]
            s = win_list[i][0]; e = win_list[best_j][1]
            merged_txt = " ".join(t for _,__,t in win_list[i:best_j+1]).strip()
            out.append((s,e,merged_txt))
            i = best_j + 1
        return out

    # switch to bucketed windows if requested
    bucket_mode = False  # default; will be wired to CLI below
    try:
        # presence of attrs means we were invoked via the CLI wrapper in this file
        bucket_mode = bool(args.bucket_mode)
        if bucket_mode:
            windows = _pack_time_buckets(
                windows,
                target=getattr(args, "bucket_target", 8.0),
                bmin=getattr(args, "bucket_min", 1.0),
                bmax=getattr(args, "bucket_max", 10.0),
                prefer_sil_ms=getattr(args, "bucket_silence_ms", 700),
            )
            logger.info("Bucketed windows: %d (target=%.1fs, min=%.1fs, max=%.1fs)",
                        len(windows),
                        getattr(args, "bucket_target", 8.0),
                        getattr(args, "bucket_min", 1.0),
                        getattr(args, "bucket_max", 10.0))
            _crumb("windows_bucketed", count=len(windows),
                   target=getattr(args, "bucket_target", 8.0),
                   min=getattr(args, "bucket_min", 1.0),
                   max=getattr(args, "bucket_max", 10.0),
                   silence_ms=getattr(args, "bucket_silence_ms", 700))
    except NameError:
        # cut_sentences() may be used directly (no args namespace). Ignore.
        pass
 
    # Build punctuation graft context if requested
    graft_ready = False
    map_idx_to_src = {}
    src_tokens = []; src_word_to_tok = []; src_words_norm = []
    if punct_mode == "graft":
        if not punct_source:
            logger.warning("--punct_mode graft requires --punct_source; falling back to 'none'")
            punct_mode = "none"
        else:
            src_text = _load_punct_source_text(punct_source)
            if not src_text:
                logger.warning("punct_source provided but empty/invalid; falling back to 'none'")
                punct_mode = "none"
            else:
                src_tokens, src_word_to_tok, src_words_norm = _tokenize_src_text_with_punct(src_text)
                all_norm = [w["__norm"] for w in all_words]
                map_idx_to_src = _align_global(all_norm, src_words_norm)
                graft_ready = True if map_idx_to_src else False
                if not graft_ready:
                    logger.warning("Could not build alignment to punctuation source; falling back to 'none'")
                    punct_mode = "none"
                else:
                    # stash per-word src indices to enable local anchoring
                    for w in all_words:
                        if w["__idx"] in map_idx_to_src:
                            w["__src_idx"] = map_idx_to_src[w["__idx"]]


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

    # Speaker purity threshold (from CLI)
    purity_min = speaker_purity_min

    # 3) Filter, slice audio, write files + manifest + QC
    spk2id: Dict[str,int] = {}
    seg_idx = 0
    qc_rows = []
    with open(manifest_path, "w", encoding="utf-8") as man_f:
        for i,(s,e,txt_pre) in enumerate(windows):

            # base normalized text from coverage-gated words
            idxs_all = _words_indices_in_range(all_words, s, e)
            idxs, _dropped = _filter_idxs_by_coverage(
                idxs_all, all_words, s, e,
                min_cov=word_min_cov, min_ms=word_min_ms
            )
            txt_norm = " ".join(all_words[j]["word"] for j in idxs).strip() or txt_pre
            # choose display text according to punct_mode (with locality + validation)
            display_text = txt_norm
            dbg_graft = None
            dbg_sim = None
            # breadcrumbs
            dbg_mode: str = ""
            dbg_src_lo: Optional[int] = None
            dbg_src_hi: Optional[int] = None
            dbg_anchor: Optional[int] = None
            # content-ish words inside this window (exclude fillers/1-char)
            idxs_all = _words_indices_in_range(all_words, s, e)
            _content = [w for j,w in enumerate(all_words) if j in idxs_all
                        if len((w.get("word","").strip())) >= 2 and not _is_filler(w.get("word",""))]
            dbg_content_n = len(_content)

            if punct_mode == "graft" and graft_ready:
                # ---- Hard guard: skip graft if window is too flimsy ----
                _chars = sum(len(w.get("word","").strip()) for j,w in enumerate(all_words) if j in idxs)
                _dur   = (all_words[idxs[-1]]["end"] - all_words[idxs[0]]["start"]) if idxs else 0.0
                fragile = (dbg_content_n < 3) or (_chars < 8) or (_dur < 1.2)
                if fragile:
                    display_text = txt_norm[:1].upper() + txt_norm[1:] if txt_norm else txt_norm
                    _crumb("graft_skip_fragile", i=i, s=round(s,3), e=round(e,3),
                           content=dbg_content_n, chars=_chars, dur=round(_dur,3))
                else:
                    # 1) locality-aware graft (preferred)
                    graft_meta = None
                    g = _graft_text_for_window_local(
                        idxs, all_words, src_tokens, src_word_to_tok, src_words_norm,
                        radius_words=graft_local_radius, return_meta=True
                    )
                    if isinstance(g, tuple): graft, graft_meta = g
                    else:                    graft = g
                    # 2) global fallback
                    if not graft:
                        g = _graft_text_for_window(
                            idxs, all_words, map_idx_to_src, src_tokens, src_word_to_tok, return_meta=True
                        )
                        if isinstance(g, tuple): graft, graft_meta = g
                        else:                    graft = g
                    if graft:
                        dbg_graft = graft
                        dbg_sim = _seq_ratio(graft, txt_norm)
                        accepted = bool(dbg_sim >= graft_min_sim)
                        if accepted:
                            display_text = graft
                        else:
                            display_text = txt_norm[:1].upper() + txt_norm[1:] if txt_norm else txt_norm
                        
                        if graft_meta and isinstance(graft_meta, dict):
                            dbg_mode   = graft_meta.get("mode","")
                            dbg_src_lo = graft_meta.get("src_lo")
                            dbg_src_hi = graft_meta.get("src_hi")
                            dbg_anchor = graft_meta.get("anchor_src")

                        _crumb("graft", i=i, s=round(s,3), e=round(e,3),
                            sim=round(float(dbg_sim),3), accepted=accepted,
                            mode=(graft_meta.get("mode") if graft_meta else None),
                            meta=graft_meta, txt_norm=txt_norm, text_graft=graft)
                    else:
                        display_text = txt_norm[:1].upper() + txt_norm[1:] if txt_norm else txt_norm
                        _crumb("graft_miss", i=i, s=round(s,3), e=round(e,3), txt_norm=txt_norm)

            elif punct_mode == "restore":
                display_text = _restore_punct_and_case(txt_norm)
            else:
                # 'none' or unknown: keep normalized but capitalize sentence-initial word heuristically
                display_text = txt_norm[:1].upper() + txt_norm[1:] if txt_norm else txt_norm


            dur = e - s
            if dur < min_dur:
                continue

            dom_spk, has_ovl, cover = dominant_speaker_and_overlap((s,e), spk_intvs, cover_thres=0.05)
            if has_ovl and overlap_policy == "drop":
                _crumb("drop_overlap", i=i, s=round(s,3), e=round(e,3))
                continue
            if dom_spk is None:
                _crumb("drop_no_speaker", i=i, s=round(s,3), e=round(e,3))
                continue

            # enforce purity
            dur_win = max(1e-9, e - s)
            purity = cover.get(dom_spk, 0.0) / dur_win
            if purity < purity_min:
                # try a surgical split at nearest internal word edge where speaker flips
                idxs = [k for k,w in enumerate(all_words) if (w["start"] < (e-0.005)) and (w["end"] > (s+0.005))]
                cut_pts = []
                for a,b in zip(idxs, idxs[1:]):
                    if all_words[a].get("spk") != all_words[b].get("spk"):
                        cut_pts.append( (all_words[a]["end"] + all_words[b]["start"]) * 0.5 )
                did_split = False
                for cp in cut_pts:
                    if (cp - s) > 0.35 and (e - cp) > 0.35:
                        # in-place split so the new windows are processed immediately
                        windows[i:i+1] = [(s, cp, txt_pre), (cp, e, txt_pre)]

                        did_split = True
                        break
                if did_split:
                    _crumb("surgical_split", i=i, s=round(s,3), e=round(e,3), cp=round(cp,3))
                    continue
                # else drop mixed window
                continue

            mean_vad = mean_vad_in_window((s,e))
            if mean_vad < mean_vad_thres:
                _crumb("drop_vad", i=i, s=round(s,3), e=round(e,3), mean_vad=float(mean_vad))
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

            if manifest_field == "ipa":
                field_val = phonemize_ipa(display_text, lang_for_ipa) if display_text else ""
            else:
                field_val = display_text
            man_f.write(f"{wav_name} | {field_val} | {spk2id[dom_spk]}\n")
            # QC uses padded/capped times for transparency
            s_pad = round(start_frame/sr24, 3)
            e_pad = round(end_frame  /sr24, 3)

            # also log raw window boundaries in ms (pre-pad) as breadcrumbs
            win_start_ms = int(round(s * 1000.0))
            win_end_ms   = int(round(e * 1000.0))
            # choose a human-friendly graft mode string when not in 'graft'
            qc_mode = dbg_mode if dbg_mode else ("restore" if punct_mode=="restore" else ("none" if punct_mode in ("none","time") else ""))

            qc_rows.append({
                "utt": wav_name,
                "spk_name": dom_spk,
                "spk_id": spk2id[dom_spk],
                "start": s_pad,
                "end": e_pad,
                "dur": round(e_pad-s_pad, 3),
                "win_start_ms": win_start_ms,
                "win_end_ms": win_end_ms,
                "mean_vad": round(mean_vad,3),
                "overlap": int(has_ovl),
                "text": display_text,
                "text_norm": txt_norm,
                "text_graft": dbg_graft if dbg_graft is not None else "",
                "graft_sim": round(dbg_sim, 3) if dbg_sim is not None else "",
                "graft_mode": qc_mode,
                "src_anchor_idx": (dbg_anchor if (dbg_anchor is not None and dbg_anchor >= 0) else ""),
                "src_span_lo": (dbg_src_lo if dbg_src_lo is not None else ""),
                "src_span_hi": (dbg_src_hi if dbg_src_hi is not None else ""),
                "content_words": dbg_content_n,
            })
            seg_idx += 1

            _crumb("emit", utt=wav_name, spk_name=dom_spk, spk_id=spk2id[dom_spk],
                start=s_pad, end=e_pad, dur=round(e_pad - s_pad,3),
                text=display_text, text_norm=txt_norm, text_graft=(dbg_graft or ""),
                graft_sim=(round(float(dbg_sim),3) if dbg_sim is not None else None))


    sf_in.close()

    with open(spk2id_path, "w", encoding="utf-8") as f:
        json.dump(spk2id, f, indent=2)

    if qc_csv:
        os.makedirs(Path(qc_csv).parent, exist_ok=True)

        with open(qc_csv, "w", newline="", encoding="utf-8") as f:
            cols = [
                "utt","spk_name","spk_id",
                "start","end","dur","win_start_ms","win_end_ms",
                "mean_vad","overlap",
                "text","text_norm","text_graft","graft_sim","graft_mode",
                "src_anchor_idx","src_span_lo","src_span_hi",
                "content_words"
            ]

            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in qc_rows: w.writerow(r)

    _crumb("summary", emitted=seg_idx)
    print(f"Done. Wrote {seg_idx} WAVs to {out_wavs_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"spk2id:   {spk2id_path}")
    if qc_csv: print(f"QC CSV:   {qc_csv}")
    if crumb_f: crumb_f.close()

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

    p.add_argument("--punct_mode", choices=["graft","time","restore","none"], default="graft",
                   help="How to obtain punctuated+capitalized text for manifest.")
    p.add_argument("--punct_source", default=None,
                   help="Parakeet text source (manifest.json or transcript.jsonl) for 'graft'/'time' modes.")
    p.add_argument("--manifest_field", choices=["ipa","text"], default="text",
                   help="Which field to write into train_list.txt: IPA (phonemized) or text.")

    p.add_argument("--min_dur", type=float, default=1.0)
    p.add_argument("--gap_thres", type=float, default=1.2)
    p.add_argument("--period_gap", type=float, default=0.8)
    p.add_argument("--ellipsis_gap", type=float, default=1.4)
    p.add_argument("--max_chars", type=int, default=None)
    p.add_argument("--min_dur_singleton", type=float, default=0.30)

    p.add_argument("--pad_ms", type=int, default=20)
    p.add_argument("--mean_vad_thres", type=float, default=0.80)
    p.add_argument("--overlap_policy", choices=["drop","dominant"], default="drop")
    p.add_argument("--fade_ms", type=int, default=10)
    p.add_argument("--lang_for_ipa", default="en-us")

    p.add_argument("--stitch_gap_ms", type=int, default=0)
    p.add_argument("--extend_tail_ms", type=int, default=0)
    p.add_argument("--join_guard_ms", type=int, default=120)
    p.add_argument("--word_min_cov", type=float, default=0.50,
                   help="Minimum fraction of a word that must lie within the window to keep it.")
    p.add_argument("--word_min_ms", type=int, default=80,
                   help="Or, minimum milliseconds of a word that must lie within the window.")
    p.add_argument("--graft_min_sim", type=float, default=0.60,
                   help="Reject graft if normalized similarity to window text is below this.")
    p.add_argument("--graft_local_radius", type=int, default=80,
                   help="Locality radius (in words) around the mapped anchor for graft.")

    # time-bucket controls (defaults: target=8s, min=1s, max=10s)
    p.add_argument("--bucket_mode", action="store_true",
                   help="Group adjacent windows to ~fixed length buckets.")
    p.add_argument("--bucket_target", type=float, default=8.0,
                   help="Target bucket duration in seconds.")
    p.add_argument("--bucket_min", type=float, default=1.0,
                   help="Minimum bucket duration in seconds.")
    p.add_argument("--bucket_max", type=float, default=10.0,
                   help="Maximum bucket duration in seconds.")
    p.add_argument("--bucket_silence_ms", type=int, default=700,
                   help="Prefer bucket boundaries with >= this much silence before them.")

    p.add_argument("--split_on_speaker_change", action="store_true", default=True,
                   help="Break sentences when speaker label changes (recommended).")
    p.add_argument("--speaker_purity_min", type=float, default=0.90,
                   help="Min fraction of a single speaker's coverage required inside a cut.")

    p.add_argument("--qc_csv", default=None)
    p.add_argument("--breadcrumbs", default=None,
                   help="Write per-utterance JSONL breadcrumbs for debugging.")
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
        punct_mode=args.punct_mode,
        punct_source=args.punct_source,
        manifest_field=args.manifest_field,
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
        word_min_cov=args.word_min_cov,
        word_min_ms=args.word_min_ms,
        speaker_purity_min=args.speaker_purity_min,
        breadcrumbs=args.breadcrumbs,
        qc_csv=args.qc_csv,
        stitch_gap_ms=args.stitch_gap_ms,
        extend_tail_ms=args.extend_tail_ms,
        join_guard_ms=args.join_guard_ms,
        graft_min_sim=args.graft_min_sim,
        graft_local_radius=args.graft_local_radius,
    )

if __name__ == "__main__":
    main()
