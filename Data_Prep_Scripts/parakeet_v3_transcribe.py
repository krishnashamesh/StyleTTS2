#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B-v3 transcription (NeMo 2.4+ compatible)
Emits:
  - transcript.jsonl   (single full-span segment with text; NFA will do precise timing)
  - words_raw.json     (optional: flat word list with times if available)
  - manifest.json      (raw text, absolute audio path)                 [for NFA]
  - manifest_norm.json (normalized text for CTC alignment, optional)   [for NFA]

Usage:
  python parakeet_v3_transcribe.py \
    --audio /opt/apps/bandit/workspace/clip_trim_16k.wav \
    --out_dir /opt/apps/NeMo/parakeet_v3_out \
    --language en \
    --emit_words \
    --write_manifest \
    --write_manifest_norm \
    --mode buffered --chunk_len 30 --chunk_step 25

Modes:
  --mode offline   (default) feeds the whole file to RNNT (may OOM on long audio)
  --mode buffered  streams overlapping chunks (constant VRAM), stitches text
"""

import argparse, json, re, tempfile
from pathlib import Path
import soundfile as sf
import nemo.collections.asr as nemo_asr

# ------------------------ text normalization for CTC aligners ------------------------

_PUNCT = r"""[\.\,\?\!\;\:\“\”\"\‘\’\'\(\)\[\]\{\}\<\>…—–\-_/\\]"""

def normalize_for_ctc(text: str, keep_apostrophes: bool = True) -> str:
    """
    Basic, deterministic normalization that matches typical CTC alignment models:
    - lowercase
    - strip most punctuation (optionally keep apostrophes)
    - collapse whitespace
    - keep digits/letters
    """
    t = text.lower()

    # replace unicode dashes/ellipsis with spaces/dots first
    t = t.replace("—", " ").replace("–", " ").replace("…", "...")

    # build punctuation regex: optionally exclude apostrophes from removal
    punct = _PUNCT
    if keep_apostrophes:
        punct = punct.replace("\\'", "")  # keep apostrophes
    t = re.sub(punct, " ", t)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--language", default="en")  # reserved for future decode hints
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--emit_words", action="store_true", help="Write words_raw.json if available")
    # manifest controls
    ap.add_argument("--write_manifest", action="store_true", help="Write raw manifest.json for NFA")
    ap.add_argument("--write_manifest_norm", action="store_true", help="Also write normalized manifest_norm.json")
    ap.add_argument("--keep_apostrophes", action="store_true", help="Keep apostrophes in normalized text (default off)")
    # long-audio controls
    ap.add_argument("--mode", choices=["offline", "buffered"], default="offline",
                    help="offline = single pass (may OOM); buffered = chunked streaming")
    ap.add_argument("--chunk_len", type=float, default=30.0,
                    help="seconds per chunk in buffered mode (default 30)")
    ap.add_argument("--chunk_step", type=float, default=25.0,
                    help="hop seconds between chunks in buffered mode (default 25)")
    # overlap de-duplication knobs
    ap.add_argument("--dedupe_min_run", type=int, default=6,
                    help="Min identical-token run to consider as overlap when stitching buffered chunks.")
    ap.add_argument("--dedupe_min_ratio", type=float, default=0.66,
                    help="Min Jaccard ratio over the candidate run to accept overlap trimming.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = Path(args.audio).resolve()
    if not wav.exists():
        raise FileNotFoundError(wav)

    # Duration for single full-span segment (NFA will refine per word)
    with sf.SoundFile(str(wav)) as f:
        dur = len(f) / float(f.samplerate)

    # Load Parakeet v3 (multilingual TDT)
    asr = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    asr.eval()


    # --------- helpers for stitching de-dup ---------
    def _simple_tokenize(text: str):
        # lower + keep letters/digits/' ; collapse whitespace
        t = re.sub(r"[^\w'\s]", " ", text.lower())
        return [tok for tok in re.split(r"\s+", t) if tok]

    def _jaccard(a, b):
        A, B = set(a), set(b)
        return len(A & B) / max(1, len(A | B))

    def _trim_overlap(prev_words, curr_words, min_run: int, min_ratio: float) -> int:
        """
        Return how many tokens to skip from the start of curr_words due to a
        matching suffix(prefix) overlap with prev_words.
        """
        if not prev_words or not curr_words:
            return 0
        max_k = min(len(prev_words), len(curr_words))
        for k in range(max_k, min_run - 1, -1):
            suf = prev_words[-k:]
            pre = curr_words[:k]
            if suf == pre:
                return k
            if _jaccard(suf, pre) >= min_ratio:
                return k
        return 0

    def _iter_chunks(wav_path: Path, chunk_s: float, step_s: float, sr_expected: int = 16000):
        """Yield (start_sec, end_sec, tmp_wav_path) for overlapping windows."""
        # Read fully (16 k mono expected). If stereo, fold to mono.
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            # simple mono fold
            data = (data[:, 0] + data[:, -1]) * 0.5
        if sr != sr_expected:
            raise ValueError(f"Expected {sr_expected} Hz input, got {sr} Hz")
        n = len(data)
        cs = int(chunk_s * sr)
        hs = int(step_s * sr)
        if cs <= 0 or hs <= 0:
            raise ValueError("chunk_len and chunk_step must be > 0")
        # ensure at least one window
        start = 0
        with tempfile.TemporaryDirectory() as td:
            while start < n:
                end = min(n, start + cs)
                tmp = Path(td) / f"chunk_{start}_{end}.wav"
                sf.write(tmp, data[start:end], sr, subtype="PCM_16")
                yield (start / sr, end / sr, tmp)
                if end >= n:
                    break
                start += hs

    # In 'offline' mode we do the original single-shot call (may OOM on long audio).
    # In 'buffered' mode we stream chunks with overlap and stitch the text to one span.
    words = []
    if args.mode == "offline":
        hyps = asr.transcribe(audio=[str(wav)], batch_size=args.batch_size, return_hypotheses=True)
        hyp = hyps[0][0] if isinstance(hyps[0], (list, tuple)) else hyps[0]
        text = getattr(hyp, "text", "") or (str(hyp) if not hasattr(hyp, "text") else "")
        # try to keep words if the model provided them
        wt = getattr(hyp, "word_timestamps", None)
        if wt and args.emit_words:
            for w in wt:
                token = (getattr(w, "word", None) or getattr(w, "text", None) or "").strip()
                if not token:
                    continue
                s = getattr(w, "start", None) or getattr(w, "start_time", None) or getattr(w, "start_offset", None)
                e = getattr(w, "end", None)   or getattr(w, "end_time", None)   or getattr(w, "end_offset", None)
                if s is None or e is None:
                    continue
                words.append({"start": float(s), "end": float(e), "word": token})
    else:
        pieces = []
        prev_tokens = []
        # Force batch_size=1 in buffered mode to keep VRAM flat
        for (ts, te, tmp) in _iter_chunks(wav, args.chunk_len, args.chunk_step, sr_expected=16000):
            hyps = asr.transcribe(audio=[str(tmp)], batch_size=1, return_hypotheses=True)
            h = hyps[0][0] if isinstance(hyps[0], (list, tuple)) else hyps[0]
            t = getattr(h, "text", "") or (str(h) if not hasattr(h, "text") else "")
            t = t.strip()
            if not t:
                continue
            toks = _simple_tokenize(t)
            # trim any overlapped prefix against previous chunk's tail
            skip = _trim_overlap(prev_tokens, toks, args.dedupe_min_run, args.dedupe_min_ratio)
            if skip > 0:
                toks = toks[skip:]
                t = " ".join(toks)
            if t:
                pieces.append(t)
                prev_tokens = (prev_tokens + toks)[-4096:]
        text = " ".join(pieces).strip()
        # word-level timestamps across overlapping chunks are non-trivial to dedupe;
        # we skip words_raw.json in buffered mode to avoid misleading timings.

    # 1) transcript.jsonl   (single segment, we rely on NFA for word timings)
    tr_jsonl = out_dir / "transcript.jsonl"
    tr_jsonl.write_text(
        json.dumps({"start": 0.0, "end": round(dur, 3), "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(f"Wrote: {tr_jsonl}")

    # 2) words_raw.json     (optional, only if model exposed it)
    if args.emit_words and args.mode == "offline" and words:
        (out_dir / "words_raw.json").write_text(
            json.dumps({"words": sorted(words, key=lambda d: (d['start'], d['end']))},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote: {out_dir / 'words_raw.json'}")
    elif args.emit_words and args.mode == "buffered":
        print("Note: Skipping words_raw.json in buffered mode (chunk overlaps make naive words misleading).")

    # 3) manifest.json      (raw text, absolute path) for NFA
    if args.write_manifest:
        mani = out_dir / "manifest.json"
        mani.write_text(
            json.dumps({"audio_filepath": str(wav), "text": text}, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"Wrote: {mani}")

    # 4) manifest_norm.json (normalized text) for CTC-friendly alignment
    if args.write_manifest_norm:
        tnorm = normalize_for_ctc(text, keep_apostrophes=args.keep_apostrophes)
        mani_n = out_dir / "manifest_norm.json"
        mani_n.write_text(
            json.dumps({"audio_filepath": str(wav), "text": tnorm}, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"Wrote: {mani_n}")

if __name__ == "__main__":
    main()
