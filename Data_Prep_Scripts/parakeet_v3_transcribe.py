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
    --write_manifest_norm
"""

import argparse, json, re
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

    # Ask for hypotheses; NeMo 2.4 returns top-N per file if return_hypotheses=True
    hyps = asr.transcribe(audio=[str(wav)], batch_size=args.batch_size, return_hypotheses=True)
    hyp = hyps[0][0] if isinstance(hyps[0], (list, tuple)) else hyps[0]
    text = getattr(hyp, "text", "") or (str(hyp) if not hasattr(hyp, "text") else "")

    # 1) transcript.jsonl   (single segment, we rely on NFA for word timings)
    tr_jsonl = out_dir / "transcript.jsonl"
    tr_jsonl.write_text(
        json.dumps({"start": 0.0, "end": round(dur, 3), "text": text}, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(f"Wrote: {tr_jsonl}")

    # 2) words_raw.json     (optional, only if model exposed it)
    if args.emit_words:
        words = []
        wt = getattr(hyp, "word_timestamps", None)
        if wt:
            for w in wt:
                token = (getattr(w, "word", None) or getattr(w, "text", None) or "").strip()
                if not token:
                    continue
                s = getattr(w, "start", None) or getattr(w, "start_time", None) or getattr(w, "start_offset", None)
                e = getattr(w, "end", None)   or getattr(w, "end_time", None)   or getattr(w, "end_offset", None)
                if s is None or e is None:
                    continue
                words.append({"start": float(s), "end": float(e), "word": token})
        if words:
            (out_dir / "words_raw.json").write_text(
                json.dumps({"words": sorted(words, key=lambda d: (d["start"], d["end"]))},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote: {out_dir / 'words_raw.json'}")

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
