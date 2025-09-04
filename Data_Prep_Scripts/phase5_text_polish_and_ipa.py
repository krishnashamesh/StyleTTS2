#!/usr/bin/env python3
"""
Phase 5 — Text Polishing & Phonemes (Custom Data)

Reads an existing StyleTTS2-style manifest (train_list.txt) produced in earlier
phases and emits:
  - train_list.txt
  - val_list.txt
  - OOD_texts.txt
  - spk2id.json
  - stats.json (counts and simple diagnostics)
  - bad_lines.txt (any input rows we couldn't parse cleanly)

Key features
------------
* Robust parser: accepts BOTH formats you've produced so far:
    A) "<wav> | <text> | <spk>"   (from phase-3 cutter)
    B) "<wav>|<spk>|<text>"       (from phase-4 ASR-per-cut option)
  It also tolerates extra spaces around the pipe and multiple '|' in text (rare).

* Light normalization (opt-in): normalizes quotes/dashes, collapses whitespace,
  preserves apostrophes, and leaves numbers/letters intact.

* Optional IPA phonemization using espeak-ng via phonemizer backend. If the
  phonemizer/espeak is missing or fails for a line, it falls back to original
  text and records the failure.

* Stable random split: OOD first, then train/val on the remainder. Seeded
  shuffling for reproducibility.

* Speaker mapping: if speakers are numeric already, we keep them; otherwise we
  map distinct speaker labels to integer IDs and emit spk2id.json.

Usage (example)
---------------
python phase5_text_polish_and_ipa.py \
  --in_manifest /opt/apps/Training/manifests/train_list.txt \
  --out_dir /opt/apps/Training/manifests_ipa \
  --ipa --lang en-us \
  --ood-frac 0.05 --train-frac-of-rest 0.95 --seed 42 \
  --normalize

Notes
-----
- Paths are written exactly as in the input manifest (absolute or relative).
  If you want to rewrite to relative paths under a root, use --make-relative-to.
- You can point --index_json to Phase-4's /asr_per_cut/index.json to optionally
  drop lines with conf < --min_conf (default: keep all).
"""

import argparse, json, os, re, random, sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Optional, Dict

# --------------------------- Utils ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_posix(p: Path, root: Path) -> str:
    try:
        return str(PurePosixPath(os.path.relpath(str(p), str(root))))
    except Exception:
        return str(PurePosixPath(str(p)))

# Very light, deterministic normalization suitable for TTS text
_PUNCT = r"""[\.,\?\!\;\:\“\”\"\‘\’\'\(\)\[\]\{\}\<\>…—–\-_/\\]"""

def normalize_text(text: str, keep_apostrophes: bool = True) -> str:
    t = (text or "")
    # Normalize fancy quotes to ASCII
    t = (t
         .replace("\u2019", "'")
         .replace("\u2018", "'")
         .replace("\u201c", '"')
         .replace("\u201d", '"')
         .replace("—", " ")
         .replace("–", " ")
         .replace("…", "...")
    )
    # Strip (most) punctuation; optionally keep apostrophes
    punct = _PUNCT
    if keep_apostrophes:
        punct = punct.replace("\\'", "")
    t = re.sub(punct, " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------------------- IPA phonemization ---------------------

_PHONEMIZER_STATUS = "init"

def phonemize_ipa(text: str, lang: str = "en-us") -> str:
    global _PHONEMIZER_STATUS
    try:
        # Prefer the explicit EspeakBackend to avoid large deps
        from phonemizer.backend import EspeakBackend  # type: ignore
        backend = EspeakBackend(language=lang)
        ipa = backend.phonemize([text], strip=True, njobs=1)[0]
        if _PHONEMIZER_STATUS != "ok":
            print(f"[info] Phonemizer ready (espeak-ng, lang={lang}).")
            _PHONEMIZER_STATUS = "ok"
        return ipa
    except Exception as e:
        if _PHONEMIZER_STATUS != "fail":
            print(f"[warn] phonemizer/espeak-ng unavailable ({e}); falling back to raw text.")
            _PHONEMIZER_STATUS = "fail"
        return text

# ------------------------- Parsing ---------------------------

@dataclass
class Item:
    wav: str
    text: str
    spk_label: str  # may be numeric string; we map to int later

_SPK_PAT = re.compile(r"^(?:spk[_-]?(\d+)|(\d+))$")

def _is_spk_field(s: str) -> Tuple[bool, Optional[int]]:
    s = (s or "").strip()
    m = _SPK_PAT.match(s)
    if m:
        g1, g2 = m.groups()
        return True, int(g1 or g2)
    return False, None

def parse_manifest_line(line: str) -> Optional[Item]:
    # Accept both with/without spaces around pipes; also handle extra '|' in text by limiting splits
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) < 3:
        return None
    # Heuristics:
    #   Format A (cutter): wav | text | spk
    #   Format B (phase4): wav | spk  | text
    # Try to identify which field is speaker using regex
    is2, spk2 = _is_spk_field(parts[1])
    is3, spk3 = _is_spk_field(parts[-1])  # last field as speaker (in case text had pipes)

    if is2 and not is3:
        wav = parts[0]
        spk_label = parts[1]
        text = "|".join(parts[2:]).strip()
        return Item(wav=wav, text=text, spk_label=spk_label)
    elif is3:
        wav = parts[0]
        spk_label = parts[-1]
        text = "|".join(parts[1:-1]).strip()
        return Item(wav=wav, text=text, spk_label=spk_label)
    else:
        # Fallback: assume A (more common with cutter)
        wav = parts[0]
        text = "|".join(parts[1:-1]).strip() if len(parts) > 3 else parts[1]
        spk_label = parts[-1]
        return Item(wav=wav, text=text, spk_label=spk_label)

# ------------------------ Main logic -------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 5: normalize + IPA + split (train/val/OOD)")
    ap.add_argument("--in_manifest", required=True, type=Path,
                    help="Input manifest produced by Phase-3/4 (train_list.txt)")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output directory for manifests and metadata")
    ap.add_argument("--ood-frac", type=float, default=0.05,
                    help="Fraction of items to hold out as OOD_texts.txt (default 0.05)")
    ap.add_argument("--train-frac-of-rest", type=float, default=0.95,
                    help="Fraction of non-OOD assigned to train (remainder to val)")
    ap.add_argument("--seed", type=int, default=42)

    # Text processing
    ap.add_argument("--normalize", action="store_true", help="Apply light normalization before IPA")
    ap.add_argument("--keep-apostrophes", action="store_true", default=True,
                    help="Keep apostrophes during normalization (default on)")
    ap.add_argument("--ipa", action="store_true", help="Convert to IPA using espeak-ng")
    ap.add_argument("--lang", type=str, default="en-us", help="espeak-ng language code for IPA (e.g., en-us, en-gb)")

    # Optional filtering using Phase-4 index.json (confidence)
    ap.add_argument("--index_json", type=Path, default=None,
                    help="Optional asr_per_cut/index.json to filter by confidence")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="Drop rows with conf < min_conf if --index_json provided")

    # Path rewriting
    ap.add_argument("--make-relative-to", type=Path, default=None,
                    help="If set, rewrite wav paths relative to this root (posix style)")

    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out_dir)

    # Load optional confidence map
    ok_by_utt: Dict[str, float] = {}
    if args.index_json and args.index_json.exists():
        try:
            idx = json.loads(args.index_json.read_text(encoding="utf-8"))
            for row in idx:
                if row.get("ok"):
                    ok_by_utt[str(row.get("utt_id"))] = float(row.get("conf", 0.0))
        except Exception as e:
            print(f"[warn] Could not read index_json: {e}")

    # Parse input manifest
    items_raw: List[Item] = []
    bad: List[str] = []
    with open(args.in_manifest, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            it = parse_manifest_line(line)
            if not it:
                bad.append(f"{ln}: {line.rstrip()}\n")
                continue
            items_raw.append(it)

    if not items_raw:
        print("[ERROR] No valid rows found in input manifest.", file=sys.stderr)
        sys.exit(2)

    # Speaker mapping -> int ids
    spk2id: Dict[str, int] = {}
    def to_spk_id(lbl: str) -> int:
        ok, sid = _is_spk_field(lbl)
        if ok and sid is not None:
            return sid
        if lbl not in spk2id:
            spk2id[lbl] = len(spk2id)
        return spk2id[lbl]

    # Prepare final items (wav, text_or_ipa, spk_id)
    final: List[Tuple[str, str, int]] = []
    phoneme_fail = 0

    for it in items_raw:
        # Optional conf filtering
        if ok_by_utt:
            utt_id = Path(it.wav).stem
            conf = ok_by_utt.get(utt_id, 1.0)
            if conf < args.min_conf:
                continue
        text = it.text
        if args.normalize:
            text = normalize_text(text, keep_apostrophes=args.keep_apostrophes)
        if args.ipa:
            before = text
            text = phonemize_ipa(text, lang=args.lang)
            if text == before:
                # We treat exact equality as potential failure if phonemizer is active
                phoneme_fail += 1
        spk_id = to_spk_id(it.spk_label)

        wav_path_out = it.wav
        if args.make_relative_to is not None:
            try:
                wav_path_out = to_posix(Path(it.wav), args.make_relative_to)
            except Exception:
                pass

        final.append((wav_path_out, text, spk_id))

    if not final:
        print("[ERROR] No rows survived filtering/processing.", file=sys.stderr)
        sys.exit(2)

    random.shuffle(final)

    N = len(final)
    n_ood = int(round(N * float(args.ood_frac)))
    rest = N - n_ood
    n_train = int(round(rest * float(args.train_frac_of_rest)))
    n_val = rest - n_train

    ood = final[:n_ood]
    train = final[n_ood:n_ood + n_train]
    val = final[n_ood + n_train:]

    # Write manifests
    (args.out_dir / 'train_list.txt').write_text('\n'.join(f"{p}|{t}|{s}" for p,t,s in train) + '\n', encoding='utf-8')
    (args.out_dir / 'val_list.txt').write_text('\n'.join(f"{p}|{t}|{s}" for p,t,s in val) + '\n', encoding='utf-8')
    (args.out_dir / 'OOD_texts.txt').write_text('\n'.join(f"{p}|{t}|{s}" for p,t,s in ood) + '\n', encoding='utf-8')
    (args.out_dir / 'spk2id.json').write_text(json.dumps(spk2id, indent=2, ensure_ascii=False), encoding='utf-8')

    # Stats and logs
    stats = {
        "total_in": len(items_raw),
        "total_out": N,
        "ood": len(ood),
        "train": len(train),
        "val": len(val),
        "phoneme_fail_est": phoneme_fail if args.ipa else 0,
        "used_index_json": bool(ok_by_utt),
        "min_conf": args.min_conf if ok_by_utt else None,
    }
    (args.out_dir / 'stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    if bad:
        (args.out_dir / 'bad_lines.txt').write_text(''.join(bad), encoding='utf-8')

    print(f"[done] total_in={len(items_raw)} total_out={N}  ood={len(ood)}  train={len(train)}  val={len(val)}")
    if bad:
        print(f"[note] bad_lines={len(bad)} -> {args.out_dir/'bad_lines.txt'}")
    print(f"[info] outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()