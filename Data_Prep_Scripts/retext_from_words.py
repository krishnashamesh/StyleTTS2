#!/usr/bin/env python3
# retext_from_words.py
import json, csv, argparse
from pathlib import Path
from collections import defaultdict

def load_words(segments_json):
    data = json.load(open(segments_json, "r", encoding="utf-8"))
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            ws = float(w["start"]); we = float(w["end"])
            tok = str(w.get("word","")).strip()
            if tok:
                words.append((ws, we, tok))
    words.sort(key=lambda t: (t[0], t[1]))
    return words

def read_qc(qc_csv):
    rows=[]
    with open(qc_csv, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r: rows.append(row)
    return rows, r.fieldnames

def overlap(a1,a2,b1,b2):
    s=max(a1,b1); e=min(a2,b2)
    return max(0.0, e-s)

def join_tokens(toks):
    toks.sort(key=lambda x:(x[0],x[1]))
    out=[]
    for _,_,t in toks:
        if not out: out.append(t); continue
        if t and t[0] in ".,?!;:…":
            out[-1] = out[-1] + t
        else:
            out.append(" " + t)
    return "".join(out).strip()

def spk_id_from_utt(utt):
    # utt_000083_spk0.wav -> 0
    core = utt.rsplit(".",1)[0]
    tag = core.split("_")[-1]
    return int(tag[3:]) if tag.startswith("spk") else 0

def main(args):
    words = load_words(args.aligned_json)
    qc_rows, headers = read_qc(args.qc_csv)

    pad_s = args.pad_ms/1000.0
    # windows from QC (unpadded)
    wins = []
    for row in qc_rows:
        s_pad = float(row["start"]); e_pad = float(row["end"])
        s_un = s_pad + pad_s; e_un = e_pad - pad_s
        if e_un <= s_un:  # safety
            mid = 0.5*(s_pad + e_pad)
            s_un, e_un = mid-0.05, mid+0.05
        wins.append((row["utt"], s_un, e_un))

    # assign each word to the single window with the largest overlap
    assign = defaultdict(list)
    MIN_OVL = 0.005  # 5 ms
    for ws,we,tok in words:
        best, best_ovl = None, 0.0
        for utt,s,e in wins:
            ovl = overlap(ws,we,s,e)
            if ovl > best_ovl:
                best, best_ovl = utt, ovl
        if best and best_ovl >= MIN_OVL:
            assign[best].append((ws,we,tok))

    # rebuild text per utt
    text_map = {utt: join_tokens(toks) for utt, toks in assign.items()}

    # write corrected QC (replace 'text')
    qc_out = Path(args.qc_out)
    with open(qc_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in qc_rows:
            row["text"] = text_map.get(row["utt"], "")
            w.writerow(row)

    # write corrected manifest (TEXT, not IPA)
    man_out = Path(args.manifest_out)
    with open(man_out, "w", encoding="utf-8") as f:
        for row in qc_rows:
            utt = row["utt"]
            f.write(f"{utt} | {text_map.get(utt,'')} | {spk_id_from_utt(utt)}\n")

    print("Wrote:", man_out, "and", qc_out)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_json", required=True)
    ap.add_argument("--qc_csv", required=True)
    ap.add_argument("--pad_ms", type=int, default=20)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--qc_out", required=True)
    args = ap.parse_args()
    main(args)
