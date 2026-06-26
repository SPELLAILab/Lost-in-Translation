#!/usr/bin/env python3
"""
Cultural Alignment Study — LaBSE semantic drift computation.

For each (model, scenario, religion) trio with valid EN + HI/PA responses,
computes cosine similarity and drift using the LaBSE multilingual model.

Input:  ../results/checkpoints/raw_responses.json
Output: ../data/embed_distances.json
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE    = Path(__file__).parent
RAW_F   = BASE / "../results/checkpoints/raw_responses.json"
OUT_F   = BASE / "../data/embed_distances.json"

with open(RAW_F) as f:
    responses = json.load(f)

# only usable records
usable = [r for r in responses if r.get("script_ok") and not r.get("truncated")]
print(f"Usable records: {len(usable)}")

# index by (model, scenario_id, religion, lang)
idx = {}
for r in usable:
    idx[(r["model"], r["scenario_id"], r["religion"], r["lang"])] = r["response"]

models    = sorted(set(r["model"] for r in usable))
scenarios = sorted(set(r["scenario_id"] for r in usable))
religions = ["sikh", "hindu"]
langs     = ["en", "hi", "pa"]

print("Loading LaBSE model...")
lm = SentenceTransformer("sentence-transformers/LaBSE")
print("LaBSE loaded.")

results = []
total = len(models) * len(scenarios) * len(religions)
done  = 0

for m in models:
    for sid in scenarios:
        for rel in religions:
            done += 1
            texts = {l: idx.get((m, sid, rel, l), "") for l in langs}
            en_text = texts["en"]
            if not en_text:
                continue

            # compute EN→HI if HI available
            for tgt in ["hi", "pa"]:
                if not texts[tgt]:
                    continue
                embs = lm.encode([en_text, texts[tgt]], normalize_embeddings=True)
                sim  = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
                results.append({
                    "model":      m,
                    "scenario_id": sid,
                    "dimension":  next(r["dimension"] for r in usable
                                       if r["model"]==m and r["scenario_id"]==sid),
                    "religion":   rel,
                    "lang_pair":  f"en→{tgt}",
                    "cosine_sim": round(sim, 4),
                    "drift":      round(1 - sim, 4),
                })

    print(f"  {m}: {len([r for r in results if r['model']==m])} pairs computed")

with open(OUT_F, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(results)} embedding distance records → {OUT_F}")

# quick summary
from collections import defaultdict
by = defaultdict(list)
for r in results:
    by[(r["model"], r["lang_pair"])].append(r["drift"])

print(f"\n{'Model':<20} {'en→hi':>8} {'en→pa':>8}")
print("-" * 40)
for m in models:
    hi = by.get((m,"en→hi"),[])
    pa = by.get((m,"en→pa"),[])
    hi_s = f"{sum(hi)/len(hi):.3f}(n={len(hi)})" if hi else "—"
    pa_s = f"{sum(pa)/len(pa):.3f}(n={len(pa)})" if pa else "—"
    print(f"{m:<20} {hi_s:>12} {pa_s:>12}")
