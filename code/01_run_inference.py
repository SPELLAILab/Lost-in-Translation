"""
Cross-Lingual Cultural Value Drift — Inference Script v5

Changes:
- Language-aware token limits: EN=600, HI=1000, PA=1400
- Truncation detection: exact token count check (n_generated >= max_new_tokens)
- Auto-retry: up to MAX_RETRIES attempts, each multiplying the budget by RETRY_MULTIPLIER
- Per-record fields: truncated, n_tokens_generated, attempts, max_tokens_used
- Verbose logging: timestamps, GPU memory, ETA, running stats, mini-summaries
- Output: raw_responses_v5.json + per-model checkpoint_v5_{name}.json
"""

import json, os, sys, time, gc, platform, datetime
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Force unbuffered stdout so every print shows up in PBS log immediately
sys.stdout.reconfigure(line_buffering=True)

# ── logging helper ─────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = {"INFO": "   ", "OK": "✓  ", "WARN": "!  ", "ERR": "ERR", "HDR": ">>>"}
    print(f"[{ts}] [{tag.get(level,'   ')}] {msg}", flush=True)

def section(title):
    width = 70
    log("=" * width)
    log(f"  {title}")
    log("=" * width)

def subsection(title):
    log(f"  ── {title} ──")

# ── GPU memory helper ──────────────────────────────────────────────────────────
def gpu_mem():
    if not torch.cuda.is_available():
        return "no-GPU"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"alloc={allocated:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB"

def gpu_mem_free():
    if not torch.cuda.is_available():
        return "no-GPU"
    free  = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"free={free:.2f}GB / {total:.2f}GB"

# ── ETA helper ─────────────────────────────────────────────────────────────────
def fmt_eta(seconds):
    if seconds < 0 or seconds > 86400:
        return "??:??:??"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_elapsed(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

# ── progress bar (log-file friendly) ──────────────────────────────────────────
def progress_bar(current, total, width=40):
    pct   = current / total if total > 0 else 0
    filled = int(width * pct)
    bar   = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({100*pct:.1f}%)"

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../results/checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── 5 paper models only ────────────────────────────────────────────────────────
PAPER_MODELS = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct",  "name": "llama31_8b",     "gated": True,  "eager": False, "no_cache": False},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "mistral_7b",     "gated": False, "eager": False, "no_cache": False},
    {"id": "Qwen/Qwen2.5-7B-Instruct",           "name": "qwen25_7b",      "gated": False, "eager": False, "no_cache": False},
    {"id": "google/gemma-2-9b-it",               "name": "gemma2_9b",      "gated": True,  "eager": False, "no_cache": False},
    {"id": "CohereForAI/aya-expanse-8b",         "name": "aya_expanse_8b", "gated": False, "eager": False, "no_cache": False},
]

# ── token budget config ────────────────────────────────────────────────────────
# Indic scripts cost 2-3x more tokens per word (BPE tokenizer bias).
# EN: ~4.8 chars/token  HI: ~2.6 chars/token  PA: ~2.0 chars/token
TOKENS_BY_LANG = {"en": 600, "hi": 1500, "pa": 2048}
MAX_RETRIES      = 3
RETRY_MULTIPLIER = 1.5
MAX_TOKENS_CAP   = 4096

LANGS     = ["en", "hi", "pa"]
RELIGIONS = ["sikh", "hindu"]

# ── quantisation ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP BANNER
# ══════════════════════════════════════════════════════════════════════════════
section("CULTURAL ALIGNMENT INFERENCE v5 — STARTUP")
log(f"Script       : {os.path.abspath(__file__)}")
log(f"Results dir  : {RESULTS_DIR}")
log(f"Start time   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")
log(f"Python       : {platform.python_version()}")
log(f"PyTorch      : {torch.__version__}")
log(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"CUDA version : {torch.version.cuda}")
    log(f"GPU count    : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log(f"  GPU {i}      : {props.name}  VRAM={props.total_memory/1024**3:.1f}GB")
log(f"GPU memory   : {gpu_mem()}")
log(f"HF_TOKEN     : {'set (len=' + str(len(HF_TOKEN)) + ')' if HF_TOKEN else 'NOT SET — gated models will be skipped'}")
log("")
log("Token budgets per language:")
for lang, toks in TOKENS_BY_LANG.items():
    max1 = min(int(toks * RETRY_MULTIPLIER**0), MAX_TOKENS_CAP)
    max2 = min(int(toks * RETRY_MULTIPLIER**1), MAX_TOKENS_CAP)
    max3 = min(int(toks * RETRY_MULTIPLIER**2), MAX_TOKENS_CAP)
    log(f"  {lang.upper()}: attempt1={max1}  attempt2={max2}  attempt3={max3}  cap={MAX_TOKENS_CAP}")
log(f"MAX_RETRIES={MAX_RETRIES}  RETRY_MULTIPLIER={RETRY_MULTIPLIER}  MAX_TOKENS_CAP={MAX_TOKENS_CAP}")
log("")
log("Models to run:")
for i, m in enumerate(PAPER_MODELS, 1):
    gated = "GATED" if m["gated"] else "public"
    log(f"  {i}. {m['name']:<20}  [{gated}]  {m['id']}")

# ── scenario bank ─────────────────────────────────────────────────────────────
section("LOADING SCENARIO BANK")
bank_path = os.path.join(BASE_DIR, "../data/scenario_bank.json")
log(f"Bank path: {bank_path}")
with open(bank_path, encoding="utf-8") as f:
    bank = json.load(f)
log(f"Scenarios loaded : {len(bank['scenarios'])}", "OK")
log(f"Languages        : {LANGS}")
log(f"Religions        : {RELIGIONS}")

# ── script fidelity helpers ───────────────────────────────────────────────────
def has_gurmukhi(text):
    return any('਀' <= c <= '੿' for c in text)

def has_devanagari(text):
    return any('ऀ' <= c <= 'ॿ' for c in text)

def script_fidelity(text, lang):
    if lang == "pa": return has_gurmukhi(text)
    if lang == "hi": return has_devanagari(text)
    return True

def build_prompt(scenario, religion, lang):
    wrapper = bank["advisor_wrapper"][lang]
    return wrapper + "\n\n" + scenario[religion][lang]

# ── build prompt list ─────────────────────────────────────────────────────────
prompt_list = []
for s in bank["scenarios"]:
    for religion in RELIGIONS:
        for lang in LANGS:
            prompt_list.append({
                "scenario_id": s["id"],
                "dimension":   s["dimension"],
                "religion":    religion,
                "lang":        lang,
                "prompt":      build_prompt(s, religion, lang),
            })

TOTAL_PROMPTS = len(prompt_list)
log(f"Total prompts/model: {TOTAL_PROMPTS}  "
    f"({len(bank['scenarios'])} scenarios x {len(RELIGIONS)} religions x {len(LANGS)} langs)", "OK")

# ── core generation ────────────────────────────────────────────────────────────
def _generate_once(model, tokenizer, user_text, max_new_tokens, no_cache=False):
    messages = [
        {"role": "system", "content": "You are a helpful advisor giving practical life advice."},
        {"role": "user",   "content": user_text},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"### Instruction:\n{user_text}\n\n### Response:"

    enc            = tokenizer(text, return_tensors="pt")
    input_ids      = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)
    n_input_toks   = input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,   # prevents Indic-script degeneration loops
            pad_token_id=tokenizer.eos_token_id,
            use_cache=not no_cache,
        )

    generated_ids = output[0][input_ids.shape[-1]:]
    n_generated   = len(generated_ids)
    truncated     = (n_generated >= max_new_tokens)   # exact check
    decoded       = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del input_ids, attention_mask, output, enc, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    return decoded.strip(), n_generated, n_input_toks, truncated


def generate_complete(model, tokenizer, user_text, lang, prompt_idx, no_cache=False):
    """
    Retry loop: start at TOKENS_BY_LANG[lang], multiply by RETRY_MULTIPLIER on
    truncation, cap at MAX_TOKENS_CAP. Returns complete response whenever possible.
    """
    base = TOKENS_BY_LANG[lang]

    for attempt in range(1, MAX_RETRIES + 1):
        max_toks = min(int(base * (RETRY_MULTIPLIER ** (attempt - 1))), MAX_TOKENS_CAP)

        log(f"    Attempt {attempt}/{MAX_RETRIES} | budget={max_toks} tokens | lang={lang.upper()}")
        t_gen = time.time()
        response, n_generated, n_input, truncated = _generate_once(
            model, tokenizer, user_text, max_toks, no_cache
        )
        t_gen = time.time() - t_gen

        chars_out  = len(response)
        chars_trunc = "TRUNCATED" if truncated else "COMPLETE"
        log(f"    Result    : {chars_trunc} | generated={n_generated}/{max_toks} tokens"
            f" | {chars_out} chars | {t_gen:.1f}s")

        if not truncated:
            log(f"    Response complete on attempt {attempt}.", "OK")
            return response, n_generated, n_input, max_toks, attempt, False

        if max_toks >= MAX_TOKENS_CAP:
            log(f"    Token cap {MAX_TOKENS_CAP} reached — accepting truncated response.", "WARN")
            break

        log(f"    Truncated — increasing budget and retrying...", "WARN")

    return response, n_generated, n_input, max_toks, attempt, True


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUME CHECK
# ══════════════════════════════════════════════════════════════════════════════
section("CHECKPOINT RESUME CHECK")
all_results      = []
completed_models = set()

for m in PAPER_MODELS:
    ckpt = os.path.join(RESULTS_DIR, f"checkpoint_v5_{m['name']}.json")
    if not os.path.exists(ckpt):
        log(f"  {m['name']:<20} — no checkpoint found, will run from scratch")
        continue
    data  = json.load(open(ckpt, encoding="utf-8"))
    valid = [r for r in data if r.get("response", "").strip()]
    if len(valid) == TOTAL_PROMPTS:
        all_results.extend(data)
        completed_models.add(m["name"])
        trunc = sum(1 for r in valid if r.get("truncated"))
        log(f"  {m['name']:<20} — COMPLETE ({len(valid)} records, {trunc} still truncated)", "OK")
    else:
        log(f"  {m['name']:<20} — PARTIAL ({len(valid)}/{TOTAL_PROMPTS}) — will re-run from scratch", "WARN")

log(f"Models already complete: {sorted(completed_models) or 'none'}")
models_to_run = [m for m in PAPER_MODELS if m["name"] not in completed_models]
log(f"Models to run now      : {[m['name'] for m in models_to_run]}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════
job_start = time.time()

for model_idx, model_cfg in enumerate(models_to_run, 1):
    model_name = model_cfg["name"]
    model_id   = model_cfg["id"]

    section(f"MODEL {model_idx}/{len(models_to_run)}: {model_name}")
    log(f"HuggingFace ID : {model_id}")
    log(f"Gated          : {model_cfg['gated']}")
    log(f"Eager attn     : {model_cfg.get('eager', False)}")
    log(f"No KV cache    : {model_cfg.get('no_cache', False)}")
    log(f"GPU memory before load: {gpu_mem()}")

    if model_cfg["gated"] and not HF_TOKEN:
        log(f"SKIPPING — gated model and HF_TOKEN is not set", "ERR")
        continue

    # ── load tokenizer ─────────────────────────────────────────────────────
    subsection("Loading tokenizer")
    t_load = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=HF_TOKEN or None, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log("  pad_token was None — set to eos_token", "WARN")
        log(f"  Tokenizer loaded | vocab_size={tokenizer.vocab_size}", "OK")
    except Exception as e:
        log(f"  FAILED to load tokenizer: {e}", "ERR")
        continue

    # ── load model ─────────────────────────────────────────────────────────
    subsection("Loading model (4-bit quantised)")
    load_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if model_cfg.get("eager"):
        load_kwargs["attn_implementation"] = "eager"
        log("  Using eager attention implementation")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()
    except Exception as e:
        log(f"  FAILED to load model: {str(e)[:300]}", "ERR")
        continue

    t_load = time.time() - t_load
    log(f"  Model loaded in {fmt_elapsed(t_load)}", "OK")
    log(f"  Device map: {next(model.parameters()).device}")
    log(f"  GPU memory after load: {gpu_mem()}")
    log(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B (all) "
        f"| {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.0f}M trainable")

    # ── inference loop ─────────────────────────────────────────────────────
    subsection(f"Running inference — {TOTAL_PROMPTS} prompts")
    log(f"  Token budgets: EN={TOKENS_BY_LANG['en']}  HI={TOKENS_BY_LANG['hi']}  PA={TOKENS_BY_LANG['pa']}  cap={MAX_TOKENS_CAP}")
    log(f"  Checkpoint saved every 10 prompts to {RESULTS_DIR}/checkpoint_v5_{model_name}.json")
    log("")

    model_results      = []
    consecutive_errors = 0
    no_cache           = model_cfg.get("no_cache", False)

    # Running counters for live stats
    stats = defaultdict(lambda: {"complete": 0, "truncated": 0, "error": 0,
                                  "total_toks": 0, "total_time": 0.0, "attempts_sum": 0})
    model_start = time.time()

    for i, p in enumerate(prompt_list):
        if consecutive_errors >= 10:
            log(f"  10 consecutive errors — aborting this model", "ERR")
            break

        prompt_start = time.time()
        lang         = p["lang"]
        log(f"")
        log(f"  ┌─ Prompt {i+1:03d}/{TOTAL_PROMPTS}  {progress_bar(i+1, TOTAL_PROMPTS, width=30)}")
        log(f"  │  scenario={p['scenario_id']}  religion={p['religion']}  lang={lang.upper()}  dim={p['dimension']}")

        try:
            response, n_tokens, n_input, max_toks_used, attempts, truncated = generate_complete(
                model, tokenizer, p["prompt"], lang, i, no_cache=no_cache
            )
            consecutive_errors = 0
        except Exception as e:
            log(f"  │  EXCEPTION: {str(e)[:200]}", "ERR")
            response, n_tokens, n_input, max_toks_used, attempts, truncated = "", 0, 0, 0, 1, False
            consecutive_errors += 1
            gc.collect()
            torch.cuda.empty_cache()

        elapsed   = time.time() - prompt_start
        fidelity  = script_fidelity(response, lang)
        chars_out = len(response)

        # Update running stats
        if response.strip():
            if truncated:
                stats[lang]["truncated"] += 1
            else:
                stats[lang]["complete"] += 1
        else:
            stats[lang]["error"] += 1
        stats[lang]["total_toks"]    += n_tokens
        stats[lang]["total_time"]    += elapsed
        stats[lang]["attempts_sum"]  += attempts

        # ETA
        done_so_far   = i + 1
        elapsed_total = time.time() - model_start
        rate          = elapsed_total / done_so_far if done_so_far > 0 else 0
        remaining     = (TOTAL_PROMPTS - done_so_far) * rate
        eta_str       = fmt_eta(remaining)

        status_icon = "OK" if (fidelity and not truncated) else ("WARN" if truncated else "ERR")
        log(f"  │  script_ok={fidelity}  chars={chars_out}  input_toks={n_input}  "
            f"out_toks={n_tokens}  attempts={attempts}  {elapsed:.1f}s", status_icon)
        log(f"  │  ETA for model: {eta_str}  |  elapsed: {fmt_elapsed(elapsed_total)}  "
            f"|  avg_time/prompt: {rate:.1f}s")
        log(f"  └─ done")

        model_results.append({
            "model":              model_name,
            "scenario_id":        p["scenario_id"],
            "dimension":          p["dimension"],
            "religion":           p["religion"],
            "lang":               lang,
            "prompt":             p["prompt"],
            "response":           response,
            "script_ok":          fidelity,
            "truncated":          truncated,
            "n_tokens_generated": n_tokens,
            "n_input_tokens":     n_input,
            "max_tokens_used":    max_toks_used,
            "attempts":           attempts,
            "elapsed_s":          round(elapsed, 2),
        })

        # ── checkpoint + mini summary every 10 prompts ─────────────────────
        if (i + 1) % 10 == 0:
            ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_v5_{model_name}.json")
            with open(ckpt_path, "w", encoding="utf-8") as cf:
                json.dump(model_results, cf, ensure_ascii=False, indent=2)

            log("")
            log(f"  ══ CHECKPOINT SAVED ({i+1}/{TOTAL_PROMPTS}) ══════════════════════════", "OK")
            log(f"     File: {ckpt_path}")
            log(f"     GPU memory: {gpu_mem()}")
            log(f"     Running stats by language:")
            log(f"     {'Lang':<6} {'Complete':>10} {'Truncated':>10} {'Error':>7} {'AvgToks':>9} {'AvgTime':>9} {'AvgAtt':>8}")
            log(f"     {'-'*60}")
            for lk in LANGS:
                s   = stats[lk]
                tot = s["complete"] + s["truncated"] + s["error"]
                if tot == 0:
                    continue
                avg_toks = s["total_toks"] / tot if tot else 0
                avg_time = s["total_time"] / tot if tot else 0
                avg_att  = s["attempts_sum"] / tot if tot else 0
                log(f"     {lk.upper():<6} {s['complete']:>10} {s['truncated']:>10} {s['error']:>7} "
                    f"{avg_toks:>9.0f} {avg_time:>8.1f}s {avg_att:>8.2f}")
            log(f"  ═══════════════════════════════════════════════════════════")
            log("")

    # ── model done — final checkpoint and summary ──────────────────────────
    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_v5_{model_name}.json")
    with open(ckpt_path, "w", encoding="utf-8") as cf:
        json.dump(model_results, cf, ensure_ascii=False, indent=2)

    model_elapsed = time.time() - model_start
    valid_count   = sum(1 for r in model_results if r["response"].strip())
    trunc_count   = sum(1 for r in model_results if r.get("truncated"))
    error_count   = sum(1 for r in model_results if not r["response"].strip())
    ok_count      = valid_count - trunc_count

    log("")
    log(f"  ══ MODEL COMPLETE: {model_name} ══════════════════════════════════", "OK")
    log(f"     Total prompts     : {len(model_results)}")
    log(f"     Fully complete    : {ok_count}  ({100*ok_count/len(model_results):.1f}%)")
    log(f"     Still truncated   : {trunc_count}  ({100*trunc_count/len(model_results):.1f}%)")
    log(f"     Errors/empty      : {error_count}  ({100*error_count/len(model_results):.1f}%)")
    log(f"     Model wall time   : {fmt_elapsed(model_elapsed)}")
    log(f"     Avg time/prompt   : {model_elapsed/len(model_results):.1f}s")
    log(f"     Checkpoint        : {ckpt_path}")
    log(f"     GPU memory after  : {gpu_mem()}")
    log("")
    log(f"     Per-language breakdown:")
    log(f"     {'Lang':<6} {'Complete':>10} {'Truncated':>10} {'Error':>7} {'AvgToks':>9} {'AvgTime':>9}")
    log(f"     {'-'*55}")
    for lk in LANGS:
        s   = stats[lk]
        tot = s["complete"] + s["truncated"] + s["error"]
        if tot == 0:
            continue
        avg_toks = s["total_toks"] / tot
        avg_time = s["total_time"] / tot
        log(f"     {lk.upper():<6} {s['complete']:>10} {s['truncated']:>10} {s['error']:>7} "
            f"{avg_toks:>9.0f} {avg_time:>8.1f}s")
    log(f"  ═══════════════════════════════════════════════════════════════════")

    all_results.extend(model_results)

    # ── unload model ───────────────────────────────────────────────────────
    log(f"  Unloading model from GPU...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU memory after unload: {gpu_mem_free()}", "OK")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
section("SAVING FINAL OUTPUT")

out_path = os.path.join(RESULTS_DIR, "raw_responses_v5.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
log(f"Saved {len(all_results)} records to: {out_path}", "OK")

valid_total  = sum(1 for r in all_results if r.get("response", "").strip())
trunc_total  = sum(1 for r in all_results if r.get("truncated"))
error_total  = sum(1 for r in all_results if not r.get("response", "").strip())
ok_total     = valid_total - trunc_total
models_done  = sorted(set(r["model"] for r in all_results))
job_elapsed  = time.time() - job_start

section("FINAL SUMMARY")
log(f"Job wall time       : {fmt_elapsed(job_elapsed)}")
log(f"Finish time         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Models completed    : {models_done}")
log(f"Total records       : {len(all_results)}")
log(f"Fully complete      : {ok_total}  ({100*ok_total/max(len(all_results),1):.1f}%)")
log(f"Still truncated     : {trunc_total}  ({100*trunc_total/max(len(all_results),1):.1f}%)")
log(f"Errors / empty      : {error_total}")
log("")
log("Per-model per-language truncation table:")
log(f"{'Model':<22} {'EN compl':>10} {'EN trunc':>10} {'HI compl':>10} {'HI trunc':>10} {'PA compl':>10} {'PA trunc':>10}")
log("-" * 84)
for mn in models_done:
    mr  = [r for r in all_results if r["model"] == mn]
    row = [mn]
    for lk in LANGS:
        lr    = [r for r in mr if r["lang"] == lk]
        comp  = sum(1 for r in lr if not r.get("truncated") and r.get("response","").strip())
        trunc = sum(1 for r in lr if r.get("truncated"))
        row  += [str(comp), str(trunc)]
    log(f"{row[0]:<22} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10} {row[6]:>10}")
log("")
log(f"Output file: {out_path}", "OK")
log("ALL DONE.", "OK")
