# Repo Analysis: Single-Price-Point-Negotiation-Domain (Janus / HyperLoRA)

**Path:** `refs/related-work/Janus-Protocol/repo/` · **Source:** https://github.com/zacharytgray/Single-Price-Point-Negotiation-Domain · **Analyzed:** 2026-08-12 (commit `c7b3a97`, authored 2026-02-23; 21 commits spanning 2026-02-02 to 2026-02-23)

## Overview

- **Paper:** Competitive, Cooperative, and Everything In-Between: The Janus Protocol for Strategy-Infused Language Models (see `paper_analysis.md`). The repo never names the paper and the paper never links the repo; the identification is by fingerprint (README carries the paper's eq. 2 verbatim at `README.md:206`; the 11 special tokens of Section 4.4.1 are `train_janus_hyperlora.py:63-75`; Table 1's hyperparameters are the argparse defaults at `train_janus_hyperlora.py:205-249`; the 23-strategy pool is `price_strategies.py:1057-1217`).
- **Framework:** custom — plain PyTorch + `transformers` (`AutoModelForCausalLM`, `get_linear_schedule_with_warmup`, `BitsAndBytesConfig`), hand-written `Dataset`/`DataLoader`/collator/training loop. No TRL, no PEFT usage despite `peft` sitting in `requirements.txt:8` (the adapter is reimplemented from scratch in `src/training/hyper_lora.py`). Opponent LLM access is via Ollama + LangChain (`src/agents/ollama_agent.py:8-10`).
- **RL Algorithm:** none. Supervised behavior cloning of scripted trajectories. Grepping the tree for `reward|grpo|ppo|advantage|rollout` returns only a hard-coded `reward=0.0` written into the dataset schema (`negotiation.py:211`, `dataset_writer.py:41,75`) — a vestige of the "offline RL" framing in the writer's docstring (`dataset_writer.py:1-5`).
- **Base Model:** `Qwen/Qwen2-7B` at training time (`train_janus_hyperlora.py:205`), but `Qwen/Qwen2-7B-Instruct` at inference and benchmark time (`janus_agent.py:49`, `config/settings.py:50,73`) — see the mismatch flagged under Component 1.
- **Key Innovation:** a per-layer 3-layer MLP maps a scalar $\rho$ (normalized settlement position in the Zone of Possible Agreement) to a rank-dimensional sigmoid gating vector inserted between the LoRA up- and down-projections, so one frozen model exposes a continuous strategy dial; a gate-separation regularizer forces distant $\rho$ to produce distant gates.

## File Structure Map

```
repo/
├── README.md                          # quick start; partly stale (see note below)
├── requirements.txt                   # torch, transformers, peft(unused), datasets, bitsandbytes, ollama, langchain, pandas, matplotlib, seaborn
├── train.py                           # 34-line subprocess wrapper -> `python -m src.training.train_janus_hyperlora`
├── negotiation.py                     # episode engine + dataset generation entry point (492 lines)
├── run_full_janus_benchmark.py        # evaluation harness: Janus/base vs all scripted strategies (907 lines)
├── analyze_benchmark_results.py       # violin / win-loss / summary plots from benchmark CSV (729 lines)
├── generate_concession_curves.py      # per-turn offer trajectories from benchmark CSV (410 lines)
├── rank_strategies_by_utility.py      # ranks scripted strategies from the generated JSONL (564 lines)
├── visualize_strategy_curves.py       # plots each strategy's concession curve solo or vs Boulware (330 lines)
├── config/
│   ├── settings.py                    # all domain + training constants (100 lines)
│   ├── price_system_instructions.txt  # free-form LLM negotiator system prompt (19 lines)
│   └── price_determ_instructions.txt  # "wrap this scripted move in natural language" prompt (17 lines)
├── src/
│   ├── training/
│   │   ├── hyper_lora.py              # RhoHyperNet, HyperLoRALinear, inject_hyperlora (240 lines) — the method
│   │   └── train_janus_hyperlora.py   # canonical prompt builder + dataset + separation loss + train loop (887 lines)
│   ├── agents/
│   │   ├── price_strategies.py        # 23-entry STRATEGY_REGISTRY + strategy implementations (1304 lines)
│   │   ├── janus_agent.py             # HyperLoRA inference agent; imports the training prompt builder (285 lines)
│   │   ├── ollama_agent.py            # base-model opponent: prompt builder + response normalizer (332 lines)
│   │   ├── basic_price_agent.py       # free-form LLM negotiator (43 lines)
│   │   ├── price_strategy_agent.py    # LLM mouthpiece wrapping a scripted move (113 lines)
│   │   ├── agent_factory.py           # agent registry / construction (190 lines)
│   │   └── base_agent.py              # abstract interface (119 lines)
│   ├── core/price_structures.py       # PriceAction / PriceState dataclasses (41 lines)
│   ├── domain/single_issue_price_domain.py  # ZOPA sampling, action parsing, outcome (272 lines)
│   ├── data_prep/prepare_data.py      # JSONL -> decision_steps.parquet (331 lines)
│   ├── logging/dataset_writer.py      # JSONL writer; stamps terminal rho on every step (255 lines)
│   ├── logging/csv_logger.py          # episode-level CSV (164 lines)
│   └── utils/thinking_model_processor.py  # strips <think> blocks from Ollama output (35 lines)
├── docs/
│   ├── hyperlora_training_report.md   # 517-line architecture+training reference (best single doc; partly stale)
│   └── price_strategies.md            # strategy doc — stale, describes a registry that no longer exists
└── strategy_curve_plots/*.png         # 13 committed concession-curve figures
```

No tests anywhere. No `datasets/`, `checkpoints/`, `logs/`, or notebook in the tree — all gitignored (`.gitignore:12-17`), so **no data and no weights ship with the repo**. Three files the README advertises do not exist: `run_janus_vs_deterministic.py`, `visualize_concessions.py`, `analysis/analyze_results_spp.ipynb` (`README.md:28-29,89`). `docs/price_strategies.md` documents `boulware_linear`, `price_fixed_strict/loose`, `time_dependent`, `micro_low/mid/high` — none of which are in the current registry. Treat `docs/hyperlora_training_report.md` as the authoritative doc and the two others as historical.

## Component Inventory

### 1. Entry Points

| Script | Role | Key arguments |
|---|---|---|
| `negotiation.py:364` | run episodes; generate the training corpus | `--num_runs`, `--max_turns`, `--dataset_out`, `--buyer_type/--seller_type`, `--buyer_strategy/--seller_strategy`, `--randomize_strategies`, `--janus_rho/--janus_adapter/--janus_model` (`negotiation.py:316-361`) |
| `src/data_prep/prepare_data.py:288` | JSONL → `decision_steps.parquet` + `trajectory_outcomes.parquet` | `--jsonl_path`, `--output_dir`, `--k` (history window, default 8) |
| `src/training/train_janus_hyperlora.py:617` (or the `train.py:28` wrapper) | supervised HyperLoRA training | see table below |
| `run_full_janus_benchmark.py:675` | evaluation | `--episodes_per_strategy` (50), `--buyer_rho` (0.2), `--seller_rho` (0.8), `--janus_rho`, `--janus_adapter`, `--strategies`, `--use_base_model`, `--model_name`, `--include_base_comparison`, `--only_base_comparison` (`run_full_janus_benchmark.py:681-728`) |
| `analyze_benchmark_results.py:602`, `generate_concession_curves.py:310`, `rank_strategies_by_utility.py:495`, `visualize_strategy_curves.py:314` | plotting / ranking | read the benchmark CSV or the dataset JSONL |

Training defaults (`train_janus_hyperlora.py:205-249`) — these are the paper's Table 1:

| Flag | Default | Flag | Default |
|---|---|---|---|
| `--model_name` | `Qwen/Qwen2-7B` | `--rank` | 16 |
| `--k_history` | 8 | `--alpha` | 32.0 |
| `--include_failures` | `true` | `--dropout` | 0.05 |
| `--batch_size` | 4 | `--hyper_hidden` | 64 |
| `--grad_accum` | 8 | `--lambda_sep` | **0.0 (off)** |
| `--lr` | 2e-4 | `--sep_exclude_failures` | `true` |
| `--max_steps` | 20000 | `--lambda_smooth` | 0.0, explicitly `[RESERVED - NOT CURRENTLY IMPLEMENTED]` (`:228-229`) |
| `--warmup_steps` | 500 | `--max_length` | 1024 |
| `--save_every` | 1000 | `--eval_every` | 1000 — **declared and never read** |

Two dead flags: `--lambda_smooth` and `--eval_every` appear only at their argparse lines; evaluation actually rides inside the `save_every` branch (`train_janus_hyperlora.py:808-826`).

**The $\lambda_{sep}$ value the paper never states is not recoverable from code either** — the default is 0.0 (regularizer off) and the only value that appears anywhere is the README's illustrative `--lambda_sep 0.01` (`README.md:152`). No config file, no run script, no logged run records the value actually used.

**Base-model mismatch to verify before reuse:** training defaults to `Qwen/Qwen2-7B` (`train_janus_hyperlora.py:205`) while `JanusAgent` defaults to `Qwen/Qwen2-7B-Instruct` (`janus_agent.py:49`) and the benchmark hard-loads `JANUS_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"` from settings (`config/settings.py:50`, used at `run_full_janus_benchmark.py:797,804`). `save_hyperlora_adapter` records the training base model in `adapter_config.json` (`train_janus_hyperlora.py:595`), but `janus_agent.py:98-103` never reads that key — it loads whatever `model_path` says. Unless training was launched with an explicit `--model_name .../Qwen2-7B-Instruct`, the evaluated system is an adapter trained on the base checkpoint and served on the instruct checkpoint.

### 2. Special Token / Action Handling

Eleven tokens, `train_janus_hyperlora.py:63-75`:

```python
SPECIAL_TOKENS = [
    "<RHO>", "<RHO_FAIL>", "<SUCCESS>", "<TURN>", "<TURNS_REMAINING>",
    "<RESERVATION_NORM>", "<LAST_OFFER_NORM>", "<HISTORY_LEN>", "<HISTORY>",
    "<INSTRUCTION>", "<OUTPUT>",
]
```

They are registered as `additional_special_tokens` and the embedding matrix is resized (`:650`, `:698`). The whole prompt is one `build_prompt()` (`train_janus_hyperlora.py:163-195`) that the inference agent imports rather than reimplements (`janus_agent.py:19-24`) — the cleanest train/inference-parity mechanism in the repo:

```python
    prompt = (
        f"{rho_line}\n"
        f"{success_line}\n"
        f"<TURN> {turn} / {max_turns}\n"
        f"<TURNS_REMAINING> {turns_remaining}\n"
        f"<RESERVATION_NORM> {reservation_norm:.4f}\n"
        f"<LAST_OFFER_NORM> {last_offer_norm_str}\n"
        f"<HISTORY_LEN> {history_len}\n"
        f"<HISTORY> {history_str}\n"
        f"<INSTRUCTION> Output exactly one of:\n"
        f"ACCEPT\n"
        f"OFFER <PRICE_NORM>\n"
        f"<OUTPUT>\n"
    )
```

`format_rho_text` (`:93-104`) is the single writer of the textual $\rho$ (`<RHO> 0.5300` or `<RHO> <RHO_FAIL>`); `format_success_text` (`:107-111`) derives `<SUCCESS>` from the same number; `build_history_str` (`:114-160`) emits exactly K slots, front-padded with the literal `EMPTY:EMPTY` (`:158`) so padding is never mistakable for a zero price. Generation is not intercepted mid-stream — there is no stop-token machinery; the model emits at most 30 new tokens and the text is regex-parsed afterwards (`janus_agent.py:218-246`).

Two observations worth carrying:

- **The special-token embeddings are never trained and never saved.** `inject_hyperlora` unfreezes only parameters whose names contain `lora_` or `hyper_net` (`hyper_lora.py:220-224`), and `save_hyperlora_adapter` persists only `requires_grad` parameters (`train_janus_hyperlora.py:584-588`). The 11 tokens therefore act as fixed arbitrary vectors from the base checkpoint, not learned markers — and inference never calls `resize_token_embeddings`, so their identity depends entirely on the served base checkpoint (compounding the mismatch in Component 1).
- **The "no `<ROLE>` token" claim does not do the work the paper says it does.** Role is trivially recoverable from `<RESERVATION_NORM>`: buyer_max $\sim \mathcal{N}(900,50)$ and seller_min = buyer_max − 500, normalized against the public range [200, 1500] (`config/settings.py:33-42`, `negotiation.py:105-108`). A buyer's reservation normalizes into roughly [0.42, 0.65]; a seller's into roughly [0.04, 0.27]. The two ranges are disjoint, so a single field in the prompt discloses the role exactly. "Posture must be inferred from $\rho$ alone" (`train_janus_hyperlora.py:23`, `README.md:235`) is not what the input actually forces.

### 3. Environment / Tool Interaction

The environment is in-process Python; there is no tool-calling, no server, no retrieval. Two loops exist:

- `negotiation.py:110-313` — `NegotiationEngine.run_episode`, used for corpus generation. Alternating turns from `starting_role`, up to `MAX_TURNS = 20` (`config/settings.py:26-27`), state assembled as a `PriceState` (`negotiation.py:173-182`), agreement recorded when an agent returns `ACCEPT` and a `last_offer` exists (`:228-235`).
- `run_full_janus_benchmark.py:204-472` — the evaluation loop, a near-duplicate that additionally supports an Ollama base-model agent on either side.

Only the base-model opponent talks over a network: `ChatOllama` at `http://localhost:11434`, temperature 0.2, stateless per turn (memory reset before every call, `ollama_agent.py:45-49, 260-309`). Janus itself runs in-process on GPU.

Domain constants (`config/settings.py:26-42`): 20 turns, buyer_max $\sim \mathcal{N}(900, 50)$, `FIXED_ZOPA_WIDTH = 500.0`, public range [200, 1500]. Sampling: `negotiation.py:105-108` and `single_issue_price_domain.py:77-78`.

**The fixed ZOPA width is not merely a leak in principle — the scripted strategies exploit it explicitly.** Every strategy reconstructs its opponent's reservation as own ± 500: `strategy_boulware` (`price_strategies.py:35-36`), `strategy_price_fixed` (`:146-161`), `strategy_tit_for_tat` (`:181-182`), `strategy_fair` (`:428-435`), `strategy_micro` (`:955-964`). The "hidden reservations" framing does not survive contact with the code.

**Acceptance is never checked against the accepter's own reservation.** In both loops an `ACCEPT` closes the deal at `last_offer` with no ZOPA test (`negotiation.py:228-235`, `run_full_janus_benchmark.py:321-330`), and utility is computed as a signed difference (`run_full_janus_benchmark.py:427-434`). The Janus agent also denormalizes its offer against the public range with no clamp to its own reservation (`janus_agent.py:239-242`). Together these make **negative normalized utility reachable** — which is what the paper's unexplained −4.3% for the Flipped configuration is. Note the asymmetry: the Ollama base model *is* clamped to its reservation before its action is emitted (`ollama_agent.py:211-214, 224-226`), so the protection exists for the baseline and not for the method under test.

### 4. Token Masking

Prompt tokens are masked out of the loss; only the target action is supervised (`train_janus_hyperlora.py:334-342`):

```python
        labels = input_ids.clone()

        prompt_enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]

        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            labels[:] = -100
```

The target is `"ACCEPT"` or `f"OFFER {price_norm:.4f}"` plus EOS (`:306-312, 326`). Padding uses `-100` for labels and 0 for attention (`:371-373`, collator `:355-381`). Note the prompt is re-tokenized separately with `add_special_tokens=False` while the full text is tokenized with defaults (`:328-330`) — for Qwen2 (no BOS) these agree, but the boundary is computed rather than tracked, so any tokenizer with a BOS would silently shift the mask by one.

### 5. Reward Function

**Absent** — this is supervised learning. The only reward-shaped object is the outcome scalar $\rho$, computed once per trajectory at episode close and stamped onto every step of that trajectory (`dataset_writer.py:124-129`):

```python
        denominator = max_acceptable - min_acceptable
        rho = 0.0
        if denominator != 0:
            rho = (final_price - min_acceptable) / denominator
```

i.e. $\rho = \frac{\text{FinalPrice} - \text{SellerMin}}{\text{BuyerMax} - \text{SellerMin}}$, with impasse trajectories written at $\rho = -1.0$ through a parallel writer (`dataset_writer.py:188-255`, constant at `:204`). Evaluation-time utility is `janus_utility / zopa_width` (`run_full_janus_benchmark.py:427-434`) — never used as a training signal.

### 6. RL Training Loop

**Absent.** No policy gradient, no rollouts, no advantages, no group sampling — confirmed by grep across the tree (only hits are the literal `reward=0.0` field and unrelated substrings). The trained artifact is a behavior-cloner of scripted play, conditioned in hindsight on the outcome that play reached.

### 7. SFT / Cold-Start Pipeline (the repo's actual training method)

The supervised phase *is* the whole method.

**Architecture** — `src/training/hyper_lora.py`. `RhoHyperNet` (`:11-74`) is `Linear(1→64) → SiLU → Linear(64→64) → SiLU → Linear(64→16) → Sigmoid` (`:33-46`); 5,328 parameters per instance, one instance per targeted linear (`:112-118`), i.e. 7 per transformer block (`:175`). An optional Fourier encoding of $\rho$ exists but is off by default (`:26-31, 50-63`). `HyperLoRALinear` (`:77-170`) implements $W = W_{base} + \frac{\alpha}{r} B\,\mathrm{diag}(g(\rho))\,A$ without ever materializing $W$:

```python
        g = self.hyper_net(rho)

        x_lora = x.to(self.lora_A.dtype)
        z = F.linear(x_lora, self.lora_A)

        g_casted = g.to(dtype=z.dtype, device=z.device)

        if z.dim() == 3:
            z_gated = z * g_casted.unsqueeze(1)
        else:
            z_gated = z * g_casted

        delta = F.linear(z_gated, self.lora_B)
        output = y_base + (self.scaling * self.dropout(delta)).to(y_base.dtype)
```

(`hyper_lora.py:152-165`). `lora_A` is Kaiming-initialized and `lora_B` zeroed so the delta starts at zero (`:123-125`). `inject_hyperlora` (`:173-226`) does in-place module surgery on every `nn.Linear` whose name ends in `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (`:175`), wires a `rho_getter` closure onto `model.current_rho` (`:187-190`), and freezes everything else (`:220-224`). The doc reports ~60M trainable / ~0.8% for Qwen2-7B rank 16 (`docs/hyperlora_training_report.md:146`); the injection count is printed at runtime (`hyper_lora.py:199`).

Note the conditioning channel: $\rho$ reaches the adapter through a **mutable attribute on the model object**, set before each forward (`train_janus_hyperlora.py:772`, `janus_agent.py:213-215`). It is not a forward argument and not part of the batch dict — a global that any caller must remember to set.

**Gate separation regularizer** — `train_janus_hyperlora.py:387-436`, the paper's main training novelty:

```python
    perm = torch.randperm(B, device=device)
    rho_perm = rho_flat[perm]
    ...
    for hlm in hyper_modules:
        g = hlm.hyper_net(rho_flat)
        g_perm = hlm.hyper_net(rho_perm)
        total_d = total_d + ((g - g_perm) ** 2).mean()

    avg_d = total_d / len(hyper_modules)
    w = torch.clamp((rho_flat - rho_perm).abs(), 0.0, 1.0).mean()

    return -(avg_d * w)
```

i.e. $\mathcal{L}_{sep} = -\bar{D}\cdot\bar{w}$ with $\bar{D}$ the mean squared gate difference across layers under a random within-batch permutation and $\bar{w}$ the mean clipped $|\Delta\rho|$. Impasse rows are dropped first when `--sep_exclude_failures` (`:410-416`); batches with fewer than two usable rows return a zero (`:407-408, 413-414`). Two properties matter for reuse: the weight $\bar{w}$ is a *batch mean*, not per-pair, so the objective is (mean gate distance) × (mean scalar distance) rather than a correlation — a batch of uniformly-far pairs and a batch mixing near and far pairs are scored identically; and the term is unbounded below (nothing stops gates saturating at 0/1), which is a plausible mechanism for the boundary instability the paper reports at $\rho \in \{0, 1\}$.

**Training loop** — `:764-848`. `model.current_rho = rhos` → forward with labels → `lm_loss / grad_accum` → optional `+ (lambda_sep * sep_loss) / grad_accum` (`:780-784`) → backward → on accumulation boundary, clip at 1.0, AdamW step, linear-warmup schedule step (`:788-792`, optimizer at `:709-713`). Checkpointing every `--save_every` steps saves trainable-only `adapter_state.pt` + `adapter_config.json` + tokenizer (`:579-603`) and appends an eval entry to `training_log.json` (`:808-845`). Resume support (`:719-751`) reloads `adapter_state.pt` with `strict=False` and fast-forwards the scheduler; it does **not** restore optimizer moments or dataloader position.

**Eval split leaks.** The split is drawn over *decision steps* after a global shuffle (`:638-645`), not over trajectories, so steps from a trajectory land on both sides — and since every step of a trajectory carries the same $\rho$ and near-identical history prefixes, the 1% eval loss is close to a training-loss readout. `run_evaluation` (`:442-516`) further attributes one batch-level loss to every $\rho$ bucket in that batch, an approximation its own comment admits (`:478-480`).

### 8. Data Pipeline

Three stages, all on disk, all gitignored.

1. **Episodes → JSONL.** `negotiation.py --dataset_out ... --buyer_type deterministic --seller_type deterministic` auto-enables `--randomize_strategies` (`:367-371`), then draws an independent uniform strategy pair per episode from the registry (`:432-458`). `DatasetWriter` buffers each step (`dataset_writer.py:37-79`) and flushes the whole trajectory at close with the terminal $\rho$ attached to every record (`:81-186` for agreements, `:188-255` for impasses).
2. **JSONL → Parquet.** `prepare_data.py:49-222` groups by `trajectory_id`, labels success as "any ACCEPT" and failure as "reached max_turns with none" (`:100-116`), drops trajectories that are neither (`:114-116`), sets `rho_train = rho_outcome if success else -1.0` (`:164`), and emits one row per decision step carrying `turn, turns_remaining, reservation_price, price_low/high, last_offer_price, history_roles, history_prices, target_action, target_price, rho_train` (`:198-214`). History is rebuilt causally — appended only after the step is recorded (`:216-218`). `validate()` (`:225-285`) checks $\rho$ constancy within a trajectory and $\rho = -1$ on all failure steps.
3. **Parquet → tensors.** `NegotiationDataset` (`train_janus_hyperlora.py:263-349`) drops degenerate price ranges (`:270-274`) and builds prompt+target per row.

Two data-generation quirks worth knowing before reusing the corpus:

- `random_zopa` is in the registry (`price_strategies.py:1137-1142`) and is *not* excluded from dataset generation (`negotiation.py:436` excludes only a `price_fixed_` prefix that no longer matches any key), but it needs `zopa_min`/`zopa_max` params that the generator never supplies — so it silently falls back to `strategy_linear` (`price_strategies.py:392-393`). The "oracle random" strategy contributed plain linear-concession trajectories to training.
- `strategy_price_fixed` (`:131`) and `strategy_time_dependent_threshold` (`:305`) are implemented but unregistered — dead code, and the source of the README/`price_strategies.md` staleness.

**Opponent pool** — `STRATEGY_REGISTRY`, 23 entries (`price_strategies.py:1057-1217`): 4 Boulware ($\beta \in \{0.2, 0.5, 2.0, 4.0\}$), 3 noisy Boulware, `tit_for_tat`, `linear_standard`, `split_difference`, `hardliner`, `random_zopa`, 3 MiCRO (step 10/25/50), 3 ChargingBoul (`charging_boul{,_aggressive,_patient}`, an adaptation of Shymanski 2025 with UBI/AUI opponent classification at `:490-557`), 3 Fair (convergence rate 0.8/1.0/0.4), `naive_concession`, `naive_boulware`. `EXCLUDED_FROM_BENCHMARK = {"random_zopa"}` (`:1224-1226`), so `get_benchmark_strategies()` returns 22 (`:1229-1236`). Registry entries are `(name, description, func, default_params)` (`:1049-1055`) and dispatch is a plain function call (`:1291-1293`) — a genuinely clean, copyable pattern.

### 9. Multi-GPU / Infrastructure

**Absent.** Single-GPU only: `device_map=None` on both load paths and an explicit `model.to("cuda")` (`train_janus_hyperlora.py:684-706`), `num_workers=0` (`:661`), no FSDP/DeepSpeed/DDP/accelerate anywhere despite `accelerate` in `requirements.txt:12`. Memory relief is QLoRA 4-bit via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` behind `--use_qlora` (`:678-689`); dtype is bfloat16 when supported (`:672-676`). Effective batch 4×8 = 32. Inference loads one full-precision-ish (fp16) copy and caches it globally across agents so buyer and seller share weights (`janus_agent.py:27, 67-72, 139`) — the two roles differ only in the `rho` attribute, which is exactly what the method claims.

### 10. Evaluation

`run_full_janus_benchmark.py`. Per opponent strategy, 50 episodes (`:55`), each with a freshly sampled ZOPA (`:598-599`) and a 4-way balanced assignment of Janus's role and who moves first (`:564-576`). Janus buyer and seller are instantiated once with `--buyer_rho` 0.2 / `--seller_rho` 0.8 (`:686-692, 794-807`). `--include_base_comparison` adds a Janus-vs-Ollama-base matchup logged under the pseudo-strategy `base_model_opponent` (`:848-877`), and `--use_base_model` reruns the identical harness with Ollama in Janus's seat (`:605-615`).

Metrics (`calculate_strategy_summary`, `:475-520`): agreement rate over all episodes; **mean normalized utility over agreements only** (`:498-516`); average turns over all episodes; win/tie/loss with a 0.01 normalized-utility dead band (`:503-505`). The printed OVERALL row weights each strategy's mean by its agreement count (`:552-556`). Outputs are a per-episode CSV including the full offer history as JSON (`:118-139`) plus a summary JSON (`:141-172`). `analyze_benchmark_results.py` re-derives everything from the CSV, again filtering to agreements (`:72`), and draws violin plots, a win/tie/loss/disagreement stacked bar (`:313-414`), and a 2×2 summary (`:417`).

What the harness does **not** do, all verifiable here: no held-out opponent set (the benchmark list is the training pool minus `random_zopa`, `:1229-1236`); no scripted-vs-scripted control, so the model is never compared against the strategies it imitates; no plot of achieved $\rho$ against requested $\rho$; no seeding of the benchmark's RNG; no confidence intervals; and no ablation that varies the weight-space channel independently of the `<RHO>` prompt text.

**Decoding is stochastic and undocumented.** The paper is silent and `docs/hyperlora_training_report.md:439` claims `max_new_tokens=20, do_sample=False`; the code actually runs `max_new_tokens=30, temperature=0.7, do_sample=True, repetition_penalty=1.2` (`janus_agent.py:218-227`). Every reported number therefore comes from sampled decoding at temperature 0.7 with a repetition penalty applied to numeric output, and unparseable generations fall back to "offer at my own reservation" (`janus_agent.py:246, 284-285`) — a silent worst-case action that inflates agreement rate and depresses utility without appearing anywhere in the metrics.

## Relevance to this project

### Components We Can Directly Reuse

- **`src/agents/price_strategies.py:1057-1217` — the 23-strategy scripted opponent pool with its `StrategySpec` registry (`:1049-1055`) and `DeterministicPriceAgent` adapter (`:1261-1304`).** Self-contained, dependency-free, one function per strategy over a `(state, params)` signature. This is the single most reusable artifact in the repo: a calibrated ladder of opponent firmness (Boulware $\beta$ spectrum, reciprocal, minimal-concession, adaptive, fair, deliberately-bad) to place a persuasion policy against. Transplant the registry pattern even where the price domain does not transfer.
- **`src/training/hyper_lora.py` in full (240 lines).** A complete conditional-LoRA implementation with no PEFT dependency: `RhoHyperNet` (`:11-74`), the gated forward (`:134-167`), and `inject_hyperlora`'s suffix-matched module surgery (`:173-226`). If we ever want an adapter whose behavior is dialed by a scalar at inference, this is a working reference that fits on two screens.
- **`compute_gate_separation_loss` (`train_janus_hyperlora.py:387-436`)** as a template for anti-collapse pressure on any conditioning input — with the two caveats above (batch-mean weighting, unbounded below) treated as things to fix rather than copy.
- **The single-`build_prompt` train/inference-parity discipline** (`train_janus_hyperlora.py:163-195` imported by `janus_agent.py:19-24`). Cheap, and it eliminates an entire class of silent evaluation bugs. Worth adopting as a convention regardless of domain.
- **`get_balanced_episode_assignment` (`run_full_janus_benchmark.py:564-576`)** — four-way counterbalancing of role and first-mover across episodes, three lines, removes two confounds we would otherwise have to argue away.

### Components We Need to Modify

- **The training loop (`train_janus_hyperlora.py:617-883`)** is single-GPU, single-process, hand-rolled. For 7B GRPO work it is a reference for the *conditioning mechanism* only; the loop itself would be replaced by our RL stack.
- **The eval split (`:638-645`)** must become trajectory-level, not step-level, before any held-out number from this pipeline means anything.
- **`JanusAgent` decoding (`janus_agent.py:218-227`)** — pin to greedy (or record the sampling config) and make the unparseable-output fallback (`:246`) an explicit logged failure rather than a silent reservation-price offer, before any comparison built on it is trustworthy.
- **The episode loops are duplicated** between `negotiation.py:110-313` and `run_full_janus_benchmark.py:204-472` and have already drifted (different logging, different agent handling). Any reuse should take one loop.
- **Acceptance and offer validity (`negotiation.py:228-235`, `run_full_janus_benchmark.py:321-330`, `janus_agent.py:239-242`)** — decide deliberately whether irrational agreement is legal; here it is legal for the method and illegal for the baseline (`ollama_agent.py:211-214`), which is the wrong way round.
- **The fixed ZOPA width (`config/settings.py:35-36`)** must be randomized if the environment is reused, or every scripted opponent's ±500 reservation inference (`price_strategies.py:35-36, 146-161, 181-182, 428-435, 955-964`) becomes an oracle.

### Components That Don't Apply

- **Everything Ollama/LangChain** (`src/agents/ollama_agent.py`, `basic_price_agent.py`, `price_strategy_agent.py`, `config/*.txt`) — we serve models directly; the base-model baseline here is a prompt-engineered Ollama wrapper whose measured weakness is partly a property of that wrapper.
- **QLoRA 4-bit loading (`train_janus_hyperlora.py:678-689`)** — a single-GPU concession we do not need.
- **The plotting scripts** (`analyze_benchmark_results.py`, `generate_concession_curves.py`, `visualize_strategy_curves.py`, ~1,500 lines) — domain-specific matplotlib, superseded by our notebook-report convention.
- **Reward / RL components** — absent by construction. Nothing to borrow on the GRPO side; this repo's interest to us is precisely that it reaches controllable strategy *without* RL, which makes it a baseline to beat rather than a component to lift.

### Key Code Snippets Worth Studying

| Location | Why |
|---|---|
| `src/training/hyper_lora.py:134-167` | the gated LoRA forward — the entire mechanism in 30 lines; shows the delta is per-rank rescaling of one fixed adapter, and that $g$ depends on $\rho$ alone, never on dialogue state |
| `src/training/hyper_lora.py:173-226` | suffix-matched in-place module replacement + selective unfreezing; the cleanest no-PEFT adapter injection pattern in our refs |
| `src/training/train_janus_hyperlora.py:387-436` | the gate separation regularizer, verbatim |
| `src/training/train_janus_hyperlora.py:279-349` | prompt construction + loss masking for one training record, end to end |
| `src/logging/dataset_writer.py:81-186` | outcome-indexed supervision in its simplest form: compute the terminal scalar once, stamp it on every step of the trajectory |
| `src/data_prep/prepare_data.py:100-218` | trajectory labeling, $\rho$ assignment, and causal history reconstruction |
| `src/agents/price_strategies.py:1049-1093, 1229-1256` | the registry pattern, its parameterization, and the benchmark-exclusion mechanism |
| `src/agents/price_strategies.py:490-557` | UBI/AUI opponent classification (ChargingBoul) — a cheap opponent model worth knowing about |
| `run_full_janus_benchmark.py:475-520` | exactly how "mean utility" is defined (agreements only) and how wins/ties are counted (0.01 dead band) — required to read the paper's Table 3 correctly |
| `src/agents/janus_agent.py:199-246` | inference-side conditioning: same `build_prompt`, `model.current_rho` tensor, sampled decoding, regex parse, reservation fallback |
## Paper–code cross-check: discrepancies and resolutions

A dedicated pass compared every checkable claim in the paper against this code (verified 2026-08-12, commit `c7b3a97`). Items already expanded in the component inventory above are kept to one line here.

| # | Paper says | Code says | Evidence |
|---|---|---|---|
| 1 | ~60M trainable (~0.8%), Table 1 | 41,414,464 (0.54%) for the actual module set: 28 blocks × 7 projections at r=16 (per block 1,441,792 LoRA params + 7 hypernets × 5,328), 196 hypernetwork instances total. The ~60M figure originates in the repo's own doc (`docs/hyperlora_training_report.md:146`), not a measurement; reaching 60M would need r≈24 | `hyper_lora.py:175,112-118`; Qwen2-7B dims |
| 2 | gate separation loss is "critical" | `--lambda_sep` defaults to 0.0 (off); the only value anywhere is a README usage example, 0.01; the value actually used in training is recorded nowhere | `train_janus_hyperlora.py:232`; `README.md:152` |
| 3 | Max Steps 40,000 (Table 1) | default 20,000 everywhere; reconcilable only via the resume flag added in the last commit | `train_janus_hyperlora.py:216`; `config/settings.py:85` |
| 4 | one base model, Qwen2-7B | adapter trained on `Qwen/Qwen2-7B` but served on `Qwen/Qwen2-7B-Instruct`; the saved `base_model` key in `adapter_config.json` is never checked at load | `train_janus_hyperlora.py:205`; `config/settings.py:50`; `janus_agent.py:49,98-103` |
| 5 | Base baseline = same model without adapter | different serving stack and protocol: Ollama `qwen2:7b` at temperature 0.2 with a coached ~30-line natural-language prompt (stateless per turn) vs Janus on HF fp16 Instruct at temperature 0.7 with the special-token prompt. The base agent's opening-offer anchor is computed from the opponent's reservation (a ZOPA oracle input Janus never gets), its offers are clamped to its own reservation (Janus's never are), and when its output fails to parse, `last_offer` goes stale and the scripted opponent can accept its own previous offer — a maximally unfavorable "agreement" charged to the base model. The +34pp headline gap spans all of these differences, not just the adapter | `ollama_agent.py:56-194,116-120,211-226`; `run_full_janus_benchmark.py:331-341,714` |
| 6 | "no role token; posture inferred solely from ρ" | `<RESERVATION_NORM>` identifies role exactly (buyer ≈0.42–0.65 vs seller ≈0.04–0.27 under the public-range normalization); additionally, data generation always lets the buyer move first, so `<TURN>` parity encodes role during training — while the benchmark alternates first mover, an unremarked train/test shift | `negotiation.py:463-467,116`; `run_full_janus_benchmark.py:564-576` |
| 7 | −4.3% mean utility at 97.0% agreement (unexplained) | fully reproducible: `ACCEPT` closes at `last_offer` with no reservation or ZOPA check, Janus offers are never clamped, utility is the signed distance to one's own reservation over the fixed width 500, and the mean is computed over agreement episodes only. Malformed generations fall back to offering the agent's own reservation — utility exactly 0, counted as agreement | `run_full_janus_benchmark.py:321-330,424-447,496-520`; `janus_agent.py:239-246` |
| 8 | Table 3 = "Aggregate Performance Across All Strategies" | the aggregate is whatever opponent rows a run happened to include: `--strategies` accepts arbitrary subsets (the Neutral row is consistent with a 3-Fair-variants + base-comparison run), the plotting script drops opponents with zero agreements (the Base chart's 21 rows), and the OVERALL line is an agreement-weighted mean of the rows present. Mean utility, agreement rate, and average turns also use different denominators (agreements vs all episodes) | `run_full_janus_benchmark.py:706-708,549-559`; `analyze_benchmark_results.py:326` |
| 9 | "n=50" per matchup; paired comparisons implied | role split is 26 buyer / 24 seller per opponent (deterministic 4-cycle), the benchmark sets no random seed anywhere, and the ZOPA is redrawn per episode — so the five configurations are evaluated on different episode draws, unpaired | `run_full_janus_benchmark.py:564-576,598-599` |
| 10 | 23-strategy pool incl. a training-only random strategy | `random_zopa` needs `zopa_min/max` params the generator never supplies and silently falls back to `strategy_linear` — the training pool is 23 names but 22 behaviors, with linear concession double-weighted | `price_strategies.py:1137-1142,392-393` |
| 11 | scripted opponents negotiate under hidden reservations | every strategy reconstructs the opponent's reservation from its own via the hard-coded width 500; the pool consists of ZOPA oracles by construction | `price_strategies.py:35-36,181-182,428-437,955-964,630` |
| 12 | ChargingBoul adapts per opponent | its memory is a module-level dict keyed by a literal `"default"` shared across all variants, roles, and episodes — opponent classification state leaks across the whole run | `price_strategies.py:487,635-646` |
| 13 | 11 special tokens added to the tokenizer | their embeddings are frozen, excluded from the checkpoint, and inference never resizes the embedding matrix; moreover `len(tokenizer)` after adding 11 tokens (151,657) is below Qwen2-7B's 152,064 embedding rows, so the training-time resize shrinks the matrix and the new ids inherit pretrained padding rows | `hyper_lora.py:220-224`; `train_janus_hyperlora.py:584-598,650,698`; `janus_agent.py:98-121` |
| 14 | decoding unstated | sampled decoding: temperature 0.7, repetition penalty 1.2, 30 new tokens, no seed (the repo doc's "greedy, 20 tokens" is stale) | `janus_agent.py:218-227`; `docs/hyperlora_training_report.md:439` |
| 15 | Algorithm 1 data generation | matches (uniform i.i.d. strategy pairs with replacement, self-pairs allowed; BuyerMax ~ N(900,50); SellerMin = BuyerMax − 500; eq. 1 stamping exact) except: BuyerMax is clamped to [700, 1500] in generation but not in the benchmark, and the README's own corpus stats (~95,000 examples, 75% agreement) disagree with the paper's (up to 200,000; 82.4%) | `negotiation.py:105-108,432-458`; `dataset_writer.py:123-129`; `README.md:126-129` |
| 16 | — (never mentioned) | an optional Fourier encoding of ρ and alternative gate activations (tanh/softplus) exist unused; a 1% step-level eval split (leaks across trajectories) runs during training; and `rank_strategies_by_utility.py` — the exact scripts-as-agents comparison the paper lacks — exists but its output is never reported | `hyper_lora.py:26-63`; `train_janus_hyperlora.py:638-645`; `rank_strategies_by_utility.py` |

Net reading: the architecture, prompt contract, loss, and data pipeline in the code match the paper's Sections 3–4 closely (equations and token formats verbatim), so the repo is a faithful implementation reference for the method. The evaluation harness, however, is materially weaker than the paper's presentation of it — asymmetric protections between method and baseline, oracle opponents, unpaired unseeded runs, aggregation over unequal opponent sets, and two unrecoverable training constants ($\lambda_{\text{sep}}$, actual step count/base checkpoint). Reuse the mechanism; do not inherit the benchmark.
