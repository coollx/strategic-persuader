# Janus-Protocol — paper analysis

**Original title:** Competitive, Cooperative, and Everything In-Between: The Janus Protocol for Strategy-Infused Language Models
**Authors:** Zachary Gray, Dale Peasley, Feyza M. Hafızoğlu, Sandip Sen (University of Tulsa; Istanbul Commerce University)
**Venue/year:** Research Square preprint rs-9000725/v1, posted 2026-06-16; under review at the Springer journal Autonomous Agents and Multi-Agent Systems since 2026-03-01 (no decision as of 2026-08-12)
**Source:** https://www.researchsquare.com/article/rs-9000725/v1 · DOI: https://doi.org/10.21203/rs.3.rs-9000725/v1
**Type:** method paper

## 1. Method Motivation

a) **Why proposed.** Deployed LLMs have a static strategic posture: once fine-tuned toward one behavioral objective, they cannot change their negotiation stance at inference time without retraining or switching models. The authors want one frozen model whose strategic posture is a continuous inference-time control input.

b) **Pain points of existing methods.** (i) RLHF-aligned base models are "concessionary by design" — conflict-averse, anchoring-prone, and exploitable by firm or adversarial concession tactics (the paper cites NegotiationArena evidence that fake-desperation tactics boost opponent payoffs by 20% against GPT-4, and its own base-model results, Section 5.2.1). (ii) SFT-based negotiation training (AgreeMate [29]) locks a model into discrete, role-specific personas (Buyer/Seller/Generalist). (iii) RL-based training (REPO [18]) optimizes a single static policy for one reward mixture. None offer continuous, interpretable control over strategy at inference.

c) **Core hypothesis.** A hypernetwork conditioned on a single outcome-grounded scalar can learn a continuous, coherent manifold of negotiation policies inside adapter weight space, so that varying only the scalar at inference produces qualitatively distinct, strategically appropriate behaviors (Section 3, opening).

## 2. Method Design

### 2.1 Domain and the control scalar (Section 3.1)

Bilateral price negotiation, single continuous price. Two agents (buyer, seller) alternate offers for at most 20 turns; an episode ends with an `ACCEPT` action or impasse. The Zone of Possible Agreement (ZOPA, the price interval where a deal is mutually acceptable) is defined by a hidden buyer reservation BuyerMax $\sim \mathcal{N}(900, 50)$ and SellerMin set so the ZOPA width is always 500. Both agents see only a shared public price range $[200, 1500]$; prices are normalized to $[0,1]$ against this public range so the policy never encodes private-reservation structure.

The control scalar is the normalized settlement position within the ZOPA at episode end (eq. 1):

$$\rho = \frac{\text{FinalPrice} - \text{SellerMin}}{\text{BuyerMax} - \text{SellerMin}}$$

$\rho \to 0$ is buyer-favorable, $\rho \to 1$ seller-favorable, $\rho = 0.5$ the midpoint. Impasse episodes get $\rho = -1.0$ and distinct treatment (failure token, exclusion from the separation loss).

### 2.2 Dataset generation (Section 4.1, Algorithm 1; Fig. 2)

Training data comes from script-vs-script play, not from the LLM (the paper's own word "self-play" is a misnomer — the model never plays during data collection; Algorithm 1 rolls out scripted strategy pairs): pairs of deterministic strategies are drawn uniformly from a pool of 23 classical/adversarial negotiation strategies (Appendix A, Table A1: Boulware at $\beta \in \{0.2, 0.5, 2.0, 4.0\}$, noisy Boulware, Tit-for-Tat, Linear, Split Difference, Hardliner, MiCRO at step sizes 10/25/50 and ChargingBoul variants borrowed from the ANAC automated-negotiation literature [10, 11], Fair variants, naive strategies, and a training-only uniform-random strategy — 22 usable at evaluation). 10,000 complete trajectories are collected (up to 20 offers each, so up to 200,000 offer-level instances). Each finished trajectory is stamped with its terminal $\rho$, and every offer in it becomes one training record carrying that trajectory-level $\rho$. Outcome distribution (Fig. 3): 8,236 of 10,000 trajectories reach agreement; over successes $\mu = 0.525$, $\sigma = 0.288$, median $= 0.500$; 1,764 impasses excluded from that plot. The authors liken this outcome-conditioned reuse of verified trajectories to Outcome-Conditioned Reasoning Distillation [42]: condition generation on the verified terminal state instead of paying for forward-time RL search.

### 2.3 Prompt format (Section 4.4, Algorithm 2)

A single canonical `build_prompt()` is shared between training and inference (train–inference alignment). Eleven special tokens are added to the tokenizer: `<RHO>`, `<RHO_FAIL>`, `<SUCCESS>`, `<TURN>`, `<TURNS_REMAINING>`, `<RESERVATION_NORM>`, `<LAST_OFFER_NORM>`, `<HISTORY_LEN>`, `<HISTORY>`, `<INSTRUCTION>`, `<OUTPUT>`. Offer history is a fixed-width $K = 8$ window of `{side}:{price_norm}` entries, front-padded with `EMPTY:EMPTY`. $\rho$ is provided through two synchronized channels — textually as `<RHO> 0.xxxx` (four decimals) in the prompt, and numerically as a tensor consumed by the hypernetwork — so contextual and weight-space conditioning always agree. No role token is given; posture must be inferred from $\rho$ alone. `<SUCCESS>` is 1/0 by outcome during training and fixed to 1 at inference for any valid $\rho \in [0,1]$ ("assume agreement will be reached, steer toward the target $\rho$").

### 2.4 Architecture (Section 4.3)

Base model: Qwen2-7B, decoder-only, all weights frozen. At each targeted linear layer the effective weight is (eq. 2):

$$W = W_{\text{base}} + \frac{\alpha}{r}\, B\, \mathrm{diag}(g(\rho))\, A$$

with $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$ trainable, $g(\rho) \in (0,1)^r$ the gating vector, $r = 16$, $\alpha = 32$. The hypernetwork is a three-layer MLP with hidden dimension 64 (eq. 3):

$$g(\rho) = \sigma\left(W_3\, \phi\left(W_2\, \phi\left(W_1 \rho\right)\right)\right)$$

with SiLU activations $\phi$ and sigmoid output $\sigma$; one hypernetwork instance per targeted projection layer. The design point borrowed from HyperLoRA [39] is that adapter weights can be generated dynamically by a plug-in network; the design point borrowed from Zhyper [41] is factorization — generate only the $r$-dimensional gating vector rather than full LoRA matrices, which for $d = 4096$, $r = 16$ cuts the per-layer output from ~131,072 values to 16 (~8000× reduction). LoRA $A, B$ are shared and learned; only their per-rank scaling varies with $\rho$.

### 2.5 Training objective (Section 4.5, Algorithm 3)

Total loss (eq. 8): $\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_{\text{sep}} \mathcal{L}_{\text{sep}}$.

- $\mathcal{L}_{\text{LM}}$ (eq. 4): standard causal cross-entropy over the target action tokens only; prompt tokens masked.
- Gate separation regularizer (eqs. 5–7), the paper's main training novelty, counteracting "scalar collapse" ($\rho$-collapse) where the hypernetwork ignores $\rho$ and emits one average gating vector: sample a random permutation $\pi$ of the batch; for each pair $(i, \pi(i))$ compute the mean squared gate difference per targeted layer $D^{(\ell)} = \frac{1}{B}\sum_{i=1}^{B}\left(\frac{1}{r}\sum_{k=1}^{r} \left(g_k^{(\ell)}(\rho_i) - g_k^{(\ell)}(\rho_{\pi(i)})\right)^2\right)$, average over layers to $\bar{D}$, weight by mean scalar distance $w_i = \min(|\rho_i - \rho_{\pi(i)}|, 1)$, and set $\mathcal{L}_{\text{sep}} = -\bar{D} \cdot \bar{w}$. Maximizing gate distance in proportion to scalar distance forces distinct adapter configurations for distant $\rho$ while leaving nearby $\rho$ smooth. Impasse records are excluded.

Hyperparameters (Table 1): LoRA rank 16, $\alpha$ 32.0, dropout 0.05, AdamW, learning rate $2 \times 10^{-4}$, warmup 500, max steps 40,000, batch size 4 with gradient accumulation 8 (effective 32), history window $K = 8$, ~60M trainable parameters (~0.8% of the model). $\lambda_{\text{sep}}$'s value is not stated — and the code (see `repo_analysis.md`) defaults it to 0.0 (regularizer off), with 0.01 appearing only in a README example; the value actually used is recorded nowhere. The code also shows the adapters sit on all 7 projection types of all 28 blocks (196 per-layer hypernetworks), for which the true trainable count is ~41.4M (0.54%), not ~60M, and the code's default max steps is 20,000, not 40,000.

### 2.6 Inference configurations (Section 4.7, Table 2)

Five evaluated configurations, written as (buyer $\rho$, seller $\rho$): Base (no adapter); Janus Standard (0.2, 0.8) — each role targets a favorable but interior settlement; Janus Extreme (0.0, 1.0) — pole testing; Janus Flipped (0.8, 0.2) — deliberate role inversion, the causal ablation; Janus Neutral (0.5, 0.5) — mediator mode.

## 3. Comparison with Existing Methods

a) **Fundamental difference.** Prior negotiation fine-tuning produces one policy per model (or per persona); prompt-based steering has no weight-space guarantee. Janus makes strategic posture an explicit continuous coordinate of adapter space: one trained artifact, a family of policies indexed by an outcome-grounded scalar. Versus HyperLoRA/Zhyper, the conditioning input is not an identity embedding or task embedding but a semantically grounded game-theoretic quantity (settlement position in the ZOPA).

b) **Key innovations and significance.** (i) Outcome-indexed supervision — labeling whole trajectories with terminal $\rho$ and cloning them conditioned on it — notable; it is hindsight outcome conditioning applied to negotiation. (ii) Gate separation regularizer preventing conditional collapse — notable and reusable wherever a conditioning signal risks being ignored. (iii) Rank-dimensional gating instead of full weight generation — incremental (inherited from Zhyper) but well-executed. (iv) The dual text+weight conditioning contract with special tokens — incremental engineering, but it is what makes the control actually bind.

c) **Applicability.** Excels where an outcome admits a natural normalized scalar axis and scripted opponents can generate dense coverage of it. Struggles where the behavior axis is not scalar, where outcomes cannot be verified/normalized, or against rigid reciprocal opponents (Tit-for-Tat, MiCRO) that structurally punish firmness with impasse.

d) **Related work the paper does not engage.** Cicero (the strategy-conditioned Diplomacy negotiator, where a planner's intent conditions the dialogue model — the closest prior in spirit); activation-steering / representation-engineering methods, the cheapest competing mechanism for continuous inference-time behavior control and the natural baseline; control-token conditioning (CTRL); and the LoRA composition/interpolation literature (LoRAHub, task arithmetic, X-LoRA-style dynamic gating over adapters), which is the paper's own mechanism family. Its parameter-efficient-conditioning citations are limited to LoRA, HyperLoRA, HyperDPO/CoS-DPO, and Zhyper.

e) **A structural note on what the mechanism actually is.** Since $A, B$ are shared and only 16 sigmoid gates depend on $\rho$, the reachable weight deltas form a one-dimensional curve through a positive combination of 16 fixed rank-1 directions per layer — per-direction scaling of a fixed adapter, closer to feature-wise modulation applied in weight space than to weight generation. The gate depends only on $\rho$, never on dialogue state. Also, despite the "conversational negotiators" framing, the evaluated system emits only structured actions (`OFFER $X` / `ACCEPT`) over special-token prompts — no free-text negotiation is ever produced or evaluated.

d) **Comparison table.**

| Method | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Janus (this paper) | Continuous inference-time posture control; one artifact; interpretable axis; +34pp mean utility over base | Single numeric-offer domain; evaluated only against its own training strategies; boundary instability at $\rho \in \{0, 1\}$ | Held-out opponents; free-text negotiation; multi-issue; boundary-aware regularization (their Section 6.3 list) |
| Base RLHF model (Qwen2-7B) | No training; fluent | Concessionary, anchoring-prone; 21.0% mean utility; 100% loss vs firm Boulware | — |
| Role-specific SFT (AgreeMate [29]) | Simple, stable | Discrete frozen personas; retraining to change posture | Replace persona switch with a control axis (what Janus does) |
| RL with mixed rewards (REPO [18]) | Optimizes utility directly | One static policy per reward mixture; costly rollouts | Condition the policy on the reward mixture (analogous to $\rho$) |
| Prompt-only steering | Zero training | No weight-space effect; fragile; exploitable | Pair textual with weight-space conditioning (Janus's dual channel) |

## 4. Experimental Validation

a) **Design.** All configurations are evaluated against the full strategy pool of Table A1 (the same strategies that generated training data; the random strategy is training-only), plus direct head-to-head matches between each Janus configuration and the base model under identical settings (Section 4.8). Metrics: mean normalized utility, agreement rate, average turns, win–tie–loss counts. Concession-curve figures use $n = 50$ episodes per matchup per role. The ANOVA (Section 5.4) pools $n > 5{,}000$ episodes.

b) **Key results** (Table 3, aggregate over the pool):

| Configuration | Mean utility | Agreement rate | Avg turns |
|---|---|---|---|
| Base | 21.0% | 84.7% | 6.9 |
| Janus Standard (0.2/0.8) | 55.3% | 90.3% | 14.8 |
| Janus Extreme (0.0/1.0) | 44.7% | 96.3% | 12.4 |
| Janus Flipped (0.8/0.2) | −4.3% | 97.0% | 7.6 |
| Janus Neutral (0.5/0.5) | 46.3% | 98.8% | 8.0 |

- Base model fragility (Section 5.2.1, Figs. 4–5): loses 100% of episodes to Boulware Firm and Boulware Hard, 96% to MiCRO Moderate, 89% to ChargingBoul Aggressive; against Tit-for-Tat, 69% impasse and 12% mean utility. Mechanism: it opens low, then immediately jumps to near the rigid opponent's demand and stays there (anchoring/capitulation).
- Standard vs base head-to-head (Section 5.3, Figs. 6, 17, 18): Janus wins 89% of direct episodes with 79% mean utility; as seller it is near-dominant, and the base buyer's offer jumps to match Janus's high anchor by turn 2.
- Causal ablation: Flipped collapses to −4.3% mean utility while still agreeing 97.0% of the time — it "sells against a seller," anchoring on the wrong side (Fig. 13). Scalar conditioning is causally necessary for directional coherence, not decorative.
- Neutral induces fairness deliberately: against Fair opponents all outcomes fall within ±8% of an even split (Fig. 15), lowest mean utility against Fair variants still 42%.
- Statistics (Section 5.4): Welch's ANOVA (Levene's test significant, $W = 409.85$, $p < .001$ all episodes) gives $F(4, 2868.8) = 713.45$, $p < .001$ (all episodes) and $F(4, 2858.1) = 695.87$, $p < .001$ (agreements only); $\eta^2 = .234$/.252 — configuration alone explains ~23–25% of utility variance. Tukey HSD: 9 of 10 configuration pairs differ at $\alpha = .05$; Standard beats every other configuration ($p < .01$); Flipped is worse than everything including Base; only Extreme vs Neutral is non-significant ($\Delta = .018$, $p = .918$).

c) **Where it shines.** Against time-based concession strategies (all Boulware variants: 58–83% utility for Standard, Fig. 6), naive strategies (91–92%), and the concessionary base model itself. Interior $\rho$ values (Standard) beat poles (Extreme, 55.3% vs 44.7%): maximal conditioning is not maximal performance.

d) **Weak spots the paper shows.** Standard still loses predominantly to Tit-for-Tat (71% opponent wins, 25% utility), the MiCRO family (11–49% utility), and ChargingBoul variants (20–29% utility) — mutual rigidity produces standoffs and impasses; Extreme shows erratic first offers at $\rho \in \{0, 1\}$ before self-correcting (Section 6.3: plausible causes are thin data density at the endpoints, sigmoid saturation in the gates, and less-reinforced extreme postures).

e) **Weaknesses the paper does not state** (verified against the PDF during this analysis):
- **Test set equals training set.** Every evaluation opponent is a training-pool strategy (Section 4.8: "evaluated against all training strategies"); only the uniform-random strategy is training-only. There is no held-out opponent, so no generalization evidence at all. MiCRO/ChargingBoul are motivated as overfitting stress-tests (Section 3.2.1) but sit inside the training pool, defeating that purpose.
- **The Table 3 rows aggregate over different opponent sets** (code-confirmed; see `repo_analysis.md` cross-check items 8–9). The Base win–loss chart (Fig. 4) has 21 opponent rows; the Standard/Extreme/Flipped charts (Figs. 6, 11, 12) have 23 rows (adding Split Difference and the winning "vs Base Model" matchup); the Neutral chart (Fig. 14) shows only 4 rows (three Fair variants + vs Base) — and averaging each figure's per-row utility column approximately reproduces its Table 3 value (exactly, 44.7%, for Extreme). The benchmark code explains the mechanism: an arbitrary `--strategies` subset per run, zero-agreement opponents dropped from charts, and an "OVERALL" line that is an agreement-weighted mean over whatever rows are present. So "Aggregate Performance Across All Strategies" is not computed over one common opponent set, and the ANOVA/Tukey comparisons across configurations (Section 5.4) compare different opponent distributions, from unpaired unseeded episode draws. No standard deviations or confidence intervals are reported for any Table 3 mean.
- **No ablation isolates the hypernetwork.** $\rho$ enters through two always-synchronized channels (prompt text and weight-space gate), and the Flipped ablation flips both at once. A plain LoRA with $\rho$ as prompt text only — the one control that would isolate the weight-space mechanism — is absent, as is any tuned baseline (the whole +34pp gap over an untuned zero-shot base model conflates "fine-tuning on this task helps" with "scalar-conditioned gating helps"). $\lambda_{\text{sep}}$ is called critical but never ablated or even reported.
- **The fixed ZOPA width leaks private information.** SellerMin is deterministically BuyerMax − 500 in every episode, so either agent can derive its opponent's reservation from its own — contradicting the stated rationale that hidden reservations prevent trivialized bargaining, and letting a policy fit narrow absolute price bands rather than general bargaining behavior.
- **Continuous control is claimed but sampled at 5 points.** Only $\rho \in \{0.0, 0.2, 0.5, 0.8, 1.0\}$ is ever tested; there is no plot of achieved settlement $\rho$ against requested $\rho$, so the dial's calibration and monotonicity — the paper's central "everything in-between" claim — are never directly demonstrated. The multimodal training distribution (spike at 0.5, mass at the poles) also means conditioning may select memorized strategy modes rather than interpolate.
- **The scripted strategies are never scored as agents**, so there is no evidence the 7B model beats the scripts it imitates; Standard's long episodes (14.8 turns vs Base 6.9) are consistent with much of the gain being "concede late," which a simple script also achieves.
- **Hindsight conditioning at inference.** $\rho$ is a terminal-outcome label the acting agent could never know mid-episode, and `<SUCCESS>` is always forced to 1 at inference; the failure-conditioned regime is explicitly untested ("We did not test this outcome," Section 4.2 implementation note).
- **Unstated utility semantics** (resolved by the code; see `repo_analysis.md` cross-check item 7). Flipped's mean utility is negative (−4.3%, per-strategy to −29%) at 97.0% agreement because the benchmark's `ACCEPT` closes at the standing offer with no reservation or ZOPA check, the Janus agent's offers are never clamped, utility is the signed distance to one's own reservation divided by the fixed width 500, and mean utility is computed over agreement episodes only (agreement rate and average turns use all episodes). "Agreement" means someone said ACCEPT, not "settled inside the ZOPA" — contradicting the paper's stated domain rule that out-of-ZOPA offers "cannot yield agreement".
- **Draft-state inconsistency.** The introduction still says "full quantitative evaluation is ongoing" and calls results "preliminary" while Section 5 reports a completed suite with ANOVA — a leftover from an earlier draft worth remembering when weighing polish.

## 5. Reproduction & Application

a) **Open source?** The paper declares nothing — no URL, footnote, or availability statement anywhere in 47 pages (Section 4.2 mentions "the offline training pipeline implemented in our repository" without a link). However, the corresponding author's public GitHub account hosts what is, with high confidence, that repository: **https://github.com/zacharytgray/Single-Price-Point-Negotiation-Domain** (cloned to `repo/` here). The identification is inferred, not declared — neither the paper nor the repo cites the other — but the fingerprint is exact: the README carries eq. 2 verbatim; the training script's defaults match Table 1 exactly (Qwen/Qwen2-7B, rank 16, alpha 32.0, dropout 0.05, warmup 500); the code defines exactly the 11 special tokens of Section 4.4.1, a `compute_gate_separation_loss()` with a `--lambda_sep` flag, and a 23-entry strategy registry matching Table A1 name-for-name; the last commit (2026-02-23) precedes the journal submission (2026-03-01) by a week. An earlier, broader repo by the same author, https://github.com/zacharytgray/MultiAgent-LLM-Negotiation-Research-Domain, contains the same core files plus the training corpus (`datasets/price_domain.jsonl`) and an additional multi-item domain. No model weights are published anywhere (no Hugging Face account; checkpoints gitignored). Details the paper omits ($\lambda_{\text{sep}}$'s value, targeted projection layers, the `build_prompt()` template, decoding settings) are recoverable from this code — see `repo_analysis.md`.

b) **Implementation details needing attention.** Dual conditioning must stay synchronized (same $\rho$ in text and tensor); `<SUCCESS>` handling differs between training (outcome-derived) and inference (fixed to 1); impasse records excluded from the separation loss; prices normalized against the public range, never private reservations; padding uses explicit `EMPTY:EMPTY` tokens rather than zeros to avoid numeric bias. From the code, two traps the paper hides: the adapter was (by defaults) trained on `Qwen/Qwen2-7B` but served on `Qwen/Qwen2-7B-Instruct`, and the base-model baseline runs on an entirely different stack (Ollama, temperature 0.2, coached natural-language prompt with an anchor computed from the opponent's reservation, offers clamped to its own reservation) — so the +34pp headline gap is not a controlled adapter-versus-no-adapter comparison.

c) **Transferability.** The authors argue (Section 6.5) the recipe transfers to any behavior with a semantically meaningful scalar axis: tone formality, persuasiveness intensity, verbosity, risk tolerance, code-comment density. The recipe: (1) define an outcome-grounded normalized scalar; (2) generate trajectories whose terminal outcomes densely cover it; (3) stamp trajectories with their terminal value; (4) supervised-train a gated adapter with a separation regularizer against conditional collapse.

## 6. Summary

a) **One-sentence core idea.** A tiny MLP maps a target-outcome scalar to LoRA gating vectors, making one frozen LLM's negotiation posture continuously steerable at inference.

b) **Quick-reference pipeline.**
1. Script-vs-script negotiations (23 classical strategies) produce 10,000 complete price-bargaining episodes.
2. Each episode's final price is normalized inside the ZOPA to a scalar $\rho \in [0,1]$; every turn of that episode becomes a training record stamped with $\rho$.
3. A frozen Qwen2-7B gets LoRA adapters whose per-rank scaling is produced by a 3-layer MLP reading $\rho$; $\rho$ also appears as text in the prompt.
4. Training = next-action cross-entropy + a penalty that forces different $\rho$ values to produce different gating vectors.
5. At inference, choosing $\rho$ chooses the strategic posture: 0.2/0.8 (favorable-but-interior) beats the base model by +34pp mean utility; reversing the scalars collapses performance, proving the control is causal.

## 7. Reviewer Reception

No public OpenReview page found (searched OpenReview and the general web for the exact title and venue variants on 2026-08-12), and no public reviews exist anywhere. The paper's review status, from the Research Square page's embedded editorial metadata (fetched 2026-08-12): submitted 2026-03-01 to the Springer **journal** *Autonomous Agents and Multi-Agent Systems* (JAAMAS — the journal, not the AAMAS conference), status "under review"; editor assigned 2026-03-03, reviewers invited 2026-06-09, two reviewers agreed (2026-06-18 and 2026-07-07), and the editor invited a further review on 2026-07-23 — the most recent event. No review has been returned and no decision recorded; zero community comments on the preprint. Net takeaway: every claim in this paper is currently unvetted by peer review; treat the results as preliminary evidence from a single group, with the unstated-weaknesses list above (Section 4e) as the de-facto review.

## Relevance to strategic-persuader

- The paper's diagnosis — RLHF models are concessionary by design and exploitable by firm scripted tactics — is quantified baseline evidence (21.0% mean utility; 100% loss vs firm Boulware) usable when motivating strategic-behavior training.
- The outcome-indexed supervision trick (stamp full trajectories with a terminal outcome scalar, clone conditioned on it) is a cheap alternative to RL for instilling controllable strategy, directly comparable to our GRPO-based approach as a baseline or complement.
- The gate separation regularizer is a general recipe against a conditioning signal being ignored — relevant wherever we condition generation on a target (e.g., persuasion intensity) and fear collapse.
- Caveats to carry when citing: numeric-offer protocol rather than free-text dialogue (despite "conversational negotiators" framing), evaluation only against the training strategy pool (no held-out opponents), single fixed-width ZOPA domain that leaks the opponent's reservation, no tuned baseline (untuned zero-shot base model only), no ablation separating the weight-space channel from the `<RHO>` prompt text, and under review at JAAMAS with no reviews returned — code exists but only via inferred attribution (see Section 5a).
