# Paper Analysis: Towards Strategic Persuasion with Language Models

- **Original title**: Towards Strategic Persuasion with Language Models
- **Authors**: Zirui Cheng, Jiaxuan You (University of Illinois Urbana-Champaign)
- **Venue / year**: ICLR 2026 (published as a conference paper)
- **Source**: arXiv:2509.22989 (v2, 2026-03-07) — https://arxiv.org/abs/2509.22989
- **OpenReview**: https://openreview.net/forum?id=aTCXvJKnkE (page exists; review content unreachable from this environment — see section 7)
- **Code**: no public repository found (searched 2026-08-11; see section 5)
- **Classification**: method/system paper — the contribution is a built, reproducible framework (evaluation environments, metrics, and an RL training pipeline); the empirical findings about frontier models serve as validation of that framework.

## 1. Method Motivation

a) **Why proposed.** LLMs already produce persuasive content rated comparable to human-written arguments, and OpenAI classified GPT-4o's persuasive capabilities as "medium" risk. Yet the field has no systematic way to measure or improve LLM persuasiveness: empirical persuasion effects are highly heterogeneous across domains (advertising, politics, health), so ad-hoc evaluation setups produce fragmented, sometimes inconsistent findings.

b) **Pain points of existing approaches.** (1) Human evaluation is subjective and expensive, and Durmus et al. (2024) found model-based persuasiveness scores do not correlate well with human judgments. (2) Existing evaluation setups lack a unified theoretical foundation, so results are hard to compare or generalize. (3) There are no scalable methods for *training* LLMs to be more persuasive (existing work is mostly evaluation-only or prompt-based).

c) **Core hypothesis.** Bayesian persuasion (Kamenica & Gentzkow, 2011) — persuasion as strategic information revelation to a rational belief-updater — provides a principled and *operationalizable* frame: if LLMs approximate Bayesian receivers well enough, one can repurpose human-human persuasion datasets into controlled Sender-vs-Receiver games that both measure and (via reinforcement learning) improve persuasive capability.

## 2. Method Design (primary focus)

### 2a. Pipeline overview

**Stage 1 — Theoretical frame (Section 2.1).** Bayesian persuasion has a Sender and a Receiver with utilities $v(a,\omega)$ and $u(a,\omega)$ over Receiver action $a \in A$ and world state $\omega \in \Omega$, a common prior $\mu_0 \in \Delta(\Omega)$, and a Sender-committed signaling scheme $\pi: \Omega \to \Delta(S)$. Protocol: Sender commits to $\pi$; state $\omega \sim \mu_0$ is drawn; signal $s \sim \pi(\cdot \mid \omega)$ is observed; Receiver forms posterior $\mu_s$ by Bayes' rule and best-responds $a^*(\mu_s) \in \arg\max_{a} \mathbb{E}_{\omega \sim \mu_s}[u(a,\omega)]$. The Sender's problem reduces to choosing a Bayes-plausible distribution over posteriors ($\mathbb{E}_{\mu \sim \tau}[\mu] = \mu_0$) maximizing $\mathbb{E}_{\mu \sim \tau}[\hat{v}(\mu)]$, where $\hat{v}(\mu)$ is the Sender's expected payoff under the Receiver's best response to belief $\mu$. The optimal value is the concave closure ("concavification") of $\hat{v}$ at $\mu_0$: $\max_\pi \mathbb{E}_{\mu\sim\tau(\pi)}[\hat v(\mu)] = \hat v^*(\mu_0)$ — this is why partial disclosure often beats both full transparency and silence. The dynamic extension (Ely, 2017; Appendix A) has a Markov state, a myopic Receiver with action threshold $p^*$, and an optimal *delayed-disclosure* policy characterized by the fixed point $V = \mathrm{cav}[(1-\delta)u + \delta(V \circ f)]$.

**Stage 2 — Metrics (Section 2.2).** Two quantities operationalize persuasive capability:
- **Persuasion gain**: if the LLM Sender induces posterior $\mu$, its benefit over the prior is $\Delta\hat{v}(\mu_0) = \hat{v}(\mu) - \hat{v}(\mu_0)$; the theoretical ceiling is $\Delta V(\mu_0) = V(\mu_0) - \hat{v}(\mu_0)$ with $V(\mu_0) = \max_\tau \mathbb{E}_{\mu\sim\tau}[\hat v(\mu)]$.
- **Persuasion signals**: in dynamic settings, the conditional mutual information $I(m_t; \omega_t \mid \mathcal{H}_{t-1})$ between message and state given interaction history measures how much state-relevant information the Sender chooses to reveal per turn (high = adaptive signaling, low = deliberate withholding). In experiments this is proxied by semantic similarity of messages across contexts (Section 5, Figure 4).

**Stage 3 — Benchmark construction (Section 2.3, Appendix D).** Opinion-change task: the state space is defined by a focal claim; Receiver actions are the seven Likert stance categories from *strongly oppose* (1) to *strongly support* (7); the Sender's utility is a score-mapping that rises as the Receiver's stance moves toward the target position, independent of the underlying state; the Receiver's utility is $u(a,\omega) = -\ell(a,\omega)$ for a loss $\ell$ measuring how well the stance reflects the true state, so $a^*(\mu) \in \arg\min_a \mathbb{E}_{\omega\sim\mu}[\ell(a,\omega)]$. Four human-persuasion datasets are repurposed: **Anthropic** persuasion (Durmus et al., 2024), **DDO** (debate.org; Durmus & Cardie, 2019), **Perspectrum** (Chen et al., 2019), and **CMV** (r/ChangeMyView; Tan et al., 2016). Focal claims are extracted from raw transcripts with Llama-3.3-70B-Instruct one-sentence summarization (Appendix D, Table 8). Section 4.1 reports evaluation on 475 instances; Appendix D.2 describes a self-contained subset of 375 instances sampled to encourage adoption (the two counts appear as written in the paper).

**Stage 4 — Environment with LLM Receivers (Section 2.3).** Both Sender and Receiver are LLMs. The Receiver is prompted as a Bayesian decision-maker who "understands the strategic nature of the Sender's communication" and outputs `<score>` (Likert 1–7), `<opinion>`, `<thinking>`, `<question>` each round (full prompts in Appendix C, Tables 4–7). Static persuasion = 1 round; dynamic = 3 rounds. Validity of the Receiver proxy is checked with a human study: 45 Prolific participants annotated 149 transcripts (DeepSeek-R1 Sender, Llama-3.1-8B-Instruct Receiver); belief-update *direction* was judged reasonable in 77.18% (turn 1) rising to 85.23% (turn 3) of cases, and proportional-update ratings averaged 4.82–5.05 on a 7-point scale, all significantly above chance/neutral ($p < 0.001$; Appendix B).

**Stage 5 — RL training (Section 3).** The Sender LLM (Llama-3.2-3B-Instruct) is trained against a frozen Receiver LLM (Llama-3.1-8B-Instruct) that acts as environment dynamics. The Sender's policy generates the message autoregressively: $\pi_\theta(m \mid \omega, \mu_0, u, v, A) = \prod_{t=1}^{T} \pi_\theta(m_t \mid \omega, \mu_0, u, v, A, m_{<t})$. The reward is the realized persuasion gain $r(\omega, m, a) = v(a,\omega) - \hat{v}(\mu_0)$, where $\hat v(\mu_0) = \max_{a'} \mathbb{E}_{\omega'\sim\mu_0}[v(a',\omega')]$ — positive reward means the Sender beat the prior benchmark. Objective: $J(\theta) = \mathbb{E}_{s_0 \sim \mathcal{D},\, m \sim \pi_\theta,\, a \sim \rho}[R(s_0, m, a)]$. Training uses PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization) implemented in verl, on roughly 2,700 instances from the same datasets.

### 2b. Architecture components

There is no new neural architecture; the components are: (1) claim-extraction preprocessing (Llama-3.3-70B summarizer); (2) prompted Sender LLM; (3) prompted, frozen Receiver LLM with structured tag output and score parsing; (4) reward computation from the parsed stance score; (5) verl-based PPO/GRPO training loop. Receiver prior confidence is measured by the log-probability the Receiver model assigns to discriminative tokens (e.g., "yes") under prompts containing the claim (Appendix D.2).

### 2c. Key formulas

- Bayes plausibility: $\mathbb{E}_{\mu \sim \tau}[\mu] = \mu_0$ — the Sender can redistribute beliefs but not shift their mean.
- Sender's static problem: $\max_\tau \mathbb{E}_{\mu \sim \tau}[\hat{v}(\mu)]$ s.t. Bayes plausibility; value = concave closure of $\hat v$ at $\mu_0$.
- Persuasion is strictly beneficial iff $V(\mu_0) > \hat{v}(\mu_0)$.
- Dynamic value fixed point (Appendix A): $V = \mathrm{cav}\left[(1-\delta)u + \delta (V \circ f)\right]$, where $f$ is the no-information belief drift $\mathrm{d}\mu_t/\mathrm{d}t = \lambda(1-\mu_t)$.
- RL reward: $r(\omega, m, a) = v(a,\omega) - \hat{v}(\mu_0)$.

## 3. Comparison with Existing Methods

a) **Fundamental differences.** Prior LLM-persuasion work measures persuasiveness with human raters or LLM judges on free-form arguments, with setup-specific metrics; this paper instead *derives* the metric (persuasion gain) from an economic theory with known optimal benchmarks, and makes both sides of the interaction LLMs so the whole pipeline is simulable and trainable. Relative to game-theory work on algorithmic Bayesian persuasion (Dughmi & Xu, 2016), it moves from abstract signal spaces to natural-language messages on real controversial claims. Relative to Li et al. (2025) ("Verbalized Bayesian Persuasion"), it provides a scalable benchmark and an RL training recipe rather than solving individual persuasion problems.

b) **Key innovations.** (1) Theory-derived evaluation (persuasion gains + information-revelation signals) — *notable*; (2) repurposing four human-persuasion datasets into Sender/Receiver game environments with a human study validating the Receiver proxy — *notable*; (3) RL training of Senders directly against LLM Receivers with persuasion gain as verifiable reward, showing transfer across Receiver architectures — *significant* (first systematic demonstration that small models can be RL-trained into stronger persuaders in this framing).

c) **Applicability.** Excels: opinion-change settings with discretizable stances, comparisons across Sender models, training with cheap simulated feedback. Struggles: settings where Receivers are not even approximately Bayesian (the paper itself concedes LLM Receivers make incoherent judgments and are not fully goal-directed), preference-based (rather than belief-based) persuasion, multi-sender or multi-receiver settings (all acknowledged in Appendix G).

d) **Comparison table.**

| Approach | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| This paper (Bayesian-persuasion environments + RL) | Principled metric with theoretical ceiling; fully simulable; trainable; cross-Receiver transfer shown | Receiver is an imperfect Bayesian proxy; opinion-change domain only; utilities hand-designed (score mapping) | Better-calibrated Receiver proxies; richer utility/loss designs; verify against human receivers at scale |
| Human-rater evaluation (Durmus et al., 2024; Salvi et al., 2024) | Ground truth for human persuasion | Expensive, subjective, not trainable; judge scores uncorrelated with human judgment | Use as periodic validation of simulated environments |
| LLM-judge persuasiveness scoring | Cheap, scalable | No theoretical grounding; known miscorrelation with humans | Anchor judges to theory-derived quantities |
| Algorithmic Bayesian persuasion (game theory) | Exact optimal schemes | No natural language; abstract states/signals | Use as upper-bound oracles for language environments |

## 4. Experimental Validation

a) **Design.** Sender models: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Llama-3.3-70B-Instruct, GPT-4o, Claude 3.7 Sonnet, DeepSeek-R1. Receiver fixed to Llama-3.1-8B-Instruct for the main evaluation (Table 1); Receiver varied (Mistral-7B, Qwen2.5-7B) for the heterogeneity analysis (Table 3). Static = 1 round, dynamic = 3 rounds; 475 evaluation instances across the four datasets; stance scores 1–7; gains reported as score change relative to the prior-implied baseline. RL: Llama-3.2-3B-Instruct Sender, PPO and GRPO, learning rate $5\times10^{-7}$ constant, batch size 4, Adam, KL coefficient 0.001, ~2,700 training instances (verl).

b) **Key results.**
- Persuasion gains scale with model capability: DeepSeek-R1 achieves average gains of **0.23 (static)** and **1.27 (dynamic)** on the 1–7 scale — about 3.29% and 18.14% of the Sender's utility scale — versus e.g. Llama-3.1-8B at 0.04/0.42 (Table 1).
- Dynamic (multi-round) persuasion is far more effective than static for all models; the gap widens with model strength — persuasive power depends on interaction structure, not just model quality.
- RL training works: with Llama-3.1-8B as Receiver, average dynamic gains go from 0.21 (base 3B) to **0.38 (PPO and GRPO)** (Table 2); trained-3B gains become comparable to much larger untrained models.
- Transfer across Receivers: the Sender trained only against Llama-3.1-8B also improves against Mistral-7B (1.34 → 1.67 PPO) and Qwen2.5-7B (0.80 → 0.86/0.87), so it is not purely exploiting one Receiver's quirks.
- Theory-consistent behavior: gains are largest at intermediate Receiver prior confidence (Figure 3), matching the Bayesian-persuasion prediction; larger models show progressively lower cross-turn semantic similarity of messages (Figure 4), read as adaptive information disclosure.

c) **Where it shines.** The strongest effects are in dynamic settings on the Perspectrum and Anthropic datasets (e.g., DeepSeek-R1 dynamic gains of 1.53 and 1.33; Table 1), and for mid-confidence priors. Receiver architecture matters a lot: Mistral-7B is the most persuadable Receiver (average dynamic gain 1.81 vs DeepSeek-R1 Sender), Llama-3.1-8B the least (1.27; Table 3).

d) **Limitations.** Acknowledged: LLM Receivers are not perfectly Bayesian (incoherent probability judgments, imperfect goal-directedness); only belief-based, single-sender/single-receiver, opinion-change persuasion is covered (Appendix G). Implicit: the Sender utility is state-independent (it just wants stance movement), which departs from the canonical Bayesian-persuasion setup where Sender payoffs depend on the state; the mutual-information "persuasion signal" is only proxied by semantic similarity, not computed; RL-trained 3B models still fall well short of frontier-model gains; evaluation instances number in the hundreds, so per-dataset differences of a few hundredths of a point are likely noise (no confidence intervals reported in the main tables).

## 5. Reproduction & Application

a) **Open source?** **No public code found.** The paper text contains no repository link; searches of GitHub (repository and code search), the authors' personal accounts (`chengzr01`), and the lab organization (`ulab-uiuc`) on 2026-08-11 found no repository for this paper. Note the adjacent-sounding `chengzr01/persuasion-gym` is the code for a *different* paper (Cheng, Shen, Griffiths & Henderson, "Using Cognitive Models to Improve Language Model Simulation of Human Persuasion Games", Princeton) — do not confuse the two. Reproduction is nevertheless feasible from the paper: all Sender/Receiver prompts are in Appendix C, the four source datasets are public, claim extraction is a one-prompt summarization (Table 8), and training is standard verl PPO/GRPO.

b) **Implementation details to watch.** Receiver outputs are parsed from structured tags (`<score>`, `<opinion>`, `<thinking>`, `<question>`); the Likert score 1–7 is the action. Reward = stance score minus the prior-implied benchmark score. Hyperparameters: constant learning rate $5\times10^{-7}$, batch size 4, Adam, KL coefficient 0.001; ~2,700 training instances; 3 rounds in dynamic settings; word limits inside prompts. Receiver prior confidence is extracted from log-probabilities of discriminative tokens (e.g., "yes") under claim-containing prompts. Example transcripts (Appendix E/F) show trained small models drift toward verbose, evidence-dense messages that sometimes get truncated — watch generation length limits.

c) **Transferability.** The environment recipe (claim + prior + Likert action space + score-mapped utility + frozen Receiver LLM) transfers to any domain with extractable claims and gradable stances. The persuasion-gain reward is a verifiable reward in the RLVR (reinforcement learning with verifiable rewards) sense and can plug into any RL post-training stack. The human-study protocol for validating a Receiver proxy (direction + proportion reasonableness annotations) is reusable whenever an LLM stands in for human belief-updaters.

## 6. Summary

a) **One-sentence core idea.** Turn human persuasion datasets into Bayesian-persuasion games between LLM Sender and LLM Receiver, then measure and RL-train the Sender's persuasion gains.

b) **Quick-reference pipeline.**
1. Extract a focal claim from a human debate/persuasion transcript; define Receiver actions as a 1–7 stance scale.
2. Prompt one LLM as a Sender who wants the stance moved toward a target, and another (frozen) LLM as a Bayesian Receiver who reports a stance score each round.
3. Run 1 round (static) or 3 rounds (dynamic) of messages; the change in the Receiver's stance-implied payoff relative to the prior is the persuasion gain.
4. Evaluate any Sender model by its average gain; stronger models gain more, especially over multiple rounds.
5. Train a small Sender with PPO/GRPO using the persuasion gain as reward; it roughly doubles its dynamic gains and the improvement transfers to unseen Receiver models.

## 7. Reviewer Reception (OpenReview)

An OpenReview page exists: https://openreview.net/forum?id=aTCXvJKnkE, and the paper is published as a conference paper at ICLR 2026 (accept; stated on the camera-ready header). The review content (scores, reviews, rebuttal) could not be retrieved on 2026-08-11: OpenReview's API and site both require a browser-verification challenge that blocked automated access from this environment. No review summary is given here rather than risk fabrication; the page is worth a manual look for reviewer concerns about the Receiver-as-Bayesian-proxy assumption and the state-independent Sender utility.
