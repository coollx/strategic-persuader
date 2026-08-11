# Reference index

One row per reference: a self-contained folder under `related-work/` holding the paper, its analyses, and (if available) the code.

| Ref | Paper | Type | Code | Summary | Analyses |
|-----|-------|------|------|------------------|----------|
| Strategic-Persuasion | *Towards Strategic Persuasion with Language Models* ([paper.pdf](related-work/Strategic-Persuasion/paper.pdf)) — Cheng & You, ICLR 2026, [arXiv:2509.22989](https://arxiv.org/abs/2509.22989) | method | ❌ no public code (searched 2026-08-11; `chengzr01/persuasion-gym` is a different paper) | Grounds LLM persuasion evaluation in Bayesian persuasion theory: repurposes four human-persuasion datasets (Anthropic, CMV, DDO, Perspectrum) into Sender-vs-Receiver opinion-change games where both sides are LLMs, measures persuasion gains on a 7-point stance scale, and validates the LLM Receiver proxy with a 45-participant human study. Frontier Senders gain most (DeepSeek-R1: 0.23 static / 1.27 dynamic), and RL (PPO/GRPO via verl) roughly doubles a Llama-3.2-3B Sender's dynamic gains with transfer across Receiver architectures. All prompts and hyperparameters are in the appendices, so the pipeline is reproducible despite no released code. | [paper_analysis.md](related-work/Strategic-Persuasion/paper_analysis.md) |

## Literature reviews and syntheses

Cross-paper documents live in `lit-review/`, listed here, never as table rows.
