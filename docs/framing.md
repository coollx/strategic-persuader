# Research framing

The research identity: what this project claims, in what setting, and the words it uses. Edited rarely — additive amendments batch after they survive; retractions are edited in immediately.

## Claim

One reinforcement-learning-trained language-model negotiator, given a short prose persuasion profile in its system prompt, lands at profile-determined points on the efficient frontier of a multi-issue negotiation — steerable at inference time from proself to prosocial. Untrained models cannot do this: fitting a single coherent welfare weight to their behavior fails (Wu et al., arXiv 2604.08525, Section 4.3), and instructions alone barely move sender behavior (arXiv 2606.01456).

Mechanism sub-claim: the seller's in-reasoning estimates of the buyer's hidden priorities, stated as rankings, improve with training and are scored directly against ground truth — opponent modeling is measured, never inferred from task success (the dissociation warning of arXiv 2602.17045).

Maintained axiom, not a claim: reaching the frontier is a requirement verified in every experiment — efficiency must stay high at every profile level; the profile owns only the surplus share.

Falsifiers, stated up front: (1) untrained prompted models already produce the same steering curve; (2) the trained model's behavior does not track its profile; (3) efficiency collapses at some profile levels; (4) untrained opponent-estimate accuracy is already at ceiling, leaving the mechanism sub-claim no headroom.

## Setting

Task: two-party multi-issue negotiation in free natural language. Each party's utility is a numeric table — ground truth by construction, hidden from the counterpart. The ledger is numeric underneath; the negotiation surface is language only, and nothing numeric ever needs to be spoken.

Environment: CaSiNo rules (arXiv 2103.15721) as a generative environment — three issues, 5/4/3 points per package by private priority order, walk-away pays 5. Trained on the natural mix of all conflict-alignment levels; analysis stratified by level, with the full-alignment stratum doubling as the near-zero-sum diagnostic. Later instantiation: sales scenarios seeded from real listings (CraigslistBargain items) with generated hidden weights — ground truth is always generated, never recovered from language.

Trained agent: one seller, a 3-8B open model, supervised fine-tuning then GRPO. Per-episode scalar reward: (welfare weight) × own points + (1 − welfare weight) × buyer points. The welfare weight is set by one of roughly five frozen prose persuasion profiles, sampled independently of everything else in the scenario; the model sees the text, and only the reward sees the number (a numeric-in-prompt variant is a planned ablation). Reward details beyond this line are deliberately unspecified here; they get their own design round before training.

Counterpart: a frozen prompted LLM buyer with per-episode sampled hidden priorities, persona, concession style, and a real walk-away rule. Evaluation on held-out buyer models plus a scripted rational buyer. Positioning: the seller computes a best response to a fixed buyer population; no equilibrium claims.

Measurement: two columns per episode — efficiency (the axiom check) and surplus share (what the profile steers). Opponent estimates scored by ranking agreement against the true priority order. The oracle-versus-inferred gap prices the cost of estimation. The Wu et al. regression (base propensity plus separate sensitivities to each side's utility) is fitted before and after training as the steerability instrument.

## Terms

The vocabulary allowlist. A term or abbreviation may be used in project documents only if listed here; only the researcher approves additions; agents never coin terms.

| Term | Meaning |
|---|---|
| persuasion profile | The short prose statement in the seller's system prompt that sets its stance from proself to prosocial; each profile maps to one welfare-weight value used by the reward. Profiles are text by default; a variant that additionally states the numeric weight in the prompt is a planned ablation. |
| welfare weight (λ) | The number λ between 0 and 1 mixing own points and buyer points in the training reward; set by the persuasion profile; visible to the model only in the numeric-profile ablation, never required in dialogue. |
| issue | One negotiable dimension of the deal (food, water, firewood; later price, delivery). The word "attribute" is not used. |
| utility, points | A party's scalar score for a deal, computed from its ground-truth priority table. |
| walk-away | The points a party receives if no deal is reached (5 under CaSiNo rules); also the boundary below which a rational party refuses a deal. |
| conflict-alignment level | How aligned the two parties' priority orders are. Full alignment means a fixed pie and pure conflict; misalignment creates room for value-creating trades. Maps to CaSiNo's "integrative potential" levels 1-3. |
| efficiency | Joint points achieved divided by the maximum joint points available in the scenario. The pie-found check. |
| surplus share | A party's points above its walk-away divided by the joint surplus above both walk-aways. The pie-split quantity the profile steers. |
| opponent estimate | The seller's stated ranking, with optional intensity words, of the buyer's hidden priorities — written in its reasoning and scored against ground truth. |
| proself, prosocial | The two ends of the persuasion-profile range: weight on own points versus weight on the buyer's points. |
| steerable | Controllable at inference time by prompt content, in the sense of steerable pluralistic alignment (arXiv 2402.05070). |
