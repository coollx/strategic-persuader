# Paper Analysis: Dialogue Act-based Breakdown Detection in Negotiation Dialogues

**Original title:** Dialogue Act-based Breakdown Detection in Negotiation Dialogues
**Authors:** Atsuki Yamaguchi (Hitachi R&D; work done as a master's student at the University of Sheffield), Kosui Iwasa, Katsuhide Fujita (Tokyo University of Agriculture and Technology)
**Venue/Year:** EACL 2021 (main volume), pages 745–757
**Source:** https://aclanthology.org/2021.eacl-main.63/ (PDF: https://aclanthology.org/2021.eacl-main.63.pdf)
**Code/Data:** https://github.com/gucci-j/negotiation-breakdown-detection
**Classification:** method/artifact paper (two built artifacts: the JobInterview negotiation corpus and a dialogue act-based breakdown detection pipeline); the empirical comparison across corpora is folded into Section 4 below.

## 1. Method Motivation

a) **Why proposed.** Human-human negotiation support needs a way to warn negotiators that a conversation is heading toward failure (a "breakdown": ending without agreement), so they can correct course before losing time and utility. Two ingredients were missing: a sufficiently complex negotiation corpus and a breakdown detection method designed for negotiation rather than open-domain chat.

b) **Pain points of existing resources/methods.** (1) The field reuses two corpora: DealOrNoDeal (DN) and CraigslistBargain (CB). DN handles item division with only 22.5 possible solutions per dialogue and a standard linear additive utility; CB is single-issue price negotiation. Both are easy regimes for machine learning models. Other corpora are far smaller (~100 dialogues). (2) Prior breakdown detection (Yamaguchi & Fujita, 2020) used text-based features; the authors show these collapse when breakdowns are rare. (3) The Dialogue Breakdown Detection Challenge (DBDC) targets a different task: judging whether a system utterance is valid in human-machine chat, not predicting the outcome of a human-human negotiation.

c) **Core intuition.** A negotiation that will break down has a distinct dialogue *flow* (for example, many disagreements), so classifying the sequence of dialogue acts — rather than the surface text — should detect breakdowns robustly, especially when breakdown examples are scarce.

## 2. Method Design

### 2a. The JobInterview (JI) corpus

- **Scenario.** Recruiter vs. job applicant negotiate 5 issues (Table 1): Salary ($20–$50/hour, integer), Weekly day off (2–5 days, integer), Position {Engineer, Designer, Manager, Sales}, Company {Google, Apple, Facebook, Amazon}, Workplace {Tokyo, Seoul, Beijing, Sydney}. Position and Company are *interdependent*: the value of a company option depends on which position option is chosen. This yields 9,920 possible solutions per dialogue (vs. 22.5 in DN).
- **Preferences (Section 3.2).** Two negotiators $A=\{a_1,a_2\}$ negotiate independent issues $\boldsymbol{I}$ and interdependent issue pairs $\boldsymbol{J}$. Each issue $i \in \boldsymbol{I}$ has weight $w_i^{a_k} \in [0.1,0.6]$ with $\sum_{i \in \boldsymbol{I}} w_i^{a_k} = 1$ per negotiator; each option $o^i \in \boldsymbol{O}^i$ has weight $w_{o^i}^{a_k} \in [0,1]$. For an interdependent pair $(j_{\text{from}}, j_{\text{to}}) \in \boldsymbol{J}^2$ (implemented as position→company), a bias $b_{(o^{j_{\text{to}}}, o^{j_{\text{from}}})}^{w_{o^{j_{\text{to}}}}} \in [0,0.5]$ raises the importance of option $o^{j_{\text{to}}}$ for a particular pairing. All weights and biases are drawn uniformly at random within their ranges.
- **Scoring function.** With $w'$ the min-max-normalized option weight after adding the bias, a draft agreement $s$ scores $$U_{a_k}(s) = \sum_{i \in \boldsymbol{I}} w_i^{a_k} w_{o_s^i}^{a_k} + \sum_{(j_{\text{from}},j_{\text{to}}) \in \boldsymbol{J}} \left( w_{j_{\text{from}}}^{a_k} w_{o_s^{j_{\text{from}}}}^{a_k} + w_{j_{\text{to}}}^{a_k} w'^{a_k}_{o_s^{j_{\text{to}}}} \right),$$ a linear additive utility plus the interdependency term. The bias prevents a purely linear additive structure, which is what makes Pareto-optimal solutions hard to find.
- **Collection protocol (Section 3.3, Appendix A).** Amazon Mechanical Turk, US-based workers with ≥1,000 approved tasks and ≥95% approval. A web interface shows each worker their own preferences and live-scores any candidate solution. Each worker may propose a draft agreement at most 3 times per dialogue and must send at least 6 messages before proposing. Pay: $0.20 per dialogue plus a bonus of $(score − 5)/5 dollars when the score exceeds 5/10, promoting efficient negotiation. If both sides exhaust their proposals without acceptance, the negotiation is a breakdown and both scores are zero.
- **Statistics (Table 3).** JI: 2,639 dialogues, 12.7 average turns per dialogue (highest of the three corpora), 6.12 average words per turn (lowest), vocabulary 4,476, 92.9% agreement ratio (vs. DN 76.2%, CB 74.9%), 13.4% of solutions Pareto optimal (vs. DN 75.0%), 0.98% of all bids Pareto optimal (vs. DN 18.0%), average score 6.4/10. The high agreement ratio plus rare Pareto-optimal bids indicate participants agreed often but struggled to find *good* agreements — the intended hard regime.

### 2b. Breakdown detection task (Section 4)

Given a dialogue $D$ of $n$ turn utterances $\{s_1, \dots, s_n\}$, label $D$ as success (0, agreement reached) or breakdown (1). Labels per corpus: DN — a `<disagree>` or `<no_agreement>` tag in the output; CB — no offer price; JI — status not "completed". Dialogues under 3 turns are removed (they rarely contain bargaining), leaving breakdown ratios of 23.8% (DN), 18.9% (CB), and 4.9% (JI). Metrics: area under the receiver operating characteristic curve (ROC-AUC), average precision (AP; chosen for label imbalance), and normalized confusion matrices.

### 2c. Dialogue act extraction (Section 5.1)

Two stages, building on He et al. (2018):

1. **Pattern matching.** Regular expressions (Table 4) map each turn to acts among `<greet>`, `<agree>`, `<disagree>`, `<inquire>`, `<propose>`, `<inform>`; unmatched turns get `<unk>`. Corpus-specific rules: in JI, rejecting an intermediate offer is `<disagree>` and submitting a new intermediate offer is `<propose>`; `<inform>` is assigned when a previous turn ends with `<inquire>` and the reply contains no other tag. If several patterns match, only the first is kept.
2. **Filtering and alignment.** Unlike He et al. (one act per turn), turns may carry multiple acts. Extracted acts are filtered to conform to a constrained per-turn flow (Figure 1) — for example `<greet>` first, `<agree>`/`<disagree>` optionally followed by `<propose>` then `<inquire>` — motivated by the alternating-offers protocol of Rubinstein (1982): one side proposes, the other accepts or counter-offers. An act violating the allowed order causes all remaining unmatched acts of the turn to be discarded, reducing regex noise.

### 2d. Classifiers (Sections 5.2, 6.1)

- **Input encoding.** Turns are concatenated with a `<sep>` tag heading each turn and `<end>` closing the dialogue. Linear models get a count vector over acts (including `<unk>`, `<sep>`, `<end>`). Sequence models get one-hot vectors $\boldsymbol{e} \in \mathbb{R}^{1 \times 10}$ (10 = 7 acts + `<sep>` + `<end>` + `<pad>`) stacked into $\boldsymbol{E} \in \mathbb{R}^{n \times 10}$.
- **Models.** LR-BOW: logistic regression on bag-of-words (TF-IDF for text; counts for acts). GRU and GRU-Att: gated recurrent unit (GRU) with a linear head, optionally with self-attention; text inputs use frozen 300-dimensional GloVe embeddings. BERT base/large: fine-tuned on text only, linear head on the `[CLS]` representation. Random: class-distribution baseline. Each model runs with either text features (TEXT) or dialogue-act features (TAG), where applicable.
- **Training protocol.** Stratified five-fold cross-validation; folds split 80/20 into train/validation; binary cross-entropy with Adam; hyperparameters tuned with Optuna over 100 trials per model on validation F1 (search spaces in Tables 8–9); NVIDIA V100.

## 3. Comparison with Existing Methods

a) **Fundamental differences.** Against text-based breakdown detection, the method replaces lexical evidence with a task-structured symbolic sequence (the act flow), which is invariant to vocabulary and robust to scarce positive labels. Against the DBDC line, the unit of prediction is the whole human-human negotiation's outcome, not the validity of one system utterance.

b) **Key innovations.** (1) JI corpus with interdependent issues and a non-purely-linear utility — *notable* (it changes the difficulty regime measurably: Pareto-optimal bids drop from 18.0% to 0.98%). (2) Constrained-flow dialogue act extraction — *incremental-to-notable* (a filtering layer over He et al.'s regexes, but it is what makes multi-act turns usable). (3) Evidence that act-based features dominate exactly in the low-breakdown-ratio regime — *notable* as an empirical claim.

c) **Applicability.** Excels when breakdowns are rare and text is short/domain-narrow (JI). Struggles when euphemism carries the signal (regex extraction misses it) and offers no advantage when breakdowns are frequent and text-rich (DN, CB, where BERT is comparable or better).

d) **Comparison table.**

| | Advantages | Disadvantages | Potential improvements |
|---|---|---|---|
| Dialogue-act + GRU (this paper) | Robust at 4.9% breakdown ratio (AP .418 vs. .172 for BERT); tiny input space; interpretable acts | Rule-based extraction misses euphemism; needs a sequence model (count vectors fail completely) | Learned act taggers; act + text hybrid inputs |
| Text + BERT | Best or tied on DN/CB (AP .779/.756); rich contextual signal | Near-random AP on JI (.172 at 4.9% breakdowns); costly | Class-imbalance training; domain pretraining |
| Text + LR-BOW | Cheap; surprisingly decent AP on DN | More false positives on successful dialogues; no sequence information | n-gram tuning only goes so far |

## 4. Experimental Validation

a) **Design.** Three corpora (DN, CB, JI), nine models, five-fold stratified cross-validation, ROC-AUC/AP/confusion matrices (Table 5).

b) **Key results.**
- JI (4.9% breakdowns): GRU-Att with act features reaches ROC-AUC .915 and GRU with act features AP .418, versus text-based GRU AP .093, BERT base AP .172, random .053. Act-based features win by roughly 2.4× AP over the best text model.
- CB: GRU-Att act features tie BERT base on ROC-AUC (.920); BERT base leads AP (.756 vs .737).
- DN: BERT large best ROC-AUC (.851); act-based GRU is within the 95% confidence interval of the best AP.
- LR-BOW on act counts is exactly random (ROC-AUC .500) on all three corpora: dialogue acts carry signal only as an ordered sequence.

c) **Where it shines.** The proposed corpus: text-based GRU models detect almost no breakdowns on JI (true-positive ratio .063–.078), BERT manages ~.19–.198, while act-based GRU models reach .354–.418 true-positive ratio with true-negative ratio ≥ .95.

d) **Ablations and error analysis (Sections 7.2–7.3).** Replacing one act type with `<unk>` shows `<agree>` matters most across all corpora despite being infrequent; `<propose>` also matters; `<greet>`/`<inform>` matter least. Swapping `<agree>` → `<disagree>` raises predicted breakdowns (true positives up, true negatives sharply down) and the reverse swap suppresses them, confirming the model uses the semantics of the two acts. Error analysis: false positives triggered by a `<disagree>` matching a harmless "not"; false negatives on euphemistic refusals the regexes cannot see. Acknowledged limitation: rule-based extraction; the authors call for act-annotated negotiation corpora.

## 5. Reproduction & Application

a) **Open source.** Yes: https://github.com/gucci-j/negotiation-breakdown-detection contains the JI dataset, the negotiation interface description, and the detection code (see `repo_analysis.md` next to this file). Key steps: extract acts with the Table 4 regexes + Figure 1 filter, encode as one-hot sequences, train a small GRU with the Table 8 search space.
- b) **Implementation details.** Stratified folds are essential at a 4.9% positive rate; class-weighted logistic regression; Optuna 100-trial search on validation F1; GRU hidden units 64–256, 1–4 layers; frozen GloVe for text baselines.
- c) **Transferability to this project.** Three pieces transfer independently of the classifiers: (1) the JI corpus is a ready-made multi-issue negotiation environment with per-side utility functions — usable to instantiate persuasion/negotiation games between LLM agents with exact scores; (2) the scoring function (linear additive + interdependency bias) is a drop-in utility model for reward design in a negotiation setting; (3) the act extraction + breakdown detector is a cheap monitor that could flag failing dialogues produced by an LLM persuader during rollout or analysis (though the regexes are tuned to these three corpora's phrasing and would need adaptation for LLM-generated text).

## 6. Summary

a) **One-sentence core idea.** Classify the sequence of rule-extracted dialogue acts with a small GRU to predict negotiation breakdown, and provide a harder negotiation corpus where this is the only approach that works.

b) **Quick-reference pipeline.**
1. Collect human-human multi-issue negotiations with randomized private preferences and a live utility display (JI corpus, 2,639 dialogues, 92.9% ending in agreement).
2. Map each turn to symbolic acts (greet/agree/disagree/inquire/propose/inform) with regular expressions, then keep only act sequences consistent with an alternating-offers ordering.
3. Feed the act sequence to a GRU classifier that outputs breakdown vs. success for the whole dialogue.
4. Result: on a corpus where only 4.9% of dialogues break down, this beats every text-based model by a wide margin (average precision .418 vs. .172 for the best BERT); on older corpora it merely matches them.

## 7. Reviewer Reception

No public OpenReview page found (searched OpenReview for the exact title and EACL 2021 on 2026-08-11; EACL 2021 did not run public reviews on OpenReview).
