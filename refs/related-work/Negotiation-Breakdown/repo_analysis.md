# Repo Analysis: negotiation-breakdown-detection

**Path:** `refs/related-work/Negotiation-Breakdown/repo/` · **Source:** https://github.com/gucci-j/negotiation-breakdown-detection · **Analyzed:** 2026-08-11 (commit `d4c2bf6`)

## Overview

- **Paper:** Dialogue Act-based Breakdown Detection in Negotiation Dialogues (Yamaguchi, Iwasa, Fujita; EACL 2021, pages 745-757) — see `paper_analysis.md`
- **Framework:** PyTorch 1.5 + torchtext 0.5 + Optuna 1.5 for the classifier; a separate React + Node/socket.io + PostgreSQL web application for data collection. No training framework (no veRL, TRL, OpenRLHF, Hugging Face Transformers).
- **RL Algorithm:** none. This is a supervised binary-classification repo plus a crowdsourcing platform. A grep across all Python, JavaScript, JSON and shell files for `ppo|grpo|reinforce|policy_gradient|advantage|rollout|openai|gpt-|llm|deepspeed|fsdp|accelerate|distributed` returns zero hits.
- **Base Model:** none pretrained. The only trained model is a small recurrent network (GRU or LSTM, optional self-attention, 64-256 hidden units) over one-hot dialogue-act symbols — `implementation/src/model/core/model.py:57-143`.
- **Key Innovation:** represent a whole negotiation dialogue as a short sequence of six dialogue acts extracted by regular expressions and constrained by a hand-written act-flow automaton, then classify that symbol sequence as agreement or breakdown. The text itself is discarded before the model sees anything.

The repository is three loosely coupled parts: (1) the JobInterview dataset, shipped inside the repo as `data.zip`; (2) `implementation/`, the preprocessing and classifier code for three corpora; (3) `interface/`, the negotiation web application and Amazon Mechanical Turk scripts that produced the dataset. The repository has three commits total, the last from 2021-10-21, and is MIT licensed (`LICENSE:1-3`).

## File Structure Map

```
repo/
├── README.md                       dataset unzip instructions, helper usage, citation
├── LICENSE                         MIT, (c) 2021 Yamaguchi, Iwasa, Fujita
├── data.zip                        4.3 MB → data.json (40.7 MB): THE JOBINTERVIEW DATASET, git-tracked
├── helper/
│   └── negotiation_ji.py           standalone dataset reader: Bid/Issue/User/Comment/Negotiation
│                                   classes, calc_score utility function, Pareto frontier
├── implementation/
│   ├── README.md                   preprocess → extract acts → train, command lines
│   ├── requirements.txt            torch 1.5.1, torchtext 0.5, optuna 1.5.0, sklearn 0.23.1, ...
│   ├── params/{ji,dn,cb}_tuning.json   run configs (all three files are byte-identical)
│   └── src/
│       ├── preprocess/
│       │   ├── load_negotiation_ji.py   JobInterview JSON → csv(raw_text, role_text, flag)
│       │   ├── load_negotiation_dn.py   DealOrNoDeal text  → csv(raw_text, text, flag)
│       │   ├── load_negotiation_cb.py   CraigslistBargain  → csv(raw_text, text, flag)
│       │   ├── create_meta_ji.py        csv → dialogue-act sequence csv (JobInterview variant)
│       │   ├── create_meta_dn.py        same, DealOrNoDeal regex set
│       │   ├── create_meta_cb.py        same, CraigslistBargain regex set
│       │   └── util/
│       │       ├── negotiation_ji.py    near-duplicate of helper/negotiation_ji.py
│       │       └── negotiation_cb.py    CraigslistBargain reader (User/Comment/Bid/Negotiation)
│       └── model/
│           ├── run.py                   single entry point: train | train_with_tuning | random
│           └── core/
│               ├── dataprocessor.py     stratified 5-fold, one-hot batching, attention mask
│               ├── model.py             GRU/LSTM ± scaled dot-product self-attention → 1 logit
│               ├── metrics.py           F1/Fbeta/precision/recall, ROC, PR, confusion matrix
│               ├── earlystopping.py     patience on validation F1, checkpoints best weights
│               └── resultlogger.py      per-fold csv of (label, prediction, probability, text)
└── interface/
    ├── README.md                   postgres + server + client setup
    ├── common/messageTypes.js      joinLobby, joinRoom, sendMessage, sendSolution,
    │                               acceptSolution, terminateNegotiation
    ├── server/                     Node + socket.io + Sequelize/PostgreSQL
    │   ├── issues.js               the five issues and their option sets
    │   ├── controllers/
    │   │   ├── onJoinLobby.js      pairs two workers, assigns roles, GENERATES PREFERENCES
    │   │   ├── onReceiveMessage.js persists an utterance, broadcasts it
    │   │   ├── onReceiveSolution.js persists an offer, broadcasts it
    │   │   ├── onAcceptSolution.js room → 'completed', scores both users
    │   │   ├── onTerminateNegotiation.js room → 'terminated'  (the breakdown label)
    │   │   └── onDisconnect.js     room → 'aborted'           (discarded, not a breakdown)
    │   ├── models/{room,user,comment,solution}.js  the four tables mirrored by data.json
    │   └── migrations/             table schemas
    ├── client/                     React + Redux + antd negotiation UI
    │   ├── src/utils/calcScore.js  the utility function, client copy
    │   ├── src/utils/calcReward.js crowd payment: $0.2 + up to $1.0 bonus
    │   ├── src/utils/constant.js   minMessages: 6
    │   └── src/pages/Talking/Talking.js  chat + offer panel, 3-proposal budget, terminate button
    ├── mturk/                      boto3 HIT creation, assignment approval, question.xml
    └── analytics/main.py           reads the live PostgreSQL, renders an HTML review page
```

There are no shell scripts anywhere; `*.sh` is in `.gitignore:32`.

## Component Inventory

### 1. Entry Points

| Script | Invocation | Produces |
|---|---|---|
| `implementation/src/preprocess/load_negotiation_ji.py:181-184` | `python load_negotiation_ji.py <data.json> <out.csv>` | csv with columns `raw_text, role_text, flag` |
| `implementation/src/preprocess/create_meta_ji.py:246-250` | `python create_meta_ji.py <in.csv> <out.csv>` | csv with columns `text, meta_text, flag` |
| `implementation/src/model/run.py:652-653` | `python src/run.py params/ji_tuning.json` | logs, per-fold csv, ROC/PR/confusion-matrix PNGs, checkpoints |
| `helper/negotiation_ji.py:255-260` | `python negotiation_ji.py` (path hardcoded to `./data/data.json`) | prints a count |
| `interface/analytics/main.py:116-132` | `python main.py` | `main.html` review page from the live database |
| `interface/mturk/create_hit.py:29-64` | `python create_hit.py` | posts a Mechanical Turk HIT |

`run.py` takes exactly one argument, a JSON config path; running it with no arguments prints the full option list (`run.py:24-52`). The three shipped configs are byte-identical and contain placeholder paths, so `hidden_dim`, `num_layers`, `lr`, `recurrent_dropout`, `dense_dropout` and `bidirectional` are absent — they are supplied by Optuna in `train_with_tuning` mode, which is the mode all three configs select.

Verification: I unzipped `data.zip` and ran both JobInterview preprocessing stages end to end. `load_negotiation_ji.py` printed `There are 2639 completed negotiations.` / `excluding: 62, breakdowns: 127`, exactly matching the counts recorded as comments at `load_negotiation_ji.py:178-179`, and `create_meta_ji.py` then produced a 2577-row act-sequence csv. The pipeline runs unmodified on Python 3 with modern pandas.

### 2. The JobInterview dataset (`data.zip` → `data.json`)

The dataset is in the repository, git-tracked, at `repo/data.zip` (4.3 MB compressed, 40.7 MB as `data.json`). No download step is needed. `README.md:7-8` states it plainly: "Please unzip `data.zip`. The unzipped data: `data.json` is our proposed JI dataset."

`data.json` is a single JSON array of 3,935 negotiation records. Status counts: 2,500 `completed`, 1,296 `aborted`, 139 `terminated`. The breakdown-detection subset drops `aborted` (a disconnect, not a decision) and is therefore $2500 + 139 = 2639$ dialogues — the paper's headline number, produced by the filter at `helper/negotiation_ji.py:244-246`. The alternative filter for "ordinary tasks" keeps only completed dialogues with an accepted offer, yielding 2,452.

Corpus scale over the 2,639-dialogue subset: 33,181 utterances, 208,124 words, 12.6 utterances per dialogue on average (median 11, max 96), 6.3 words per utterance, 2.52 offers per dialogue (max 6, consistent with a 3-offer budget per side). 1,501 distinct Mechanical Turk worker identifiers appear. Every record has exactly two users, one `worker` and one `recruiter`.

**Exact schema.** Every record has five keys, `id`, `status`, `users`, `comments`, `solutions`, mirroring the four database tables in `interface/server/models/`. A trimmed real record:

```json
{
  "id": "886d9792-1d6e-11e9-982d-6c96cfdf0295",
  "status": "completed",
  "users": [
    {
      "id": "886b88c6-1d6e-11e9-903e-6c96cfdf0295",
      "assignment_id": "3M23Y66PO4OJF27PGT26HQ81BHB6SB",
      "worker_id": "ARKB541D42E8W",
      "role": "worker",
      "utilities": [
        {"name": "Salary", "type": "INTEGER", "min": 20, "max": 50, "weight": 0.2556392516896328},
        {"name": "Position", "type": "DISCRETE", "relatedTo": "Company",
         "options": [{"names": {"Position": "Engineer", "Company": "Google"}, "weight": 0},
                     {"names": {"Position": "Manager", "Company": "Apple"}, "weight": 0.9194294698682506}],
         "weight": 0.23846211486659388},
        {"name": "Weekly holiday", "type": "INTEGER", "min": 2, "max": 6, "weight": 0.14387722975857642},
        {"name": "Workplace", "type": "DISCRETE",
         "options": [{"name": "Tokyo", "weight": 0}, {"name": "Seoul", "weight": 1}],
         "weight": 0.18692161128810048},
        {"name": "Company", "type": "DISCRETE",
         "options": [{"name": "Google", "weight": 0.974752936023897}],
         "weight": 0.17509979239709647}
      ]
    }
  ],
  "comments": [
    {"id": "886d564c-1d6e-11e9-ad20-6c96cfdf0295", "body": "Hello",
     "user_id": "886cd8a2-1d6e-11e9-8655-6c96cfdf0295",
     "created_at": "2018-11-07 02:11:05.909000+09:00"}
  ],
  "solutions": [
    {"id": "886d9670-1d6e-11e9-abf6-6c96cfdf0295",
     "body": {"Salary": 35, "Position": "Designer", "Weekly holiday": 4,
              "Workplace": "Tokyo", "Company": "Apple"},
     "accepted": true,
     "user_id": "886b88c6-1d6e-11e9-903e-6c96cfdf0295",
     "created_at": "2018-11-07 02:12:04.243000+09:00"}
  ]
}
```

Notes on the schema that matter for reuse:

- `status` carries the outcome label directly: `completed` (someone clicked accept, `onAcceptSolution.js:55-58`), `terminated` (someone clicked terminate, `onTerminateNegotiation.js:10-13` — this is the breakdown class), `aborted` (a socket disconnected mid-negotiation, `onDisconnect.js:26-29`).
- `comments` is the flat chat transcript in timestamp order; `solutions` is the separate offer channel. Utterances and offers are interleaved only by comparing `created_at`, which is what `load_negotiation_ji.py:83` does via `get_keys_from_value`.
- `accepted` is on the offer, not the dialogue. 2,468 records have exactly one accepted offer, 1,457 have none, and 10 have two (a race in the accept handler, since `onAcceptSolution.js` does not guard against a second acceptance). `Negotiation.__init__` (`helper/negotiation_ji.py:140-144`) silently keeps the last one.
- Per-user issue weights always sum to exactly 1.0. This is by construction: `onJoinLobby.js:55-57` draws five uniform randoms and normalizes them, then `onJoinLobby.js:63` maps each to `0.1 + w * 0.5`, so the five weights sum to $5 \times 0.1 + 0.5 = 1$.
- Option weights inside a discrete issue are min-max normalized to span $[0, 1]$ (`onJoinLobby.js:65-68`), so every issue's best option scores 1 and its worst scores 0 for that negotiator.
- The dependent issue (Position, `relatedTo: Company`) is stored in the released JSON as a flat list of 16 entries keyed by a `names` object naming both issues, whereas the live server stored a nested `biasedWeights` map keyed by the related option (`onJoinLobby.js:87-100`). The release was reshaped; code written against the server representation (including `interface/analytics/main.py:94-99`) will not read `data.json` directly, but `helper/negotiation_ji.py:86-93` handles the released shape.

### 3. Utility / scoring function

Implemented four times in the repo, identically in substance: `helper/negotiation_ji.py:63-102` (and its duplicate `implementation/src/preprocess/util/negotiation_ji.py:63-102`), `interface/server/controllers/onAcceptSolution.js:7-36`, `interface/client/src/utils/calcScore.js:1-43`, and `interface/analytics/main.py:78-109`. The Python version that reads the released JSON:

```python
    def calc_score(self, bid: Bid):
        """Calculate a score that the user can earn by the bid."""

        score = 0
        user_role = self.context["role"]
        user_utilities = self.context["utilities"]
        for issue_name, option in bid.options.items():
            issue_utility = None
            for u in user_utilities:
                if u["name"] == issue_name:
                    issue_utility = u
            issue_weight = issue_utility["weight"]

            if issue_utility["type"] == "INTEGER":
                option_max = issue_utility["max"]
                option_min = issue_utility["min"]
                if user_role == "recruiter":
                    score += issue_weight * (option_max - option) / (option_max - option_min)
                elif user_role == "worker":
                    score += issue_weight * (option - option_min) / (option_max - option_min)
```

In words: the score of an offer is $\sum_i w_i \, v_i(o_i)$ over the five issues, where $\sum_i w_i = 1$ and each $v_i \in [0,1]$, so a score is always in $[0,1]$. For the two integer issues the value is a linear ramp, rising in the option value for the worker and falling for the recruiter — the roles are exactly opposed on salary and days off. For an independent discrete issue the value is the stored option weight. For the dependent issue, the value is looked up jointly on (Position, Company), which is where the interdependence lives: `helper/negotiation_ji.py:86-93` matches the option entry whose `names` agree with the offer on *both* issues.

The "bias" the paper mentions is in the generator, not the scorer: `onJoinLobby.js:91-96` draws a base weight per Position and then adds a per-Company bias in $[0, 0.5)$ before renormalizing, so Position preferences are correlated across Companies rather than independent.

`Negotiation.get_all_bids` (`helper/negotiation_ji.py:146-163`) enumerates the joint offer space and `get_pareto_bids` (`:165-189`) sweeps the Pareto frontier. Measured on the shipped data: the offer space is 9,920 offers ($31 \times 4 \times 5 \times 4 \times 4$) and the frontier for the first negotiation holds 69 offers, computed in 0.2 s. Over the first 1,000 completed negotiations the accepted offer gives each side 0.673 on average, a joint score of 1.35 out of a possible 2, with individual scores spanning $[0.122, 1.000]$.

One inert inconsistency: `IntegerIssue.__init__` builds `self.options` as `range(option_min, option_max + 1)` while the constructor is called with `IntegerIssue("Salary", 20, 51)` (`helper/negotiation_ji.py:209, 211`), one past the data's `max` of 50. `get_all_bids` ignores `IntegerIssue.options` and uses `range(option_min, option_max)` (`:158`), which lands on the correct 20-50 and 2-6 ranges, so nothing downstream is affected — but anyone reading `IntegerIssue.options` directly gets an off-by-one.

The crowd payment function is separate from the utility: `calcReward.js:1` pays $0.20 flat plus a linear bonus up to $1.00 once the displayed score (the utility times 100) passes 50. `interface/analytics/main.py:40-49` uses different thresholds (0.5 / 0.7 and a $0.50 base), so the two disagree; the client version is what participants saw.

### 4. Dialogue-act extraction

Two stages. `load_negotiation_ji.py` linearizes a negotiation into a role-tagged string, and `create_meta_ji.py` converts that string into a dialogue-act sequence.

**Stage 1 — linearization** (`load_negotiation_ji.py:41-179`). Dialogues under 3 utterances are dropped (62 of 2,639). Utterances are lowercased, newlines stripped, and punctuation `,.!?` split off as separate tokens (`:8-22`). Offers are spliced into the utterance stream by timestamp (`:83`); each offer emits `<propose>` for the offerer and then `<agree>` or `<disagree>` for the counterparty depending on `bid.accepted` (`:93-108`). Two parallel strings are built: `raw_text`, where every turn boundary is `<sep>`, and `role_text`, where it is `WORKER:` or `RECRUITER:`. The label is `flag = 1` when `status != "completed"` or no offer was accepted (`:165-167`). A comment at `:78-81` flags a deliberate divergence from the paper:

```
            # -> We include *all* intermediate bid information (agree, disagree,
            #    and propose) for breakdown detection to alert negotiators
            #    when a negotiation has a possibility to be broken down before being finalised.
            #    This is different from the Table description in the paper in that
            #    we also consider agree bids.
```

The `ratio` argument (`:41`, called with `1.0` at `:184`) truncates each dialogue to a leading fraction of its turns — the early-detection ablation, exposed only by editing the call site.

**Stage 2 — acts** (`create_meta_ji.py`). The tag alphabet is nine symbols (`create_meta_ji.py:8`):

```python
nego_units = ('<sep>', '<end>', '<greet>', '<agree>', '<disagree>', '<inquire>', '<propose>', '<inform>', '<unk>')
```

`divide_utterance` (`:186-227`) merges consecutive same-speaker turns into one block and hands each block to `convert_to_units`. Six regular expressions do the matching (`:42-54`), verbatim:

```python
    # greet
    greet_result = re.search(r'\bhi\b|\bhello\b|\bhey\b|\bhiya\b|\bhowdy\b| (how are you) | (good day) | (good afternoon) | (good morning) |\byo\b', text)
    # disagree
    disagree_result = re.search(r"\bisn\'t\b|\bworse\b|\bbad\b|\bsorry\b|\bno\b|\bnot\b|\bnothing\b|\bdon\'t\b|\bcan`t\b|\bcan\'t\b|\bcant\b|\bcannot\b|\bafraid\b| (a lot lower) | (too much) | (too high) | (too low)", text)
    # agree
    agree_result = re.search(r"\bok\b| (no problem) |\bokay\b|\byes\b|\bgreat\b|\bperfect\b|\bthanks\b|\bgracias\b|\bthx\b| (thank you) |\bpleasure\b|\bfine\b|\bdeal\b|\bcool\b| (sounds good) | (very good) | (looks good) | (that works) | (that will work) | (it will work) | (i can do)", text)
    # inquire
    inquire_result = re.search(r"\bwhat\b|\bwhen\b|\bwhich\b|\bwhere\b| (how about) |\b(how\'s)\b| (how does) |\?| (do you) | (did you) | (did we) | (do we) | (do i) | (are you) | (would you) | (will you) | (could we) |  (could you) | (let me know) ", text)
    # propose
    propose_result = re.search(r"(\$\d{1,}) | (\d{1,}\$) | \d{1,} | (come down) | (highest) | (lowest) | (go higher) | (go lower) | (i would like)", text)
```

Each match contributes `(start_index, tag)` to a queue, which is sorted by position (`:93-96`) so acts appear in the order the cues appear in the text. Disagree dominates agree: they are an if/elif chain (`:73-78`), so an utterance containing both cues is disagreement. Explicit `<agree>`/`<disagree>`/`<propose>` markers injected by stage 1 outrank the lexical cues (`:66-72, 88-90`), and a block that is nothing but such a marker short-circuits immediately (`:24-36`).

The sorted queue is then filtered by a small automaton (`:119-183`) that encodes the alternating-offers flow: an utterance may open with any act; after `<greet>` only `<inquire>` or `<propose>` may follow; after `<agree>`/`<disagree>` only `<inquire>` or `<propose>`; after `<propose>` only `<inquire>`; `<inquire>` always terminates the utterance. Anything else breaks the loop and is dropped. Two fallbacks fire when nothing matched: `<inform>` if the previous utterance ended in `<inquire>` (`:110-112`), otherwise `<unk>` (`:115-117`). So `<inform>` is defined operationally as "an unclassifiable reply to a question".

Two design facts worth knowing before reuse. First, speaker identity is **not** encoded: both roles emit `nego_units[0]` = `<sep>` as the utterance boundary, with the role-specific alternative commented out at `:103-107`. The model sees turn boundaries but not who spoke. Second, `create_meta_ji.py` prints every intermediate match to stdout (`:56, 94, 96, 203, 221, 224, 240`) — my run over 2,577 dialogues produced a 6.7 MB, 160k-line log. Redirect it.

Measured act frequencies over the full JobInterview corpus after both stages (2,577 dialogues, mean sequence length 34.5 symbols, median 27, max 366):

| Act | Count |
|---|---|
| `<sep>` | 39,552 |
| `<propose>` | 13,261 |
| `<disagree>` | 11,706 |
| `<inquire>` | 8,061 |
| `<agree>` | 5,840 |
| `<greet>` | 3,984 |
| `<end>` | 2,577 |
| `<unk>` | 2,240 |
| `<inform>` | 1,607 |

A real extracted sequence for a breakdown dialogue: `['<sep>', '<greet>', '<sep>', '<greet>', '<sep>', '<inquire>', '<sep>', '<agree>', '<inquire>', '<sep>', '<inquire>', '<sep>', '<disagree>', '<sep>', '<disagree>', '<inquire>', '<sep>', '<disagree>', ...]`.

The DealOrNoDeal and CraigslistBargain variants (`create_meta_dn.py`, `create_meta_cb.py`) are copies of the same file. Diffing them against `create_meta_ji.py` shows the only substantive changes are the speaker tag names (`YOU:`/`THEM:`, `BUYER:`/`SELLER:`), the DealOrNoDeal propose regex gaining item words (`\bball\b|\bhat\b|\bbook\b|...`), and the removal of the injected-marker handling that JobInterview needs because only JobInterview has a structured offer channel. Note that both non-JobInterview copies contain the typo `(go hiher)` where the JobInterview file has `(go higher)`.

### 5. Breakdown-detection classifier

The task is binary classification of a whole dialogue: `flag = 1` means breakdown. The class is rare — 127 positives in 2,577 JobInterview rows, 4.9 %.

**Input.** `batch_input_make` (`dataprocessor.py:9-15`) turns the act-index sequence into one-hot vectors of width `input_dim = 10`, which the comment at `:10` explains as "10 tags -> 9 tags + 1 for padding". There is no embedding layer and no pretrained vector; the model consumes one-hot act symbols directly.

**Model** (`model.py:57-143`). A GRU or LSTM (`rnn_type` from config), 1-4 layers, 64-256 hidden units, optionally bidirectional, with recurrent and dense dropout. The final hidden state is dropped out and, when `attention: true`, used as the query of a single scaled dot-product self-attention over all timesteps (`model.py:8-54`). A single linear layer maps to one logit; the loss is `nn.BCEWithLogitsLoss` (`run.py:130, 245, 372`) with `optim.Adam`.

**Cross-validation** (`dataprocessor.py:91-128`). `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)` over the whole dataset; within each training fold, a further stratified 80/20 train/validation split with the same seed. `seed: 1234` in all three shipped configs.

**Hyperparameter search** (`run.py:262-263, 327-347`). One Optuna study per fold, 100 trials each, maximizing validation F1:

```python
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    bidirectional = trial.suggest_int('bidirectional', 0, 1)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    recurrent_dropout_rate = trial.suggest_uniform('recurrent_dropout_rate', 0.0, 1.0)
    dense_dropout_rate = trial.suggest_uniform('dense_dropout_rate', 0.0, 1.0)
```

Early stopping watches validation F1 with `patience: 8` and checkpoints the best weights (`earlystopping.py:52-86`); a maximum of 100 epochs.

**Metrics** (`metrics.py`). Accuracy, F1, F-beta with $\beta = 2$, precision, recall at a 0.5 threshold (`:14-51`); ROC curves and ROC-AUC per fold with an interpolated mean curve (`:92-158`); precision-recall curves with average precision as the reported area (`:161-215`, note `:174, 179` where trapezoidal PR-AUC is commented out in favor of average precision); row-normalized confusion matrices (`:54-89`). Per-fold results go to a `result.csv` and are averaged with standard deviations (`run.py:190-211`). `ResultLogger` (`resultlogger.py:40-47`) additionally dumps every test dialogue with its label, prediction, probability, text and act sequence — useful for error analysis.

**Baseline.** `mode: "random"` (`run.py:528-638`) fits an sklearn `DummyClassifier` with a configurable `strategy` (`prior`, `most_frequent`, `stratified`) on constant features, and pushes it through the same metric machinery.

**What is missing relative to the paper.** The logistic-regression bag-of-words, GRU-over-text and BERT classifiers described in the paper are not in this repository. A grep for `bert|transformers|LogisticRegression|CountVectorizer|tfidf` across all Python files returns only `sklearn.dummy`, `sklearn.metrics` and `sklearn.model_selection`. `run.py` always feeds `batch.meta_text` to the model (`:429, 467, 502`) and never `batch.text`; the GloVe vectors loaded at `:146-147` are used only so `ResultLogger` can decode token indices back to words. Reproducing the paper's text-based rows requires writing those models.

**Two bugs to be aware of if adapting this code.** In `run.py:438-440` the gradient-norm clip runs *after* `optimizer.step()`, so it never affects an update:

```python
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```

And in `model.py:43` the attention mask is computed and discarded, because `masked_fill` is the out-of-place form and its return value is dropped:

```python
            attention_weight.masked_fill((1 - mask).bool(), float('-inf'))
```

So the attention variant attends over padding. Neither bug is fatal to the reported results, but both mean the released code does not do what its structure suggests.

### 6. Data-collection environment (`interface/`)

The negotiation protocol, as implemented, is a useful specification even independent of the code. Two participants are paired in a lobby, roles `recruiter` and `worker` are shuffled, and each is given an independently drawn private preference profile (`onJoinLobby.js:47-104`). The five issues and their options are fixed (`server/issues.js:1-69`): Salary 20-50, Weekly holiday 2-6, Workplace ∈ {Tokyo, Seoul, Beijing, Sydney}, Company ∈ {Google, Amazon, Facebook, Apple}, Position ∈ {Engineer, Manager, Designer, Sales} with Position dependent on Company.

Interaction is free-form chat plus a structured offer channel; the six socket message types are enumerated in `common/messageTypes.js:1-8`. Constraints enforced by the client (`Talking.js:245-310`, `utils/constant.js:1-4`): at least 6 total messages before an offer may be sent, and at most 3 offers per participant. Accepting the counterparty's offer ends the negotiation as `completed` and reveals both scores (`onAcceptSolution.js:55-73`); pressing terminate ends it as `terminated`. Participants see their own issue weights as importance badges and a live score for the offer they are composing (`Talking.js:135-145`), but never the counterparty's preferences.

### 7. Components that are absent

The protocol's RL-oriented components do not exist in this repository and are noted here rather than forced into the template: **special action tokens intercepting generation** (the `<propose>`/`<agree>` markers are inserted by an offline preprocessor, never emitted by a model), **environment/tool interaction** (no model ever calls anything; the socket server is a human-to-human relay), **token masking for loss** (the only mask is the attention mask, and it is inert — see above), **reward function** (`calcReward.js` is a crowd payment schedule, not a training signal), **RL training loop**, **SFT / cold start**, and **multi-GPU infrastructure** (`run.py:138-142` selects a single device by index; no DDP, FSDP or DeepSpeed).

## Relevance to this project

This project trains language-model agents to persuade an interlocutor over multi-turn dialogue with reinforcement learning. This repository contributes no method we would adopt, but it contributes a dataset, a utility model, and a cheap dialogue-level signal that all transfer.

### Components We Can Directly Reuse

- **The JobInterview corpus itself** — `refs/related-work/Negotiation-Breakdown/repo/data.zip` → `data.json`, MIT licensed, no download gate. 2,639 usable human-human negotiations (2,500 agreements, 139 breakdowns) with full transcripts, timestamped offers, per-negotiator private preference weights and outcome labels. Two immediate uses: as a corpus of human persuasion attempts to measure our agents against, and as a source of 2,639 concrete preference profiles for instantiating negotiation episodes without inventing our own distribution.
- **The offer-space and preference generator** — `interface/server/controllers/onJoinLobby.js:47-104` is a self-contained recipe for drawing a fresh private preference profile over the five issues, including the interdependent Position-Company structure. Fifty lines of JavaScript, straightforward to port, and it gives an unbounded supply of episodes rather than the 2,639 recorded ones.
- **The utility function** — `helper/negotiation_ji.py:63-102` reads the released JSON directly and returns a score in $[0,1]$ for any offer under any negotiator's profile. As a shaped reward or an evaluation metric for a negotiating agent this is ready to use, and it is already validated against the shipped data (I ran it: accepted offers score 0.673 per side on average, joint 1.35 of a possible 2). The dependent-issue branch is the part worth keeping — it is what makes the offer space non-separable, so an agent cannot optimize issue by issue.
- **The Pareto frontier computation** — `helper/negotiation_ji.py:165-189`. Over a 9,920-offer space it returns a 69-offer frontier in 0.2 s, giving a principled efficiency denominator: how close to Pareto-optimal did our agent's agreement land, and did it capture more or less than half the joint surplus.
- **The dialogue-act extractor** — `implementation/src/preprocess/create_meta_ji.py:42-183`. Six regular expressions and a 60-line automaton, no model, no dependencies beyond `re`. Applied to dialogues our agents generate it produces a compact symbol stream in which the run-up to a failed negotiation is visible (in the corpus, `<disagree>` at 11,706 occurrences is the second most common act after `<propose>`). Cheap enough to run inside a rollout loop as an auxiliary signal, and cheap enough to run over thousands of generated dialogues as an analysis tool.
- **The negotiation protocol specification** — the 6-message minimum, the 3-offer budget, the separate chat and offer channels, and the three terminal states (`completed` / `terminated` / `aborted`) are a tested design for a multi-turn negotiation environment. If we build an LLM negotiation environment, matching this protocol makes our transcripts directly comparable to the human ones.

### Components We Need to Modify

- **The dependent-issue lookup.** `helper/negotiation_ji.py:86-93` reads the released `names`-keyed format, but `onJoinLobby.js`, `calcScore.js` and `analytics/main.py` all read the server's nested `biasedWeights` format. Porting the generator and the scorer together requires picking one representation and converting.
- **The dialogue-act extractor's speaker blindness.** Both roles emit `<sep>` (`create_meta_ji.py:102-107`, with the role-specific branch commented out). For our purposes — attributing persuasive moves to the persuader — restoring a per-role boundary symbol is a two-line change and probably necessary.
- **The extractor's vocabulary.** The regexes are tuned to job-interview and haggling language; `\bdeal\b` as agreement and bare digits as proposal will misfire in a different domain. The automaton at `:119-183` is domain-independent and can stay; the lexicons at `:42-50` would need rewriting per setting.
- **The extractor's stdout.** Roughly ten `print` calls fire per utterance (`:56, 94, 96, 203, 221, 224, 240`); 2,577 dialogues produced 160k log lines. Strip them before calling this in any loop.
- **The classifier, if we want it.** The released model consumes only act sequences; adding a text-conditioned or model-based breakdown detector means writing it from scratch, and the two bugs noted above (post-step gradient clipping, inert attention mask) should be fixed rather than inherited.

### Components That Don't Apply

- The entire `interface/` web stack as software — React, Redux, socket.io, Sequelize, PostgreSQL and the Mechanical Turk scripts exist to collect data from human pairs. We would read it as a specification, not run it.
- The DealOrNoDeal and CraigslistBargain loaders (`load_negotiation_dn.py`, `load_negotiation_cb.py` and their `create_meta_*` siblings) require datasets not shipped here and cover corpora we are not currently targeting.
- The GRU/attention classifier and its Optuna search — a 5-fold × 100-trial search over a 10-symbol one-hot input is not a modeling approach we would adopt for anything.
- Every RL-shaped component of the analysis protocol (action tokens, rollouts, advantage computation, loss masking, SFT, multi-GPU) is simply absent; there is nothing here to compare our training setup against.

### Key Code Snippets Worth Studying

| Location | Why |
|---|---|
| `helper/negotiation_ji.py:63-102` | the multi-issue utility function, including the interdependent-issue branch |
| `helper/negotiation_ji.py:198-252` | the full JSON reader and the two outcome filters that define the 2,639- and 2,452-dialogue subsets |
| `helper/negotiation_ji.py:146-189` | offer-space enumeration and the Pareto sweep |
| `interface/server/controllers/onJoinLobby.js:47-104` | how a private preference profile is drawn, including the Company-conditioned Position bias |
| `implementation/src/preprocess/create_meta_ji.py:42-54` | the six dialogue-act regular expressions, verbatim |
| `implementation/src/preprocess/create_meta_ji.py:119-183` | the act-flow automaton — the paper's central mechanism, in 65 lines |
| `implementation/src/preprocess/load_negotiation_ji.py:62-108` | how the offer channel is interleaved into the utterance stream by timestamp |
| `implementation/src/model/core/dataprocessor.py:91-128` | stratified 5-fold plus nested validation split, for a 4.9 %-positive label |
| `implementation/src/model/core/metrics.py:161-215` | precision-recall evaluation for a rare positive class, with average precision as the summary |
| `interface/client/src/pages/Talking/Talking.js:109-170` | the offer-budget and confirmation flow that shaped how humans in this corpus behaved |
