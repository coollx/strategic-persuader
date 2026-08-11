# Repo Analysis: GTAlign

**Path:** `refs/related-work/GTAlign/repo/` · **Source:** https://github.com/ulab-uiuc/GTAlign · **Analyzed:** 2026-08-11 (commit `1e353602`)

## Overview

- **Paper:** GTAlign: Game-Theoretic Alignment of LLM Assistants for Social Welfare (see `paper_analysis.md`)
- **Framework:** a fork of [verl](https://github.com/volcengine/verl) at version `0.5.0.dev` (`verl/version/version`). The entire verl tree is vendored (274 Python files under `verl/`, 944 tracked files total, package still named `verl` in `setup.py:78`); all method-specific code lives in `recipe/gamellm/` plus two touched verl files (`verl/workers/reward_manager/gamellm.py`, `verl/trainer/ppo/reward.py`).
- **RL Algorithm:** PPO with generalized advantage estimation and a separately trained critic — every launcher sets `algorithm.adv_estimator=gae` and a `critic.model.path`; rollout count per prompt stays at the verl default `n: 1` (`verl/trainer/config/rollout/rollout.yaml:83`), so there is no group-relative baseline. One recipe script (`recipe/genrm_remote/run_gamellm_grpo.sh`) uses GRPO with `rollout.n=8`, but it points at a data path not shipped in the repo and is not referenced by the README.
- **Base Model:** Qwen2.5-3B-Instruct for the single-dataset runs (`recipe/gamellm/math.sh:21`); the multi-dataset and out-of-domain scripts start from a supervised fine-tuning checkpoint whose code is not in the repo (`recipe/gamellm/full_zscore.sh:22`). Judge model: Qwen3-32B served by SGLang on separate GPUs, reached over an OpenAI-compatible HTTP endpoint.
- **Key Innovation:** the training reward is the geometric mean of two hand-weighted welfare terms — one for the user, one for the assistant — computed from a language-model judge score, a regular-expression format check, a payoff-matrix quality score, and two length terms; the assistant's output is forced into four tagged blocks whose `<payoff>` block is a strict JSON matrix that can be intercepted, edited, and fed back mid-generation to steer behavior at inference time.

## File Structure Map

```
repo/
├── README.md                              install, SGLang judge launch, five training scripts, three evaluation scripts
├── requirements.txt                       verl deps (transformers==4.53.0, ray, vllm path)
├── requirements_sglang.txt                sglang[all]==0.4.10.post2 for the judge server
├── data/                                  all training and evaluation data, as parquet, committed to git
│   ├── gamellm_wildbreak_math_medium_abgqa/{train,test}.parquet    10806 train rows, four sources mixed
│   │   └── by_source/{math-hard,medium,wildguard,ambig_qa}/        per-dataset splits used by the single-dataset runs
│   ├── minerva_math/test.parquet          out-of-domain math
│   ├── abg_coqa/test.parquet              out-of-domain ambiguity
│   └── advbench/test.parquet              out-of-domain safety
├── recipe/gamellm/                        THE METHOD
│   ├── math.sh, medium.sh, abgqa.sh, wildguard.sh, full_zscore.sh   PPO launchers, one per training setting
│   ├── ablation_{minerva,coqa,advbench}.sh                          out-of-domain evaluation (same trainer, val_only=True)
│   ├── reward_function.py                 dispatcher: data_source -> tasks/<name>.compute_score (has a broken import, see below)
│   ├── reward_function_abgqa_v2.py        the working dispatcher used by math.sh / abgqa.sh / wildguard.sh
│   ├── reward_function_ablation.py        dispatcher -> tasks_ablation/ (adds the Pareto-consistency judge term)
│   ├── reward_function_baseline.py        dispatcher -> tasks_baseline_llm/ (assistant welfare only)
│   ├── reward_function_baseline_user.py   dispatcher -> tasks_baseline_user/ (user welfare only)
│   ├── reward_function_linear_comb.py     dispatcher -> tasks_linear_comb/ (arithmetic instead of geometric mean)
│   ├── tasks/                             the main reward: one module per dataset + format.py shared by all
│   │   ├── format.py                      tag extraction, payoff-JSON parsing and validation, format reward, Pareto frontier, length terms
│   │   ├── math_hard.py, medium.py, wildguard.py, abgqa_v2.py      judge prompt + welfare arithmetic per dataset
│   │   ├── gpqa.py, repeat.py             unused helpers (no import site)
│   │   └── __init__.py
│   ├── tasks_ablation/                    same four modules + abg_coqa.py, advbench.py, minerva.py; adds the Pareto judge
│   ├── tasks_baseline_llm/, tasks_baseline_user/, tasks_linear_comb/   near-identical copies differing in one line
│   ├── inference/inference.py             first sketch of the halt-at-payoff loop
│   ├── inference/inference_v2.py          the payoff-intervention orchestrator described in the README
│   └── experiments/                       post-hoc analysis of dumped validation jsonl
│       ├── pareto_ratio.py                dominance counts, epsilon-Pareto coverage, hypervolume, regret, Nash welfare
│       ├── basic_metrics.py, heatmap.py, plot_csv.py, comprehensive.py, pareto_graph.py
│       └── wildguard.py, abgqa.py, class_games.py
├── verl/workers/reward_manager/gamellm.py   the two custom reward managers (judge client, thread pool, normalization)
├── verl/trainer/ppo/reward.py               modified to pass judge_ip / judge_port / normalize_method through
└── verl/trainer/ppo/ray_trainer.py          stock verl loop; _dump_generations writes the jsonl the analysis scripts read
```

## Component Inventory

### 1. Entry points

There is no custom trainer. Every run — training and evaluation alike — is `python3 -m verl.trainer.main_ppo` with a long flag list; the eight shell scripts in `recipe/gamellm/` differ only in data path, reward-function path, reward manager, and experiment name.

| Script | Data | Reward manager | Reward function | Purpose |
|---|---|---|---|---|
| `math.sh` | `by_source/math-hard` | `gamellm` | `reward_function_abgqa_v2.py` | MATH level-5 training |
| `abgqa.sh` | `by_source/ambig_qa` | `gamellm` | `reward_function_abgqa_v2.py` | Ambig-QA training |
| `wildguard.sh` | `by_source/wildguard` | `gamellm` | `reward_function_abgqa_v2.py` | WildGuard training |
| `medium.sh` | `by_source/medium` | `gamellm` | `reward_function_baseline_user.py` | Medium writing — **as shipped this runs the user-welfare-only ablation, not the geometric-mean reward** |
| `full_zscore.sh` | all four mixed | `gamellm_normalize` | `reward_function_ablation.py` | mixed-dataset training with per-source z-score normalization |
| `ablation_minerva.sh`, `ablation_coqa.sh`, `ablation_advbench.sh` | out-of-domain test parquet | `gamellm` | `reward_function_ablation.py` | evaluation only (`trainer.val_only=True`) |

Shared hyperparameters across the training scripts (`recipe/gamellm/math.sh:11-62`): `data.train_batch_size=512`, `actor.ppo_mini_batch_size=32`, actor learning rate $1 \times 10^{-6}$, critic learning rate $2 \times 10^{-6}$, `use_kl_loss=False` and `use_kl_in_reward=False` (no reference-model penalty at all), `max_prompt_length=8192`, `max_response_length=8192`, vLLM rollout at `gpu_memory_utilization=0.6`, four GPUs, `total_epochs=30`, `test_freq=5`. The mixed-dataset script shortens the prompt and response budgets to 2048 and 4096 (`full_zscore.sh:17-18`).

The inference-time entry point is `recipe/gamellm/inference/inference_v2.py`, run as a plain script against a served policy.

### 2. Special token / action handling

No new tokens are added to the tokenizer. The action structure is entirely a text protocol enforced by the prompt and scored by regular expressions.

The assistant must emit exactly four blocks in order. The format reward is a single anchored regular expression plus a successful parse of the payoff JSON (`recipe/gamellm/tasks/format.py:158-192`):

```python
def format_reward(completions, **kwargs):
    pattern = (
        r"^<thinking>.*?</thinking>\s*"
        r"<payoff>.*?</payoff>\s*"
        r"<analyze>.*?</analyze>\s*"
        r"<response>.*?</response>$"
    )
```

Note the tag is `<analyze>`, not `<analysis>`. Block extraction is a non-greedy regex that raises when the tag is missing (`format.py:50-55`):

```python
def extract_tag_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Missing <{tag}> block")
    return m.group(1).strip()
```

The payoff block is parsed leniently and then validated strictly. `jsonish_to_json` (`format.py:57-81`) first tries `json.loads`, then repairs curly quotes, then falls back to `ast.literal_eval` — so Python-style dictionaries with single quotes and `True`/`None` are accepted. `parse_payoff_block` (`format.py:83-120`) then requires the key set to equal exactly `{"DQ_AQ", "DQ_CQ", "DQ_DA", "VQ_AQ", "VQ_CQ", "VQ_DA"}` (`format.py:9`), each mapping to a dictionary with exactly the sub-keys `{"LLM", "user"}` (`format.py:10`) holding numbers. Any deviation raises, and every `compute_score` wraps the whole parse in a bare `try/except` that returns an all-zero reward dictionary (for example `tasks/math_hard.py:106-123`).

Generation interception happens only at inference. `LLMConfig.stop_on_payoff = "</payoff>"` (`inference/inference_v2.py:133`) is passed as the API `stop` sequence, the truncated text is completed with the missing closing tag before parsing, and the matrix is recovered (`inference_v2.py:168-194`):

```python
        messages = [{"role": "user", "content": user_prompt}]
        partial_text, finish_reason = self._chat(messages, stop=[self.cfg.stop_on_payoff])
        ...
        # 补齐 </payoff> 用于解析
        payoff_block = extract_tag_block(partial_text + "</payoff>", "payoff")
        payoff_json = parse_payoff_block(payoff_block)
```

`transform_payoffs` (`inference_v2.py:197-246`) applies the intervention: an additive offset `scale_llm` to every assistant payoff (with a strategy-dependent bump to the user column), or symmetrically an offset `scale_user` to every user payoff. The Pareto-optimal assistant actions of the edited matrix are computed, one is sampled, and the corresponding instruction sentence is appended to the original question for a second, unconstrained generation call (`inference_v2.py:248-277`). The resume is therefore a fresh two-message exchange, not a true continuation of the partial assistant turn — the commented-out line at `inference_v2.py:262` shows the authors also tried feeding the edited `<thinking>`/`<payoff>`/`<analyze>` prefix back as an assistant message.

One discrepancy worth knowing: `pareto_optimal_keys` (`inference_v2.py:65-85`) is documented as Pareto dominance but implemented as a comparison of products, `if uj*lj > u*l`, so it returns the maximizers of the Nash product rather than the non-dominated set. The genuine dominance test lives in `format.py:126-156` and is the one used by the reward path.

### 3. Environment / tool interaction

The only external service is the judge. The reward manager constructs one `openai.OpenAI` client at init and hands it to every scoring call (`verl/workers/reward_manager/gamellm.py:155-162`):

```python
        if self.judge_ip == "localhost":
            self.judge_url = f"http://{self.judge_ip}:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        else:
            self.judge_url = f"http://[{self.judge_ip}]:{self.judge_port}/v1" # ipv6的ip需要用中括号括起来
        self.judge_client = OpenAI(
            api_key="EMPTY",
            base_url=self.judge_url,
        )
```

The `else` branch wraps the host in square brackets because the authors ran the judge on a second node addressed by IPv6; `judge_ip` and `judge_port` arrive as ad-hoc config keys (`+reward_model.judge_ip=localhost`, `+reward_model.judge_port=36485`) threaded through `verl/trainer/ppo/reward.py:119-166`.

Each per-sample judge call is a chat completion with retries (three attempts, linear back-off) and Qwen3 thinking mode enabled (`tasks/math_hard.py:73-103`):

```python
            output = judge.chat.completions.create(
                model="random",
                messages=[
                        {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
```

`model="random"` works because SGLang serves a single model. The judge is asked to put its verdict in `\boxed{}` (math, writing, safety) or in `<abg>...</abg>` / `<po>...</po>` tags (ambiguity, Pareto consistency), and the score is recovered with a regex that returns 0 when nothing matches (`tasks/math_hard.py:61-71`).

Concurrency is asyncio over a thread pool: `parallel_compute_score_async` (`gamellm.py:46-110`) fans the whole batch out with `num_processes=1024` workers (`gamellm.py:185`) and a 300-second per-sample timeout, and the comment at `gamellm.py:52-54` records that the switch from processes to threads was made to avoid pickling the OpenAI client. Failures and timeouts fall through to a zero-filled reward dictionary rather than raising.

The README documents the judge launch itself: `python3 -m sglang.launch_server --model-path Qwen/Qwen3-32B --mem-fraction-static 0.8 --port <port> --tp 4 --dp 1 --reasoning-parser qwen3` on the same node using four GPUs, or `--tp 1 --dp 8` on a dedicated node (`README.md:36-62`).

### 4. Token masking

**Absent.** Nothing in the recipe touches loss masks, and nothing needs to: the rollout is a single assistant turn, so every generated token is a policy token. Prior conversation turns are baked into the prompt string (see Data pipeline), and the judge never writes into the trajectory. The reward is placed on the last valid response token in the standard verl way (`gamellm.py:251`):

```python
            reward_tensor[i, valid_response_length[i].item() - 1] = scores["score"][i]
```

### 5. Reward function

This is the centre of the repo. Every dataset module follows the same five steps: format check, block extraction, payoff-matrix scoring, one judge call, welfare arithmetic.

**Payoff-matrix quality.** A cheap, judge-free heuristic that rewards matrices whose two players' payoffs actually differ. Each of the six cells can contribute twice, so the term lands in $[0, 1]$ (`tasks/math_hard.py:124-130`):

```python
    payoff_reward = 0
    for key, value in payoff_json.items():
        if value["user"] != value["LLM"]:
            payoff_reward += 1/12
        denom = max(abs(value["user"]), abs(value["LLM"]), 1e-8)  # 避免除零
        if abs(value["user"] - value["LLM"]) / denom > 0.1:
            payoff_reward += 1/12
```

**Length terms.** Two closed-form shaping functions in `format.py:397-413`. `reasoning_penalty(text, scale)` is $e^{-\text{len}(text)/\text{scale}}$ over characters — shorter reasoning scores higher, standing in for the user's waiting cost. `response_reward_logistic(text, L_min, L_max, k)` is a product of two logistic gates over word count, so a response is rewarded for sitting inside a band:

```python
def response_reward_logistic(text: str, L_min=100, L_max=2000, k=0.005):
    length = len(text.split())
    left = 1 / (1 + math.exp(k * (L_min - length)))
    right = 1 / (1 + math.exp(k * (length - L_max)))
    return left * right
```

**Welfare aggregation.** With user welfare $U$, assistant welfare $L$, and $\epsilon = 10^{-4}$ keeping the product away from zero, the training score is the geometric mean $W = \sqrt{U \cdot L}$ (`tasks/math_hard.py:134-137`):

```python
    epsilon = 1e-4
    user_reward = epsilon + 0.5 * answer_reward + 0.2 * response_reward_logistic(solution_str, L_min=20, L_max=500, k=0.005) + 0.3 * reasoning_penalty(reasoning_str,scale=500)
    llm_reward = epsilon + 0.2 * format_reward1 + 0.2 * payoff_reward + 0.4 * answer_reward + 0.2 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005)
    score = (user_reward * llm_reward) ** 0.5
```

The two welfare terms differ in what they measure and over what text. User welfare weighs answer quality 0.5, response length 0.2 (band 20 to 500 words, measured on the `<response>` block only) and reasoning brevity 0.3 (measured on everything before `<response>`). Assistant welfare weighs format 0.2, payoff-matrix quality 0.2, answer quality 0.4 and total-output length 0.2 (band 500 to 1000 words, measured on the whole four-block output). So the same answer score enters both sides, the length terms pull in opposite directions, and only the assistant side pays for format and matrix quality.

**Per-dataset answer score.** Each module supplies its own judge prompt and its own reading of "answer quality":

| Dataset | Module | Judge prompt | Scale | Answer term |
|---|---|---|---|---|
| MATH level-5 | `tasks/math_hard.py:27-59` | binary correctness against ground truth | `\boxed{0/1}` | same score on both welfare sides |
| Medium writing | `tasks/medium.py:30-70` | 0-4 helpfulness, explicitly telling the judge that clarifying questions are acceptable and even preferable | `\boxed{0..4}`, divided by 4 | **asymmetric**: user side uses sentence BLEU against the reference, assistant side uses the normalized judge score (`medium.py:148-155`) |
| WildGuard | `tasks/wildguard.py:32-96` | three-way safety verdict: 1 refusal, 0.5 refusal plus a constructive alternative, 0 unsafe | `\boxed{0/0.5/1}` | user side scores 1 only when the harmful request was *not* complied with; assistant side binarizes any non-zero judge score to 1 (`wildguard.py:174-183`) |
| Ambig-QA | `tasks/abgqa_v2.py:43-80` | binary ambiguity handling; ground truth carries the `multipleQAs###answer` marker split at `###` | `<abg>0/1</abg>` | same score on both sides |

**Reward variants.** The four parallel task packages are byte-identical to `tasks/` except for one line each, which makes the ablation set unusually easy to read:

```
tasks_baseline_llm/math_hard.py:139     "score": llm_reward,           # assistant welfare only
tasks_baseline_user/math_hard.py:139    "score": user_reward,          # user welfare only
tasks_linear_comb/math_hard.py:137-138  score = (user_reward + llm_reward) / 2   # arithmetic mean, geometric mean kept as "utility"
```

**The evaluation-time reward** (`tasks_ablation/`) adds a fifth component: a second judge call that scores whether the `<analyze>` block names an action that is actually on the Pareto frontier of the model's own matrix. The frontier is computed in Python from the parsed matrix, stringified, and handed to the judge together with the analysis text (`tasks_ablation/math_hard.py:73-171` for the prompt and call, `:226-230` for the wiring), and the assistant welfare is reweighted to make room for it (`tasks_ablation/math_hard.py:243`):

```python
    llm_reward = epsilon + 0.2 * pareto_score + 0.2 * payoff_reward + 0.2 * format_reward1 + 0.35 * answer_reward + 0.05 * response_reward_logistic(ori_solution_str, L_min=500, L_max=1000, k=0.005)
```

**Return shape.** Every `compute_score` returns a dictionary — `score`, `answer_reward`, `format_reward`, `user_reward`, `llm_reward`, `utility`, `reward_gap` — and the dispatcher stamps `source` onto it (`reward_function_abgqa_v2.py:31-34`). verl's `reward_extra_info` channel carries all of these into the metric logger and into the dumped jsonl, which is what makes the downstream Pareto analysis possible.

**Reward normalization.** The second manager, `gamellm_normalize` (`gamellm.py:286-537`), groups the batch by `source` and rewrites `ppo_score` before it reaches the advantage computation; the selection is a string config key `normalize_method` with eight branches — `zscore`, `minmax`, `robustz` (median and scaled median absolute deviation, then `tanh`), `cdf`, `quantile`, `james` (shrinkage of per-source statistics toward global ones), `decentralize`, `selfadapt` (`gamellm.py:377-493`). Only `zscore` is used by a shipped script. The point of this machinery is that the four datasets have different reward scales, so mixed-batch PPO would otherwise chase whichever source has the widest spread.

### 6. RL training loop

Stock verl. No file under `verl/trainer/ppo/` was modified except `reward.py`, which only widens the reward-manager constructor call. The loop is therefore: sample one response per prompt with vLLM, score the batch through the reward manager, place the scalar at the last response token, compute GAE against the learned critic, run PPO updates with `use_dynamic_bsz=True` and a 32768-token per-GPU budget.

Consequences of `adv_estimator=gae` with `rollout.n=1`: a full critic (a second copy of the 3B model) is trained alongside the actor, there is no group baseline, and the reward scale matters directly — which is exactly what the normalization manager exists to control. No KL penalty is applied in either the loss or the reward.

The custom reward function is loaded by path, not by import, through verl's `get_custom_reward_fn` (`verl/trainer/ppo/reward.py:42-90`), which `importlib`-loads the file named by `custom_reward_function.path` and looks up `custom_reward_function.name`.

### 7. Supervised fine-tuning / cold start

**Absent from the repo.** The paper's cold-start pipeline (synthesis with Qwen3-32B and gpt-oss-20b, best-of-9 selection by social welfare) has no code here. Its output is referenced only as a checkpoint path in the launchers, `gamellm_new/outputs/sft/gamellm/thinking-full-parallel-test3/checkpoint-6` (`full_zscore.sh:22`) and `checkpoint-18` (`ablation_minerva.sh:22`). verl's generic `verl/trainer/fsdp_sft_trainer.py` ships but is not wired to anything in `recipe/gamellm/`, and it imports the missing dataset package (see the defects note below).

### 8. Data pipeline

Data is committed to git as parquet — no preprocessing scripts survive (`data_process/` holds only a wandb download helper). The mixed training file has 10806 rows over four sources: `math-hard` 3176, `wildguard` 2700, `ambig_qa` 2700, `medium` 2230. Columns: `prompt`, `adversarial`, `source`, `conv_id`, `unique_conv_id`, `turn_id`, `data_source`, `ability`, `reward_model`, `extra_info`.

The important structural fact is that **multi-turn conversations are flattened into a single prompt**. Each row's `prompt` is a two-message chat: a fixed system message about being proactive and asking for clarification, and one user message that concatenates the prior conversation, the full game-theory instruction block, and finally `Question: <last user turn>`. `turn_id` reaches 5 for `medium` and `math-hard` (mean 2.7 and 2.2) and is always 1 for `ambig_qa` and `wildguard`, but every row is still one prompt and one response. verl's multi-turn rollout machinery is not used; `actor_rollout_ref.rollout.multi_turn.format=hermes` is set in the scripts while `rollout.mode=sync` and multi-turn is never enabled.

The instruction block embedded in the data is character-for-character the same text as `GAME_THEORY_PREFIX` in `inference/inference_v2.py:26-60` — it defines the two user types (DQ definite, VQ vague), the three assistant strategies (DA direct answer, CQ clarifying question, AQ answer plus one follow-up), the required JSON shape, the value range roughly $[-5, 5]$ with one decimal place, and the tie-break rule for choosing a strategy ("1) maximize social welfare (LLM + user), 2) if tied, maximize user, 3) if still tied, maximize LLM").

`extra_info` carries what the reward needs: `chat_history` (the message list), `single_turn_prompt` (the user's underlying intent or the bare question, used as the judge's view of the problem), `single_turn_completion`, `adversarial`, `split`. `reward_model.ground_truth` holds the reference answer, the harmfulness label, or the `multipleQAs###answer` string.

### 9. Multi-GPU and infrastructure

FSDP for actor and critic (`fsdp_config.param_offload=False`, `optimizer_offload=False`), vLLM for rollout with `tensor_model_parallel_size=1`, four GPUs on one node (`trainer.n_gpus_per_node=4`, `trainer.nnodes=1`, `CUDA_VISIBLE_DEVICES=0,1,2,3`). The judge takes the other four GPUs of an eight-GPU machine, or a whole second node; the README recommends eight H20 GPUs. Trainer and judge are separate processes talking over HTTP, so the judge is a fixed external service, never sharded with the trainer. Gradient checkpointing and `use_remove_padding=True` are on everywhere.

### 10. Evaluation

Out-of-domain evaluation reuses the training entry point with `trainer.val_only=True` and the same test parquet in both the train and val slots (`ablation_minerva.sh:14-15,62`). Scoring runs through `tasks_ablation/`, so the evaluation reward includes the Pareto-consistency judge term that training does not use.

The bridge to analysis is `trainer.validation_data_dir`. When set, verl's `_dump_generations` (`verl/trainer/ppo/ray_trainer.py:431-457`) writes `<global_step>.jsonl` with one object per validation sample carrying `input`, `output`, `gts`, `score`, `step`, plus every key of `reward_extra_info` — which is how `user_reward` and `llm_reward` end up on disk per sample (`ray_trainer.py:580-611`).

`recipe/gamellm/experiments/pareto_ratio.py` is the main analysis module. It joins two jsonl files on the `input` field and computes, over the two-dimensional points $(U, L)$: pairwise dominance counts with an $\epsilon$ tolerance (`:36-46`), the joint Pareto frontier of both methods (`:72-84`), two-dimensional hypervolume against the data-driven reference point $(U_{\min}, L_{\min})$ (`:86-102`), average normalized $L_\infty$ regret to the joint frontier (`:107-122`), $\epsilon$-Pareto coverage (`:124-142`), and Nash social welfare and Cobb-Douglas aggregates (`:144-162`). `heatmap.py` runs it over all pairs from {cobb-douglas, linear, user, llm} to produce advantage matrices; `basic_metrics.py` builds the comparison table; `wildguard.py` and `abgqa.py` compute per-dataset precision, recall and F1 from the same dumps. All of these hardcode absolute jsonl paths from the authors' machine and none of the referenced files ship.

### Defects and traps in the published tree

These matter for anyone trying to run it:

- **`verl/utils/dataset/` is missing from the published commit**, along with `verl/experimental/dataset/sampler.py`. Both are still imported: `verl/trainer/main_ppo.py:25,282,324` and `verl/trainer/ppo/ray_trainer.py:39,377` need `RLHFDataset`, `collate_fn` and `AbstractSampler`. The tree as cloned cannot start a run without restoring those files from upstream verl 0.5.0. `verl/trainer/fsdp_sft_trainer.py:49-50` is broken the same way.
- `recipe/gamellm/reward_function.py:26` imports `from recipe.gamellm.tasks import abgqa`, which does not exist (the module is `abgqa_v2.py`, and the corresponding line in `tasks/__init__.py:17` is commented out). This dispatcher is unused by the shell scripts, which all point at `reward_function_abgqa_v2.py` or the ablation variants.
- `medium.sh:49` points at `reward_function_baseline_user.py`, so the shipped Medium-writing launcher trains against user welfare alone rather than the geometric mean.
- `recipe/gamellm/tasks/format.py:11` hardcodes a stale IPv6 judge URL as a module constant; it is dead code, but every task module also carries its own `BASE_URL = "http://localhost:36485/v1"` used only by the `test()` functions.
- `pareto_frontier_from_json` has diverged between copies: `tasks/format.py:139-156` takes an already-parsed dictionary and returns strategy names, while the copies in `tasks_linear_comb/format.py`, `tasks_baseline_llm/format.py` and `tasks_baseline_user/format.py` call `json.loads` on the argument and return `(name, point)` pairs. The `__main__` demo in `tasks/format.py:415-429` unpacks names as pairs and would crash.
- Comments and docstrings are largely in Chinese throughout `recipe/gamellm/`.
- `tasks/format.py:282-395` carries a TF-IDF semantic-redundancy scorer (pulling in scikit-learn) and `tasks/repeat.py` an n-gram repetition metric; neither has an import site in the reward path.

## Relevance to this project

The project trains persuader agents with reinforcement learning against an interlocutor, scored by a language-model judge over multi-component rewards. GTAlign is the closest available template for the *scoring and serving* half of that problem and a weak template for the *interaction* half.

### Components we can directly reuse

- **The judge-backed reward manager**, `verl/workers/reward_manager/gamellm.py:32-263`. This is the single most transferable artifact: a verl reward manager that owns an OpenAI-compatible client, fans a whole batch of judge calls out over asyncio plus a thread pool with per-sample timeouts, degrades to zero rewards instead of crashing the training step, and returns a dictionary of components that verl carries through as `reward_extra_info`. The pattern of registering with `@register("name")` and threading extra constructor arguments through `verl/trainer/ppo/reward.py:119-166` is the minimal way to add a service-backed reward to verl.
- **The judge-call wrapper**, `recipe/gamellm/tasks/math_hard.py:73-103`. Retry loop, `model="random"` against a single-model SGLang server, thinking mode enabled through `extra_body`, verdict recovered by regex from `\boxed{}` with a zero fallback. Directly reusable for a persuasion judge that rates opinion change or argument quality.
- **Judge serving over the network**, `README.md:36-62`. The SGLang launch commands, the same-node four-GPU split versus dedicated-node layout, and the IPv6 bracket handling at `gamellm.py:155-158` are hard-won operational details.
- **The tagged-block protocol and its parser**, `recipe/gamellm/tasks/format.py:50-192`. An anchored regex over the whole completion for the format reward, tolerant JSON recovery through `ast.literal_eval`, then strict key-set validation. A persuader emitting, say, a strategy block and a message block needs exactly this shape, and the lenient-then-strict split is the part worth copying — models emit Python-style dictionaries often enough that a bare `json.loads` throws away usable rollouts.
- **Per-source reward normalization**, `gamellm.py:377-397`. If persuader training mixes topics or receiver types with different reward scales, grouping by `source` and z-scoring within the batch before the advantage step is the fix, and it is about fifteen lines.
- **The dump-then-analyze split**, `verl/trainer/ppo/ray_trainer.py:431-457` feeding `recipe/gamellm/experiments/pareto_ratio.py`. Every reward component is persisted per sample per validation step, and all interpretation happens later from those files. That matches how this project separates runs from analyses.

### Components we need to modify

- **The welfare aggregation.** $W = \sqrt{U \cdot L}$ over two hand-weighted sums is the paper's contribution, and its shape transfers to a persuasion setting with a sender term and a receiver term — but every weight (0.5/0.2/0.3 and 0.2/0.2/0.4/0.2) and every band constant (`L_min=20, L_max=500, scale=500`) is tuned to a helpfulness task and carries no meaning here. What transfers is the *structure*: one judge-derived quality term, one cheap structural term, one length term, combined multiplicatively so that neither party's welfare can be traded to zero.
- **The rollout shape.** GTAlign is single-turn: prior conversation is flattened into the prompt string and the interlocutor never responds inside the rollout. A persuasion setting needs a real receiver in the loop, so verl's multi-turn or agent-loop path has to be used, and with it the loss masking that GTAlign never needed. Nothing in this repo helps with that; the sibling reference (Cheng and You) is the place to look.
- **The RL algorithm.** PPO with GAE and `rollout.n=1` means training a full critic copy of the policy. If the project uses GRPO, the reward manager and reward functions carry over unchanged (they are algorithm-agnostic) but the normalization manager becomes largely redundant, since group-relative advantages already remove per-prompt scale.
- **The dispatcher-by-copy pattern.** Four near-identical task packages differing in one line each is readable but is four times the maintenance. A single module with the aggregation selected by config is the version to write.
- **The launcher scripts.** Absolute paths from the authors' machine, no config file, no run isolation. Replace with this project's `defaults.yaml` plus per-run overlay.

### Components that don't apply

- The payoff matrix itself, its six-key schema, the DQ/VQ and DA/CQ/AQ strategy sets, and the Pareto-frontier reasoning — these encode the assistant-helpfulness game, not a persuasion game. The *mechanism* of having the model emit a machine-readable strategic object that the reward can score independently of the final message is what generalizes; the specific matrix does not.
- The four datasets and their judge prompts (math correctness, writing helpfulness, safety, ambiguity) are unrelated to opinion change.
- The Pareto analysis scripts in `experiments/` assume a two-objective trade-off between two parties' welfare. Useful only if the project frames sender and receiver welfare as competing objectives; otherwise the hypervolume and $\epsilon$-coverage machinery is overhead.
- The supervised cold-start is absent, so there is nothing to reuse there.

### Key code snippets worth studying

| What | Where |
|---|---|
| Reward manager: batch fan-out, timeout handling, failure fallback | `verl/workers/reward_manager/gamellm.py:32-110` |
| Reward manager: judge client, `verify`, reward placement, extra-info return | `verl/workers/reward_manager/gamellm.py:140-263` |
| Per-source reward normalization before the advantage step | `verl/workers/reward_manager/gamellm.py:377-397, 496, 525` |
| Registering a custom reward manager and threading its config | `verl/trainer/ppo/reward.py:117-173` |
| Loading a reward function by file path | `verl/trainer/ppo/reward.py:42-90` |
| Tagged-block format reward and payoff parsing | `recipe/gamellm/tasks/format.py:50-192` |
| Length-shaping terms in closed form | `recipe/gamellm/tasks/format.py:397-413` |
| Full welfare computation for one dataset, end to end | `recipe/gamellm/tasks/math_hard.py:105-149` |
| Judge call with retries and thinking mode | `recipe/gamellm/tasks/math_hard.py:73-103` |
| Judge scoring a model's own reasoning against a computed frontier | `recipe/gamellm/tasks_ablation/math_hard.py:73-171, 226-230, 243` |
| Halt at `</payoff>`, edit the matrix, resume | `recipe/gamellm/inference/inference_v2.py:168-321` |
| Per-sample validation dump that feeds all downstream analysis | `verl/trainer/ppo/ray_trainer.py:431-457, 580-611` |
| Pareto and social-welfare metrics over dumped rewards | `recipe/gamellm/experiments/pareto_ratio.py:36-162` |
| A complete PPO launcher with a judge-backed reward | `recipe/gamellm/math.sh:11-62` |
