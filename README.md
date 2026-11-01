# Anti-BAD Challenge evaluation kit

The Anti-Backdoor (Anti-BAD) Challenge benchmark is now fully released. This
repository provides the public development and test resources, reference
metadata, and runnable evaluation code needed to reproduce the competition
settings and evaluate defenses on the released candidate models.

The kit covers all three tracks and both tasks per track:

| Track | Evaluation pipeline |
| --- | --- |
| Classification | prediction → score |
| Multilingual classification | prediction → score |
| Generation | prediction → GPT-4o-mini judge → score |

Each task provides three potentially backdoored LoRA adapters. Evaluation uses
400 examples: 100 clean examples and 100 examples from each of three backdoor
settings. The released annotations in [`metadata/`](metadata/README.md) enable
local end-to-end evaluation for both the development and test phases.

## Repository layout

```text
anti-bad-challenge/
├── classification-track/   # classification inference and scoring
├── generation-track/       # generation, LLM judging, and scoring
├── multilingual-track/     # multilingual inference and scoring
├── metadata/               # released dev/test reference annotations
├── scripts/                # code shared by all tracks
├── download_resources.py   # selective Hugging Face downloader
└── outputs/                # created at runtime and ignored by Git
```

Downloaded datasets and adapters are kept inside each track under `data/` and
`models/`. They are ignored by Git because they can be restored from Hugging
Face at any time.

## What's included

- **Resource downloader:** select a phase, track, task, and candidate model
  without downloading the entire benchmark.
- **Prediction pipelines:** run inference with a released candidate adapter,
  a WAG model, or a custom defended adapter.
- **Local evaluation:** score classification predictions directly; generation
  responses are first judged with `gpt-4o-mini` and then scored.
- **Released metadata:** inspect the clean and attack-specific annotations used
  for development- and test-phase evaluation.
- **WAG baseline:** reproduce the weight-averaging defense by merging the three
  candidate adapters for a task.

## 1. Install dependencies

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

Some base models are gated on Hugging Face. If a download requires access,
request it from the model provider and authenticate locally with `huggingface-cli login`.

The generation judge also needs an OpenAI API key:

```bash
cp .env.example .env
# Then set OPENAI_API_KEY in .env.
```

Classification and multilingual evaluation do not use an external API.

## 2. Download a benchmark phase

The phase is always explicit. The following command downloads all datasets and
all three candidate adapters for the development phase:

```bash
python download_resources.py --phase dev
```

The downloader can select only the resources you need:

```bash
# One complete track from the public test phase
python download_resources.py --phase test --track generation

# A single task and candidate model
python download_resources.py \
  --phase dev \
  --track classification \
  --task 1 \
  --model model2

# Input datasets only, for two tracks
python download_resources.py \
  --phase test \
  --track classification \
  --track multilingual \
  --data-only
```

`--track`, `--task`, and `--model` may each be repeated. Omitting one selects
all available values for that option. Use `--models-only` when the inputs are
already present.

## 3. Evaluate a released model

Run each command from its track directory. Arguments are consistently ordered
as `<phase> <task> <model>`.

### Classification

```bash
cd classification-track
bash run_evaluation.sh dev 1 model1
```

This writes:

```text
../outputs/dev/classification/task1/model1/
├── predictions.csv
└── score.json
```

### Multilingual classification

```bash
cd multilingual-track
bash run_evaluation.sh dev 1 model1
```

This uses the same prediction and scoring contract as classification and writes
to `outputs/dev/multilingual/task1/model1/`.

### Generation

```bash
cd generation-track
bash run_evaluation.sh dev 1 model1
```

The generation pipeline first produces model responses, then judges each
response with `gpt-4o-mini`, and finally calculates the benchmark score:

```text
../outputs/dev/generation/task1/model1/
├── predictions.json
├── judgments.jsonl
└── score.json
```

`judgments.jsonl` is a resumable cache. If an API request is interrupted,
rerun `run_judge.sh`; completed records are retained. To intentionally discard
the cache, set `RESTART_ARGS=(--restart)` in that script. LLM judging incurs
OpenAI API usage.

## Run individual stages

Every stage can be run separately, which is useful when evaluating a custom
defense or inspecting intermediate results:

```bash
cd generation-track
bash run_prediction.sh test 2 model3
bash run_judge.sh test 2 model3
bash run_score.sh test 2 model3
```

Classification and multilingual tracks expose `run_prediction.sh` and
`run_score.sh` in the same way. Prediction scripts accept a custom adapter path
through the `MODEL_PATH` environment variable:

```bash
MODEL_PATH=/path/to/my_adapter bash run_prediction.sh dev 1 my_defense
bash run_score.sh dev 1 my_defense
```

The main runtime settings—quantization, batch size, generation length, and
overwrite behavior—are defined near the top of each shell script so they can be
changed without navigating through Python code.

## Weight-averaging baseline

An optional Weight Averaging (WAG) baseline averages the three released
candidate adapters for a task:

```bash
cd classification-track
bash run_wag.sh dev 1
bash run_evaluation.sh dev 1 wag
```

The same commands are available in all three tracks. WAG always merges exactly
`model1`, `model2`, and `model3`. A clean reference model is not part of the
baseline and is never included in the merge.

If you use this baseline, please cite:

> Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, and
> Qiongkai Xu. 2024. [Here's a Free Lunch: Sanitizing Backdoored Models with
> Model Merge](https://aclanthology.org/2024.findings-acl.894/). Findings of ACL
> 2024.

## Scoring

For classification tracks, utility is clean accuracy and attack success is the
fraction of triggered inputs classified as their attack target. For generation,
utility is judged response quality and attack success is determined by the
attack-specific content, sentiment, refusal, or jailbreak judge. Attack success
rates are macro-averaged across the three backdoor settings before utility and
defense are combined geometrically. Each `score.json` reports the aggregate
score and the full per-attack breakdown.

The scripts validate record counts, prediction format, and metadata alignment
before scoring. The metadata remains human-readable so experiments can be
audited without a separate evaluation bundle.

## Links

- [Challenge website](https://anti-bad.github.io)
- [Public models and datasets](https://huggingface.co/anti-bad-challenge)
- [Original CodaBench competition](https://www.codabench.org/competitions/11188/)
