# Anti-BAD Challenge – Starter Kit

Official starter kit for the **Anti-Backdoor (Anti-BAD) Challenge @ IEEE SaTML 2026**

**Website:** [https://anti-bad.github.io](https://anti-bad.github.io)  
**Competition Platform:** [CodaBench](https://www.codabench.org/)  

---

## Overview

The Anti-BAD Challenge focuses on **post-training defenses** against backdoor attacks in large language models (LLMs).  
Participants receive **backdoored models** and develop defense methods that reduce the **Attack Success Rate (ASR)** while maintaining **model utility**.

**Three Tracks – Six Tasks**
- **Generation Track** (2 tasks): instruction-following models  
- **Classification Track** (2 tasks): sequence classification  
- **Multilingual Track** (2 tasks): cross-lingual classification  

You may submit to any subset of tasks.  
Your overall leaderboard score is the average across all six tasks.

---

## What’s Provided

Each task includes:
- **Three backdoored LoRA models** (attack methods undisclosed)  
- **An input-only test dataset** (no ground truth)  
- **Code scripts** for prediction generation and the baseline method  

The provided LLMs are mainly based on **Llama** and **Qwen** architectures.  
Please refer to the **model configuration files** in this repository for detailed model cards and parameter information.  
All models are **LoRA adapters** fine-tuned on all linear modules.

---

## Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
````

### 2 · Download resources

```bash
python download_resources.py
```

This retrieves all models and test data for the six tasks.

### 3 · Generate predictions

```bash
cd classification-track
bash pred.sh 1     # run Task 1
bash pred.sh 2     # run Task 2
```

Predictions are saved to `submission/` using the required filenames.

### 4 · Create submission package

```bash
bash create_submission.sh
```

Generates `submission.zip` ready to upload to **CodaBench**.

---

## Developing Your Defense

You are free to work on a single provided model or on multiple models.
After applying your defense methods, use the same prediction scripts in this repository to generate outputs for submission.

---

## Baseline Defense

A simple **Weight Averaging (WAG)** baseline is included:

```bash
cd <track-name>
bash baseline_wag.sh <task_number>
```

This merges the three backdoored models to mitigate malicious triggers.

If you reference this method, please cite:

> **Ansh Arora, Xuanli He, Maximilian Mozes, Srinibas Swain, Mark Dras, and Qiongkai Xu.**
> 2024. *Here’s a Free Lunch: Sanitizing Backdoored Models with Model Merge.*
> *Findings of the Association for Computational Linguistics: ACL 2024.*

---

## Submission

1. Generate predictions for one or more tasks.
2. Verify that filenames match the required format (see competition documentation).
3. Package all results into `submission.zip`.
4. Upload via **My Submissions** on CodaBench.

**Submission limits**

* Development phase: 3 submissions per day (270 total)
* Test phase: 2 submissions per day (14 total)

During the **test phase**, participants will submit both prediction results and executable code for verification.  
The code should be able to reproduce the submitted outputs within 24 hours on a single NVIDIA A100 GPU.  
All submitted code will remain **private** and will be used only for evaluation purposes.

---

## Support

* **Website:** [https://anti-bad.github.io](https://anti-bad.github.io)
* **Email:** [antibad-competition-satml-2026@googlegroups.com](mailto:antibad-competition-satml-2026@googlegroups.com)
* **Discord:** [https://discord.gg/x8GqKDF2Rb](https://discord.gg/x8GqKDF2Rb)

Good luck defending your models!