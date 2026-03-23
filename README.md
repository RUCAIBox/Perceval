<div align="center">

<h1>🔍 Perceval</h1>

<h3><b>Improving Vision-language Models with Perception-centric Process Reward Models</b></h3>

<p>
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv"></a>
  <a href="https://github.com/RUCAIBox/Perceval"><img src="https://img.shields.io/badge/GitHub-Code-black?style=flat-square&logo=github" alt="Code"></a>
  <img src="https://img.shields.io/badge/CVPR-2026-blue?style=flat-square" alt="CVPR 2026">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

<p>
  <a href="#">Yingqian Min</a><sup>1,2*</sup>&nbsp;·&nbsp;
  <a href="#">Kun Zhou</a><sup>3*</sup>&nbsp;·&nbsp;
  <a href="#">Yifan Li</a><sup>1,2*</sup>&nbsp;·&nbsp;
  <a href="#">Yuhuan Wu</a><sup>4</sup>&nbsp;·&nbsp;
  <a href="#">Han Peng</a><sup>1</sup>&nbsp;·&nbsp;
  <a href="#">Yifan Du</a><sup>1</sup><br>
  <a href="#">Wayne Xin Zhao</a><sup>1†</sup>&nbsp;·&nbsp;
  <a href="#">Min Yang</a><sup>2</sup>&nbsp;·&nbsp;
  <a href="#">Ji-Rong Wen</a><sup>1</sup>
</p>

<p>
  <sup>1</sup>Gaoling School of AI, Renmin University of China &nbsp;·&nbsp;
  <sup>2</sup>Bytedance &nbsp;·&nbsp;
  <sup>3</sup>UC San Diego &nbsp;·&nbsp;
  <sup>4</sup>HKUST
</p>

<p><sup>*</sup>Equal contributions &nbsp;&nbsp;<sup>†</sup>Corresponding author</p>

</div>

---

## 📖 TL;DR

We propose **Perceval**, a **perception-centric Process Reward Model (PRM)** that tackles the sparse-reward bottleneck in RLVR for vision-language models. Perceval detects image–text misalignments at the token level, enabling fine-grained training supervision and test-time error correction.

---

## 🚀 News

- **[2026.03]** 🎉 Paper accepted at **CVPR 2026**!
- **[2026.03]** Code and models will be released soon. Stay tuned!

---

## 💡 Motivation

Existing RLVR methods for VLMs rely on **outcome-level (sequence-level) rewards**, which are too coarse to:
- Identify *which step* in the reasoning chain went wrong
- Distinguish perceptual errors from logical errors
- Provide corrective gradients to specific hallucinated spans

This creates a **hard credit-assignment problem** that bottlenecks learning.

---

## 🔧 Method Overview

<div align="center">
<img src="assets/framework.png" width="85%" alt="Perceval Framework">
<p><em>Overview of Process-Supervised GRPO with Perceval.</em></p>
</div>

Perceval operates in three stages:

### 1. 🧠 Perception-Centric PRM Training
- Trained on perception-intensive data (visual search, referring-expression grounding)
- Uses a `<think>...</think>` → `<answer>...</answer>` schema to ground claims against visual evidence
- Identifies **hallucinated spans** as Python lists of exact offending strings

### 2. 🎯 Token-Level Advantage Reallocation (RLVR)
We replace GRPO's sequence-level advantage $\hat{A}_i$ with a **token-level** variant:

$$\hat{A}'_{i,t} := \hat{A}_i - \alpha \cdot m_{i,t} \cdot |\hat{A}_i|$$

where $m_{i,t} = 1$ for tokens in hallucinated spans. This directly penalizes hallucinatory tokens while preserving signal for correct tokens.

### 3. 🔄 Test-Time Scaling via Truncation–Regeneration
- **Truncate–then–Regenerate**: truncate at the first erroneous span, then resample
- **Truncate–Thinking–then–Regenerate**: additionally inject a reflection prompt before regeneration
- Both loops iterate up to $k$ times, consistently outperforming majority voting

---

## 📊 Main Results

### RL Training with PRM (vs. GRPO baseline)

| Model | V\* All | BLINK | MMStar | MathVision | ChartQA |
|:------|:-------:|:-----:|:------:|:----------:|:-------:|
| Qwen2.5-VL-3B + GRPO | 80.10 | 49.13 | 55.3 | 23.36 | 83.32 |
| **+ Ours (3B)** | **83.25** | **48.75** | **55.8** | **26.32** | **86.48** |
| Qwen2.5-VL-7B + GRPO | 84.29 | 53.55 | 62.0 | 27.96 | 85.16 |
| **+ Ours (7B)** | **86.39** | **54.49** | **63.8** | **30.92** | **84.44** |

### Test-Time Scaling (3B, k=16)

| Method | V\* All | BLINK |
|:-------|:-------:|:-----:|
| Major Voting | 85.86 | 48.41 |
| Truncate | **89.53** | **49.45** |
| Truncate-Thinking | 88.48 | 49.38 |

> **Key finding**: Perceval's fine-grained perceptual supervision generalizes to math & chart reasoning tasks, even without PRM intervention during their RL training — a surprising capability transfer effect.

---

## 🏗️ Installation

```bash
git clone https://github.com/RUCAIBox/Perceval.git
cd Perceval
pip install -r requirements.txt
```

---

## 🤗 Model Zoo

| Model | Size | Description | Download |
|:------|:----:|:------------|:--------:|
| Perceval-PRM | 3B | Perception-centric PRM | Coming soon |
| Perceval-PRM | 7B | Perception-centric PRM | Coming soon |
| Perceval-Policy | 3B | Policy model trained with PRM | Coming soon |
| Perceval-Policy | 7B | Policy model trained with PRM | Coming soon |

---

## 🧪 Quick Start

```python
# Coming soon
from perceval import Perceval

prm = Perceval.from_pretrained("RUCAIBox/Perceval-PRM-7B")
result = prm.verify(image=image, query=query, response=response)
# Returns hallucinated spans, e.g.:
# ["The main color of the desk appears to be dark brown or black"]
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{min2026perceval,
  title     = {Improving Vision-language Models with Perception-centric Process Reward Models},
  author    = {Min, Yingqian and Zhou, Kun and Li, Yifan and Wu, Yuhuan and Peng, Han and Du, Yifan and Zhao, Wayne Xin and Yang, Min and Wen, Ji-Rong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgement

This work was partially supported by the National Natural Science Foundation of China (No. 92470205) and Beijing Major Science and Technology Project (No. Z251100008425002).

We thank the authors of [DeepEyes](https://github.com/Visual-Agent/DeepEyes), [SophiaVL-R1](https://github.com/kxfan2002/SophiaVL-R1), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source contributions.

---

<div align="center">
<sub>⭐ Star this repo if you find it helpful!</sub>
</div>