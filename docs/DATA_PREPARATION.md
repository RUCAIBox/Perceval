# Data preparation

The trainer consumes parquet files in the format produced by verl's data
preprocessing utilities (`verl/utils/dataset/rl_dataset.py` is the loader).
This document describes the schema you need to produce and points at the
public sources we used in the paper. The repository does **not** ship parquet
files.

## Parquet schema

Each row corresponds to one prompt and must contain at least these columns:

| Column | Type | Description |
| --- | --- | --- |
| `data_source` | `string` | Drives reward-function dispatch. We use: `vstar`, `vstar_test`, `chart`, `geoguessr`, `math`, `ViRL`, `thinklite_eureka`, `xince`, `critic`. **Any string containing `vstar` triggers the PRM process verifier**; everything else falls back to outcome-only verification (which makes TRM degenerate to vanilla GRPO for those rows — see the README). Strings containing `test` skip process verification entirely. |
| `prompt` | `list<dict>` | OpenAI-style message list (`role`/`content`). Image placeholders inside the user message are rendered into image tokens by verl. |
| `images` | `list<bytes \| string>` | One or more images, either raw bytes (preferred) or filesystem paths reachable from every trainer worker. |
| `reward_model.ground_truth` | `string` | Final answer text used by the LLM-as-judge. |
| `extra_info` | `dict` | The reward function looks up `question`, `images`, and (for PRM evaluation) `original_response`. |

The validation files used in the paper also include a per-question `idx`
column (file names end `_idx.parquet`) so per-question metrics can be
aggregated across `rollout.val_kwargs.n` samples.

## Public sources

| Dataset | Used as | Source |
| --- | --- | --- |
| **DeepEyes** | Training (perception-heavy multi-image reasoning) | https://huggingface.co/datasets/yanyq/DeepEyes-data |
| **V\*-bench** (direct_attributes + relative_position) | Validation | https://github.com/penghao-wu/vstar |
| **MathVista** (testmini) | Validation (math reasoning OOD) | https://huggingface.co/datasets/AI4Math/MathVista |
| **ViRL-39k** | Optional training mix-in | https://huggingface.co/datasets/TIGER-Lab/ViRL39K |
| **ChartQA** | Optional eval | https://huggingface.co/datasets/HuggingFaceM4/ChartQA |

After preparing parquet files that satisfy the schema above, point
`configs/perceval.env` at them via `PERCEVAL_TRAIN_DATA` and
`PERCEVAL_VAL_DATA`.

## Writing your own converter

`verl/utils/dataset/rl_dataset.py` shows exactly what columns the loader
touches; the data-source-aware dispatch is in
[`verl/utils/reward_score/hallu_token_reward_vstar.py::adaptive_comput_score`](../verl/utils/reward_score/hallu_token_reward_vstar.py).
Read those two before writing a converter — the schema is small but the
field names matter.

A minimal converter looks like:

```python
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

rows = []
for sample in load_your_dataset():
    rows.append({
        "data_source": "vstar",                    # triggers PRM
        "prompt": [
            {"role": "user",
             "content": [{"type": "image", "image": "<image>"},
                         {"type": "text",  "text": sample["question"]}]},
        ],
        "images": [open(sample["image_path"], "rb").read()],
        "reward_model": {"ground_truth": sample["answer"]},
        "extra_info": {
            "question": sample["question"],
            "images":   [sample["image_path"]],     # path or bytes, same as above
        },
    })
pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), "train.parquet")
```

For validation files, append an `idx` column whose values are unique per
question (any stable hash works) and write to a `*_idx.parquet`.

## Process labels for PRM training

The PRM itself is trained with an SFT objective on responses annotated with
sub-sentence error spans inside `<answer>...</answer>` blocks
(see `evaluate_process_verification` in
[`verl/utils/reward_score/hallu_token_reward_vstar.py`](../verl/utils/reward_score/hallu_token_reward_vstar.py)
for the exact expected format). The data construction pipeline for the PRM
itself is outside the scope of this RL training repo; refer to the paper for
details, or use the released `<YOUR_HF_USER>/perceval-prm-7b` weights.
