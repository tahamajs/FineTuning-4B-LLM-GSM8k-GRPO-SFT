# QLoRA + GRPO on Qwen/Qwen3-4B for openai/gsm8k

> **Two‑stage training pipeline (SFT ➜ GRPO)** built with **Unsloth**, **Transformers/PEFT**, and **TRL**.
> Trains a compact reasoning model using **QLoRA SFT** on `openai/gsm8k` (config: `main`), then **GRPO** with multi‑signal rewards that enforce *formatting* and *answer correctness*.

---

## ✨ Features

- **SFT (Supervised Fine‑Tuning)**

  - Loads **Qwen/Qwen3-4B** in **4‑bit** (QLoRA) with **Unsloth** for fast/low‑VRAM training
  - Custom **Qwen‑style chat template** with `<reasoning>` and `<answer>` blocks
  - Clean mapping from `openai/gsm8k` (“main”) into chat messages + ground‑truth final numeric answer
  - **PEFT/LoRA** with rank **64**, cosine LR, fused AdamW, bf16 when available
  - Handy loss‑curve plotting utilities
- **GRPO (Group Relative Policy Optimization)**

  - **Multiple reward functions** used together:
    - `r_strict`: well‑formed XML (`<reasoning>…</reasoning>`, `<answer>…</answer>`)
    - `r_soft`: relaxed format check
    - `r_xmlcount`: counts/weights the presence of required tags
    - `r_correctness`: compares the **last number** in the model’s output to the gold answer
    - Optional **length shaping** and **repeat‑penalty**
  - Grouped sampling with **NUM_GENERATIONS = 4** per prompt
  - Efficient dataloaders (pin/persistent workers), fused optimizers
- **Hub Integration**

  - Save LoRA adapters to `outputs_sft` / `outputs_grpo`
  - Push adapters to the Hugging Face Hub via `huggingface_hub.HfApi`

---

## 📁 Repo Layout

```
.
├─ main_sft_final.ipynb        # Stage 1: QLoRA SFT on openai/gsm8k
├─ main_grpo_final.ipynb       # Stage 2: GRPO with multi-reward training
├─ outputs_sft/                  # Saved SFT adapter (created after training)
├─ outputs_grpo/                 # Saved GRPO adapter (created after training)
└─ README.md                   # You are here
```

---

## 🚀 Quickstart

### 1) Environment

```bash
# Python 3.10+ recommended
pip install --upgrade pip

# Core stack
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu index if needed
pip install transformers==4.* peft datasets accelerate bitsandbytes trl
pip install "unsloth[colab-new]"  # fast model loading & QLoRA helpers
pip install scikit-learn pandas matplotlib huggingface_hub
```

> On Windows without a proper CUDA setup, `bitsandbytes` may fall back to CPU or fail. Consider Linux or Colab for best results.

### 2) (Optional) Login to the Hub

```python
from huggingface_hub import notebook_login
notebook_login()
```

### 3) Run SFT (Stage 1)

Open **`main_sft_final.ipynb`** and execute cells top‑to‑bottom.Default key settings:

- `MODEL_NAME = "Qwen/Qwen3-4B"`
- `DATASET_NAME = "openai/gsm8k"` (config: `main`)
- `MAX_SEQ_LENGTH = 1024`
- `LORA_RANK = 64`
- `per_device_train_batch_size = 10`
- `gradient_accumulation_steps = 2`
- `learning_rate = 9e-4`
- `num_train_epochs = 2`
- Outputs saved to **`outputs_sft`**

The notebook:

- Loads the base model in **4‑bit** and prepares it for **LoRA**
- Ensures a **Qwen chat template** and adds `<reasoning>/<answer>` supervision
- Trains with **SFTTrainer**
- Optionally **uploads** `outputs_sft` to your Hub repo

### 4) Run GRPO (Stage 2)

Open **`main_grpo_final.ipynb`** and execute.Key defaults detected:

- `MODEL_NAME = "Qwen/Qwen3-4B"` (loaded via Unsloth)
- `NUM_GENERATIONS = 4`
- Max lengths: prompt `512`, completion `700`
- `learning_rate = 5e-4` (cosine schedule, fused AdamW)
- Outputs saved to **`outputs_grpo`**

This stage:

- Reuses the SFT‑tuned model + LoRA
- Builds rewards: **strict/soft/XML‑count/correctness/len‑shaping/repeat‑penalty**
- Trains with **TRL.GRPOTrainer** using multiple generations per prompt

---

## 🧱 Data & Formatting

The code expects **GSM8K** samples and maps them to:

```python
example = {
  "messages": [
    {"role": "system", "content": "SYSTEM_PROMPT"},
    {"role": "user",   "content": "question_text"},
  ],
  "answer_text": "full_solution_rationale",   # original GSM8K "answer"
  "answer": "final_numeric_answer",           # extracted from the line starting with '####'
}
```

During SFT, the final training text is created by:

```text
<reasoning>
{answer_text}
</reasoning>
<answer>
The final answer is {answer}.
</answer>
```

During GRPO, **`r_correctness`** compares the **last number** in the model’s response to `example["answer"]`.XML‑related rewards encourage clean blocks and allow gentle penalties or bonuses for length and repetition.

> **Using a custom dataset?** Provide the same fields (`messages`, `answer_text`, `answer`) or adapt the mapping/formatting helpers in the notebooks.

---

## ⚙️ Important Hyperparameters (defaults)

| Stage | Key                         | Value                                                |
| ----: | --------------------------- | ---------------------------------------------------- |
|   SFT | `MODEL_NAME`              | `Qwen/Qwen3-4B`                                    |
|   SFT | `DATASET_NAME`            | `openai/gsm8k` (`main`)                          |
|   SFT | `LORA_RANK`               | `64`                                               |
|   SFT | `batch_size x grad_accum` | `10 x 2` (effective global batch scales with GPUs) |
|   SFT | `learning_rate`           | `9e-4`                                             |
|   SFT | `num_train_epochs`        | `2`                                                |
|  GRPO | `NUM_GENERATIONS`         | `4` (candidates per prompt)                        |
|  GRPO | `learning_rate`           | `5e-4`                                             |
|  GRPO | Lengths                     | prompt `512`, completion `700`                   |

Other goodies: `bf16` when available, TF32 matmul, fused AdamW, cosine LR, pin/persistent dataloader workers.

---

## 🧪 Evaluation & Inference (adapters)

**Load LoRA adapter for inference:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", device_map="auto", load_in_4bit=True)
tok  = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Replace with your trained adapter path or Hub repo
adapter_path = "outputs_grpo"  # or "outputs_sft"
model = PeftModel.from_pretrained(base, adapter_path)

prompt = "A train travels 120 km in 2 hours. What is its average speed?"
messages = [
    {"role": "system", "content": "You are a careful math tutor that answers with <reasoning> and <answer> blocks."},
    {"role": "user", "content": prompt},
]

def apply_qwen_template(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

inp = apply_qwen_template(tok, messages)
out = model.generate(**tok(inp, return_tensors="pt").to(model.device), max_new_tokens=300)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 📈 Logging & Plots

Both notebooks include small utilities to plot **loss curves** from the trainer logs.
If you want **Weights & Biases**, disable `WANDB_DISABLED` and set `report_to="wandb"` in the training args.

---

## 🧩 Tips & Troubleshooting

- **CUDA OOM**: reduce `batch_size`, `NUM_GENERATIONS`, or sequence lengths; keep 4‑bit loading on.
- **bf16 not used**: requires Ampere+ GPUs; the code falls back safely.
- **bitsandbytes on Windows**: consider WSL/Colab; CPU fallback is very slow.
- **GRPO “flat” losses**: check reward aggregation, ensure `answer` exists, and verify the **last‑number extraction** is working for your data.
- **Bad formatting**: XML rewards encourage `<reasoning>`/`<answer>`. Verify your chat template matches Qwen formatting.

---

## 🔐 License

This project is released under the **MIT License** (see `LICENSE`).
Make sure your dataset/model licenses are compatible with your use case.

---

## 🙏 Acknowledgements

- Unsloth for blazing‑fast 4‑bit loading and QLoRA helpers
- Hugging Face Transformers / PEFT / Datasets / TRL for the training stack
- `openai/gsm8k` authors and community
