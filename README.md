# local-llm

Run large language models (LLMs) **entirely on your own computer** — no cloud API keys required for inference.

This repository contains a single Python script, [`inference.py`](inference.py), that:

1. Downloads a model from [Hugging Face Hub](https://huggingface.co/models) into a local `models/` folder.
2. Loads the model on your hardware (NVIDIA, AMD, Intel, Apple Silicon, or CPU).
3. Streams generated text to your terminal in real time.
4. Runs an **interactive chat** by default (multi-turn conversation with history).
5. Prints a summary with elapsed time, token counts, and generation speed.

You do **not** need prior experience with AI or Python to follow this guide. Every step is explained below.

---

## Table of contents

- [What is this project?](#what-is-this-project)
- [What you need before starting](#what-you-need-before-starting)
- [Step 1: Install Python](#step-1-install-python)
- [Step 2: Download this project](#step-2-download-this-project)
- [Step 3: Create an isolated Python environment](#step-3-create-an-isolated-python-environment)
  - [Option A — `venv` (lightweight)](#option-a--venv-lightweight)
  - [Option B — Miniconda (recommended for GPU workflows)](#option-b--miniconda-recommended-for-gpu-workflows)
- [Step 4: Install dependencies](#step-4-install-dependencies)
- [Step 5: Install PyTorch for your hardware](#step-5-install-pytorch-for-your-hardware)
- [Step 6: Run your first inference](#step-6-run-your-first-inference)
- [Understanding the output](#understanding-the-output)
- [All command-line options](#all-command-line-options)
- [Choosing a model](#choosing-a-model)
  - [Model families on Hugging Face](#model-families-on-hugging-face)
  - [Pick by hardware (VRAM / RAM)](#pick-by-hardware-vram--ram)
  - [Gated models and licenses](#gated-models-and-licenses)
- [Platform setup guide (NVIDIA, AMD, Intel, Apple, CPU)](#platform-setup-guide-nvidia-amd-intel-apple-cpu)
  - [Identify your GPU (model, architecture, generation)](#identify-your-gpu-model-architecture-generation)
- [Gated / private models](#gated--private-models)
- [Troubleshooting](#troubleshooting)
- [FAQ for beginners](#faq-for-beginners)
- [Project layout](#project-layout)
- [License](#license)

---

## What is this project?

| Term | Plain-English meaning |
|------|----------------------|
| **LLM** | A Large Language Model — software trained on text that can answer questions, write code, summarize documents, etc. |
| **Hugging Face** | A website that hosts thousands of open models you can download. |
| **Inference** | Using a trained model to generate new text (as opposed to *training*, which teaches the model). |
| **Token** | A small piece of text (often a word fragment). Models read and write tokens, not whole words. |
| **Streaming** | Showing each piece of output as it is generated, like a typewriter, instead of waiting for the full answer. |

`inference.py` is a **single-file script**. You run it from the command line and pass a model name plus optional settings. It handles download, device selection, and printing results.

---

## What you need before starting

- A computer running **Windows**, **macOS**, or **Linux**.
- An internet connection (only for the first download of each model).
- At least **8 GB RAM** for small models; **16 GB+** recommended.
- Enough free disk space for the model (small models ≈ 1–3 GB; large models can be 10–80+ GB).
- **Python 3.11 or 3.12** recommended (**3.10+** may work; see `requirements.txt` notes), **or** Miniconda (see [Step 3 Option B](#option-b--miniconda-recommended-for-gpu-workflows)).

Optional but helpful:

- An NVIDIA, AMD, Intel, or Apple Silicon GPU for faster generation.
- A free [Hugging Face account](https://huggingface.co/join) if you want gated models (e.g. some Llama variants).

---

## Step 1: Install Python

You need **Python 3.11 or 3.12** for the best PyTorch wheel compatibility. Two common paths:

| Path | When to use |
|------|-------------|
| **A — System Python** (below) | Quick setup with `venv` |
| **B — Miniconda** | Skip installing Python separately — [go to Step 3 Option B](#option-b--miniconda-recommended-for-gpu-workflows) after downloading this repo |

### Windows

1. Go to [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
2. Download the latest **Python 3** installer.
3. Run the installer.
4. **Important:** check **“Add python.exe to PATH”** at the bottom of the first screen.
5. Click **Install Now**.

Verify in **Command Prompt** or **PowerShell**:

```bash
python --version
```

You should see something like `Python 3.12.x`.

### macOS

```bash
# If you have Homebrew:
brew install python@3.12

python3 --version
```

Or download the macOS installer from [python.org](https://www.python.org/downloads/macos/).

### Linux (Ubuntu / Debian example)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

---

## Step 2: Download this project

### Option A — Git (if you have Git installed)

```bash
git clone https://github.com/cw1997/local-llm.git
cd local-llm
```

### Option B — ZIP download

1. Open the repository page on GitHub.
2. Click **Code → Download ZIP**.
3. Extract the ZIP file.
4. Open a terminal in the extracted `local-llm` folder.

All commands below assume your terminal’s **current directory** is the project root (the folder that contains `inference.py`).

---

## Step 3: Create an isolated Python environment

You need an **isolated environment** so this project’s packages do not conflict with other Python projects on your system. Pick one approach below.

| Approach | Best for | Notes |
|----------|----------|-------|
| **Option A — `venv`** | Quick try-out, already have Python 3.11/3.12 | Built into Python; no extra installer |
| **Option B — Miniconda** | GPU workflows, multiple Python versions, data/ML stacks | **Recommended** if you plan to experiment with CUDA / ROCm / XPU wheels |

---

### Option A — `venv` (lightweight)

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your prompt usually shows `(.venv)`.

To leave the environment later:

```bash
deactivate
```

---

### Option B — Miniconda (recommended for GPU workflows)

[Miniconda](https://docs.anaconda.com/miniconda/) is a minimal installer for **Conda** — a package and environment manager widely used for machine-learning stacks. It lets you create a dedicated environment with a specific Python version, install PyTorch builds matched to your GPU, and keep everything separate from system Python.

#### B.1 — Install Miniconda

Download the installer for your OS from the official page:

**[https://docs.anaconda.com/miniconda/](https://docs.anaconda.com/miniconda/)**

| OS | Installer tips |
|----|----------------|
| **Windows** | Run the `.exe` installer. Optionally check “Add Miniconda3 to PATH” (or use **Anaconda Prompt** / **Miniconda Prompt** from the Start menu). |
| **macOS (Apple Silicon)** | Use the **arm64** (M1/M2/M3/M4) installer. |
| **macOS (Intel Mac)** | Use the **x86_64** installer — inference will be **CPU-only** (no MPS). |
| **Linux** | Use the latest Linux x86_64 installer; on ARM64 Linux, use the matching arm64 build if available. |

Verify after installation (open a **new** terminal):

```bash
conda --version
```

You should see something like `conda 25.x.x`.

#### B.2 — Create a project environment

From the project root (`local-llm/` folder that contains `inference.py`):

```bash
conda create -n local-llm python=3.12 -y
conda activate local-llm
```

Use **Python 3.11** instead if a specific PyTorch wheel you need only ships for 3.11:

```bash
conda create -n local-llm python=3.11 -y
conda activate local-llm
```

Your prompt should show `(local-llm)`.

#### B.3 — Install dependencies inside the environment

Still in the activated `local-llm` environment:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- On **NVIDIA CUDA** machines, `requirements.txt` installs **PyTorch cu128** by default.
- On **other platforms**, see [Step 5](#step-5-install-pytorch-for-your-hardware) and the [Platform setup guide](#platform-setup-guide-nvidia-amd-intel-apple-cpu) — you may need to install a different PyTorch build *before* the remaining packages.

#### B.4 — Verify hardware detection

```bash
python inference.py --list-devices
```

Example on an NVIDIA machine:

```text
Detected inference backends on this machine:
  - cuda
  - cpu
```

If you expect a GPU backend but only see `cpu`, use [Identify your GPU](#identify-your-gpu-model-architecture-generation) to confirm your hardware, then jump to the matching section in the [Platform setup guide](#platform-setup-guide-nvidia-amd-intel-apple-cpu).

#### B.5 — Run inference

Interactive chat (default):

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
```

Single-shot test:

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --no-interactive --prompt "Hello!"
```

Force a specific backend while debugging, e.g.:

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device cuda --no-interactive --prompt "GPU test"
```

#### B.6 — Daily workflow (Conda cheat sheet)

```bash
# Enter project folder
cd path/to/local-llm

# Activate environment (do this every new terminal session)
conda activate local-llm

# Run inference …
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct

# Leave environment
conda deactivate
```

Remove the environment entirely (does **not** delete downloaded models in `models/`):

```bash
conda deactivate
conda env remove -n local-llm -y
```

#### B.7 — Conda + platform-specific PyTorch (common pattern)

When the default `requirements.txt` CUDA wheel is **not** right for your hardware, use this pattern inside `conda activate local-llm`:

```bash
# 1) Install PyTorch for YOUR platform (examples — see platform guide for details)
# NVIDIA Blackwell / RTX 50xx:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (MPS):
pip install torch torchvision

# Intel GPU (XPU):
pip install torch --index-url https://download.pytorch.org/whl/xpu

# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest of this project’s dependencies
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

# 3) Verify and run
python inference.py --list-devices
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
```

#### B.8 — Conda on Windows: SSL certificate gotcha

Conda sometimes sets `SSL_CERT_FILE` to a path that does not exist, which breaks Hugging Face downloads in **Git Bash**. `inference.py` auto-clears invalid cert paths and falls back to **certifi**, but if downloads still fail:

```bash
conda activate local-llm
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

Or run commands from **Anaconda Prompt** / **PowerShell** instead of Git Bash.

---

## Step 4: Install dependencies

With your environment activated (`conda activate local-llm` **or** `source .venv/bin/activate` / `.venv\Scripts\activate`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- **PyTorch** (CUDA 12.8 build by default — see Step 5) — deep learning runtime.
- **transformers** — loads Hugging Face models.
- **accelerate** — helps spread large models across GPUs.
- **huggingface_hub** — downloads models from the Hub.
- **certifi** — SSL certificates (also used to fix broken cert paths on some Windows setups).
- **sentencepiece**, **protobuf**, **safetensors** — common tokenizer / model file support.

---

## Step 5: Install PyTorch for your hardware

`requirements.txt` is configured for **NVIDIA GPUs** out of the box: it pulls PyTorch from the **CUDA 12.8** wheel index (`cu128`), which supports recent cards such as RTX 40xx and 50xx (Blackwell) series. If you have a supported NVIDIA GPU, the Step 4 command alone is usually enough.

**Not on NVIDIA CUDA?** Do **not** blindly run `pip install -r requirements.txt` first — install the PyTorch build that matches your hardware, then install the remaining packages. The detailed per-vendor, per-generation instructions are in the **[Platform setup guide](#platform-setup-guide-nvidia-amd-intel-apple-cpu)** below.

Quick fallback for any non-default platform after installing the correct `torch`:

```bash
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors
```

Official PyTorch install selector (copy the command for your OS / hardware):

**[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)**

### Quick reference

| Hardware | Typical install approach |
|----------|--------------------------|
| **NVIDIA GPU (CUDA)** | Default `pip install -r requirements.txt` (cu128). See [NVIDIA](#nvidia-gpus-cuda-backend) for older generations. |
| **Apple Silicon (M1–M4)** | Native macOS PyTorch with **MPS**. See [Apple](#apple-silicon-mps-backend). |
| **AMD GPU (Linux ROCm)** | ROCm PyTorch wheels. See [AMD](#amd-gpus-rocm-on-linux--directml-on-windows). |
| **Intel GPU (XPU)** | PyTorch `xpu` wheels. See [Intel](#intel-gpus-xpu-backend). |
| **AMD / Intel GPU on Windows** | `torch-directml` + `--device dml`. See [AMD Windows](#amd--intel-gpus-on-windows-directml). |
| **CPU only** | PyTorch CPU wheels. See [CPU-only](#cpu-only-inference). |

If you already ran `pip install -r requirements.txt` but need a different PyTorch build, reinstall `torch` for your platform, then refresh the rest with the command above or `pip install -r requirements.txt` again.

### Check what your machine supports

```bash
python inference.py --list-devices
```

Not sure which GPU you have? See [Identify your GPU](#identify-your-gpu-model-architecture-generation) for `nvidia-smi`, `lspci`, `system_profiler`, and other commands.

Example output:

```text
Detected inference backends on this machine:
  - cuda
  - cpu
```

---

## Step 6: Run your first inference

Try a **small** instruct model (fast download, works on modest hardware):

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
```

On macOS/Linux, if `python` is not found, use `python3` instead.

After the model loads, you enter **interactive chat** mode. Type a question at the `You>` prompt and press Enter. The model streams its reply, then waits for your next message.

**Interactive commands**

| Command | Action |
|---------|--------|
| `/exit`, `/quit`, `exit`, or `quit` | End the session |
| `/clear` or `clear` | Reset conversation history (useful if replies get slow or off-topic) |

### What happens

1. The script prints which **device** it will use (GPU or CPU).
2. It downloads model files into `./models/` (skipped on later runs if already cached).
3. It loads weights into memory (can take 10 seconds to several minutes).
4. It starts an interactive chat loop — each turn **streams** the answer token by token.
5. After each reply it prints **statistics** (prompt tokens, generation time, tokens per second).

### Single-shot mode (one prompt, then exit)

To run a single question without entering chat, pass `--no-interactive`:

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --no-interactive --prompt "What is the capital of France?"
```

### Another example with a system prompt

```bash
python inference.py ^
  --model-id Qwen/Qwen2.5-0.5B-Instruct ^
  --system-prompt "You are a concise science tutor." ^
  --no-interactive ^
  --prompt "Explain photosynthesis in three sentences." ^
  --max-new-tokens 200 ^
  --temperature 0.3
```

On macOS/Linux, replace `^` line continuations with `\`:

```bash
python inference.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --system-prompt "You are a concise science tutor." \
  --no-interactive \
  --prompt "Explain photosynthesis in three sentences." \
  --max-new-tokens 200 \
  --temperature 0.3
```

For interactive chat with a system prompt, omit `--no-interactive` and `--prompt`:

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --system-prompt "You are a concise science tutor."
```

---

## Understanding the output

### Device banner (before inference)

```text
============================================================
Local LLM inference
============================================================
Selected device : cuda:0 (NVIDIA GeForce RTX 3060)
Backend         : cuda
Weight dtype    : float16
Model ID        : Qwen/Qwen2.5-0.5B-Instruct
============================================================
```

This confirms **where** the model runs. Faster devices (GPUs) yield higher tokens/s.

### Per-turn info (interactive mode)

Before each reply in chat mode, the script prints the **prompt token count** for the full conversation so far (grows as history accumulates). Use `/clear` if the count gets too large and replies slow down.

### Streaming section

```text
--- Model output (streaming) ---
Paris is the capital of France.
--- End of output ---
```

Text appears gradually as the model generates it.

### Summary (after inference)

```text
============================================================
Inference summary
============================================================
Model ID           : Qwen/Qwen2.5-0.5B-Instruct
Device             : cuda:0 (NVIDIA GeForce RTX 3060)
Backend            : cuda
Prompt tokens      : 12
Generated tokens   : 9
Total tokens       : 21
Generation time    : 0.85 s
Generation speed   : 10.59 tokens/s
============================================================
```

| Field | Meaning |
|-------|---------|
| **Prompt tokens** | How many tokens your input used |
| **Generated tokens** | How many tokens the model produced |
| **Generation time** | Wall-clock seconds for generation only |
| **Generation speed** | Generated tokens ÷ time (higher is faster) |

---

## All command-line options

Only **`--model-id`** is required. Everything else has a sensible default.

| Option | Default | Description |
|--------|---------|-------------|
| `--model-id` | *(required)* | Hugging Face repo id, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| `--interactive` / `--no-interactive` | Interactive on | Multi-turn chat (default) or single-shot with `--prompt` |
| `--prompt` | Introduce-yourself greeting | Question for single-shot mode (only with `--no-interactive`) |
| `--system-prompt` | None | Optional system role text for chat-tuned models |
| `--models-dir` | `models` | Local folder for downloaded weights |
| `--revision` | `main` | Git branch / tag / commit on the Hub |
| `--max-new-tokens` | `256` | Maximum length of the generated reply |
| `--temperature` | `0.7` | Randomness (0 = focused, 1+ = more creative) |
| `--top-p` | `0.9` | Nucleus sampling cutoff |
| `--top-k` | `50` | Limits sampling to top K tokens |
| `--repetition-penalty` | `1.05` | Reduces repeated phrases (>1.0 enables penalty) |
| `--do-sample` / `--no-do-sample` | Sampling on | Turn off for greedy (deterministic) decoding |
| `--seed` | None | Fixed random seed for reproducibility |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `xpu`, `dml`, or `cpu` |
| `--dtype` | `auto` | `auto`, `float16`, `bfloat16`, or `float32` |
| `--trust-remote-code` | off | Required for some custom model architectures |
| `--token` | env var | Hugging Face token (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`) |
| `--force-download` | off | Re-download even if files exist |
| `--low-cpu-mem-usage` / `--no-low-cpu-mem-usage` | on | Memory-efficient loading |
| `--list-devices` | off | Print detected backends and exit |

View full help anytime:

```bash
python inference.py --help
```

---

## Choosing a model

Browse and filter models at [https://huggingface.co/models](https://huggingface.co/models). For chat with `inference.py`, use **text-generation** models that are **decoder-only causal LMs** with an **Instruct** or **Chat** fine-tune.

### Quick rules

| Rule | Why |
|------|-----|
| Prefer **`-Instruct`** or **`-Chat`** suffix | Tuned for Q&A and dialogue; base models often ignore instructions |
| Match model size to your **VRAM / RAM** | See [Pick by hardware](#pick-by-hardware-vram--ram) |
| Check the **model card** on Hugging Face | License, languages, context length, hardware notes |
| Start small, then scale up | `Qwen/Qwen2.5-0.5B-Instruct` is the fastest sanity check |
| Use `--trust-remote-code` only when the card says so | Some architectures need custom code from the Hub |

Chat-tuned models use each tokenizer’s **chat template** when available; otherwise `inference.py` falls back to a simple plain-text format.

### Good starter models (small, chat-ready)

Best first downloads — fast, low disk/RAM, work on most hardware:

| Model ID | Params | Approx. disk | VRAM (fp16) | License | Notes |
|----------|--------|--------------|-------------|---------|-------|
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | ~1 GB | 2 GB | Apache 2.0 | Default demo model; very fast |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | ~3 GB | 4 GB | Apache 2.0 | Better quality than 0.5B |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1B | ~2 GB | 3 GB | Llama 3.2 | **Gated** — HF token required |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 3B | ~6 GB | 6 GB | Llama 3.2 | **Gated**; strong small Llama |
| [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | 2B | ~5 GB | 5 GB | Gemma | **Gated**; `-it` = instruction-tuned |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | ~3 GB | 4 GB | Apache 2.0 | HF’s compact instruct model |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | ~7 GB | 8 GB | MIT | Strong reasoning for its size |
| [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | ~7 GB | 8 GB | MIT | Updated Phi mini series |

```bash
# Try any starter model (replace model-id as needed)
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
python inference.py --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct --no-interactive --prompt "Explain recursion briefly."
```

### Model families on Hugging Face

Below are the **most popular open LLM families** for local inference with Transformers. Sizes marked **Gated** need a Hugging Face account, license acceptance, and `HF_TOKEN` (see [Gated models](#gated--private-models)).

#### Qwen (Alibaba)

Widely used for multilingual chat (strong **Chinese + English**), coding, and math. **Apache 2.0** on most Qwen2.5 weights.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Ultra-light | Fastest smoke test |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Light | Sweet spot on 8 GB machines |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Light | Noticeably smarter than 1.5B |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Mainstream | Default “real” local LLM for 16 GB VRAM |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Large | Needs 24 GB+ VRAM |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | Large | 48 GB+ VRAM or multi-GPU |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | XL | Datacenter / multi-GPU only |
| `Qwen/Qwen3-1.7B` | 1.7B | Light | Qwen3 generation; conversational |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | Light | Explicit Qwen3 instruct build |
| `Qwen/Qwen3-8B` | 8B | Mainstream | Qwen3 flagship small; tagged conversational |
| `Qwen/Qwen3-32B` | 32B | Large | High-end Qwen3; 48 GB+ VRAM |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Code | Code generation / completion focus |

```bash
python inference.py --model-id Qwen/Qwen2.5-7B-Instruct --system-prompt "You are a helpful coding assistant."
```

#### Llama (Meta)

Industry-standard open weights. **Llama 3.x** uses the **Llama Community License** — gated on Hugging Face.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Ultra-light | Smallest gated Llama |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Light | Good mobile / laptop model |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Mainstream | Most popular mid-size Llama |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | XL | Multi-GPU; excellent but heavy |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | XL | Updated 70B; top open Llama tier |

```bash
export HF_TOKEN=hf_your_token_here
python inference.py --model-id meta-llama/Llama-3.1-8B-Instruct --token $HF_TOKEN
```

#### Gemma (Google)

Compact, efficient models. Instruction-tuned variants use **`-it`** suffix. Most are **gated**.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `google/gemma-2-2b-it` | 2B | Light | Efficient 2B instruct |
| `google/gemma-2-9b-it` | 9B | Mainstream | Strong 9B general model |
| `google/gemma-2-27b-it` | 27B | Large | High quality; needs 48 GB+ VRAM |
| `google/gemma-3-1b-it` | 1B | Ultra-light | Gemma 3 generation |
| `google/gemma-3-4b-it` | 4B | Light | Multimodal variants exist — use **text-only** cards for this script |
| `google/gemma-3-12b-it` | 12B | Large | Gemma 3 mid-large |

#### Mistral & Ministral (Mistral AI)

European lab; strong **English** and **code**. Apache 2.0 on many releases.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Mainstream | Classic efficient 7B |
| `mistralai/Ministral-8B-Instruct-2410` | 8B | Mainstream | Newer compact Mistral |
| `mistralai/Mistral-Small-Instruct-2409` | 22B | Large | “Small” in name, large in size |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 47B MoE | Large | Mixture-of-experts; needs more VRAM |
| `mistralai/Devstral-Small-2505` | 24B | Large | Code-focused instruct model |

MoE models (Mixtral) load more weights than active parameters but still need substantial VRAM.

#### Microsoft Phi

Small models with strong **reasoning per parameter**; good on laptops.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Light | 4k context; very popular |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Light | Improved mini |
| `microsoft/Phi-3-medium-4k-instruct` | 14B | Large | Heavier Phi tier |
| `microsoft/Phi-4-mini-instruct` | 3.8B | Light | Phi-4 generation |
| `microsoft/phi-4` | 14B | Large | Full Phi-4 (check card for chat usage) |

#### DeepSeek

Known for **reasoning** and **coding**. Distilled R1 models are popular for local use.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | Light | Smallest R1 distill |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | Mainstream | R1-style reasoning distill |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | Mainstream | Llama-based R1 distill |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | Large | Mid-large distill |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B | Large | High-end distill |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 70B | XL | Needs multi-GPU |
| `deepseek-ai/DeepSeek-V2.5` | 236B MoE | XL | Too large for most local setups |
| `deepseek-ai/DeepSeek-V3` | 671B MoE | XL | API / cluster scale only |

For `inference.py`, start with **Distill** checkpoints (7B / 8B), not full V2/V3 MoE.

#### Yi (01.AI)

Bilingual **Chinese + English** models.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `01-ai/Yi-1.5-6B-Chat` | 6B | Light | Efficient chat model |
| `01-ai/Yi-1.5-9B-Chat` | 9B | Mainstream | Strong mid-size |
| `01-ai/Yi-1.5-34B-Chat` | 34B | Large | High quality; 48 GB+ VRAM |

#### SmolLM (Hugging Face)

Tiny models for education, edge devices, and fast experiments.

| Model ID | Params | Tier | Highlights |
|----------|--------|------|------------|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | Tiny | Runs almost anywhere |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | Tiny | Still sub-1B |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Light | Best SmolLM2 quality |

#### Other notable families

| Family | Example model ID | Notes |
|--------|------------------|-------|
| **TinyLlama** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Community 1.1B chat baseline |
| **Zephyr** (H4 fine-tune) | `HuggingFaceH4/zephyr-7b-beta` | Popular Mistral-7B chat fine-tune |
| **OpenChat** | `openchat/openchat-3.5-0106` | Strong conversational fine-tune |
| **Code-focused** | `Qwen/Qwen2.5-Coder-7B-Instruct`, `codellama/CodeLlama-7b-Instruct-hf` | Programming tasks; weaker at open chat |
| **Falcon** | `tiiuae/falcon-7b-instruct` | Older but still referenced |
| **OLMo** | `allenai/OLMo-7B-Instruct-hf` | Open research LM from Allen AI |

### Pick by hardware (VRAM / RAM)

Rough **fp16** guidance for `inference.py` (no quantization). For CPU, treat “VRAM” as **system RAM**.

| Your hardware | Suggested param range | Example models |
|---------------|----------------------|----------------|
| 4–6 GB VRAM / 8 GB RAM | ≤1.5B | `Qwen2.5-0.5B`, `SmolLM2-1.7B`, `Llama-3.2-1B` |
| 8 GB VRAM / 16 GB RAM | ≤3B | `Qwen2.5-3B`, `Llama-3.2-3B`, `Phi-3-mini` |
| 12 GB VRAM | ≤7–8B | `Qwen2.5-7B`, `Llama-3.1-8B`, `Mistral-7B` |
| 16 GB VRAM | ≤9–14B | `Qwen2.5-14B`, `gemma-2-9b-it`, `Phi-3-medium` |
| 24 GB VRAM | ≤14–22B | `Qwen2.5-14B`, `Mistral-Small`, `Yi-34B` (tight) |
| 48 GB+ / multi-GPU | 32B–70B | `Qwen2.5-32B`, `Llama-3.1-70B`, `DeepSeek-R1-Distill-32B` |

If a model does not fit:

1. Pick a smaller checkpoint from the same family (e.g. 7B → 3B).
2. Lower `--max-new-tokens` and use `/clear` often in chat.
3. Force `--device cpu` with a ≤1.5B model as last resort.

> **Note:** This script loads full-precision (fp16/bf16/fp32) weights via Transformers. For **GGUF / quantized** models (Q4_K_M, etc.), use tools like [llama.cpp](https://github.com/ggml-org/llama.cpp) instead — they are not loaded by `inference.py` as-is.

### Gated models and licenses

| License type | Examples | What you need |
|--------------|----------|---------------|
| **Apache 2.0 / MIT** | Qwen2.5, Mistral, SmolLM, Phi | Usually download without login |
| **Llama Community License** | Llama 3.x | HF account + accept license + token |
| **Gemma license** | Gemma 2 / 3 | HF account + accept license + token |
| **Custom / research** | Some fine-tunes | Read model card carefully |

Always read the **model card** on Hugging Face before commercial use. Weights are **not** covered by this repo’s LICENSE.

### Browse Hugging Face like a pro

On [huggingface.co/models](https://huggingface.co/models), useful filters:

- **Task:** `Text Generation`
- **Library:** `Transformers`
- **Sort:** `Trending` or `Most downloads`
- **Search examples:** `qwen2.5 instruct`, `llama 3.1 8b`, `mistral instruct`

Official task-specific leaderboards (coding, math, chat) live on the Hub — search “benchmark” datasets or visit model cards’ **Evaluation** sections. For automated leaderboard lookup, Hugging Face’s API and `hf` CLI can query benchmark datasets (see [Hugging Face docs](https://huggingface.co/docs)).

### Models that may need extra flags

| Situation | What to do |
|-----------|------------|
| Model card says “custom architecture” | Add `--trust-remote-code` |
| Gated model | Set `HF_TOKEN` or `--token` |
| Very slow first reply | Normal — weights are loading; later turns are faster |
| Multimodal-only (vision/audio) | Pick a **text-only** `-Instruct` variant; avoid `*-VL-*` vision models in this script |

```bash
python inference.py --model-id <org/model> --trust-remote-code --no-interactive --prompt "Hello"
```

### Size vs. quality (summary)

- **Smaller models** (0.5B–3B): fastest, lowest RAM, simpler answers, more hallucination on hard tasks.
- **Mid models** (7B–9B): best balance for most desktop GPUs.
- **Large models** (14B–70B): best reasoning and instruction-following; need high VRAM or multi-GPU.

If generation is slow or you run out of memory, pick a smaller model from the tables above or see [Identify your GPU](#identify-your-gpu-model-architecture-generation) and [VRAM / RAM vs model size](#vram--ram-vs-model-size-rule-of-thumb).

---

## Platform setup guide (NVIDIA, AMD, Intel, Apple, CPU)

This section is the **detailed reference** for every major inference backend that `inference.py` supports. All examples assume you have already:

1. Cloned this repo and `cd local-llm`
2. Created and activated an environment (`conda activate local-llm` or `venv`)
3. Installed the correct **PyTorch** build for your hardware (then the other dependencies from Step 4)

Use `python inference.py --list-devices` to see what your current PyTorch install exposes.

### How `inference.py` picks a device (`--device auto`)

Priority order when you do not pass `--device`:

1. **CUDA** — NVIDIA GPUs, and AMD GPUs on Linux with a ROCm PyTorch build (ROCm exposes GPUs through PyTorch’s CUDA API).
2. **MPS** — Apple Silicon Macs (Metal Performance Shaders).
3. **XPU** — Intel discrete / integrated Arc graphics with PyTorch XPU wheels.
4. **DirectML (`dml`)** — Windows GPUs via `torch-directml` (AMD / Intel / some NVIDIA on Windows).
5. **CPU** — always available as fallback.

Override anytime: `--device cuda`, `--device mps`, `--device xpu`, `--device dml`, or `--device cpu`.

| Vendor / scenario | `--device` flag | PyTorch package you need | OS notes |
|-------------------|-----------------|--------------------------|----------|
| NVIDIA discrete GPU | `cuda` (auto) | `cu118` / `cu124` / `cu128` wheel | Windows, Linux |
| AMD discrete GPU (ROCm) | `cuda` (auto) | ROCm PyTorch wheel | Linux primary; Windows ROCm improving |
| Apple M1–M4 | `mps` (auto) | macOS arm64 PyTorch | macOS only |
| Intel Arc / Core Ultra iGPU | `xpu` (auto) | `xpu` wheel from pytorch.org | Windows 11, Linux |
| AMD / Intel on Windows (no ROCm) | `dml` | `torch-directml` | Windows |
| No usable GPU / old hardware | `cpu` | `cpu` wheel | Everywhere |

### Identify your GPU (model, architecture, generation)

Not sure which vendor or generation you have? Run the commands below **before** picking a PyTorch wheel. Then match the model name to the tables in [NVIDIA](#nvidia-gpus-cuda-backend), [Apple](#apple-silicon-mps-backend), [AMD](#amd-gpus-rocm-on-linux--directml-on-windows), or [Intel](#intel-gpus-xpu-backend).

#### Step 1 — List every graphics device (OS-level)

**Windows (Command Prompt)**

```bat
wmic path win32_VideoController get Name,AdapterRAM,DriverVersion
```

**Windows (PowerShell)**

```powershell
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
```

**Linux**

```bash
lspci | grep -iE 'vga|3d|display'
# More detail:
lspci -v | grep -iE 'vga|3d|display' -A 12
```

**macOS**

```bash
# GPU / display info
system_profiler SPDisplaysDataType

# Apple Silicon chip (M1 / M2 / M3 / M4 …)
system_profiler SPHardwareDataType | grep -E 'Chip|Model Name|Memory'
```

GUI alternative on Windows: press `Win + R`, type `dxdiag`, open the **Display** tab.

#### Step 2 — Vendor-specific detail commands

##### NVIDIA (model, driver, VRAM, compute capability)

Requires [NVIDIA drivers](https://www.nvidia.com/drivers) installed. Works on Windows and Linux.

```bash
# Summary table
nvidia-smi

# GPU name + compute capability (architecture version) + driver + VRAM
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv

# Short list of GPU names
nvidia-smi -L
```

Example `compute_cap` → architecture mapping:

| `compute_cap` | NVIDIA architecture | Example products |
|---------------|---------------------|------------------|
| `12.0` | Blackwell | RTX 5090, 5080, 5070 Ti, 5060 Ti |
| `8.9` | Ada Lovelace | RTX 4090, 4080, 4070, 4060 |
| `8.6` / `8.0` | Ampere | RTX 3090, 3080, 3060; A100 |
| `7.5` | Turing | RTX 2080 Ti, 2060; GTX 1660 Ti; T4 |
| `7.0` | Volta | Tesla V100, Titan V |
| `6.1` / `6.0` | Pascal | GTX 1080 Ti, 1070, 1060; P100 |
| `5.2` / `5.3` | Maxwell | GTX 980, 970, 960 |

After PyTorch + CUDA are installed, double-check from Python:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'  [{i}]', torch.cuda.get_device_name(i), 'capability', torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]"
```

##### AMD (model, ROCm architecture `gfx*`)

**Linux — ROCm tools** (install ROCm / driver first):

```bash
# Product name + VRAM (similar to nvidia-smi)
rocm-smi

# Architecture identifier (e.g. gfx1100 = RDNA 3, gfx1200 = RDNA 4)
rocminfo | grep -E 'Marketing Name|Name:|gfx'

# Quick grep for gfx target
rocminfo | grep gfx
```

Common `gfx` codes for consumer Radeon:

| `gfx` ID | Architecture | Example products |
|----------|--------------|------------------|
| `gfx1200`, `gfx1201` | RDNA 4 | RX 9070 XT, 9060 XT |
| `gfx1100`, `gfx1101`, `gfx1102` | RDNA 3 | RX 7900 XTX/XT, 7800 XT, 7700 XT |
| `gfx1030`, `gfx1031` | RDNA 2 | RX 6900 XT, 6800 XT, 6700 XT |
| `gfx1010` | RDNA 1 | RX 5700 XT |

With ROCm PyTorch installed:

```bash
python -c "import torch; print('available:', torch.cuda.is_available()); print('name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Windows** — ROCm tools may be unavailable; use OS listing + DirectML path:

```powershell
Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'AMD|Radeon' } | Select Name, DriverVersion
```

##### Intel (discrete Arc vs integrated)

**Windows (PowerShell)**

```powershell
Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' } | Select Name, DriverVersion
```

**Linux**

```bash
lspci | grep -i vga
# Intel GPU usage monitor (if intel-gpu-tools installed):
sudo intel_gpu_top -L
```

Look for names containing **Arc** (discrete A/B series), **Iris Xe**, or **Core Ultra** with Arc branding.

With PyTorch XPU installed:

```bash
python -c "import torch; xpu=getattr(torch,'xpu',None); print('XPU available:', xpu.is_available() if xpu else False); print('device:', xpu.get_device_name(0) if xpu and xpu.is_available() else 'N/A')"
```

##### Apple Silicon (chip generation)

```bash
# Chip name: Apple M1, M2 Pro, M3 Max, M4, etc.
sysctl -n machdep.cpu.brand_string 2>/dev/null || system_profiler SPHardwareDataType | grep Chip

# Unified memory size (GB) — important for model size limits
system_profiler SPHardwareDataType | grep Memory
```

| Chip name pattern | Generation | Release era |
|-------------------|------------|-------------|
| Apple M1 / M1 Pro / Max / Ultra | M1 | 2020–2021 |
| Apple M2 / M2 Pro / Max / Ultra | M2 | 2022 |
| Apple M3 / M3 Pro / Max | M3 | 2023 |
| Apple M4 / M4 Pro / Max | M4 | 2024–2025 |

With PyTorch installed on macOS:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

#### Step 3 — Map model name → this project’s backend

| If you see … | Vendor | Likely generation | Use backend | Jump to |
|--------------|--------|-------------------|-------------|---------|
| GeForce RTX 50xx, RTX PRO Blackwell | NVIDIA | Blackwell | `cuda` + cu128 | [NVIDIA](#nvidia-gpus-cuda-backend) |
| GeForce RTX 40xx | NVIDIA | Ada Lovelace | `cuda` + cu128 | [NVIDIA](#nvidia-gpus-cuda-backend) |
| GeForce RTX 30xx | NVIDIA | Ampere | `cuda` | [NVIDIA](#nvidia-gpus-cuda-backend) |
| GeForce RTX 20xx / GTX 16xx | NVIDIA | Turing | `cuda` + cu118/cu124 | [NVIDIA](#nvidia-gpus-cuda-backend) |
| GeForce GTX 10xx | NVIDIA | Pascal | `cuda` + cu118 | [NVIDIA](#nvidia-gpus-cuda-backend) |
| GeForce GTX 9xx or older | NVIDIA | Maxwell / older | Often `cpu` | [CPU](#cpu-only-inference) |
| Radeon RX 9xxx / 7xxx (Linux) | AMD | RDNA 4 / 3 | `cuda` (ROCm) | [AMD](#amd-gpus-rocm-on-linux--directml-on-windows) |
| Radeon RX 6xxx (Linux) | AMD | RDNA 2 | ROCm or `cpu` | [AMD](#amd-gpus-rocm-on-linux--directml-on-windows) |
| Radeon / AMD (Windows) | AMD | various | `dml` | [AMD Windows](#amd--intel-gpus-on-windows-directml) |
| Intel Arc A770 / B580 / etc. | Intel | Alchemist / Battlemage | `xpu` or `dml` | [Intel](#intel-gpus-xpu-backend) |
| Intel Iris Xe / UHD | Intel | integrated | `xpu` / `dml` / `cpu` | [Intel](#intel-gpus-xpu-backend) |
| Apple M1 / M2 / M3 / M4 | Apple | Apple Silicon | `mps` | [Apple](#apple-silicon-mps-backend) |
| No discrete GPU / virtual machine | — | — | `cpu` | [CPU](#cpu-only-inference) |

#### Step 4 — What PyTorch / inference.py can actually use

OS detection tells you **what hardware exists**; PyTorch tells you **what is usable right now**:

```bash
# Backends this project can select
python inference.py --list-devices

# Full environment report (CUDA / ROCm / MPS / CPU versions)
python -m torch.utils.collect_env
```

If hardware is present but the backend is missing (e.g. GPU listed in `nvidia-smi` but `--list-devices` shows only `cpu`), your PyTorch wheel does not match the GPU — reinstall per the vendor section above.

### VRAM / RAM vs model size (rule of thumb)

These are **approximate** minimums for **FP16** weights at inference. Real usage depends on context length, tokenizer, and overhead.

| Model size | GPU VRAM (comfortable) | System RAM if CPU-only |
|------------|------------------------|-------------------------|
| 0.5B–1B | 2–4 GB | 4–8 GB |
| 3B | 6–8 GB | 8–12 GB |
| 7B | 12–16 GB | 16–24 GB |
| 13B+ | 24 GB+ | 32 GB+ |

If you run out of memory: use a smaller model, lower `--max-new-tokens`, set `--dtype float16`, or fall back to `--device cpu` with a tiny model.

---

### NVIDIA GPUs (CUDA backend)

**Backend in this project:** `cuda`  
**What you need:** recent **NVIDIA driver** only — PyTorch CUDA wheels bundle their own CUDA runtime. You do **not** need to install the full CUDA Toolkit for basic inference.

> **Don't know your exact card?** Run `nvidia-smi` and see [Identify your GPU](#identify-your-gpu-model-architecture-generation).

PyTorch wheels are labeled by CUDA *flavor*: `cu118`, `cu124`, `cu126`, `cu128`, etc. Pick a wheel **equal to or newer than** what your GPU generation needs. When in doubt, use the latest stable `cu128` (this repo’s default).

#### NVIDIA generations at a glance

| Generation | Example products | Architecture (compute capability) | Recommended PyTorch CUDA wheel | Notes |
|------------|------------------|-------------------------------------|-------------------------------|-------|
| **Blackwell** | RTX 5090, 5080, 5070 Ti, 5060 Ti; RTX PRO Blackwell | `sm_120` | **cu128**, PyTorch **2.7+** | RTX 50-series **requires** cu128; older cu124 wheels lack `sm_120` kernels |
| **Ada Lovelace** | RTX 4090, 4080, 4070 Ti, 4060; RTX 6000 Ada | `sm_89` | cu124 or **cu128** | Default `requirements.txt` (cu128) works well |
| **Ampere** | RTX 3090, 3080, 3070, 3060; A100, A40, RTX A6000 | `sm_86`, `sm_80` | cu118, cu124, or cu128 | Workhorse generation; all modern wheels supported |
| **Turing** | RTX 2080 Ti, 2070, 2060; GTX 1660 Ti; T4 | `sm_75` | cu118+ | Still well supported |
| **Volta** | Tesla V100, Titan V | `sm_70` | cu118+ | Datacenter / legacy enthusiast |
| **Pascal** | GTX 1080 Ti, 1070, 1060; P100 | `sm_61`, `sm_60` | cu118 | Supported but aging; prefer smaller models |
| **Maxwell** | GTX 980, 970, 960, 750 Ti | `sm_52`, `sm_53` | Often **CPU fallback** | Very old; PyTorch support is limited — use `--device cpu` + tiny models |

#### Miniconda + NVIDIA: full example (RTX 40xx / 50xx)

```bash
conda create -n local-llm python=3.12 -y
conda activate local-llm
cd local-llm

# Default path — cu128 (good for RTX 40xx and 50xx)
pip install --upgrade pip
pip install -r requirements.txt

python inference.py --list-devices
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct
```

#### Older NVIDIA (GTX 10xx / RTX 20xx) — cu118 example

If cu128 causes issues on very old drivers, try CUDA 11.8 wheels:

```bash
conda activate local-llm
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device cuda
```

#### Blackwell (RTX 50xx) checklist

1. Update to the **latest NVIDIA driver** for your RTX 50-series card.
2. Install **PyTorch 2.7+** with **cu128** (this repo’s `requirements.txt` targets this).
3. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

4. If you see `sm_120 is not compatible`, you likely have an older `cu124` or CPU-only torch — reinstall cu128.

#### Multi-GPU NVIDIA

When `--device auto` detects **multiple** CUDA devices, `inference.py` loads the model with `device_map="auto"` (via `accelerate`) to spread layers across GPUs. Useful for 7B+ models that do not fit on one card.

```bash
python inference.py --model-id Qwen/Qwen2.5-7B-Instruct --device cuda
```

#### NVIDIA laptop / low-VRAM tips

- Start with `Qwen/Qwen2.5-0.5B-Instruct` or `meta-llama/Llama-3.2-1B-Instruct`.
- Use `--dtype float16` (default on GPU with `--dtype auto`).
- Close browser GPU tabs and other CUDA apps before loading 7B models.

---

### Apple Silicon (MPS backend)

**Backend:** `mps`  
**Supported chips:** Apple **M1**, **M1 Pro**, **M1 Max**, **M1 Ultra**, **M2** family, **M3** family, **M4** family (MacBook Air/Pro, Mac mini, Mac Studio, Mac Pro with Apple Silicon).

> **Check your chip:** `system_profiler SPHardwareDataType | grep Chip` — see [Identify your GPU](#identify-your-gpu-model-architecture-generation).

**Not supported:** Intel-based Macs (pre-2020) — no MPS; use [CPU-only](#cpu-only-inference).

Apple Silicon uses **unified memory** (RAM = VRAM). A 16 GB Mac can run ~7B models in FP16 with care; 8 GB Macs should stick to ≤3B models.

#### Apple chip generations

| Chip | Year | GPU cores (approx.) | LLM inference notes |
|------|------|---------------------|---------------------|
| **M1** | 2020 | 7–8 (base) | Good for ≤3B; 7B possible on 16 GB+ with patience |
| **M1 Pro / Max / Ultra** | 2021 | 14–64 | Strong for 3B–7B; Ultra handles 13B with tuning |
| **M2** family | 2022 | 8–76 | Similar to M1 generation, modestly faster |
| **M3** family | 2023 | 8–40 | Improved efficiency; good 7B experience on 18 GB+ |
| **M4** family | 2024–25 | 8–40 | Fastest Apple Silicon; 7B–8B comfortable on 16 GB+ |

#### Miniconda + Apple Silicon: full example

```bash
# Use arm64 Miniconda installer from anaconda.com
conda create -n local-llm python=3.12 -y
conda activate local-llm
cd local-llm

pip install --upgrade pip
pip install torch torchvision
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python inference.py --list-devices
# Expected: mps, cpu

python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device mps
```

#### macOS requirements

- **macOS 12.3+** for MPS (Metal).
- Keep PyTorch updated — MPS bugs are fixed in newer releases.
- If MPS errors occur, retry with `--device cpu` or upgrade macOS / PyTorch.

#### Apple Silicon dtype notes

`--dtype auto` selects **float16** on MPS. Some models behave better with `--dtype float32` on MPS if you see NaNs (rare):

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype float32
```

---

### AMD GPUs (ROCm on Linux + DirectML on Windows)

AMD support splits by operating system.

> **Check your card:** `lspci | grep -i vga` (Linux) or Device Manager / `wmic` (Windows) — see [Identify your GPU](#identify-your-gpu-model-architecture-generation).

#### Linux — ROCm (uses `cuda` backend in PyTorch)

ROCm builds expose AMD GPUs through `torch.cuda`. In `inference.py`, `--device auto` therefore picks **`cuda`** when ROCm PyTorch is installed — this is expected.

##### AMD GPU generations (Linux ROCm)

| Architecture | Example products | ROCm support (2025–26) | Notes |
|--------------|------------------|------------------------|-------|
| **RDNA 4** | RX 9070 XT, 9060 XT; Radeon AI PRO R9700 | ROCm **7.2+**, PyTorch **2.9+** | Newest consumer cards; follow [AMD ROCm Radeon docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/) |
| **RDNA 3** | RX 7900 XTX/XT, 7800 XT, 7700 XT; PRO W7900/W7800 | ROCm 6.x / 7.x (see matrix) | Well supported on recent ROCm; use AMD-tested wheels from [repo.radeon.com](https://repo.radeon.com) when possible |
| **RDNA 2** | RX 6900 XT, 6800 XT, 6700 XT | Limited / legacy | May need older ROCm or CPU fallback — check AMD compatibility matrix |
| **RDNA 1** | RX 5700 XT | Mostly **unsupported** | Use CPU or consider different hardware |
| **CDNA / Instinct** | MI100, MI250, MI300 | Full ROCm datacenter support | Excellent for large models in datacenter |

##### Miniconda + AMD ROCm (Linux) — outline

AMD recommends wheels from **repo.radeon.com** (version-matched to ROCm). Exact filenames change with releases — always check:

- [Install PyTorch for Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
- [Compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)

Typical workflow:

```bash
conda create -n local-llm python=3.12 -y
conda activate local-llm
cd local-llm

# Install ROCm PyTorch per AMD docs (example — URLs change with releases)
# pip install torch-2.x.x+rocm...whl torchvision-...whl ...

pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python inference.py --list-devices
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device cuda
```

##### AMD Ryzen AI APUs (Linux / Windows)

Ryzen AI 300 / 400 series APUs with RDNA graphics are gaining ROCm support on Linux and Windows. Treat them like other AMD GPUs — install the ROCm PyTorch build AMD documents for your APU SKU.

#### AMD / Intel GPUs on Windows — DirectML

When ROCm is not available (most consumer AMD + Intel GPUs on Windows), use **DirectML**:

```bash
conda activate local-llm
pip install torch-directml
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python inference.py --list-devices
# Expected: dml, cpu

python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device dml --no-interactive --prompt "DirectML test"
```

| GPU type on Windows | DirectML expectation |
|---------------------|----------------------|
| AMD RX 6000 / 7000 / 9000 | Usually works via `--device dml` |
| Intel Arc A770 / A750 / B-series | Usually works via `--device dml` |
| Intel integrated (Iris Xe) | May work; performance varies |
| NVIDIA on Windows | Prefer native **CUDA** (`--device cuda`) over DirectML |

DirectML is slower than native CUDA/ROCm but much faster than CPU for many models.

---

### Intel GPUs (XPU backend)

**Backend:** `xpu`  
Modern Intel GPU support is built into **official PyTorch XPU wheels** (the separate Intel Extension for PyTorch package is deprecated as of 2.8 — use native `xpu`).

> **Check your GPU:** `Get-CimInstance Win32_VideoController` (Windows) or `lspci | grep -i vga` (Linux) — see [Identify your GPU](#identify-your-gpu-model-architecture-generation).

#### Supported Intel hardware (validated by PyTorch)

| Product line | Codename | Examples | OS |
|--------------|----------|----------|-----|
| **Intel Arc discrete** | Alchemist | Arc A770, A750, A580, A380 | Windows 11, Linux |
| **Intel Arc discrete** | Battlemage | Arc B580, B570 | Windows 11, Linux |
| **Core Ultra integrated** | Meteor Lake-H | Core Ultra 7 155H (Arc iGPU) | Windows 11, Linux |
| **Core Ultra Series 2** | Arrow Lake-H, Lunar Lake | Core Ultra 200V mobile | Windows 11, Linux |
| **Core Ultra Series 3** | Panther Lake | Upcoming mobile | Windows 11, Ubuntu 25.10+ |
| **Data center** | Ponte Vecchio | Intel Data Center GPU Max 1550 | Linux |

Older **Intel UHD / Iris Xe** without Arc branding: try XPU wheels on Windows 11 / recent Linux; if `xpu` is not listed, use [DirectML](#amd--intel-gpus-on-windows-directml) on Windows or [CPU](#cpu-only-inference).

#### Driver prerequisite

Install Intel GPU drivers **before** PyTorch. See Intel’s guide:

**[PyTorch prerequisites for Intel GPU](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)**

#### Miniconda + Intel XPU: full example

```bash
conda create -n local-llm python=3.12 -y
conda activate local-llm
cd local-llm

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python inference.py --list-devices
# Expected: xpu, cpu

python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device xpu
```

#### Intel XPU dtype notes

- `--dtype auto` → float16 on XPU (same as CUDA/MPS).
- FP64 is limited on some Arc cards; stick to float16 / bfloat16 / float32.

---

### CPU-only inference

Use CPU when:

- You have **no discrete GPU**, or your GPU is **too old** for PyTorch kernels (e.g. Maxwell GTX 900 series).
- You are on **Intel Mac**, **old laptop**, or **virtual machine** without GPU passthrough.
- GPU setup fails and you need a reliable fallback.

> **Confirm you have no usable GPU:** run the OS commands in [Identify your GPU](#identify-your-gpu-model-architecture-generation); if only integrated graphics or nothing appears, CPU is the safe path.

**Backend:** `cpu`  
**Expectation:** Slow but universal. A 0.5B–1B instruct model on a modern 8-core CPU is fine for testing; 7B+ is often impractical without patience.

#### Miniconda + CPU: full example

```bash
conda create -n local-llm python=3.12 -y
conda activate local-llm
cd local-llm

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors

python inference.py --list-devices
# Expected: cpu

python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu
```

#### CPU performance tips

| Tip | Command / action |
|-----|------------------|
| Use the smallest model | `--model-id Qwen/Qwen2.5-0.5B-Instruct` |
| Shorter replies | `--max-new-tokens 128` |
| Deterministic / slightly faster | `--no-do-sample` |
| Lower memory during load | `--low-cpu-mem-usage` (default on) |
| More RAM headroom | `--dtype float32` uses *more* RAM; try `--dtype float16` only if your CPU path supports it — on CPU, `auto` picks float32 |

#### CPU by vendor (all generations)

| CPU vendor | Generations | Notes |
|------------|-------------|-------|
| **Intel** | Core 6th gen–14th gen; Core Ultra 100/200 | Works everywhere; more cores = faster token/s |
| **AMD** | Ryzen 2000–9000 series, Threadripper, EPYC | Excellent CPU inference scaling with core count |
| **Apple Intel Mac** | Core i5/i7/i9 pre-2020 | No MPS; CPU-only |
| **ARM (non-Apple)** | Ampere Altra, Raspberry Pi 5 | Possible but slow; stick to ≤1B models |

Query CPU model from the terminal:

```bash
# Windows
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors

# Linux
lscpu

# macOS
sysctl -n machdep.cpu.brand_string
```

---

### Platform quick-start matrix (copy-paste)

One-page reference after `conda activate local-llm` and `cd local-llm`:

| Your hardware | Install PyTorch | Run inference |
|---------------|-----------------|---------------|
| NVIDIA RTX 40xx / 50xx | `pip install -r requirements.txt` | `python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct` |
| NVIDIA GTX 10xx / RTX 20xx | `pip install torch --index-url https://download.pytorch.org/whl/cu118` + other deps | add `--device cuda` |
| Apple M1–M4 | `pip install torch torchvision` + other deps | `--device mps` (or auto) |
| AMD RX 7000/9000 Linux | ROCm wheel from AMD docs + other deps | `--device cuda` (ROCm) |
| AMD / Intel Windows | `pip install torch-directml` + other deps | `--device dml` |
| Intel Arc / Core Ultra | `pip install torch --index-url https://download.pytorch.org/whl/xpu` + other deps | `--device xpu` |
| Any CPU-only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` + other deps | `--device cpu` |

“Other deps” = `pip install transformers accelerate huggingface_hub certifi sentencepiece protobuf safetensors`

---

### Verify your setup (all platforms)

```bash
# 0) OS-level GPU identity (see full command list in "Identify your GPU" above)
# NVIDIA:
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
# AMD Linux:
rocm-smi && rocminfo | grep gfx
# Apple Silicon:
system_profiler SPHardwareDataType | grep -E 'Chip|Memory'
# Generic Linux / fallback:
lspci | grep -iE 'vga|3d|display'

# 1) Backends exposed to inference.py
python inference.py --list-devices

# 2) Low-level PyTorch device check (examples)
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available()); [print(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]"  # NVIDIA / AMD ROCm
python -c "import torch; print('mps', torch.backends.mps.is_available())"  # Apple
python -c "import torch; xpu=getattr(torch,'xpu',None); print('xpu', xpu.is_available() if xpu else False)"  # Intel

# 3) Full diagnostic bundle
python -m torch.utils.collect_env

# 4) End-to-end smoke test
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --no-interactive --prompt "Say hello in one sentence."
```

---

## Gated / private models

Some models require you to:

1. Create a Hugging Face account.
2. Accept the model’s license on its model page.
3. Provide an access token.

### Create a token

1. Log in at [https://huggingface.co](https://huggingface.co).
2. Go to **Settings → Access Tokens**.
3. Create a token with **Read** permission.

### Use the token

**Option A — environment variable (recommended)**

Either `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` works.

```bash
# Windows Command Prompt
set HF_TOKEN=hf_your_token_here

# Windows PowerShell
$env:HF_TOKEN="hf_your_token_here"

# macOS / Linux
export HF_TOKEN=hf_your_token_here
```

**Option B — command-line flag**

```bash
python inference.py --model-id meta-llama/Llama-3.2-1B-Instruct --token hf_your_token_here --no-interactive --prompt "Hello"
```

Never commit tokens to Git or share them publicly.

---

## Troubleshooting

### `python: command not found`

Use `python3` instead of `python`, or reinstall Python and enable **Add to PATH**.

### `CUDA was requested but is not available`

- Reinstall the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) (default `requirements.txt` targets cu128).
- Update NVIDIA drivers.
- Or run on CPU: `--device cpu` (slower).

### SSL / certificate errors when downloading models

On some Windows setups (especially **Git Bash** with **conda**), `SSL_CERT_FILE` may point to a missing file. `inference.py` clears invalid cert paths and falls back to **certifi** when installed. If downloads still fail, try:

```bash
pip install certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

### Out of memory (CUDA OOM / system freeze)

- Use a smaller model.
- Close other GPU applications.
- Reduce `--max-new-tokens`.
- Try `--dtype float16` or `--dtype bfloat16`.
- Use CPU: `--device cpu` (much slower but uses system RAM).

### Model download is very slow or fails

- Check your network / VPN / firewall.
- Retry with a stable connection; downloads resume automatically in most cases.
- For corporate proxies, configure `HTTP_PROXY` / `HTTPS_PROXY` environment variables.

### `trust_remote_code` error

Some models need custom Python code from the Hub:

```bash
python inference.py --model-id <model> --trust-remote-code --no-interactive --prompt "Hi"
```

Only use this with models you trust.

### Garbled or repetitive output

- Lower `--temperature` (e.g. `0.2`).
- Increase `--repetition-penalty` (e.g. `1.1`).
- Use an **Instruct** / **Chat** model, not a base model.
- Add a clear `--system-prompt`.

### `conda: command not found`

- Open a **new** terminal after installing Miniconda.
- On Windows, use **Anaconda Prompt** / **Miniconda Prompt**, or re-run the installer and enable “Add to PATH”.
- On macOS/Linux, run `conda init bash` (or `zsh`), restart the shell, then retry.

### Wrong PyTorch wheel installed (GPU shows as CPU)

Symptoms: `--list-devices` only shows `cpu` but you have a GPU.

1. Check what you installed: `python -c "import torch; print(torch.__version__)"`
2. Uninstall and reinstall the correct wheel per the [Platform setup guide](#platform-setup-guide-nvidia-amd-intel-apple-cpu).
3. Common mistake: CPU-only `torch` from PyPI when `cu128` was intended — use `--index-url` from pytorch.org or this repo’s `requirements.txt`.

### NVIDIA `sm_120` / Blackwell (RTX 50xx) not supported

You need **PyTorch 2.7+** with **cu128**. Older `cu124` builds do not include Blackwell kernels:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Update your NVIDIA driver to the latest version for RTX 50-series.

### AMD ROCm `hipErrorNoBinaryForGPU` (Linux)

Your ROCm PyTorch build does not match the GPU architecture. Check AMD’s compatibility matrix and install the ROCm version documented for your card (e.g. RDNA 3 vs RDNA 4). Ensure your user is in the `render` and `video` groups on Linux.

### Intel XPU not listed

1. Install Intel GPU drivers first ([Intel prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html)).
2. Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/xpu`
3. On Windows, try DirectML as fallback: `pip install torch-directml` and `--device dml`.

### Apple MPS fallback

Upgrade PyTorch to the latest stable version for your macOS version. If errors persist:

```bash
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --device cpu
```

Or try `--dtype float32` with MPS (see [Apple Silicon](#apple-silicon-mps-backend)).

### Windows AMD GPU not detected

Install DirectML support:

```bash
pip install torch-directml
python inference.py --device dml --model-id Qwen/Qwen2.5-0.5B-Instruct --no-interactive --prompt "Test"
```

---

## FAQ for beginners

**Do I need to pay for API usage?**  
No. Models are downloaded once and run locally. You only pay in electricity and disk space.

**Where are models stored?**  
In the `models/` folder under this project (ignored by Git). Delete a subfolder to free space.

**Can I use this offline?**  
After the first download, yes — as long as the model is already in `models/`.

**Is my data sent to the cloud during inference?**  
No. Generation happens on your machine. (The initial download comes from Hugging Face.)

**What is a “parameter” (0.5B, 7B, …)?**  
Rough measure of model size. 0.5B ≈ 500 million learned weights. Bigger often means smarter but heavier.

**Why is the first run slow?**  
Downloading and loading multi-gigabyte weights takes time. Later runs reuse the cache.

**How do I find my GPU model and generation?**  
See [Identify your GPU](#identify-your-gpu-model-architecture-generation): `nvidia-smi` (NVIDIA), `rocm-smi` / `rocminfo` (AMD Linux), `system_profiler` (Apple), `lspci` or `wmic` (generic).

**Can I use Miniconda instead of venv?**  
Yes — [Option B in Step 3](#option-b--miniconda-recommended-for-gpu-workflows) is recommended for GPU setups. Models still download to `models/` in the project folder regardless of environment type.

**Which NVIDIA CUDA wheel should I pick?**  
For RTX 40xx / 50xx: **cu128** (this repo default). For GTX 10xx / RTX 20xx on older drivers: try **cu118**. PyTorch wheels bundle CUDA — you only need a recent NVIDIA driver.

**Does AMD use CUDA in this script?**  
On Linux with ROCm, PyTorch exposes AMD GPUs through the `torch.cuda` API, so `inference.py` reports backend `cuda` — that is normal.

**Can I run on Intel integrated graphics?**  
Recent **Core Ultra** with Arc iGPU: try `--device xpu`. Older Iris Xe on Windows: try `--device dml`. Otherwise use CPU.

**Can I run multiple GPUs?**  
With several NVIDIA/AMD CUDA devices, `--device auto` uses `accelerate` to spread layers across GPUs.

---

## Project layout

```text
local-llm/
├── inference.py       # Main script — download + interactive chat + stream
├── requirements.txt   # Python package dependencies
├── README.md          # This file
├── LICENSE
└── models/            # Created automatically; downloaded models live here
```

There is intentionally **no package split** — everything you need is in `inference.py`.

---

## License

See [LICENSE](LICENSE) for the project license.

Model weights are subject to **their own licenses** on Hugging Face (Apache 2.0, Llama Community License, etc.). Always read the model card before use.
