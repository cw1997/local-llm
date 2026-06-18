# local-llm

Run large language models (LLMs) **entirely on your own computer** — no cloud API keys required for inference.

This repository contains a single Python script, [`inference.py`](inference.py), that:

1. Downloads a model from [Hugging Face Hub](https://huggingface.co/models) into a local `models/` folder.
2. Loads the model on your hardware (NVIDIA, AMD, Intel, Apple Silicon, or CPU).
3. Streams generated text to your terminal in real time.
4. Prints a summary with elapsed time, token counts, and generation speed.

You do **not** need prior experience with AI or Python to follow this guide. Every step is explained below.

---

## Table of contents

- [What is this project?](#what-is-this-project)
- [What you need before starting](#what-you-need-before-starting)
- [Step 1: Install Python](#step-1-install-python)
- [Step 2: Download this project](#step-2-download-this-project)
- [Step 3: Create a virtual environment (recommended)](#step-3-create-a-virtual-environment-recommended)
- [Step 4: Install dependencies](#step-4-install-dependencies)
- [Step 5: Install PyTorch for your hardware](#step-5-install-pytorch-for-your-hardware)
- [Step 6: Run your first inference](#step-6-run-your-first-inference)
- [Understanding the output](#understanding-the-output)
- [All command-line options](#all-command-line-options)
- [Choosing a model](#choosing-a-model)
- [Hardware support](#hardware-support)
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
- **Python 3.10 or newer** (installation steps below).

Optional but helpful:

- An NVIDIA, AMD, Intel, or Apple Silicon GPU for faster generation.
- A free [Hugging Face account](https://huggingface.co/join) if you want gated models (e.g. some Llama variants).

---

## Step 1: Install Python

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

## Step 3: Create a virtual environment (recommended)

A *virtual environment* keeps this project’s Python packages separate from the rest of your system. This avoids version conflicts.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

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

## Step 4: Install dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- **PyTorch** — deep learning runtime.
- **transformers** — loads Hugging Face models.
- **accelerate** — helps spread large models across GPUs.
- **huggingface_hub** — downloads models from the Hub.

---

## Step 5: Install PyTorch for your hardware

The default `pip install -r requirements.txt` installs a CPU build of PyTorch. For **much faster** inference on a GPU, install the build that matches your hardware.

Visit the official selector and copy the command for your system:

**[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)**

### Quick reference

| Hardware | Typical install approach |
|----------|--------------------------|
| **NVIDIA GPU (CUDA)** | Use PyTorch’s CUDA wheel, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **Apple Silicon (M1/M2/M3/M4)** | Default PyTorch from pip includes **MPS** (Metal) support on macOS |
| **AMD GPU (Linux, ROCm)** | Use PyTorch’s ROCm build from [pytorch.org](https://pytorch.org/get-started/locally/) |
| **Intel GPU (XPU)** | `pip install intel-extension-for-pytorch` (see [Intel docs](https://intel.github.io/intel-extension-for-pytorch/)) |
| **AMD / Intel GPU on Windows** | `pip install torch-directml` for **DirectML** backend |
| **CPU only** | Default `requirements.txt` is sufficient |

After installing a GPU-specific PyTorch build, reinstall other packages if needed:

```bash
pip install -r requirements.txt
```

### Check what your machine supports

```bash
python inference.py --list-devices
```

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
python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct --prompt "What is the capital of France?"
```

On macOS/Linux, if `python` is not found, use `python3` instead.

### What happens

1. The script prints which **device** it will use (GPU or CPU).
2. It downloads model files into `./models/` (skipped on later runs if already cached).
3. It loads weights into memory (can take 10 seconds to several minutes).
4. It **streams** the answer token by token.
5. It prints **statistics**: time, token counts, tokens per second.

### Another example with a system prompt

```bash
python inference.py ^
  --model-id Qwen/Qwen2.5-0.5B-Instruct ^
  --system-prompt "You are a concise science tutor." ^
  --prompt "Explain photosynthesis in three sentences." ^
  --max-new-tokens 200 ^
  --temperature 0.3
```

On macOS/Linux, replace `^` line continuations with `\`:

```bash
python inference.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --system-prompt "You are a concise science tutor." \
  --prompt "Explain photosynthesis in three sentences." \
  --max-new-tokens 200 \
  --temperature 0.3
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
| `--prompt` | Short hello message | Your question or instruction |
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
| `--token` | env var | Hugging Face token for gated models |
| `--force-download` | off | Re-download even if files exist |
| `--low-cpu-mem-usage` / `--no-low-cpu-mem-usage` | on | Memory-efficient loading |
| `--list-devices` | off | Print detected backends and exit |

View full help anytime:

```bash
python inference.py --help
```

---

## Choosing a model

Browse models at [https://huggingface.co/models](https://huggingface.co/models).

### Good starter models (small, chat-ready)

| Model ID | Approx. size | Notes |
|----------|--------------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | Very fast, low RAM |
| `microsoft/Phi-3-mini-4k-instruct` | ~7 GB | Strong quality for its size |
| `meta-llama/Llama-3.2-1B-Instruct` | ~2 GB | Gated — requires HF account + token |

Prefer model cards whose names end in **`-Instruct`** or **`-Chat`** for question-answering. Base models (no suffix) are usually not tuned for dialogue.

### Size vs. quality

- **Smaller models** (0.5B–3B parameters): faster, less RAM, simpler answers.
- **Larger models** (7B+): better reasoning, but need more VRAM/RAM and time.

If generation is slow or you run out of memory, pick a smaller model or use `--device cpu` with a tiny model.

---

## Hardware support

`inference.py` tries to use the **best available accelerator** when `--device auto` (the default).

| Vendor | Backend flag | When it is used |
|--------|--------------|-----------------|
| **NVIDIA** | `cuda` | CUDA-capable GPU + CUDA PyTorch |
| **AMD (Linux ROCm)** | `cuda` | ROCm build of PyTorch exposes AMD GPUs via the CUDA API |
| **Apple M-series** | `mps` | macOS with Apple Silicon |
| **Intel discrete / integrated** | `xpu` | With Intel Extension for PyTorch |
| **AMD / Intel on Windows** | `dml` | With `torch-directml` installed |
| **CPU** | `cpu` | Fallback when no GPU backend is available |

### Default behavior (`--device auto`)

Priority order:

1. **CUDA** (NVIDIA or AMD ROCm) — including multi-GPU via `device_map=auto` when several CUDA devices exist.
2. **Apple MPS** (Metal on M1/M2/M3/M4 Macs).
3. **Intel XPU**.
4. **DirectML** (Windows GPUs).
5. **CPU** if nothing else works.

You can force a backend, e.g. `--device cpu` for debugging.

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
python inference.py --model-id meta-llama/Llama-3.2-1B-Instruct --token hf_your_token_here --prompt "Hello"
```

Never commit tokens to Git or share them publicly.

---

## Troubleshooting

### `python: command not found`

Use `python3` instead of `python`, or reinstall Python and enable **Add to PATH**.

### `CUDA was requested but is not available`

- Install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).
- Update NVIDIA drivers.
- Or run on CPU: `--device cpu` (slower).

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
python inference.py --model-id <model> --trust-remote-code --prompt "Hi"
```

Only use this with models you trust.

### Garbled or repetitive output

- Lower `--temperature` (e.g. `0.2`).
- Increase `--repetition-penalty` (e.g. `1.1`).
- Use an **Instruct** / **Chat** model, not a base model.
- Add a clear `--system-prompt`.

### Apple Silicon: MPS errors on older PyTorch

Upgrade PyTorch to the latest stable version for your macOS version.

### Windows AMD GPU not detected

Install DirectML support:

```bash
pip install torch-directml
python inference.py --device dml --model-id Qwen/Qwen2.5-0.5B-Instruct --prompt "Test"
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

**Can I run multiple GPUs?**  
With several NVIDIA/AMD CUDA devices, `--device auto` uses `accelerate` to spread layers across GPUs.

---

## Project layout

```text
local-llm/
├── inference.py       # Main script — download + infer + stream
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
