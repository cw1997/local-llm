#!/usr/bin/env python3
"""
Local LLM inference script.

Downloads a Hugging Face model into ./models and runs text generation with
streaming output. Supports NVIDIA CUDA, Apple Metal (MPS), Intel XPU,
AMD ROCm/CUDA-compatible stacks, Windows DirectML, and CPU fallback.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved runtime device and human-readable description."""

    torch_device: str
    label: str
    backend: str
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None


def _try_import_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed.\n"
            "Install dependencies first:\n"
            "  pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc


def _cuda_device_count(torch) -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _cuda_device_labels(torch) -> list[str]:
    labels: list[str] = []
    for index in range(_cuda_device_count(torch)):
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = f"CUDA device {index}"
        labels.append(f"cuda:{index} ({name})")
    return labels


def _mps_available(torch) -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _xpu_available(torch) -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def _directml_available() -> bool:
    try:
        import torch_directml  # type: ignore

        _ = torch_directml.device()
        return True
    except Exception:
        return False


def _directml_device():
    import torch_directml  # type: ignore

    return torch_directml.device()


def list_available_backends() -> list[str]:
    """Return backends that appear usable on this machine."""
    torch = _try_import_torch()
    backends: list[str] = []
    if _cuda_device_count(torch) > 0:
        backends.append("cuda")
    if _mps_available(torch):
        backends.append("mps")
    if _xpu_available(torch):
        backends.append("xpu")
    if _directml_available():
        backends.append("dml")
    backends.append("cpu")
    return backends


def resolve_device(requested: str, dtype: str) -> DeviceInfo:
    """
    Pick the best device for inference.

    ``auto`` prefers discrete-class accelerators in this order:
    CUDA (NVIDIA / AMD ROCm) -> Apple MPS -> Intel XPU -> DirectML -> CPU.
    """
    torch = _try_import_torch()
    req = requested.strip().lower()

    resolved_dtype = resolve_dtype(dtype, torch)

    if req == "auto":
        cuda_count = _cuda_device_count(torch)
        if cuda_count > 0:
            names = _cuda_device_labels(torch)
            if cuda_count == 1:
                return DeviceInfo(
                    torch_device="cuda:0",
                    label=names[0],
                    backend="cuda",
                    device_map=None,
                    torch_dtype=resolved_dtype,
                )
            joined = ", ".join(names)
            return DeviceInfo(
                torch_device="cuda",
                label=f"{cuda_count} CUDA device(s): {joined}",
                backend="cuda",
                device_map="auto",
                torch_dtype=resolved_dtype,
            )

        if _mps_available(torch):
            return DeviceInfo(
                torch_device="mps",
                label="Apple Metal (MPS)",
                backend="mps",
                device_map=None,
                torch_dtype=resolved_dtype,
            )

        if _xpu_available(torch):
            return DeviceInfo(
                torch_device="xpu",
                label="Intel XPU",
                backend="xpu",
                device_map=None,
                torch_dtype=resolved_dtype,
            )

        if _directml_available():
            return DeviceInfo(
                torch_device="privateuseone",
                label="DirectML (AMD / Intel GPU on Windows)",
                backend="dml",
                device_map=None,
                torch_dtype=resolved_dtype,
            )

        return DeviceInfo(
            torch_device="cpu",
            label="CPU (no GPU accelerator detected)",
            backend="cpu",
            device_map=None,
            torch_dtype=resolved_dtype,
        )

    if req in {"cuda", "gpu"}:
        if _cuda_device_count(torch) == 0:
            print("Warning: CUDA was requested but is not available. Falling back to CPU.", file=sys.stderr)
            return DeviceInfo("cpu", "CPU (CUDA unavailable)", "cpu", None, resolved_dtype)
        count = _cuda_device_count(torch)
        if count == 1:
            return DeviceInfo("cuda:0", _cuda_device_labels(torch)[0], "cuda", None, resolved_dtype)
        return DeviceInfo("cuda", ", ".join(_cuda_device_labels(torch)), "cuda", "auto", resolved_dtype)

    if req == "mps":
        if not _mps_available(torch):
            print("Warning: MPS was requested but is not available. Falling back to CPU.", file=sys.stderr)
            return DeviceInfo("cpu", "CPU (MPS unavailable)", "cpu", None, resolved_dtype)
        return DeviceInfo("mps", "Apple Metal (MPS)", "mps", None, resolved_dtype)

    if req == "xpu":
        if not _xpu_available(torch):
            print("Warning: Intel XPU was requested but is not available. Falling back to CPU.", file=sys.stderr)
            return DeviceInfo("cpu", "CPU (XPU unavailable)", "cpu", None, resolved_dtype)
        return DeviceInfo("xpu", "Intel XPU", "xpu", None, resolved_dtype)

    if req in {"dml", "directml"}:
        if not _directml_available():
            print("Warning: DirectML was requested but is not available. Falling back to CPU.", file=sys.stderr)
            return DeviceInfo("cpu", "CPU (DirectML unavailable)", "cpu", None, resolved_dtype)
        return DeviceInfo(
            "privateuseone",
            "DirectML (AMD / Intel GPU on Windows)",
            "dml",
            None,
            resolved_dtype,
        )

    if req == "cpu":
        return DeviceInfo("cpu", "CPU", "cpu", None, resolved_dtype)

    raise SystemExit(
        f"Unknown device '{requested}'. "
        f"Choose one of: auto, cuda, mps, xpu, dml, cpu. "
        f"Detected on this machine: {', '.join(list_available_backends())}"
    )


def resolve_dtype(dtype: str, torch) -> str:
    """Map user dtype string to a concrete torch dtype name."""
    value = dtype.strip().lower()
    if value == "auto":
        if torch.cuda.is_available() or _mps_available(torch) or _xpu_available(torch):
            return "float16"
        return "float32"
    if value in {"fp16", "float16", "half"}:
        return "float16"
    if value in {"bf16", "bfloat16"}:
        return "bfloat16"
    if value in {"fp32", "float32", "full"}:
        return "float32"
    raise SystemExit(f"Unknown dtype '{dtype}'. Use auto, float16, bfloat16, or float32.")


def torch_dtype_from_name(torch, name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


# ---------------------------------------------------------------------------
# Model download / load
# ---------------------------------------------------------------------------


def sanitize_model_dir_name(model_id: str) -> str:
    """Turn a Hugging Face repo id into a safe folder name."""
    return re.sub(r"[^\w.\-]+", "--", model_id.strip()).strip("-") or "model"


def download_model(
    model_id: str,
    models_dir: Path,
    revision: str,
    token: Optional[str],
    force_download: bool,
) -> Path:
    """Download model snapshot into models_dir if not already present."""
    from huggingface_hub import snapshot_download

    target = models_dir / sanitize_model_dir_name(model_id)
    target.mkdir(parents=True, exist_ok=True)

    print(f"Model cache directory: {target.resolve()}")
    print(f"Downloading (or reusing cached) model '{model_id}' from Hugging Face Hub...")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        force_download=force_download,
    )

    print("Model files are ready.\n")
    return target


def load_model_and_tokenizer(
    model_path: Path,
    device_info: DeviceInfo,
    trust_remote_code: bool,
    token: Optional[str],
    low_cpu_mem_usage: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch = _try_import_torch()
    torch_dtype = torch_dtype_from_name(torch, device_info.torch_dtype or "float32")

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
    }
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "torch_dtype": torch_dtype,
    }

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights (this may take a while on first run)...")

    if device_info.device_map == "auto":
        try:
            from accelerate import infer_auto_device_map  # noqa: F401

            model_kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        except Exception as exc:
            print(
                f"Warning: device_map='auto' failed ({exc}). Loading on cuda:0 instead.",
                file=sys.stderr,
            )
            model_kwargs.pop("device_map", None)
            model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
            model = model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
        target = device_info.torch_device
        if device_info.backend == "dml":
            model = model.to(_directml_device())
        elif target != "cpu":
            model = model.to(target)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt building & generation
# ---------------------------------------------------------------------------


def build_prompt(tokenizer, user_prompt: str, system_prompt: Optional[str]) -> str:
    """Build a chat or plain prompt depending on tokenizer capabilities."""
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    return user_prompt


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
    seed: Optional[int],
) -> tuple[str, int, float]:
    """
    Generate text with streaming stdout output.

    Returns (full_text, completion_token_count, generation_seconds).
    """
    import torch
    from transformers import TextIteratorStreamer

    torch = _try_import_torch()

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_token_count = int(input_ids.shape[-1])

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 1e-5) if do_sample else 1.0,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("--- Model output (streaming) ---")
    chunks: list[str] = []
    for text in streamer:
        if text:
            print(text, end="", flush=True)
            chunks.append(text)

    thread.join()
    elapsed = time.perf_counter() - start
    print("\n--- End of output ---\n")

    completion = "".join(chunks)
    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    return completion, completion_tokens, elapsed


def print_stats(
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_seconds: float,
    device_info: DeviceInfo,
    model_id: str,
) -> None:
    total_tokens = prompt_tokens + completion_tokens
    tokens_per_second = completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0

    print("=" * 60)
    print("Inference summary")
    print("=" * 60)
    print(f"Model ID           : {model_id}")
    print(f"Device             : {device_info.label}")
    print(f"Backend            : {device_info.backend}")
    print(f"Prompt tokens      : {prompt_tokens}")
    print(f"Generated tokens   : {completion_tokens}")
    print(f"Total tokens       : {total_tokens}")
    print(f"Generation time    : {elapsed_seconds:.2f} s")
    print(f"Generation speed   : {tokens_per_second:.2f} tokens/s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description=(
            "Download a Hugging Face causal language model into ./models and "
            "run local text generation with streaming output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py --model-id microsoft/Phi-3-mini-4k-instruct "
            '--prompt "Explain gravity in one sentence."\n'
            "  python inference.py --model-id Qwen/Qwen2.5-0.5B-Instruct "
            "--max-new-tokens 512 --temperature 0.2 --device auto\n"
            "  python inference.py --model-id meta-llama/Llama-3.2-1B-Instruct "
            "--system-prompt \"You are a helpful assistant.\" --device cuda\n"
        ),
    )

    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model repository id (required unless --list-devices), e.g. Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        default="Hello! Please introduce yourself in two short sentences.",
        help="User prompt / question sent to the model",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system instruction (used when the tokenizer supports chat templates)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory (relative to current working directory) where models are stored",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Git revision / branch / tag on the Hugging Face repo",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability mass (top-p)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling limit",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty for repeating tokens (1.0 = disabled)",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stochastic sampling; disable for greedy decoding",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "xpu", "dml", "cpu"],
        help=(
            "Execution device. 'auto' prefers discrete GPUs (CUDA/ROCm), then MPS, "
            "Intel XPU, DirectML, then CPU"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model weight dtype: auto, float16, bfloat16, or float32",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing custom model code from the Hugging Face repo",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download all model files even if they already exist locally",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load weights in a memory-efficient way on CPU while preparing the model",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print detected hardware backends and exit",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        backends = list_available_backends()
        print("Detected inference backends on this machine:")
        for name in backends:
            print(f"  - {name}")
        return 0

    if not args.model_id:
        parser.error("--model-id is required unless --list-devices is used")

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    device_info = resolve_device(args.device, args.dtype)

    print("=" * 60)
    print("Local LLM inference")
    print("=" * 60)
    print(f"Selected device : {device_info.label}")
    print(f"Backend         : {device_info.backend}")
    print(f"Weight dtype    : {device_info.torch_dtype}")
    print(f"Model ID        : {args.model_id}")
    print("=" * 60)
    print()

    models_dir = Path(args.models_dir)
    model_path = download_model(
        model_id=args.model_id,
        models_dir=models_dir,
        revision=args.revision,
        token=token,
        force_download=args.force_download,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device_info=device_info,
        trust_remote_code=args.trust_remote_code,
        token=token,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    prompt = build_prompt(tokenizer, args.prompt, args.system_prompt)
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    print(f"Prompt token count: {prompt_tokens}")
    print("Starting generation...\n")

    _, completion_tokens, elapsed = stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        seed=args.seed,
    )

    print_stats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=elapsed,
        device_info=device_info,
        model_id=args.model_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
