"""LLM backends. All of them expose generate(messages, images=None) -> str.

transformers: text model through Hugging Face, default Qwen2.5-1.5B-Instruct.
Runs on CPU but slowly.
transformers_vl: vision-language model, default Qwen2.5-VL-3B-Instruct. It
also receives the retrieved figures. Meant for the Kaggle GPU.
ollama: a model served by a local Ollama, e.g. qwen2.5vl:3b. Fastest option
on a CPU laptop and accepts images.
extractive: no LLM, returns the top evidence passage. Used by the tests.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from functools import lru_cache

from src.generation.prompts import ABSTAIN


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _dtype(torch):
    """bf16 where supported, fp16 on T4/P100 which lack bf16, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class TransformersGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        cuda = _device() == "cuda"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_dtype(torch),
            device_map="auto" if cuda else None).eval()

    def generate(self, messages: list[dict], images=None) -> str:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.eos_token_id)
        kw.update(dict(do_sample=True, temperature=self.temperature) if self.temperature > 0 else dict(do_sample=False))
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class TransformersVLGenerator:
    """For Qwen2-VL / Qwen2.5-VL: the images are placed before the text in the user message."""

    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVL
        except ImportError:  # transformers < 4.50
            from transformers import AutoModelForVision2Seq as AutoVL
        self.torch = torch
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        cuda = _device() == "cuda"
        # limit image tokens to keep memory and latency bounded
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=128 * 28 * 28, max_pixels=640 * 28 * 28)
        self.model = AutoVL.from_pretrained(model_name, torch_dtype=_dtype(torch),
                                            device_map="auto" if cuda else None).eval()

    def generate(self, messages: list[dict], images=None) -> str:
        images = images or []
        msgs = []
        for m in messages:
            if m["role"] == "user" and images:
                content = [{"type": "image"} for _ in images] + [{"type": "text", "text": m["content"]}]
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=self.max_new_tokens)
        kw.update(dict(do_sample=True, temperature=self.temperature) if self.temperature > 0 else dict(do_sample=False))
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kw)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


class OllamaGenerator:
    def __init__(self, model: str, url: str = "http://localhost:11434", max_new_tokens: int = 256,
                 temperature: float = 0.0):
        self.model, self.url = model, url.rstrip("/")
        self.options = {"temperature": temperature, "num_predict": max_new_tokens}

    def generate(self, messages: list[dict], images=None) -> str:
        msgs = [dict(m) for m in messages]
        if images:
            enc = []
            for im in images:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                enc.append(base64.b64encode(buf.getvalue()).decode())
            msgs[-1]["images"] = enc
        body = json.dumps({"model": self.model, "messages": msgs, "stream": False, "options": self.options}).encode()
        req = urllib.request.Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama call failed ({e}). Is Ollama running and is `ollama pull {self.model}` done?") from e


class ExtractiveGenerator:
    """Returns the first context passage, or the abstain text when evidence is weak."""

    def generate(self, messages: list[dict], images=None) -> str:
        user = messages[-1]["content"]
        if "Context:" not in user:
            return ABSTAIN
        ctx = user.split("Context:", 1)[1].split("\n\nQuestion:", 1)[0]
        if "weakly related" in user:
            return ABSTAIN
        first = ctx.strip().split("\n\n")[0]
        body = first.split("\n", 1)[1] if "\n" in first else first
        return f"{body.strip()[:300]} [1]"


@lru_cache(maxsize=2)
def get_generator(backend: str, model: str | None = None, max_new_tokens: int = 256, temperature: float = 0.0,
                  ollama_url: str = "http://localhost:11434"):
    if backend == "transformers":
        return TransformersGenerator(model, max_new_tokens, temperature)
    if backend == "transformers_vl":
        return TransformersVLGenerator(model, max_new_tokens, temperature)
    if backend == "ollama":
        return OllamaGenerator(model, ollama_url, max_new_tokens, temperature)
    if backend == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown generation backend: {backend}")


def supports_images(backend: str) -> bool:
    return backend in ("transformers_vl", "ollama")
