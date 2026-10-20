"""LLM backends. All expose generate(messages, images=None) -> str.

transformers: Text model through Hugging Face.
transformers_vl: Vision-language model through Hugging Face.
ollama: Locally served model, e.g. qwen2.5vl:3b.
extractive: Returns the top evidence passage; used by tests.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
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
    """Use a GPU-supported dtype, or fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        cuda = _device() == "cuda"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_dtype(torch),
            device_map="auto" if cuda else None,
        ).eval()

    def generate(self, messages: list[dict], images=None) -> str:
        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tok.eos_token_id,
        }
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False)

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        return self.tok.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()


class TransformersVLGenerator:
    """Vision-language backend for Qwen2-VL / Qwen2.5-VL."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVL
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVL

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        cuda = _device() == "cuda"
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=128 * 28 * 28,
            max_pixels=640 * 28 * 28,
        )
        self.model = AutoVL.from_pretrained(
            model_name,
            torch_dtype=_dtype(torch),
            device_map="auto" if cuda else None,
        ).eval()

    def generate(self, messages: list[dict], images=None) -> str:
        images = images or []
        formatted_messages = []

        for message in messages:
            if message["role"] == "user" and images:
                content = (
                    [{"type": "image"} for _ in images]
                    + [{"type": "text", "text": message["content"]}]
                )
            else:
                content = [{"type": "text", "text": message["content"]}]

            formatted_messages.append({
                "role": message["role"],
                "content": content,
            })

        prompt = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=images or None,
            return_tensors="pt",
        ).to(self.model.device)

        kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False)

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
        )[0].strip()


def _ns_to_s(value):
    return round(value / 1e9, 3) if value else None


def _ollama_stats(result: dict) -> dict:
    """Timing and stop-reason fields from an Ollama /api/chat response."""
    return {
        "done_reason": result.get("done_reason"),
        "load_s": _ns_to_s(result.get("load_duration")),
        "prompt_tokens": result.get("prompt_eval_count"),
        "prompt_eval_s": _ns_to_s(result.get("prompt_eval_duration")),
        "gen_tokens": result.get("eval_count"),
        "gen_s": _ns_to_s(result.get("eval_duration")),
    }


class OllamaGenerator:
    """Local Ollama backend with an 8192-token context window.

    After each generate() call, last_stats holds Ollama's timing fields and
    done_reason ("length" means the num_predict limit cut the answer off).
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.options = {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "num_ctx": 8192,
        }
        self.last_stats: dict = {}

    def generate(self, messages: list[dict], images=None) -> str:
        self.last_stats = {}
        msgs = [dict(message) for message in messages]

        if images:
            encoded_images = []
            for image in images:
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded_images.append(
                    base64.b64encode(buffer.getvalue()).decode("utf-8")
                )

            # Ollama's chat API expects images on a user message.
            user_index = next(
                (
                    i for i in range(len(msgs) - 1, -1, -1)
                    if msgs[i].get("role") == "user"
                ),
                None,
            )
            if user_index is None:
                raise ValueError(
                    "Images were supplied, but no user message was found."
                )

            msgs[user_index]["images"] = encoded_images

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": self.options,
        }
        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))

            if "message" not in result or "content" not in result["message"]:
                raise RuntimeError(
                    f"Unexpected Ollama response: {result}"
                )

            self.last_stats = _ollama_stats(result)
            return result["message"]["content"].strip()

        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {error.code}: {detail}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.url}: {error.reason}. "
                "Check that Ollama is running."
            ) from error


class ExtractiveGenerator:
    """Return the first context passage, or abstain when evidence is weak."""

    def generate(self, messages: list[dict], images=None) -> str:
        user = messages[-1]["content"]

        if "Context:" not in user:
            return ABSTAIN

        context = user.split("Context:", 1)[1].split(
            "\n\nQuestion:", 1
        )[0]

        if "weakly related" in user:
            return ABSTAIN

        first = context.strip().split("\n\n")[0]
        body = first.split("\n", 1)[1] if "\n" in first else first
        return f"{body.strip()[:300]} [1]"


@lru_cache(maxsize=2)
def get_generator(
    backend: str,
    model: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    ollama_url: str = "http://localhost:11434",
):
    if backend == "transformers":
        return TransformersGenerator(model, max_new_tokens, temperature)

    if backend == "transformers_vl":
        return TransformersVLGenerator(model, max_new_tokens, temperature)

    if backend == "ollama":
        return OllamaGenerator(
            model,
            ollama_url,
            max_new_tokens,
            temperature,
        )

    if backend == "extractive":
        return ExtractiveGenerator()

    raise ValueError(f"Unknown generation backend: {backend}")


def supports_images(backend: str) -> bool:
    return backend in ("transformers_vl", "ollama")