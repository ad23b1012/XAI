"""
VLM Engine — LLaVA-7B-based Explanation Generator.

Uses LLaVA-1.5-7B with 4-bit quantization (via bitsandbytes) to generate
natural language explanations for emotion predictions.

VRAM Usage: ~4.5GB at 4-bit quantization.
Strategy: Sequential loading — classifier is unloaded before VLM is loaded,
keeping peak VRAM under 5GB on the RTX 4050 (6GB).
"""

import torch
from typing import Optional, List, Dict
from PIL import Image
import gc


class VLMEngine:
    """
    LLaVA-7B Vision-Language Model engine for explanation generation.

    Loads the model with 4-bit quantization to fit on 6GB VRAM.
    Generates grounded natural language explanations from face images
    and structured evidence prompts.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        quantization: str = "4bit",
        max_new_tokens: int = 150,
        temperature: float = 0.3,
        do_sample: bool = False,
        device: str = "auto",
    ):
        """
        Args:
            model_name: HuggingFace model ID for LLaVA.
            quantization: "4bit", "8bit", or "none".
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            do_sample: Whether to use sampling (False = greedy).
            device: Device to use.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = device

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """
        Load the LLaVA model with quantization.

        Call this AFTER freeing VRAM from the classifier
        (torch.cuda.empty_cache()).
        """
        if self._loaded:
            return

        from transformers import LlavaForConditionalGeneration, AutoProcessor

        print(f"[VLM] Loading {self.model_name} with {self.quantization} quantization...")

        # Configure quantization
        model_kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        )

        self._loaded = True
        print(f"[VLM] Model loaded successfully!")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"[VLM] VRAM usage: {vram_used:.2f} GB")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Generate an explanation for the given face image and prompt.

        Args:
            image: Face image as PIL Image.
            prompt: Structured evidence prompt (from PromptBuilder).

        Returns:
            Generated explanation text.
        """
        if not self._loaded:
            self.load()

        # Format as LLaVA conversation with image
        # LLaVA expects "<image>" token in the prompt
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        # Process inputs
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        )

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                use_cache=True,
            )

        # Decode — only get the newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        explanation = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        return explanation

    def unload(self):
        """
        Unload the model and free VRAM.

        Call this before loading the classifier for the next image.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        self._loaded = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[VLM] Model unloaded, VRAM freed.")

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
