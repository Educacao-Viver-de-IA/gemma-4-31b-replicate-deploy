import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "google/gemma-4-31B-it"
MODEL_PATH = "/src/weights/model"

DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente de IA útil, prestativo e seguro. "
    "Responda sempre em português brasileiro, de forma clara, objetiva e "
    "estruturada. Quando receber imagens, analise-as cuidadosamente e "
    "integre o conteúdo visual à resposta. Quando não tiver certeza de "
    "algo, diga explicitamente em vez de inventar."
)

DEFAULT_PROMPT = "Como posso te ajudar hoje?"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] MODEL_PATH={MODEL_PATH}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(MODEL_PATH))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list MODEL_PATH: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}", flush=True)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"[setup]   GPU {i}: {p.name} | {p.total_memory / 1e9:.1f} GB", flush=True)

        print(f"[setup] loading processor... (t={time.time()-t0:.1f}s)", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print(f"[setup] processor OK (t={time.time()-t0:.1f}s)", flush=True)

        print(f"[setup] loading model (31B bf16 ~62GB, this can take 3-5 min)... (t={time.time()-t0:.1f}s)", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def predict(
        self,
        prompt: str = Input(
            description="Pergunta ou instrução pro modelo (texto).",
            default=DEFAULT_PROMPT,
        ),
        image: Path = Input(
            description="(Opcional) Imagem para análise multimodal.",
            default=None,
        ),
        image_2: Path = Input(
            description="(Opcional) 2ª imagem.",
            default=None,
        ),
        image_3: Path = Input(
            description="(Opcional) 3ª imagem.",
            default=None,
        ),
        image_4: Path = Input(
            description="(Opcional) 4ª imagem.",
            default=None,
        ),
        system_prompt: str = Input(
            description="Instrução de sistema/persona do modelo.",
            default=DEFAULT_SYSTEM_PROMPT,
        ),
        max_new_tokens: int = Input(
            description="Limite de tokens gerados.",
            default=2048,
            ge=64,
            le=4096,
        ),
        temperature: float = Input(
            description="0 = greedy. 0.2 = recomendado (evita loops).",
            default=0.2,
            ge=0.0,
            le=2.0,
        ),
    ) -> str:
        image_paths = [p for p in [image, image_2, image_3, image_4] if p is not None]
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        user_content: list = []
        for idx, pil in enumerate(pil_images, start=1):
            if len(pil_images) > 1:
                user_content.append({"type": "text", "text": f"Imagem {idx}:"})
            user_content.append({"type": "image", "image": pil})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        do_sample = temperature > 0.0
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **gen_kwargs)

        generation = generation[0][input_len:]
        return self.processor.decode(generation, skip_special_tokens=True)
