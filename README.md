# gemma-4-31b-assistente

Deploy do **[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)** (Gemma 4 31B Dense Instruction-Tuned) no Replicate via Cog + GitHub Actions. Assistente de IA multimodal (texto + imagem) com saída em português brasileiro.

## Modelo

- **Família**: Gemma 4 (Google DeepMind)
- **Tamanho**: 30.7B parâmetros (Dense)
- **Modalidades**: Texto + Imagem (entrada) → Texto (saída)
- **Context window**: 256K tokens
- **Linguagens**: 140+ idiomas
- **Licença**: Apache 2.0

## API

### Endpoint

```
csviverdeia/gemma-4-31b-assistente
```

### Inputs

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `prompt` | string | "Como posso te ajudar hoje?" | Pergunta ou instrução |
| `image` | Path/URL | null | (Opcional) Imagem 1 |
| `image_2` | Path/URL | null | (Opcional) Imagem 2 |
| `image_3` | Path/URL | null | (Opcional) Imagem 3 |
| `image_4` | Path/URL | null | (Opcional) Imagem 4 |
| `system_prompt` | string | Persona PT-BR | Instrução de sistema |
| `max_new_tokens` | int | 2048 | 64-4096 |
| `temperature` | float | 0.2 | 0.0-2.0 (0.2 evita loops) |

### Exemplo curl

```bash
curl -s -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<version_id>",
    "input": {
      "prompt": "Explique o que é speculative decoding em 3 parágrafos.",
      "max_new_tokens": 1500
    }
  }'
```

### Exemplo Python

```python
import replicate
output = replicate.run(
    "csviverdeia/gemma-4-31b-assistente:<version_id>",
    input={
        "prompt": "Descreva esta imagem.",
        "image": open("foto.jpg", "rb"),
        "temperature": 0.2,
    }
)
print(output)
```

## Hardware e custo

- **GPU**: Nvidia H100 (80 GB VRAM) — ~US$ 5,04/h
- **Inferência típica**: 5-15 s por resposta de tamanho médio → ~US$ 0,007-0,02 por chamada
- **Cold start**: 3-5 min na primeira chamada após ociosidade (Replicate desliga GPU em idle)

## Build automático

Mudanças em `cog.yaml`, `predict.py` ou `script/**` na branch `main` disparam novo build automaticamente via GitHub Actions.

Runner usado: `ubuntu-latest-4-cores` (150 GB de disco — necessário pros 62 GB de pesos).

## Importante

Modelo de uso geral. Não validado para casos críticos (médico, jurídico, financeiro) — para esses, integre com workflow humano de revisão.
