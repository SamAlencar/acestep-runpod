# ACE-Step 1.5 + Stable Audio Open Small — RunPod Serverless worker

Worker para gerar trilhas e SFX no RunPod Serverless.

| Modelo (`model`) | Licença | Uso |
|---|---|---|
| `acestep-v15-turbo` (padrão) | MIT | Trilhas 10–60 s, 48 kHz |
| `acestep-v15-xl-turbo` | MIT | Trilhas, mais VRAM |
| `stable-audio-open-small` | Stability AI Community License | SFX e clips até 11 s, 44,1 kHz. Exige `HF_TOKEN` no *endpoint*, aceite dos termos no Hugging Face e registro em https://stability.ai/community-license se o uso for comercial. |

- Código deste worker: MIT
- Imagem: `ghcr.io/samalencar/acestep-runpod:latest`

O token do Hugging Face entra só como variável de ambiente do endpoint RunPod. Não commitar chaves, não passar `HF_TOKEN` como build-arg, não gravar segredos no repositório.
