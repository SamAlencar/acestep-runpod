# ACE-Step 1.5 RunPod Serverless worker

Worker local para gerar trilhas com [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (MIT) no RunPod Serverless.

- Código deste worker: MIT
- Pesos padrão `acestep-v15-turbo`: MIT
- Imagem: `ghcr.io/samalencar/acestep-runpod:latest`

O handler aceita o parâmetro `model`. O modelo `stable-audio-open-small` está reservado e ainda não é executado (exige aceite de termos, registro Community License e `HF_TOKEN`).

Não commitar chaves. `HF_TOKEN` e `RUNPOD_API_KEY` entram só como variáveis de ambiente do endpoint.
