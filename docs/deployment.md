# Deployment

**Status: deployment-ready files, NOT built or deployed.** No Docker daemon was available while developing.

## Local Docker

`docker-compose up --build` (from the repository root) is the alternative to the commands below.

```bash
docker build -t dp-ecg .
```
```bash
docker run --rm -p 7860:7860 dp-ecg
```
```bash
curl -s localhost:7860/health
```
```bash
curl -s localhost:7860/experiments | head -c 400
```

The image copies only `src/`, `configs/` and `artifacts/{metrics,figures,reports}`. It contains no data and no
checkpoints, so `/predict` returns 404 by design.

To enable `/predict` locally, mount a DP checkpoint read-only. Never publish the non-private baseline checkpoint.

## Hugging Face Spaces (Docker SDK)

1. Create a Space at https://huggingface.co/new-space and choose the **Docker** SDK.
2. Copy `deployment/huggingface/README.md` (which has the Space metadata, `sdk: docker`, `app_port: 7860`) to the
   Space's `README.md`.
3. Push this repository's files to the Space's git remote.
   - Do not push `data/` or `artifacts/checkpoints/`.
   - Use a token with write scope through `git credential` or the `HF_TOKEN` environment variable. Never put it in a
     file.
4. No secrets are needed: the app reads only committed artifacts.
5. The free CPU hardware is enough for the dashboard. Check current hardware limits and prices on the Space settings
   page.
6. Watch the build logs on the Space page. When the build finishes, open `/health`, then `/`.
7. To update the Space, regenerate the artifacts locally, commit them, and push again.

Check https://huggingface.co/docs/hub/en/spaces-sdks-docker for any changes before deploying.
