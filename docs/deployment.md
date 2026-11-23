# Deployment

**Status: the Docker image builds and runs locally (checked 23 Nov 2026); nothing is deployed yet.**

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

## GitHub Pages (static results page, live)

https://pkjx29.github.io/dp-ecg/ is a single static page built from the committed artifacts:

```bash
python scripts/build_site.py --out _site
```

`.github/workflows/pages.yml` runs that on every push to `main` that touches `artifacts/` or the script, and
deploys `_site/` with `actions/deploy-pages`. The repository's Pages source is set to "GitHub Actions". The page
needs no Python packages, data or checkpoints, and has no server code.

## Hugging Face Spaces (Docker SDK)

**Note (Nov 2026):** Hugging Face now requires a PRO subscription to host Docker and Gradio Spaces, including on
free CPU hardware; only static Spaces are free. That is why the results are published on GitHub Pages instead.


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
