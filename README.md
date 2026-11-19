# Strata + MiniMax Studio

A local-first browser studio that serializes Strata Qwen3.8-Flash-Next Q2_0 and ComfyUI MiniMax H3 Image on one GPU. MiniMax prompt enhancement is disabled; the API workflow uses Qwen3-VL for image conditioning and contains no Qwen3.5 prompt-enhancement branch.

The Strata prompt composer follows the local MiniMax-H3-Image T2I prompt-writing skill. Image controls mirror ComfyUI's ResolutionSelector and KSampler options: aspect ratio, megapixel target, 32-pixel alignment, sampler, scheduler, and seed. Dimensions are derived from ratio and megapixels; fixed width/height settings are not used.

中文网页操作说明见 [docs/USER_GUIDE_ZH.md](docs/USER_GUIDE_ZH.md)。

## Setup

Create the project environment and active config first. The initializer copies the API key from the configured Strata JSON file and generates the app access token. It does not read environment variables and does not print either secret.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python studio.py init
.venv/bin/python studio.py migrate-config
.venv/bin/python studio.py prepare
.venv/bin/python studio.py validate
.venv/bin/python studio.py set-vram-threshold 12000
.venv/bin/python studio.py start comfyui
.venv/bin/python studio.py start strata
.venv/bin/python studio.py status
.venv/bin/python studio.py serve
.venv/bin/python studio.py stop strata
.venv/bin/python studio.py stop comfyui
```

`config.json` is ignored by Git and must stay owner-only (`0600`). Edit its endpoint, path, listener and memory settings directly as needed. The ComfyUI YAML is generated from JSON because ComfyUI's CLI requires YAML; the JSON remains the source of truth.

The current active config serves the token-protected app at `http://10.178.234.58:8765`. Strata and ComfyUI use their default ports, privately on `127.0.0.1:8080` and `127.0.0.1:8188`.

For sign-in, read `app.access_token` from the private `config.json` and enter it in the browser's token prompt. Do not copy it into chat or share it.

`start strata` starts the API with lazy model loading. Services bind only to their configured loopback endpoints; start/stop checks process identity before acting. The first MiniMax acceptance run should use the configured 4MP size. The app binds to localhost by default; LAN HTTP requires an access token and explicit `allow_insecure_lan_http` opt-in.