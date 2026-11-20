# Strata + MiniMax Studio

A local-first browser studio that runs Strata Qwen3.8-Flash-Next Q2_0 and ComfyUI MiniMax H3 Image on a single GPU. MiniMax prompt enhancement is disabled; the API workflow uses Qwen3-VL for image conditioning and contains no Qwen3.5 prompt-enhancement branch.

The prompt composer follows the local MiniMax-H3-Image text-to-image prompt-writing skill. Image controls mirror ComfyUI's ResolutionSelector and KSampler: aspect ratio, megapixel target, 32-pixel alignment, sampler, scheduler, and seed. Width and height are derived from the selected ratio and megapixel target.

## Setup

On first start, if `config.json` is missing, the app prompts for the Strata and ComfyUI repository paths. It expects the standard repository layouts, including each repository's `.venv` and service entry points. Model locations and other settings come from `data/config.example.json` and can be adjusted in the private `config.json` if needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python main.py start
.venv/bin/python main.py status
.venv/bin/python main.py stop
```

First start creates a private `config.json`, imports the API key from the Strata repository's configured source file, and generates the app access token. It does not print either secret or overwrite an existing config.

`start` validates the configured files, prepares the private Strata runtime config, then starts ComfyUI, Strata, and the Studio app. `status` reports all three services. `stop` stops them in reverse order. Open the browser at the address configured under `app` in `config.json` and enter the access token when prompted.

Keep `config.json` and the access token private. The config file is ignored by Git and must have owner-only permissions (`0600`). The app listens on loopback by default; enabling LAN HTTP requires explicit opt-in in the local config. ComfyUI's YAML configuration is generated from JSON; JSON remains the source of truth.