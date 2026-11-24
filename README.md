# Comfy Studio

Comfy Studio is a local-first browser studio for ComfyUI image generation. The current setup runs Strata Qwen3.8-Flash-Next Q2_0 and ComfyUI MiniMax H3 Image on a single GPU. MiniMax prompt enhancement is disabled; the API workflows use Qwen3-VL for image conditioning and contain no Qwen3.5 prompt-enhancement branch.

The prompt composer follows the local MiniMax-H3-Image text-to-image prompt-writing skill. Image controls mirror ComfyUI's ResolutionSelector: aspect ratio, megapixel target, 32-pixel alignment, and seed. New generations use the Euler sampler and simple scheduler by default. Width and height are derived from the selected ratio and megapixel target.

Edit mode accepts one base image and up to four ordered reference images in PNG, JPEG, or WebP format (25 MiB and 100 megapixels maximum per image). It preserves the base image's aspect ratio and supports a 0.5–16 MP target. Studio uploads images through its authenticated API to ComfyUI's input directory and passes their local paths, in order, to Strata as OpenAI `image_url` parts. The managed Strata source config enables the Qwen3.8 `mmproj` with the CPU `strata-vision` reader, allowing Strata to inspect the base and reference images while preparing the edit prompt. ComfyUI writes generated images to its standard `output/` directory, then Studio moves each completed PNG to `temp/comfyui/<conversation-id>/<generation-id>.png`. Deleting a conversation removes its uploaded source images, while generated PNG files remain.

## Adding image modes

Mode IDs, input capabilities, resolution strategy, workflow requirements, and UI metadata are registered in `src/comfy_studio/modes.py`. The API publishes that registry to the browser, which builds the mode selector and mode-specific controls from it; storage validation and startup preflight also iterate the same registry. Configuration schema 3 keeps each mode's workflow, Strata system prompt, and model selections together under `generation_modes.<mode-id>`. Each model names a directory key from `comfyui.model_directories` and a filename, so different modes can use different model sets or directories without adding mode-specific top-level config fields. To add a mode, add a `ModeDefinition` and its `generation_modes` entry in `data/config.example.json`, then add a workflow assembler in `src/comfy_studio/workflow.py` if it needs a new graph strategy. Add focused tests for its input rules, dimensions, model requirements, and workflow mapping. Modes can reuse an existing workflow strategy when their graph structure matches. Existing schema 1 and 2 private configs are normalized to schema 3 in memory when loaded.

## Setup

On first start, if `config.json` is missing, the app prompts for the Strata and ComfyUI repository paths. It expects the standard repository layouts, including each repository's `.venv` and service entry points. Model locations and other settings come from `data/config.example.json` and can be adjusted in the private `config.json` if needed.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python main.py
```

On Windows, use the matching PowerShell commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe main.py
```

First start creates a private `config.json`, imports an API key from the Strata source config when present (otherwise it generates a private key for the managed Strata service), and generates the app access token. It does not print either secret or overwrite an existing config.

Run `main.py` without arguments. It validates the configured files, prepares the private Strata runtime config, preflights all service PID files and ports, then starts or takes over ComfyUI, Strata, and the Studio app before staying in the foreground. A running service is taken over only when its process matches the configured service; Studio records its PID and stops it along with services it started when you press Ctrl+C, in reverse order. If Studio exits unexpectedly, the foreground process performs the same cleanup. If a Strata or ComfyUI process matches the configured server script and port but is not recorded in the local PID file, Studio adopts it without restarting it. An external Strata endpoint whose process cannot be identified, multiple matching processes, an invalid or mismatched live PID, or an unrelated process occupying a configured port causes startup to abort; Studio does not stop processes it cannot verify. Service availability is visible in the Studio UI. Open the browser at the address configured under `app` in `config.json` and enter the access token when prompted.

On Windows, repository virtual environments are expected at `.venv\Scripts\python.exe`; for the Strata terminal layout, Studio also checks the Strata repository root if `terminal\.venv` is absent. Windows private-file ACLs are restricted to the current user with `icacls`.

Keep `config.json` and the access token private. The config file is ignored by Git and must have owner-only permissions (`0600`). The app listens on loopback by default; enabling LAN HTTP requires explicit opt-in in the local config. ComfyUI's YAML configuration is generated from JSON; JSON remains the source of truth.