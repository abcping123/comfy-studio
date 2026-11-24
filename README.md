# Comfy Studio

Comfy Studio is a local-first browser studio for ComfyUI image generation. It can use either Strata Qwen3.8-Flash-Next Q2_0 or the local Bonsai 2 27B llama-server as its image-prompt chat backend, alongside ComfyUI MiniMax H3 Image. Select the backend when starting Studio. MiniMax prompt enhancement is disabled; the API workflows use Qwen3-VL for image conditioning and contain no Qwen3.5 prompt-enhancement branch.

The prompt composer follows the local MiniMax-H3-Image text-to-image prompt-writing skill. Image controls mirror ComfyUI's ResolutionSelector: aspect ratio, megapixel target, 32-pixel alignment, and seed. New generations use the Euler sampler and simple scheduler by default. Width and height are derived from the selected ratio and megapixel target.

Thinking-effort controls follow the selected chat backend. Strata exposes Off, Low, Medium, and High. Studio defaults Bonsai 2 to Off while exposing Medium and XHigh; the model's native default is XHigh. Its template does not support High, while Low is accepted but does not meaningfully reduce reasoning.

Edit mode accepts one base image and up to four ordered reference images in PNG, JPEG, or WebP format (25 MiB and 100 megapixels maximum per image). It preserves the base image's aspect ratio and supports a 0.5–16 MP target. Studio uploads images through its authenticated API to the ComfyUI input directory and sends them in order to the selected chat backend. Strata receives local image paths; Bonsai receives data URLs for its llama.cpp multimodal projector. Each conversation stores its database, context, input images, and generated images directly in `temp/session/<conversation-id>/`. ComfyUI reads and writes in that session directory. Studio also copies each completed PNG to the repository-root `output/` directory with a `yyyy-mm-dd_hh<h>mm_ss.png` filename. Deleting a conversation removes its session directory but keeps the archived copy in `output/`.

## Adding image modes

Mode IDs, input capabilities, resolution strategy, workflow requirements, and UI metadata are registered in `src/comfy_studio/modes.py`. The API publishes that registry to the browser, which builds the mode selector and mode-specific controls from it; storage validation and startup preflight also iterate the same registry. Configuration schema 6 keeps each mode's workflow, system prompt, and model selections together under `generation_modes.<mode-id>`, and nests provider configuration under `chat_backends.<backend-id>`. Each ComfyUI model names a directory key from `comfyui.model_directories` and a filename, so different modes can use different model sets or directories without adding mode-specific top-level config fields. To add a mode, add a `ModeDefinition` and its `generation_modes` entry in `data/config.example.json`, then add a workflow assembler in `src/comfy_studio/workflow.py` if it needs a new graph strategy. Add focused tests for its input rules, dimensions, model requirements, and workflow mapping. Modes can reuse an existing workflow strategy when their graph structure matches. Private configurations must use schema 6.

## Adding chat backends

Chat providers are described in `src/comfy_studio/chat_backends.py`. The registry is the source of truth for backend IDs, UI labels, supported/default reasoning efforts, client construction, service-management adapters, initialization requirements, context persistence, image transport, and GPU handoff behavior. API options, CLI choices, context validation, and UI service labels use that metadata instead of maintaining separate Strata/Bonsai lists.

To add a provider:

1. Implement the `ChatBackend` protocol in `src/comfy_studio/clients.py` (reuse `OpenAIChatClient` when the endpoint uses the compatible chat-completions API), then add its client factory and `ChatBackendDefinition`.
2. Add its configuration under `chat_backends.<backend-id>` in `data/config.example.json` and register a validator in `src/comfy_studio/settings.py`. Keep secrets in the private `config.json`.
3. If it runs as a local process, set a service adapter and implement its command and process-identity rules in `src/comfy_studio/services.py`. For hosted/stateless providers, use the appropriate non-managed service and preparation capabilities instead.
4. Choose context persistence, image transport, initialization, and generation handoff capabilities deliberately; add any provider-specific runtime preparation to `src/comfy_studio/initialize.py`.
5. Add focused tests for client requests/streaming, config validation, API options and defaults, process lifecycle (if managed), context persistence, and image input (if supported).

Prefer adding a new adapter only when a provider's protocol or lifecycle differs; providers with the same service or context strategy can reuse the existing adapter.

## Setup

On first start, if `config.json` is missing, Strata mode prompts for the Strata and ComfyUI repository paths; Bonsai mode only needs the ComfyUI repository path. The Bonsai paths and server options in `data/config.example.json` follow the supplied `run-bonsai-27b.sh` reference. Adjust the selected provider's section under `chat_backends` in the private `config.json` if the local model or binary paths differ. A leading `~` in configured filesystem paths expands to the current user's home directory.

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

Run `main.py` with a registered backend ID such as `strata` or `bonsai` to select the chat backend. Omitting the argument keeps Strata as the default. Studio validates the selected backend's configured files, prepares its runtime config when required, and preflights ComfyUI, every registered managed backend, and the Studio app port before changing services. It then runs the Studio app directly in the foreground process without creating a Studio PID file.

At startup, Studio reuses a service only when its PID file identifies the single matching service process and, on POSIX, that process is isolated from Studio's terminal process group. On Windows, existing services are restarted because the PID file cannot prove console-process-group isolation. If the PID is missing or mismatched, matching service processes are stopped and restarted in a new process group; duplicate instances and instances sharing Studio's process group are also restarted. A mismatched PID is never signaled unless its process identity confirms it is the configured service. Other running managed backends are stopped before the selected one starts so large models do not compete for GPU memory. Studio stops managed services in reverse order when you press Ctrl+C. If Studio exits unexpectedly, the foreground process performs the same cleanup. Unknown processes are never stopped, and an unidentifiable process occupying a configured service port causes startup to abort. Bonsai's model process is stopped before ComfyUI generation to release GPU memory, then started again on the next chat; transcript-based backends resume from Studio's stored conversation history. Native-session backends keep their own save/restore context. Service availability is visible in the Studio UI. Open the browser at the address configured under `app` in `config.json` and enter the access token when prompted.

```sh
.venv/bin/python main.py strata
.venv/bin/python main.py bonsai
```

On Windows, repository virtual environments are expected at `.venv\Scripts\python.exe`; for the Strata terminal layout, Studio also checks the Strata repository root if `terminal\.venv` is absent. Windows private-file ACLs are restricted to the current user with `icacls`.

Keep `config.json` and the access token private. The config file is ignored by Git and must have owner-only permissions (`0600`). The app listens on loopback by default; enabling LAN HTTP requires explicit opt-in in the local config. ComfyUI's YAML configuration is generated from JSON; JSON remains the source of truth.