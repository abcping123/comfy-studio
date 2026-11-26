from __future__ import annotations

import json
from typing import Any, Callable

from .chat_backends import get_chat_backend
from .clients import BackendError
from .coordination import ChatOutcome, CoordinationError
from .modes import DEFAULT_MODE_ID, get_mode
from .resolution import normalize_aspect_ratio
from .session_assets import SessionAssets
from .settings import Settings
from .storage import ConversationStore, StorageError


def chat_content(response: dict[str, Any]) -> tuple[str, str]:
    try:
        message = response["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise CoordinationError("chat backend returned no assistant message") from error
    if isinstance(content, str):
        answer = content
    elif isinstance(content, list):
        answer = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        raise CoordinationError("chat backend returned an unsupported message content type")
    if not answer.strip():
        raise CoordinationError("chat backend returned an empty assistant message")
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        raise CoordinationError("chat backend returned unsupported reasoning content")
    return answer, reasoning


def parse_chat_response(response: dict[str, Any]) -> tuple[str, str, str, str, str]:
    raw, reasoning = chat_content(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CoordinationError("chat backend response was not the configured JSON object") from error
    if not isinstance(parsed, dict) or set(parsed) != {"reply", "image_prompt", "aspect_ratio"}:
        raise CoordinationError("chat backend response must contain only reply, image_prompt, and aspect_ratio")
    reply, image_prompt, aspect_ratio = parsed["reply"], parsed["image_prompt"], parsed["aspect_ratio"]
    if (not isinstance(reply, str) or not reply.strip() or not isinstance(image_prompt, str)
            or not isinstance(aspect_ratio, str)):
        raise CoordinationError("chat backend reply, image_prompt, and aspect_ratio must be strings")
    return reply.strip(), image_prompt.strip(), aspect_ratio.strip(), raw, reasoning


def build_chat_messages(
    settings: Settings,
    store: ConversationStore,
    assets: SessionAssets,
    conversation_id: str,
    user_text: str,
    current_image_prompt: str | None,
    current_aspect_ratio: str | None,
) -> tuple[str, str, list[dict[str, Any]], str | None]:
    conversation_id = store.validate_id(conversation_id)
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise CoordinationError("conversation does not exist")
    if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 20000:
        raise CoordinationError("message must contain 1 to 20000 characters")
    if current_image_prompt is None:
        current_image_prompt = conversation["image_prompt"]
    if not isinstance(current_image_prompt, str) or len(current_image_prompt) > 20000:
        raise CoordinationError("current image prompt must contain at most 20000 characters")
    generation = settings.data["generation"]
    if current_aspect_ratio is None or not current_aspect_ratio.strip():
        current_aspect_ratio = conversation["aspect_ratio"] or generation["default_aspect_ratio"]
    current_aspect_ratio = current_aspect_ratio.strip()
    available_ratios = {item["label"] for item in generation["aspect_ratios"]}
    if current_aspect_ratio not in available_ratios:
        raise CoordinationError("current aspect ratio is not supported")
    image_mode = conversation.get("image_mode", DEFAULT_MODE_ID)
    try:
        mode = get_mode(image_mode)
    except ValueError as error:
        raise CoordinationError(str(error)) from error
    source_images = (
        store.source_images(conversation_id)
        if mode.send_source_images_to_chat else []
    )
    source_image_count = len(source_images)
    system_prompt_path = settings.path(f"generation_modes.{mode.id}.system_prompt_file")
    try:
        system_prompt_text = system_prompt_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise CoordinationError(f"cannot read system prompt file: {system_prompt_path}") from error
    if not system_prompt_text:
        raise CoordinationError(f"system prompt file is empty: {system_prompt_path}")
    system_prompt = system_prompt_text + (
        "\n最新用户消息开头包含 current_editor_state JSON，其中 current_image_prompt 是右侧当前草稿，"
        "current_aspect_ratio 是当前画幅选择。除非用户本轮明确要求更改，否则延续草稿并保留该画幅。"
    )
    editor_state: dict[str, Any] = {
        "current_image_prompt": current_image_prompt,
        "current_aspect_ratio": current_aspect_ratio,
    }
    preserve_aspect_ratio = None
    if mode.send_source_images_to_chat:
        editor_state.update({"image_mode": mode.id, "source_image_count": source_image_count})
    if mode.resolution_strategy == "source-image":
        preserve_aspect_ratio = current_aspect_ratio
    current_editor_state = json.dumps(editor_state, ensure_ascii=False)
    current_user_message = (
        f"current_editor_state:\n{current_editor_state}\n\n"
        f"user_request:\n{user_text.strip()}"
    )
    message_content: str | list[dict[str, Any]] = current_user_message
    if mode.send_source_images_to_chat and source_images:
        backend = get_chat_backend(settings.chat_backend)
        if backend.image_transport == "none":
            raise CoordinationError(f"{backend.label} does not accept source images for chat")
        message_content = assets.encode_source_images_for_chat(
            conversation_id, mode, source_images, backend.label, current_user_message,
        )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(store.messages(conversation_id))
    messages.append({"role": "user", "content": message_content})
    return conversation_id, user_text.strip(), messages, preserve_aspect_ratio


def finish_chat(
    settings: Settings,
    store: ConversationStore,
    save_context: Callable[[str], None],
    conversation_id: str,
    user_text: str,
    response: dict[str, Any],
    preserve_aspect_ratio: str | None = None,
) -> ChatOutcome:
    answer, image_prompt, suggested_ratio, raw_answer, reasoning_content = parse_chat_response(response)
    if preserve_aspect_ratio is not None:
        aspect_ratio, was_adjusted = preserve_aspect_ratio, False
    else:
        try:
            aspect_ratio, was_adjusted = normalize_aspect_ratio(
                suggested_ratio, settings.data["generation"]["aspect_ratios"],
            )
        except ValueError as error:
            raise CoordinationError(str(error)) from error
    display_answer = answer
    ratio_short = aspect_ratio.split(" ", 1)[0]
    explained_adjustment = suggested_ratio in answer and ratio_short in answer
    if was_adjusted and not explained_adjustment:
        display_answer += f"（{suggested_ratio} 不在原生选项中，已采用最接近的 {aspect_ratio}。）"
    store.append_exchange(
        conversation_id, user_text, raw_answer, image_prompt,
        aspect_ratio, display_answer, reasoning_content,
    )
    try:
        save_context(conversation_id)
        return ChatOutcome(
            answer=display_answer,
            image_prompt=image_prompt,
            aspect_ratio=aspect_ratio,
            context_saved=True,
            reasoning_content=reasoning_content,
        )
    except (BackendError, StorageError, CoordinationError) as error:
        return ChatOutcome(
            answer=display_answer,
            image_prompt=image_prompt,
            aspect_ratio=aspect_ratio,
            context_saved=False,
            warning=f"Reply received, but the conversation context was not saved: {error}",
            reasoning_content=reasoning_content,
        )
