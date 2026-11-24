from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from .clients import ChatBackend
    from .settings import Settings


@dataclass(frozen=True)
class ChatBackendDefinition:
    id: str
    label: str
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str
    config_schema: str
    client_factory: Callable[[Settings], ChatBackend]
    managed_service: bool
    service_adapter: str | None
    setup_adapter: Literal["strata", "comfyui", "preconfigured"]
    runtime_preparer: Callable[[Settings], None] | None
    loaded_state: Literal["client", "process", "stateless"]
    endpoint_probe: Literal["strata-model"] | None
    chat_preparation: Literal["native-session", "on-demand-service", "stateless"]
    context_strategy: Literal["native-session", "transcript"]
    image_transport: Literal["local-path", "data-url", "none"]
    generation_handoff: Literal["native-session", "stop-service", "none"]
    required_files: tuple[tuple[str, str], ...] = ()
    executable_files: tuple[str, ...] = ()
    requires_api_key: bool = False
    startup_timeout_key: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"invalid chat backend ID: {self.id!r}")
        if not self.reasoning_efforts or self.default_reasoning_effort not in self.reasoning_efforts:
            raise ValueError(f"{self.id} must define a supported default reasoning effort")
        if self.managed_service != (self.service_adapter is not None):
            raise ValueError(f"{self.id} service management and adapter settings disagree")
        if self.chat_preparation == "native-session" and self.context_strategy != "native-session":
            raise ValueError(f"{self.id} native-session preparation requires native-session context")
        if self.chat_preparation == "on-demand-service" and not self.managed_service:
            raise ValueError(f"{self.id} on-demand preparation requires a managed service")
        if self.generation_handoff == "stop-service" and not self.managed_service:
            raise ValueError(f"{self.id} stop-service handoff requires a managed service")
        if self.startup_timeout_key and not self.managed_service:
            raise ValueError(f"{self.id} startup timeout requires a managed service")


def _strata_client(settings: Settings) -> ChatBackend:
    from .clients import StrataClient

    return StrataClient(settings)


def _bonsai_client(settings: Settings) -> ChatBackend:
    from .clients import BonsaiClient

    return BonsaiClient(settings)


def _prepare_strata_runtime(settings: Settings) -> None:
    from .initialize import prepare_runtime_files

    prepare_runtime_files(settings)


CHAT_BACKEND_DEFINITIONS = (
    ChatBackendDefinition(
        id="strata",
        label="Strata",
        reasoning_efforts=("none", "low", "medium", "high"),
        default_reasoning_effort="none",
        config_schema="strata",
        client_factory=_strata_client,
        managed_service=True,
        service_adapter="strata",
        setup_adapter="strata",
        runtime_preparer=_prepare_strata_runtime,
        loaded_state="client",
        endpoint_probe="strata-model",
        chat_preparation="native-session",
        context_strategy="native-session",
        image_transport="local-path",
        generation_handoff="native-session",
        required_files=(
            ("python", "Strata Python"),
            ("server_script", "Strata server script"),
            ("source_config_json", "Strata source config"),
        ),
        executable_files=("python",),
        requires_api_key=True,
    ),
    ChatBackendDefinition(
        id="bonsai",
        label="Bonsai",
        reasoning_efforts=("none", "medium", "xhigh"),
        default_reasoning_effort="none",
        config_schema="bonsai",
        client_factory=_bonsai_client,
        managed_service=True,
        service_adapter="llama-server",
        setup_adapter="comfyui",
        runtime_preparer=None,
        loaded_state="process",
        endpoint_probe=None,
        chat_preparation="on-demand-service",
        context_strategy="transcript",
        image_transport="data-url",
        generation_handoff="stop-service",
        required_files=(
            ("server_binary", "Bonsai llama-server"),
            ("model_file", "Bonsai model"),
            ("mmproj_file", "Bonsai multimodal projector"),
        ),
        executable_files=("server_binary",),
        startup_timeout_key="startup_timeout_s",
    ),
)

CHAT_BACKEND_REGISTRY = {definition.id: definition for definition in CHAT_BACKEND_DEFINITIONS}
CHAT_BACKENDS = tuple(CHAT_BACKEND_REGISTRY)


def get_chat_backend(backend_id: str) -> ChatBackendDefinition:
    try:
        return CHAT_BACKEND_REGISTRY[backend_id]
    except KeyError as error:
        choices = ", ".join(CHAT_BACKENDS)
        raise ValueError(f"unknown chat backend {backend_id!r}; expected one of: {choices}") from error


def create_chat_backend(settings: Settings) -> ChatBackend:
    return get_chat_backend(settings.chat_backend).client_factory(settings)
