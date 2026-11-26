from __future__ import annotations

import json
from typing import Any, Callable

from .clients import ComfyUIClient
from .coordination import CoordinationError


def report_generation_event(
    callback: Callable[[dict[str, Any]], None] | None,
    event: dict[str, Any],
) -> None:
    if callback is not None:
        callback(event)


def generation_node_label(
    workflow: dict[str, Any],
    node_id: str,
    event_data: dict[str, Any],
) -> str:
    node = workflow.get(node_id, {})
    metadata = node.get("_meta", {}) if isinstance(node, dict) else {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    return str(
        title
        or (node.get("class_type") if isinstance(node, dict) else None)
        or event_data.get("display_node")
        or f"Node {node_id}"
    )


def read_comfy_progress(
    progress_socket: Any,
    prompt_id: str,
    workflow: dict[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
    completed_nodes: set[str],
    active_node: str | None,
) -> tuple[bool, str | None]:
    if progress_socket is None:
        return False, active_node
    total_nodes = len(workflow)
    sampler_node_id = next(
        (node_id for node_id, node in workflow.items()
         if isinstance(node, dict) and node.get("class_type") == "KSampler"),
        None,
    )
    sampler_node = workflow.get(sampler_node_id) if sampler_node_id is not None else None
    sampler_inputs = sampler_node.get("inputs", {}) if isinstance(sampler_node, dict) else {}
    configured_steps = sampler_inputs.get("steps") if isinstance(sampler_inputs, dict) else None
    sampler_steps = (
        configured_steps
        if isinstance(configured_steps, int) and not isinstance(configured_steps, bool) and configured_steps > 0
        else 1 if sampler_node_id is not None else 0
    )
    total_units = total_nodes - (1 if sampler_node_id is not None else 0) + sampler_steps

    def completed_work_units() -> int:
        sampler_completed = sampler_node_id is not None and sampler_node_id in completed_nodes
        return len(completed_nodes) - int(sampler_completed) + (sampler_steps if sampler_completed else 0)

    def overall_percent(work_units: float) -> float:
        return min(99.0, work_units / total_units * 100) if total_units else 0.0

    while True:
        try:
            raw = progress_socket.recv(timeout=0)
        except TimeoutError:
            return True, active_node
        except Exception:
            report_generation_event(callback, {
                "type": "phase",
                "message": "Live progress connection lost; waiting for ComfyUI completion",
            })
            return False, active_node
        if not isinstance(raw, str):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
            continue
        node_id = data.get("node")
        if not isinstance(node_id, str):
            continue
        label = generation_node_label(workflow, node_id, data)
        if event_type == "progress":
            value, maximum = data.get("value"), data.get("max")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
                    and maximum > 0):
                step_progress = min(1.0, max(0.0, value / maximum))
                active_units = 0.0
                if node_id not in completed_nodes:
                    active_units = step_progress * (sampler_steps if node_id == sampler_node_id else 1)
                completed_units = completed_work_units()
                report_generation_event(callback, {
                    "type": "progress", "node": node_id, "label": label,
                    "value": value, "max": maximum,
                    "completed_nodes": len(completed_nodes), "total_nodes": total_nodes,
                    "completed_units": completed_units, "total_units": total_units,
                    "overall_percent": overall_percent(completed_units + active_units),
                    "message": f"{label}: {value:g}/{maximum:g}",
                })
        elif event_type == "executing":
            if active_node is not None and active_node != node_id and active_node in workflow:
                completed_nodes.add(active_node)
            active_node = node_id
            completed_units = completed_work_units()
            report_generation_event(callback, {
                "type": "node", "state": "started", "node": node_id,
                "label": label, "completed_nodes": len(completed_nodes),
                "total_nodes": total_nodes,
                "completed_units": completed_units, "total_units": total_units,
                "overall_percent": overall_percent(completed_units),
                "message": f"Running {label}",
            })
        elif event_type == "executed":
            if node_id in workflow:
                completed_nodes.add(node_id)
            if active_node == node_id:
                active_node = None
            completed_units = completed_work_units()
            report_generation_event(callback, {
                "type": "node", "state": "complete", "node": node_id,
                "label": label, "completed_nodes": len(completed_nodes),
                "total_nodes": total_nodes,
                "completed_units": completed_units, "total_units": total_units,
                "overall_percent": overall_percent(completed_units),
                "message": f"Completed {label}",
            })


def wait_for_generation(
    comfy: ComfyUIClient,
    coordination: dict[str, Any],
    prompt_id: str,
    workflow: dict[str, Any],
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
    progress_socket: Any = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, str]:
    deadline = clock() + coordination["generation_timeout_s"]
    completed_nodes: set[str] = set()
    active_node: str | None = None
    while clock() < deadline:
        if progress_socket is not None:
            connected, active_node = read_comfy_progress(
                progress_socket, prompt_id, workflow, progress_callback,
                completed_nodes, active_node,
            )
            if not connected:
                progress_socket = None
        response = comfy.history(prompt_id)
        entry = response.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status", {})
            if isinstance(status, dict) and status.get("status_str") == "error":
                raise CoordinationError("ComfyUI reported a generation error")
            outputs = entry.get("outputs", {})
            for output in outputs.values() if isinstance(outputs, dict) else ():
                images = output.get("images", []) if isinstance(output, dict) else []
                if images:
                    image = images[0]
                    if all(isinstance(image.get(key), str) for key in ("filename", "subfolder", "type")):
                        return {key: image[key] for key in ("filename", "subfolder", "type")}
            if isinstance(status, dict) and status.get("status_str") == "success":
                raise CoordinationError("ComfyUI finished without a saved image")
        sleeper(coordination["poll_interval_s"])
    raise CoordinationError("ComfyUI generation timed out")
