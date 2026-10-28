"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by rewriting a request or
wrapping the actual execution callback. Agent-loop call sites and plugins share this vocabulary.
"""

from __future__ import annotations

import ipaddress
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"
LLM_ADMISSION_MIDDLEWARE = "llm_admission"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE, TOOL_EXECUTION_MIDDLEWARE, LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE, LLM_ADMISSION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LLMAdmissionDecision:
    """Trusted pre-request decision for a local model route.

    ``policy_ids`` and ``evidence_ids`` identify the controller records that allowed a
    request. A denial is terminal for the current turn and must never fall through to the
    provider or a configured model fallback.
    """

    allowed: bool
    code: str
    reason: str
    policy_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    lease_ids: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None


class LocalModelAdmissionBlockedError(RuntimeError):
    """Terminal refusal raised at a provider adapter before local inference begins."""


class LocalGPUAdmissionController:
    """Lifecycle contract for a trusted local-inference admission backend.

    Implementations must atomically validate capability and acquire a task/run-bound GPU
    reservation in ``acquire``. The provider adapter must validate that reservation immediately
    before each inference request. ``release`` is called from the outer turn finalizer and must
    be idempotent and holder-qualified. This interface intentionally has no default backend:
    without an installed controller, local inference remains denied.
    """

    def __call__(self, *, action: str, lease_ids: tuple[str, ...] = (), **context: Any) -> dict[str, Any]:
        """Plugin middleware entry point used by the pre-request gate and turn finalizer."""
        if action == "acquire":
            return self.acquire(**context)
        if action == "validate":
            return self.validate(lease_ids=lease_ids, **context)
        if action == "release":
            return self.release(lease_ids=lease_ids, **context)
        if action == "verify_completion":
            return self.verify_completion(**context)
        return {
            "decision": "block", "policy_id": "invalid-action",
            "reason": "Unknown local GPU admission lifecycle action",
        }

    def acquire(self, **scope: Any) -> dict[str, Any]:  # pragma: no cover - interface contract
        raise NotImplementedError

    def validate(self, *, lease_ids: tuple[str, ...], **scope: Any) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def release(self, *, lease_ids: tuple[str, ...], **scope: Any) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def verify_completion(self, **scope: Any) -> dict[str, Any]:  # pragma: no cover
        """Independently verify the exact task artifact before Kanban terminal mutation."""
        raise NotImplementedError


def observer_payload(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    An LLM request can carry clients/callbacks/file handles; a hard ``deepcopy`` failure would
    otherwise abort the whole request-middleware pass.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        return dict(payload) if isinstance(payload, dict) else payload


def _apply_request_chain(
    kind: str, payload_key: str, trace: list[dict[str, Any]], original: Any, **kwargs: Any
) -> RequestMiddlewareResult:
    """Feed ``kwargs[payload_key]`` through every ``kind`` middleware; each may return ``{payload_key: {...}}``."""
    from hermes_cli.plugins import invoke_middleware

    current = kwargs[payload_key]
    for result in invoke_middleware(kind, _payload_key=payload_key, **middleware_payload(**kwargs)):
        if not isinstance(result, dict):
            continue
        next_payload = result.get(payload_key)
        if not isinstance(next_payload, dict):
            continue
        current = _safe_copy(next_payload)
        entry = {
            key: value
            for key in ("source", "reason", "name")
            if isinstance(value := result.get(key), str) and value
        }
        trace.append(entry or {"source": "plugin"})
    return RequestMiddlewareResult(
        payload=current, original_payload=original, changed=bool(trace), trace=trace,
    )


def apply_llm_request_middleware(request: dict[str, Any], **context: Any) -> RequestMiddlewareResult:
    """Apply registered LLM request middleware; ``{"request": {...}}`` replaces the provider kwargs."""
    from hermes_cli.plugins import has_middleware

    if not has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(payload=request, original_payload=request)

    original_request = _safe_copy(request)
    return _apply_request_chain(
        LLM_REQUEST_MIDDLEWARE, "request", [], original_request,
        request=_safe_copy(original_request), original_request=original_request, **context,
    )


def apply_tool_request_middleware(
    tool_name: str, args: dict[str, Any], **context: Any
) -> RequestMiddlewareResult:
    """Apply registered tool request middleware; ``{"args": {...}}`` replaces the effective tool
    arguments before hooks, guardrails, approvals, and execution see them."""
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: list[dict[str, Any]] = []

    session_id = str(context.get("session_id") or "")
    skip_relay = bool(context.pop("skip_relay", False))
    if session_id and not skip_relay:
        from agent import relay_runtime

        relay_args = relay_runtime.apply_tool_request_intercepts(
            session_id=session_id, tool_name=tool_name, args=current_args)
        if relay_args != current_args:
            current_args = _safe_copy(relay_args)
            trace.append({"source": "nemo_relay"})

    from hermes_cli.plugins import has_middleware

    if not has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args if not trace else current_args, original_payload=args,
            changed=bool(trace), trace=trace,
        )
    return _apply_request_chain(
        TOOL_REQUEST_MIDDLEWARE, "args", trace, original_args,
        tool_name=tool_name, args=current_args, original_args=original_args, **context,
    )


def run_llm_execution_middleware(
    request: dict[str, Any], next_call: Callable[[dict[str, Any]], Any], **context: Any) -> Any:
    """Run provider execution through registered LLM execution middleware."""
    return _run_execution_chain(
        LLM_EXECUTION_MIDDLEWARE, next_call,
        request=request, original_request=context.pop("original_request", request), **context)


def _is_local_llm_route(provider: Any, base_url: Any) -> bool:
    """Conservatively identify local inference routes that require explicit admission."""
    provider_name = str(provider or "").strip().lower()
    if any(name in provider_name for name in ("ollama", "llama.cpp", "llamacpp", "lmstudio", "mlx", "vllm")):
        return True
    try:
        host = urlsplit(str(base_url or "")).hostname
    except ValueError:
        # A configured but malformed endpoint is not evidence that the route is remote.
        return bool(str(base_url or "").strip())
    if not host:
        return False
    normalized_host = host.lower().rstrip(".")
    if normalized_host == "localhost" or normalized_host.endswith((
        ".localhost", ".local", ".lan", ".internal", ".home.arpa",
    )):
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_private or address.is_link_local
    except ValueError:
        # Single-label names are commonly used for machines on a local network.
        return "." not in normalized_host


def _is_type_only_json_schema(schema: Any) -> bool:
    """Allow only the structural/type subset used by the local extraction capability."""
    if type(schema) is not dict:
        return False
    if set(schema) == {"anyOf"}:
        choices = schema.get("anyOf")
        return type(choices) is list and bool(choices) and all(
            _is_type_only_json_schema(choice) for choice in choices
        )
    if not isinstance(schema.get("type"), str):
        return False
    kind = schema["type"]
    if kind == "object":
        if set(schema) == {"type"}:
            return True
        properties = schema.get("properties")
        required = schema.get("required")
        return (
            set(schema) == {"type", "properties", "required", "additionalProperties"}
            and type(properties) is dict and type(required) is list
            and required == list(properties) and schema.get("additionalProperties") is False
            and all(isinstance(key, str) and _is_type_only_json_schema(value)
                    for key, value in properties.items())
        )
    if kind == "array":
        if set(schema) == {"type"}:
            return True
        return set(schema) == {"type", "items"} and _is_type_only_json_schema(schema["items"])
    if kind in {"string", "integer", "number", "boolean", "null"}:
        return set(schema) == {"type"}
    return False


def _valid_response_format(value: Any) -> bool:
    return (
        type(value) is dict and set(value) == {"type", "json_schema"}
        and value.get("type") == "json_schema"
        and type(value.get("json_schema")) is dict
        and set(value["json_schema"]) == {"schema"}
        and _is_type_only_json_schema(value["json_schema"].get("schema"))
    )


def _admission_scope_string(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def run_llm_admission_middleware(**context: Any) -> LLMAdmissionDecision:
    """Resolve fail-closed admission before a local model provider request.

    The single registered ``llm_admission`` callback receives ``action=acquire`` for the
    first request in a turn and ``action=validate`` on subsequent requests. It must return a
    mapping with ``decision`` (``allow`` or ``block``), ``policy_id``, ``reason``, and, for
    allow, ``evidence_id``, ``lease_ids``, and an exact current-request ``scope`` binding.
    Callback errors, missing identity, stale/replayed scope, and
    malformed results deny the request. Non-local provider routes are unaffected. The callback
    receives only route/task/session metadata, not prompt contents.
    """
    if not _is_local_llm_route(context.get("provider"), context.get("base_url")):
        if context.get("capability_id") and (context.get("kanban_task_id") or context.get("task_id")):
            return LLMAdmissionDecision(
                False, "local_capability_route_changed",
                "A local-GPU task cannot fall through to a remote model route or fallback",
            )
        return LLMAdmissionDecision(True, "not_local_route", "Admission is only required for local routes")

    scope_fields = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db", "kanban_workspace",
        "kanban_claim_lock", "capability_id", "task_contract_sha256", "api_request_id",
        "context_length", "ollama_num_ctx", "session_id", "model", "provider", "base_url",
        "streaming", "has_tools", "api_call_count", "existing_response_format",
    )
    expected_scope = {key: _admission_scope_string(context.get(key)) for key in scope_fields}
    if not all(expected_scope.values()):
        return LLMAdmissionDecision(
            False, "missing_admission_identity",
            "Local model admission requires task/run, capability, contract, request, session, model, and endpoint identity",
        )

    try:
        from hermes_cli.plugins import _delivery_manager, has_middleware

        if not has_middleware(LLM_ADMISSION_MIDDLEWARE):
            return LLMAdmissionDecision(
                False, "controller_missing", "No local-model admission controller is enabled",
            )
        callbacks = list(_delivery_manager()._middleware.get(LLM_ADMISSION_MIDDLEWARE, []))
    except Exception as exc:
        logger.error("Local-model admission discovery failed closed: %s", exc)
        return LLMAdmissionDecision(
            False, "controller_unavailable", "Local-model admission controller could not be loaded",
        )
    if len(callbacks) != 1:
        return LLMAdmissionDecision(
            False, "controller_configuration_invalid",
            "Exactly one local-model admission controller must be enabled",
        )

    callback = callbacks[0]
    existing_lease_ids = tuple(context.pop("existing_lease_ids", ()) or ())
    action = "validate" if existing_lease_ids else "acquire"
    try:
        result = callback(action=action, lease_ids=existing_lease_ids, **middleware_payload(**context))
    except Exception as exc:
        logger.error("Local-model admission callback failed closed: %s", exc)
        return LLMAdmissionDecision(
            False, "controller_error", "Local-model admission check failed; no model request was sent",
        )
    if not isinstance(result, dict):
        return LLMAdmissionDecision(
            False, "invalid_controller_result", "Admission controller returned no structured decision",
        )
    decision = result.get("decision")
    policy_id = result.get("policy_id")
    reason = result.get("reason")
    if decision not in {"allow", "block"} or not all(
        isinstance(value, str) and value.strip() for value in (policy_id, reason)
    ):
        return LLMAdmissionDecision(
            False, "invalid_controller_result", "Admission controller returned an incomplete decision",
        )
    if decision == "block":
        return LLMAdmissionDecision(False, "controller_block", reason.strip(), (policy_id.strip(),))
    evidence_id = result.get("evidence_id")
    lease_ids = result.get("lease_ids")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        return LLMAdmissionDecision(
            False, "missing_admission_evidence", "Admission allowed a request without an evidence ID",
            (policy_id.strip(),),
        )
    if not isinstance(lease_ids, (list, tuple)) or not lease_ids or not all(
        isinstance(item, str) and item.strip() for item in lease_ids
    ):
        return LLMAdmissionDecision(
            False, "missing_gpu_lease", "Admission allowed a local request without GPU lease IDs",
            (policy_id.strip(),), (evidence_id.strip(),),
        )
    lease_ids = tuple(item.strip() for item in lease_ids)
    if existing_lease_ids and lease_ids != existing_lease_ids:
        return LLMAdmissionDecision(
            False, "admission_lease_changed", "Controller did not validate the current task/run lease",
            (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
        )
    if result.get("scope") != expected_scope:
        return LLMAdmissionDecision(
            False, "admission_scope_mismatch",
            "Admission is not bound to the current task run, API request, and exact model route",
            (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
        )
    response_format = result.get("response_format")
    if response_format is not None:
        if not _valid_response_format(response_format):
            return LLMAdmissionDecision(
                False, "invalid_admission_response_format",
                "Admission returned a malformed or value-constraining JSON Schema",
                (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
            )
        if context.get("streaming") or context.get("has_tools"):
            return LLMAdmissionDecision(
                False, "request_format_incompatible",
                "This local capability requires a non-streaming, tool-free request",
                (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
            )
        phase = context.get("request_format_phase")
        if phase == "prepare":
            if context.get("existing_response_format") or int(context.get("api_call_count") or 0) != 1:
                return LLMAdmissionDecision(
                    False, "request_format_conflict",
                    "This local capability permits one first request with its registry-derived schema",
                    (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
                )
        elif phase == "verify":
            if context.get("supplied_response_format") != response_format:
                return LLMAdmissionDecision(
                    False, "request_format_not_applied",
                    "Provider request does not carry the registry-pinned JSON Schema",
                    (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
                )
        else:
            return LLMAdmissionDecision(
                False, "request_format_phase_missing",
                "Local JSON Schema admission was not applied at the request boundary",
                (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
            )
    return LLMAdmissionDecision(
        True, "admitted", "Admission passed", (policy_id.strip(),), (evidence_id.strip(),), lease_ids,
        response_format,
    )


def require_local_model_admission(agent: Any, api_kwargs: dict[str, Any], *, streaming: bool = False) -> None:
    """Revalidate the task-bound GPU lease at each sync or streaming provider seam."""
    if not _is_local_llm_route(getattr(agent, "provider", ""), getattr(agent, "base_url", "")):
        return
    import os
    from uuid import uuid4

    prior = getattr(agent, "_local_gpu_admission_scope", {})
    if not isinstance(prior, dict):
        prior = {}
    context = {
        **prior,
        "task_id": prior.get("task_id") or os.environ.get("HERMES_KANBAN_TASK", ""),
        "kanban_task_id": os.environ.get("HERMES_KANBAN_TASK", ""),
        "task_run_id": os.environ.get("HERMES_KANBAN_RUN_ID", ""),
        "kanban_board": os.environ.get("HERMES_KANBAN_BOARD", ""),
        "kanban_db": os.environ.get("HERMES_KANBAN_DB", ""),
        "kanban_workspace": os.environ.get("HERMES_KANBAN_WORKSPACE", ""),
        "kanban_claim_lock": os.environ.get("HERMES_KANBAN_CLAIM_LOCK", ""),
        "capability_id": os.environ.get("HERMES_KANBAN_CAPABILITY_ID", ""),
        "task_contract_sha256": os.environ.get("HERMES_KANBAN_CONTRACT_SHA256", ""),
        "api_request_id": uuid4().hex,
        "context_length": str(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or ""),
        "ollama_num_ctx": str(getattr(agent, "_ollama_num_ctx", 0) or ""),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
        "base_url": str(getattr(agent, "base_url", "") or ""),
        "api_mode": str(getattr(agent, "api_mode", "") or ""),
        "api_call_count": int(getattr(agent, "_api_call_count", 0) or 0),
        "streaming": streaming,
        "has_tools": bool(api_kwargs.get("tools")),
        "existing_response_format": bool(api_kwargs.get("response_format") or
                                          ((api_kwargs.get("extra_body") or {}).get("response_format")
                                           if isinstance(api_kwargs.get("extra_body"), dict) else False)),
        "request_format_phase": "verify",
        "supplied_response_format": api_kwargs.get("response_format"),
        "existing_lease_ids": tuple(getattr(agent, "_local_gpu_admission_lease_ids", ()) or ()),
    }
    decision = run_llm_admission_middleware(**context)
    if not decision.allowed:
        raise LocalModelAdmissionBlockedError(
            f"Local model request stopped before provider I/O ({decision.code}): {decision.reason}"
        )
    agent._local_gpu_admission_lease_ids = decision.lease_ids
    agent._local_gpu_admission_scope = {
        key: str(context.get(key) or "") for key in (
            "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db",
            "kanban_workspace", "kanban_claim_lock", "capability_id", "task_contract_sha256",
            "api_request_id", "context_length", "session_id", "model", "provider", "base_url",
            "ollama_num_ctx",
        )
    }


def release_local_gpu_admission_leases(*, lease_ids: tuple[str, ...], scope: dict[str, Any]) -> bool:
    """Release turn-owned leases through the single configured controller.

    Release is idempotent and holder-qualified in the controller contract. Returning False
    leaves backend TTL/reconciliation responsible for eventual cleanup; callers must log it.
    """
    from hermes_cli.plugins import _delivery_manager, has_middleware

    if not lease_ids or not has_middleware(LLM_ADMISSION_MIDDLEWARE):
        return False
    callbacks = list(_delivery_manager()._middleware.get(LLM_ADMISSION_MIDDLEWARE, []))
    if len(callbacks) != 1:
        return False
    try:
        result = callbacks[0](action="release", lease_ids=tuple(lease_ids), **middleware_payload(**scope))
    except Exception as exc:
        logger.error("Local-model admission release failed closed: %s", exc)
        return False
    return bool(
        isinstance(result, dict)
        and result.get("decision") == "released"
        and tuple(result.get("lease_ids") or ()) == tuple(lease_ids)
        and result.get("scope") == {key: str(value or "") for key, value in scope.items()}
    )


def verify_local_model_task_completion(**context: Any) -> dict[str, str] | None:
    """Require exactly one trusted controller to verify a local task before it can complete.

    The controller must re-read the pinned task/run, validate the active lease and task digest,
    constrain artifacts to the task workspace, verify the pinned external oracle, and return an
    exact scope echo plus immutable evidence identifiers. This helper never accepts a model's
    self-reported PASS.
    """
    required = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board", "kanban_db",
        "kanban_workspace", "capability_id", "task_contract_sha256", "workspace_path", "assignee",
        "model_override", "provider_override", "completion_contract", "artifacts",
    )
    scope = {key: str(context.get(key) or "") for key in required if key != "artifacts"}
    artifacts = context.get("artifacts")
    required_identity = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board", "kanban_db",
        "capability_id", "task_contract_sha256", "workspace_path", "assignee",
    )
    if (not all(scope.get(key) for key in required_identity)
            or not isinstance(artifacts, (list, tuple)) or not artifacts):
        raise RuntimeError("Local task completion lacks an exact task/run or artifact scope")
    normalized_artifacts = [str(path) for path in artifacts]
    if any(not path.strip() for path in normalized_artifacts):
        raise RuntimeError("Local task completion contains an empty artifact path")
    expected_scope = {**scope, "artifacts": normalized_artifacts}

    from hermes_cli.plugins import _delivery_manager, has_middleware

    if not has_middleware(LLM_ADMISSION_MIDDLEWARE):
        raise RuntimeError("No local-model controller is available to verify this task")
    callbacks = list(_delivery_manager()._middleware.get(LLM_ADMISSION_MIDDLEWARE, []))
    if len(callbacks) != 1:
        raise RuntimeError("Exactly one local-model controller is required for task verification")
    result = callbacks[0](action="verify_completion", **middleware_payload(**expected_scope))
    if not isinstance(result, dict):
        raise RuntimeError("Local-model controller returned no verification receipt")
    expected_text = ("policy_id", "reason", "evidence_id", "verifier_sha256", "artifact_sha256")
    if (result.get("decision") != "verified"
            or result.get("scope") != expected_scope
            or any(not isinstance(result.get(key), str) or not result[key].strip()
                   for key in expected_text)):
        reason = result.get("reason") if isinstance(result.get("reason"), str) else "invalid receipt"
        raise RuntimeError(f"Independent local-task verification refused completion: {reason}")
    return {key: result[key].strip() for key in expected_text}


def run_tool_execution_middleware(
    tool_name: str, args: dict[str, Any], next_call: Callable[[dict[str, Any]], Any], **context: Any,
) -> Any:
    """Run tool execution through registered tool execution middleware."""
    return _run_execution_chain(
        TOOL_EXECUTION_MIDDLEWARE, next_call,
        tool_name=tool_name, args=args, original_args=context.pop("original_args", args), **context)


class _DownstreamExecutionError(Exception):
    """Marks an exception raised BELOW a middleware frame so the frame's own failure handling
    (skip-and-continue) doesn't swallow it."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _run_execution_chain(kind: str, terminal_call: Callable[[Any], Any], **kwargs: Any) -> Any:
    from hermes_cli.plugins import _delivery_manager

    payload_key = "request" if "request" in kwargs else "args"
    manager = _delivery_manager()
    callbacks = list(manager._middleware.get(kind, []))
    if not callbacks:
        return terminal_call(kwargs[payload_key])

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # Single-use per frame: a second call would re-run the downstream provider/tool, so it
            # is a contract violation, not a retry.
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            next_called = True
            try:
                next_result = call_at(index + 1, payload if next_payload is None else next_payload)
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            return callback(**call_kwargs)
        except _DownstreamExecutionError as exc:
            raise exc.original
        except Exception as exc:
            # Runs once per tool/LLM call: a mis-declared callback fails identically every time,
            # so it goes through the manager's warn-once reporter (#111922).
            manager._report_hook_failure(kind, callback, call_kwargs, exc, surface="Middleware")
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])
