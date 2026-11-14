"""Jarvis IBA (Infinite Balanced Autonomy) SQL Runtime — ported from Dory LLM.

Every step in a task executes as a separate stateless model conversation built
from durable SQLite state (task, step, action, artifact, evidence, code_map).
"""
from .store import Store, Conflict, uid, encode, target_tokens
from .sandbox import WorkspaceSandbox, SandboxViolationError
from .tools import Tools, TOOL_HELP
from .protocol import TOOL_ARGUMENTS, outcome_schema, visible_evidence, compact_outcome_contract
from .provider import (
    LMStudioHTTP,
    DoryLlamaServer,
    DynamicProvider,
    JarvisIBAProvider,
    get_default_provider,
)
from .engine import Runtime, OutcomeRejected
from .manager import generate_task_title, generate_step_title, extract_script_map, WorkspaceManager

__all__ = [
    "Store",
    "Conflict",
    "uid",
    "encode",
    "target_tokens",
    "WorkspaceSandbox",
    "SandboxViolationError",
    "Tools",
    "TOOL_HELP",
    "TOOL_ARGUMENTS",
    "outcome_schema",
    "visible_evidence",
    "compact_outcome_contract",
    "LMStudioHTTP",
    "DoryLlamaServer",
    "DynamicProvider",
    "JarvisIBAProvider",
    "get_default_provider",
    "Runtime",
    "OutcomeRejected",
    "generate_task_title",
    "generate_step_title",
    "extract_script_map",
    "WorkspaceManager",
]
