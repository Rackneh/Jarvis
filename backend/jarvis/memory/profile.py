"""User profile — what Jarvis knows about you.

Filled by the onboarding interview and refined in the background by an
extraction pass that runs while you keep talking. Rendered into the system
prompt of every conversation and every task worker.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .. import config as cfg
from . import store

# The interview walks these slots in order.
REQUIRED_SLOTS = [
    ("call_name", "What should I call you?"),
    ("user_role", "What do you do — work, studies, projects?"),
    ("assistant_name_pref", "Anything you'd like to call me besides my default name?"),
    ("active_projects", "What projects or areas should I keep track of? Paths help."),
    ("preferences", "How do you prefer updates — brief spoken summaries, or full reports?"),
    ("permissions", "Am I allowed to run commands on this machine when working on tasks?"),
]

OPTIONAL_SLOTS = [
    ("working_hours", "When are you usually at the computer?"),
    ("personality", "How would you like me to talk — formal, casual, terse?"),
]


class Profile:
    """Typed view over profile_facts with an onboarding progress metric."""

    def __init__(self) -> None:
        self.facts: Dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        self.facts = store.all_facts()

    # ---------------- reads

    def get(self, key: str, default: str = "") -> str:
        f = self.facts.get(key)
        return (f or {}).get("value", default) if self.facts else store.get_fact(key) or default

    def is_complete(self) -> bool:
        return all(self.get(k) for k, _ in REQUIRED_SLOTS)

    def missing_slots(self) -> List[str]:
        return [k for k, _ in REQUIRED_SLOTS if not self.get(k)]

    def progress(self) -> float:
        have = sum(1 for k, _ in REQUIRED_SLOTS if self.get(k))
        return have / len(REQUIRED_SLOTS)

    def summary(self) -> str:
        """Compact render for system prompts."""
        if not self.facts:
            return "No profile yet."
        lines = ["USER PROFILE:"]
        for k in ("call_name", "user_role", "active_projects", "preferences",
                  "permissions", "working_hours", "personality",
                  "assistant_name_pref"):
            v = self.get(k)
            if v:
                lines.append(f"- {k}: {v}")
        extra = {k: v["value"] for k, v in self.facts.items()
                 if k not in ("call_name", "user_role", "active_projects",
                              "preferences", "permissions", "working_hours",
                              "personality", "assistant_name_pref")}
        if extra:
            lines.append("Additional facts:")
            for k, v in sorted(extra.items()):
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    # ---------------- writes

    def set(self, key: str, value: str, source: str = "onboarding",
            confidence: float = 1.0) -> None:
        value = (value or "").strip()
        if not value:
            return
        store.set_fact(key, value, source=source, confidence=confidence)
        self.facts[key] = {"value": value, "confidence": confidence,
                           "source": source, "updated": 0}
        from ..events import publish
        publish("profile_updated", key=key, value=value, source=source)

    def merge_extracted(self, extracted: Dict[str, str], source: str) -> int:
        """Background parser results land here; never overwrite an explicit
        onboarding answer with a lower-confidence inference."""
        merged = 0
        for k, v in extracted.items():
            v = (v or "").strip()
            if not v or not isinstance(v, str):
                continue
            existing = self.facts.get(k)
            if existing and existing.get("source") == "onboarding" and v.lower() != existing["value"].lower():
                continue
            self.set(k, v, source=source, confidence=0.8)
            merged += 1
        return merged

    # ---------------- persistence of derived state

    def environment_summary(self) -> str:
        return self.get("environment", "")

    def set_environment_summary(self, text: str) -> None:
        self.set("environment", text, source="env_scan")


_profile: Profile | None = None


def get_profile() -> Profile:
    global _profile
    if _profile is None:
        _profile = Profile()
    return _profile


def reset_profile() -> None:
    global _profile
    _profile = None
