"""Provider-default regression test (no live LLM required).

Run: python tests/test_provider_default.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from jarvis import config as cfg
import jarvis.providers.registry as reg


async def _up() -> bool:
    return True


async def _down() -> bool:
    return False


def main() -> None:
    old_env = os.environ.pop("JARVIS_PROVIDER", None)
    old_state_path = reg.PROVIDER_CFG_PATH
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-provider-test-"))
    try:
        reg.PROVIDER_CFG_PATH = tmp / "provider.json"

        assert cfg.get("providers.default") == "lmstudio", "config default must be lmstudio"
        assert reg.selected_kind() == "lmstudio", "missing provider state must default to lmstudio"

        reg.PROVIDER_CFG_PATH.write_text(json.dumps({"provider": "auto"}), encoding="utf-8")
        reg._dory_up = _up
        reg._lmstudio_up = _up
        assert asyncio.run(reg.effective_kind()) == "lmstudio", "auto must prefer LM Studio"

        reg._lmstudio_up = _down
        assert asyncio.run(reg.effective_kind()) == "dory", "auto must fall back to Dory"

        reg.PROVIDER_CFG_PATH.write_text(json.dumps({"provider": "dory"}), encoding="utf-8")
        reg._dory_up = _down
        reg._lmstudio_up = _up
        assert asyncio.run(reg.effective_kind()) == "lmstudio", "explicit dory should fall back when Dory is down"

        os.environ["JARVIS_PROVIDER"] = "lmstudio"
        assert reg.selected_kind() == "lmstudio", "env override must win"
    finally:
        reg.PROVIDER_CFG_PATH = old_state_path
        if old_env is None:
            os.environ.pop("JARVIS_PROVIDER", None)
        else:
            os.environ["JARVIS_PROVIDER"] = old_env

    print("PROVIDER DEFAULT OK")


if __name__ == "__main__":
    main()
