"""Download the Breeze-TTS-2.cpp runtime + a GGUF model for Jarvis's Breeze TTS.

The official BreezeBlue/Breeze-TTS-2 checkpoint needs a CUDA NVIDIA GPU. The
Breeze-TTS-2.cpp reimplementation runs the same model on ggml/Vulkan (NVIDIA,
AMD, Intel) or CPU, which is what this script sets up for Windows.

Idempotent: existing files are kept. It also points config.json at the binary
and model so the backend can auto-start `breeze-server`.

    python scripts/setup_breeze.py [--quant q4_k]
"""
import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from jarvis import config as cfg  # noqa: E402

REPO = cfg.REPO_ROOT
MODELS = cfg.home() / "models" / "breeze"
BIN = MODELS / "bin"
VOICES = MODELS / "voices"
HF_REPO = "HoppouAI/Breeze-TTS-2.cpp"
GH_REPO = "HoppouAI/Breeze-TTS-2.cpp"

QUANTS = ["f16", "q8_0", "q6_k", "q4_k", "q8_0-dd4", "q8_0-dd2", "q4_k-dd2"]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[..] downloading {url}\n     -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis-setup"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dest)
    print(f"[ok] {dest.name} ({dest.stat().st_size / 1e9:.2f} GB)")


def _latest_windows_asset() -> str:
    api = f"https://api.github.com/repos/{GH_REPO}/releases/latest"
    try:
        with urllib.request.urlopen(api) as r:  # noqa: S310
            data = json.load(r)
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if "windows" in name and "vulkan" in name and name.endswith(".zip"):
                return asset["browser_download_url"]
    except Exception as exc:
        print(f"[!!] GitHub release lookup failed ({exc}); using pinned v0.1.7")
    return (f"https://github.com/{GH_REPO}/releases/download/v0.1.7/"
            "breeze-tts-2-v0.1.7-windows-x64-vulkan.zip")


def ensure_server() -> Path:
    BIN.mkdir(parents=True, exist_ok=True)
    exe = BIN / "breeze-server.exe"
    if exe.exists():
        print(f"[ok] breeze-server already present: {exe}")
        return exe
    url = _latest_windows_asset()
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "breeze.zip"
        _download(url, zip_path)
        print(f"[..] extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(BIN)
    # the archive may nest binaries under a folder; flatten so DLLs stay together
    found = next(BIN.rglob("breeze-server.exe"), None)
    if found is not None and found.parent != BIN:
        for child in found.parent.iterdir():
            target = BIN / child.name
            if target.exists():
                continue
            shutil.move(str(child), target)
    exe = BIN / "breeze-server.exe"
    if not exe.exists():
        raise RuntimeError("breeze-server.exe not found after extraction")
    print(f"[ok] breeze-server: {exe}")
    return exe


def ensure_model(quant: str) -> Path:
    if quant not in QUANTS:
        raise ValueError(f"unknown quant {quant!r}; choose from {QUANTS}")
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS / f"breeze-tts-2-{quant}.gguf"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] model already present: {dest}")
        return dest
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/breeze-tts-2-{quant}.gguf"
    _download(url, dest)
    return dest


def point_config(exe: Path, model: Path) -> None:
    cfg.update("voice.breeze_server_exe", str(exe))
    cfg.update("voice.breeze_model", str(model))
    cfg.update("voice.breeze_voices_dir", str(VOICES))
    # NOTE: tts_engine is intentionally NOT forced here — Kokoro is the
    # latency-safe default; Breeze is opt-in via voice.tts_engine = "breeze".
    print("[ok] config.json -> breeze auto-start paths set "
          "(tts_engine left unchanged; set 'breeze' to enable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="q4_k", choices=QUANTS,
                    help="GGUF quantization (q4_k is the safest for 4 GB GPUs)")
    args = ap.parse_args()
    exe = ensure_server()
    model = ensure_model(args.quant)
    VOICES.mkdir(parents=True, exist_ok=True)
    point_config(exe, model)
    print("\nDone. Jarvis will auto-start breeze-server on first TTS use, or run:")
    print(f"  {exe} {model} --host 127.0.0.1 --port 8137 "
          f"--voices-dir {VOICES}")


if __name__ == "__main__":
    main()
