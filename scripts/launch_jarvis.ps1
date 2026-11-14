# Launch Jarvis: backend (FastAPI + voice + tasks) and the native HUD.
param(
    [switch]$NoGui,
    [switch]$VoiceClone,      # kept for compatibility; clone setup is now default
    [switch]$NoVoiceClone,   # skip the heavy XTTS/coqui install + reference build
    [switch]$NoBreeze,       # skip Breeze TTS 2 runtime/model setup (uses fallbacks)
    [string]$BreezeQuant = "q4_k",
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent (Split-Path -Parent $root)   # repositories root

# venv setup
$venv = Join-Path $root "venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Write-Host "[jarvis] creating venv..."
    python -m venv $venv
}
$py = "$venv\Scripts\python.exe"

# dependencies (skip if already installed)
$depsOk = & $py -c "import fastapi, uvicorn, httpx, sounddevice" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[jarvis] installing requirements..."
    & $py -m pip install --upgrade pip -q
    & $py -m pip install -r (Join-Path $root "requirements.txt") -q
}

# voice models (idempotent)
& $py (Join-Path $root "download_voice_models.py")

# Jarvis cloned voice is the default TTS. Install/build it unless explicitly skipped;
# failures are non-fatal because Kokoro/Piper/SAPI remain as fallbacks.
if (-not $NoVoiceClone) {
    try {
        Write-Host "[jarvis] ensuring Jarvis voice clone deps..."
        & $py -m pip install "coqui-tts[cpu,codec]==0.28.0" pyarrow "transformers<5" six -q
        if ($LASTEXITCODE -eq 0) {
            & $py (Join-Path $root "scripts\build_jarvis_voice.py")
        } else {
            Write-Warning "coqui-tts install failed; continuing with fallback TTS"
        }
    } catch {
        Write-Warning "voice clone setup failed: $_ - continuing with fallback TTS"
    }
}

# Breeze TTS 2 (GGUF/Vulkan) is available as an opt-in TTS engine. Download the
# runtime + model once so it's ready if selected; failures are non-fatal
# because Kokoro/Piper/SAPI remain as fallbacks.
if (-not $NoBreeze) {
    try {
        Write-Host "[jarvis] ensuring Breeze TTS 2 runtime ($BreezeQuant)..."
        & $py (Join-Path $root "scripts\setup_breeze.py") --quant $BreezeQuant
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Breeze setup failed; continuing with fallback TTS"
        }
    } catch {
        Write-Warning "Breeze setup failed: $_ - continuing with fallback TTS"
    }
}

# Ensure LM Studio is running Gemma 4 with optimal performance parameters (8k ctx, parallel 1, max GPU)
$lmsCmd = Get-Command "lms" -ErrorAction SilentlyContinue
if ($lmsCmd) {
    try {
        $loaded = & lms ps --json 2>$null | ConvertFrom-Json
        $gemma = $loaded | Where-Object { $_.modelKey -eq "gemma-4-e4b-uncensored-hauhaucs-aggressive" -or $_.identifier -eq "gemma-4-e4b-uncensored-hauhaucs-aggressive" }
        if (-not $gemma -or $gemma.contextLength -ne 8192 -or $gemma.parallel -ne 1) {
            Write-Host "[jarvis] optimizing LM Studio model (Gemma 4: 8k ctx, parallel 1, max GPU)..."
            & lms load gemma-4-e4b-uncensored-hauhaucs-aggressive --gpu max -c 8192 --parallel 1 -y
        } else {
            Write-Host "[jarvis] LM Studio Gemma 4 is active with fastest settings (8k ctx, 1 slot)."
        }
    } catch {
        Write-Warning "[jarvis] could not verify LM Studio status: $_"
    }
}

if ($NoGui) {
    Write-Host "[jarvis] starting backend on :$Port"
    $backendProc = Start-Process -FilePath $py -ArgumentList @(
        "-m", "uvicorn", "jarvis.server:app", "--host", "127.0.0.1", "--port", "$Port"
    ) -WorkingDirectory (Join-Path $root "backend") -PassThru -WindowStyle Minimized

    Write-Host "[jarvis] backend pid=$($backendProc.Id). Console chat:"
    & $py (Join-Path $root "backend\run.py") --cli
    Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    exit 0
}

Write-Host "[jarvis] starting JarvisDeck HUD (backend is owned by the GUI)"
$gui = Start-Process -FilePath "dotnet" -ArgumentList @(
    "run", "--project", (Join-Path $root "native-gui\JarvisDeck.csproj")
) -WorkingDirectory $root -PassThru -WindowStyle Hidden

Write-Host "[jarvis] running. gui pid=$($gui.Id)"
Write-Host "[jarvis] stop with .\scripts\stop_jarvis.ps1"
