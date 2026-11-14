# Stop Jarvis backend + HUD.
$ErrorActionPreference = "SilentlyContinue"

Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "jarvis\.server|uvicorn.*8010|run\.py --cli"
} | ForEach-Object {
    Write-Host "stopping backend pid=$($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force
}

Get-Process -Name "JarvisDeck", "dotnet" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path -match "Rackneh's Jarvis" } |
    ForEach-Object {
        Write-Host "stopping GUI pid=$($_.Id)"
        Stop-Process -Id $_.Id -Force
    }

# Breeze TTS server spawned by the backend (GGUF/Vulkan) - stop it too.
Get-Process -Name "breeze-server" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path -match "models\\breeze" } |
    ForEach-Object {
        Write-Host "stopping breeze-server pid=$($_.Id)"
        Stop-Process -Id $_.Id -Force
    }

Write-Host "[jarvis] stopped. (Dory engine / LM Studio are left untouched.)"
