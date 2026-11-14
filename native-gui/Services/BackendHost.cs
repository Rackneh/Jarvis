using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace JarvisDeck.Services;

/// <summary>
/// Owns the Jarvis backend process lifecycle for the HUD.
/// Starts uvicorn when the backend is not already reachable and stops the
/// process it started when the GUI closes.
/// </summary>
public sealed class BackendHost : IDisposable
{
    private const string BaseUrl = "http://127.0.0.1:8010";
    private readonly HttpClient _probe = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly string _logPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "JarvisDeck",
        "backend.log");
    private readonly object _logLock = new();
    private Process? _process;
    private bool _ownsProcess;

    public bool Owned => _ownsProcess;
    public int? ProcessId => _process?.Id;

    public async Task<bool> EnsureRunningAsync(CancellationToken ct = default)
    {
        if (await IsBackendUpAsync(ct).ConfigureAwait(false))
        {
            _ownsProcess = false;
            return true;
        }

        var root = FindRepoRoot();
        var python = Path.Combine(root, "venv", "Scripts", "python.exe");
        var backendDir = Path.Combine(root, "backend");
        var serverPy = Path.Combine(backendDir, "jarvis", "server.py");
        if (!File.Exists(python))
            throw new FileNotFoundException("Backend Python was not found.", python);
        if (!File.Exists(serverPy))
            throw new FileNotFoundException("Backend entrypoint was not found.", serverPy);

        Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!);
        var psi = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-m uvicorn jarvis.server:app --app-dir \"{backendDir}\" --host 127.0.0.1 --port 8010",
            WorkingDirectory = backendDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start the backend process.");
        _process = process;
        _ownsProcess = true;
        process.EnableRaisingEvents = true;
        process.OutputDataReceived += (_, e) => Log(e.Data);
        process.ErrorDataReceived += (_, e) => Log(e.Data);
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        var deadline = DateTime.UtcNow.AddSeconds(90);
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            if (process.HasExited)
                throw new InvalidOperationException($"Backend exited during startup (code {process.ExitCode}). See {_logPath}");
            if (await IsBackendUpAsync(ct).ConfigureAwait(false))
                return true;
            await Task.Delay(500, ct).ConfigureAwait(false);
        }

        throw new TimeoutException($"Backend did not become ready within 90 seconds. See {_logPath}");
    }

    private async Task<bool> IsBackendUpAsync(CancellationToken ct)
    {
        try
        {
            using var response = await _probe.GetAsync(BaseUrl + "/api/tasks", ct).ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    private static string FindRepoRoot()
    {
        foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var dir = new DirectoryInfo(start);
            while (dir is not null)
            {
                var candidate = dir.FullName;
                if (File.Exists(Path.Combine(candidate, "backend", "jarvis", "server.py")) &&
                    File.Exists(Path.Combine(candidate, "venv", "Scripts", "python.exe")))
                {
                    return candidate;
                }
                dir = dir.Parent;
            }
        }

        throw new DirectoryNotFoundException(
            "Could not locate the Jarvis repository root (expected backend/jarvis/server.py and venv/Scripts/python.exe).");
    }

    private void Log(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return;
        try
        {
            lock (_logLock)
            {
                File.AppendAllText(_logPath, $"[{DateTime.Now:HH:mm:ss}] {line}{Environment.NewLine}");
            }
        }
        catch
        {
            // Logging must never take down the HUD.
        }
    }

    public void Stop()
    {
        var process = _process;
        _process = null;
        _ownsProcess = false;
        if (process is null)
            return;

        try
        {
            if (!process.HasExited)
                process.Kill(true);
        }
        catch
        {
            try
            {
                if (!process.HasExited)
                    process.Kill();
            }
            catch
            {
                // Process may already be gone.
            }
        }

        try
        {
            process.WaitForExit(3000);
        }
        catch
        {
            // Best effort only.
        }
        process.Dispose();
    }

    public void Dispose()
    {
        Stop();
        _probe.Dispose();
    }
}
