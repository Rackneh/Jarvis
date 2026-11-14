using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using System.Windows.Media;

namespace JarvisDeck.Audio;

/// <summary>
/// Non-blocking polyphonic futuristic HUD sound player.
/// Loads stereo WAV cues from jarvis_home/sounds/ (with automatic on-the-fly
/// synth fallback if any file is missing) and plays them over a pool of
/// MediaPlayer instances so rapid overlapping effects never clip or block UI.
/// </summary>
public static class SoundEffects
{
    private const int PoolSize = 8;
    private static readonly List<MediaPlayer> _pool = new();
    private static int _poolIdx;
    private static readonly Dictionary<string, DateTime> _lastPlayed = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Dictionary<string, int> _cooldownMs = new(StringComparer.OrdinalIgnoreCase)
    {
        ["thought"] = 1800,
        ["task_step"] = 500,
        ["tool"] = 220,
        ["cache_hit"] = 250,
        ["listen_start"] = 250,
        ["listen_end"] = 250,
        ["wake"] = 400,
        ["task_spawn"] = 300,
        ["task_done"] = 300,
        ["alert"] = 350,
        ["pulse"] = 200,
        ["boot"] = 500,
    };

    private static string? _soundsDir;
    public static bool Enabled { get; set; } = true;
    public static double Volume { get; set; } = 0.55;

    public static void Play(string cue)
    {
        if (!Enabled || string.IsNullOrWhiteSpace(cue)) return;
        try
        {
            var now = DateTime.UtcNow;
            var cd = _cooldownMs.TryGetValue(cue, out var ms) ? ms : 150;
            if (_lastPlayed.TryGetValue(cue, out var prev) && (now - prev).TotalMilliseconds < cd)
                return;
            _lastPlayed[cue] = now;

            var path = EnsureCueFile(cue);
            if (path == null) return;

            var disp = Application.Current?.Dispatcher;
            if (disp != null && !disp.CheckAccess())
            {
                disp.BeginInvoke(() => PlayOnUiThread(path));
            }
            else
            {
                PlayOnUiThread(path);
            }
        }
        catch
        {
            // Audio hardware issues must never disrupt HUD rendering
        }
    }

    private static void PlayOnUiThread(string path)
    {
        try
        {
            while (_pool.Count < PoolSize)
            {
                var p = new MediaPlayer();
                _pool.Add(p);
            }
            var player = _pool[_poolIdx % _pool.Count];
            _poolIdx++;
            player.Stop();
            player.Volume = Math.Clamp(Volume, 0.0, 1.0);
            player.Open(new Uri(path, UriKind.Absolute));
            player.Play();
        }
        catch
        {
        }
    }

    private static string? EnsureCueFile(string cue)
    {
        _soundsDir ??= ResolveSoundsDirectory();
        var file = Path.Combine(_soundsDir, cue + ".wav");
        if (File.Exists(file) && new FileInfo(file).Length > 1024)
            return file;

        // Synthesize fallback sci-fi WAV on the fly if missing
        try
        {
            Directory.CreateDirectory(_soundsDir);
            WriteFallbackSynthWav(file, cue);
            return File.Exists(file) ? file : null;
        }
        catch
        {
            return null;
        }
    }

    private static string ResolveSoundsDirectory()
    {
        var cur = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        for (int i = 0; i < 6 && cur != null; i++)
        {
            var candidate = Path.Combine(cur.FullName, "jarvis_home", "sounds");
            if (Directory.Exists(candidate))
                return candidate;
            if (File.Exists(Path.Combine(cur.FullName, "config.json")))
                return candidate;
            cur = cur.Parent;
        }
        return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "sounds");
    }

    private static void WriteFallbackSynthWav(string path, string cue)
    {
        const int sr = 44100;
        double dur = cue switch
        {
            "boot" => 0.65,
            "task_done" => 0.45,
            "task_spawn" => 0.38,
            "alert" => 0.35,
            _ => 0.22,
        };
        double f0 = cue switch
        {
            "boot" => 440.0,
            "wake" => 660.0,
            "listen_start" => 580.0,
            "listen_end" => 480.0,
            "thought" => 740.0,
            "tool" => 1175.0,
            "cache_hit" => 1320.0,
            "task_spawn" => 520.0,
            "task_step" => 880.0,
            "task_done" => 784.0,
            "alert" => 700.0,
            _ => 620.0,
        };
        double f1 = cue switch
        {
            "listen_start" => 960.0,
            "listen_end" => 320.0,
            "boot" => 880.0,
            "task_done" => 1174.0,
            _ => f0 * 1.25,
        };

        int n = (int)(sr * dur);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs);

        // RIFF WAVE header (44.1kHz 16-bit stereo)
        int dataBytes = n * 4;
        bw.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + dataBytes);
        bw.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
        bw.Write(16);
        bw.Write((short)1);
        bw.Write((short)2);
        bw.Write(sr);
        bw.Write(sr * 4);
        bw.Write((short)4);
        bw.Write((short)16);
        bw.Write(System.Text.Encoding.ASCII.GetBytes("data"));
        bw.Write(dataBytes);

        for (int i = 0; i < n; i++)
        {
            double t = (double)i / sr;
            double env = Math.Min(1.0, t / 0.01) * Math.Min(1.0, Math.Max(0.0, (dur - t) / 0.08));
            double freq = f0 + (f1 - f0) * (t / dur);
            double sample = env * 0.45 * (Math.Sin(2 * Math.PI * freq * t) + 0.25 * Math.Sin(2 * Math.PI * freq * 1.5 * t));
            short s = (short)Math.Clamp(sample * 32767.0, -32767.0, 32767.0);
            bw.Write(s);
            bw.Write(s);
        }
    }
}
