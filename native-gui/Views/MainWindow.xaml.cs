using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;
using System.Windows.Threading;
using JarvisDeck.Audio;
using JarvisDeck.Models;
using JarvisDeck.Services;

namespace JarvisDeck.Views;

public sealed class BrushConverter : System.Windows.Data.IValueConverter
{
    public object Convert(object value, Type t, object p, System.Globalization.CultureInfo c)
        => new SolidColorBrush((Color)ColorConverter.ConvertFromString(value?.ToString() ?? "#5B6B85"));
    public object ConvertBack(object value, Type t, object p, System.Globalization.CultureInfo c) => throw new NotSupportedException();
}

public partial class MainWindow : Window
{
    private readonly Backend _backend = new();
    private readonly BackendHost _backendHost = new();
    private readonly ObservableCollection<TaskVm> _tasks = new();
    private readonly DispatcherTimer _refresh;
    private string _assistantName = "Jarvis";
    private bool _micOn = true;
    private bool _coreFocusMode;
    private bool _pttActive;
    private bool _consciousReasoning = false;
    private string _voiceState = "idle";
    private string _engineSummary = "jarvis_clone";
    private string _providerSummary = "initializing";
    private string _cacheSummary = "CACHE: STANDBY";

    private readonly Brush _userBrush;
    private readonly Brush _aiBrush;
    private readonly Brush _ambientBrush;
    private readonly List<Rectangle> _userBars = new();
    private readonly List<Rectangle> _aiBars = new();
    private readonly List<Rectangle> _ambientBars = new();
    private readonly DispatcherTimer _waveTimer = new() { Interval = TimeSpan.FromMilliseconds(33) };
    private readonly Random _waveRng = new();
    private double _userLevel;
    private double _aiLevel;
    private double _ambientLevel;
    private double _ambientTarget;
    private double _userTarget;
    private double _userPhase;
    private double _aiPhase;
    private double _ambientPhase;
    private DateTime _lastPartialEffectAt = DateTime.MinValue;
    private DateTime _lastThoughtEffectAt = DateTime.MinValue;
    private DateTime _lastThoughtEffectAtAgent = DateTime.MinValue;
    private string _activeThoughtSession = "";
    private TextBlock? _activeThoughtTb;
    private string _lastShownUserText = "";
    private DateTime _lastShownUserAt = DateTime.MinValue;

    private static readonly Color ColCyan = Color.FromRgb(0x22, 0xD3, 0xEE);
    private static readonly Color ColBlue = Color.FromRgb(0x38, 0xBD, 0xF8);
    private static readonly Color ColMagenta = Color.FromRgb(0xF4, 0x72, 0xB6);
    private static readonly Color ColPurple = Color.FromRgb(0xA7, 0x8B, 0xFA);
    private static readonly Color ColGreen = Color.FromRgb(0x4A, 0xDE, 0x80);
    private static readonly Color ColAmber = Color.FromRgb(0xFB, 0xBF, 0x24);
    private static readonly Color ColGold = Color.FromRgb(0xFA, 0xCC, 0x15);
    private static readonly Color ColRed = Color.FromRgb(0xF8, 0x71, 0x71);

    public ObservableCollection<TaskVm> Tasks => _tasks;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        _backend.EventReceived += OnEvent;
        _backend.ConnectionChanged += OnConnection;
        _ = StartBackendAsync();

        _refresh = new DispatcherTimer { Interval = TimeSpan.FromSeconds(4) };
        _refresh.Tick += async (_, _) => await RefreshTasksAsync();
        _refresh.Start();

        _userBrush = FindBrush("Green");
        _aiBrush = FindBrush("Blue");
        _ambientBrush = FindBrush("Gray");
        _waveTimer.Tick += (_, _) => WaveTick();
        _waveTimer.Start();
        UserWave.SizeChanged += (_, _) => BuildWave(UserWave, _userBars, _userBrush);
        AiWave.SizeChanged += (_, _) => BuildWave(AiWave, _aiBars, _aiBrush);
        AmbientWave.SizeChanged += (_, _) => BuildWave(AmbientWave, _ambientBars, _ambientBrush);
        BuildWave(UserWave, _userBars, _userBrush);
        BuildWave(AiWave, _aiBars, _aiBrush);
        BuildWave(AmbientWave, _ambientBars, _ambientBrush);

        Loaded += (_, _) =>
        {
            UpdateReasoningButton();
            SoundEffects.Play("boot");
            Core3D.TriggerShockwave(ColCyan, 380, 1.5);
            Core3D.SpawnSystemEffect("BOOT", "JARVIS CORE V2 ONLINE", "3D Holographic Telemetry Active", ColCyan, 6.5);
            LogSystemEvent("BOOT", "Jarvis Core V2 holographic telemetry initialized", ColCyan);
        };
        Closed += (_, _) => _waveTimer.Stop();
    }

    private void OnConnection(bool connected)
    {
        Dispatcher.Invoke(() =>
        {
            ConnDot.Fill = connected ? FindBrush("Green") : FindBrush("Red");
            ConnDot.Effect = connected ? FindEffect("GreenGlow") : FindEffect("RedGlow");
            Core3D.SpawnSystemEffect(
                "LINK",
                connected ? "CORE TELEMETRY LINKED" : "TELEMETRY RECONNECTING",
                connected ? "WebSocket event stream online" : "Re-establishing backend socket",
                connected ? ColGreen : ColRed,
                4.5);
            LogSystemEvent("LINK", connected ? "Backend WebSocket connected" : "Backend WebSocket disconnected", connected ? ColGreen : ColRed);
        });
    }

    private async Task StartBackendAsync()
    {
        try
        {
            StateText.Text = "STARTING BACKEND…";
            CoreSubStatusText.Text = "INITIALIZING NEURAL CORE LINK…";
            await _backendHost.EnsureRunningAsync();
            var msg = _backendHost.Owned ? "backend started" : "backend attached";
            AddEventLine(msg, FindBrush("Green"));
            LogSystemEvent("BOOT", msg, ColGreen);
            _ = Task.Run(_backend.ReconnectForeverAsync);
        }
        catch (Exception ex)
        {
            StateText.Text = "BACKEND OFFLINE";
            CoreSubStatusText.Text = "BACKEND OFFLINE";
            AddEventLine("backend failed: " + ex.Message, FindBrush("Red"));
            Core3D.SpawnSystemEffect("ERR", "BACKEND OFFLINE", ex.Message, ColRed, 7.5);
            LogSystemEvent("ERR", "backend failed: " + ex.Message, ColRed);
        }
    }

    // ------------------------------------------------------------------ events

    private void OnEvent(JarvisEvent ev)
    {
        Dispatcher.Invoke(() => Handle(ev.Type, ev.Data));
    }

    private void Handle(string type, JsonElement d)
    {
        switch (type)
        {
            case "snapshot":
                ApplySnapshot(d);
                Core3D.SpawnSystemEffect("SYS", "SYSTEM SNAPSHOT", "Core telemetry synchronized", ColCyan, 4.5);
                LogSystemEvent("SYS", "Snapshot synchronized", ColCyan);
                break;

            case "assistant_renamed":
                if (TryStr(d, "name", out var n)) _assistantName = n;
                UpdateName();
                Core3D.SpawnSystemEffect("SYS", "IDENTITY UPDATED", $"Assistant renamed to {_assistantName}", ColCyan, 5.0);
                LogSystemEvent("SYS", $"Renamed to {_assistantName}", ColCyan);
                break;

            case "voice_state":
                {
                    var st = TryStr(d, "state", out var s) ? s : "idle";
                    SetVoiceState(st);
                    if (TryStr(d, "tts", out var tts))
                    {
                        var stt = TryStr(d, "stt", out var s2) ? s2 : "?";
                        var wakeOk = d.TryGetProperty("wake", out var w) && w.ValueKind == JsonValueKind.True;
                        EngineText.Text = $"engines: tts={tts} stt={stt} wake={(wakeOk ? "ok" : "none")}";
                        _engineSummary = $"{tts} • {stt}";
                        Core3D.SetEngineTelemetry(_engineSummary, _providerSummary, _cacheSummary);
                    }
                    break;
                }

            case "mic_level":
                if (TryGet(d, "level", out var ml) && ml.ValueKind == JsonValueKind.Number)
                {
                    var level = ml.GetDouble();
                    var speech = d.TryGetProperty("speech", out var sp) &&
                                 sp.ValueKind == JsonValueKind.True;
                    _userTarget = speech ? Math.Clamp(level * 5.0, 0, 1) : 0.0;
                    _ambientTarget = Math.Clamp(level * (speech ? 0.7 : 1.8), 0, 1);
                }
                break;

            case "wake_detected":
                {
                    var score = TryGet(d, "score", out var sc) ? sc.ToString() : "";
                    var detail = string.IsNullOrEmpty(score) ? "Acoustic trigger matched" : $"Confidence score {score}";
                    AddEventLine("wake word detected", FindBrush("Magenta"));
                    CoreSubStatusText.Text = "WAKE MATRIX TRIGGERED // LISTENING";
                    SoundEffects.Play("wake");
                    Core3D.TriggerShockwave(ColMagenta, 380, 1.3);
                    Core3D.SpawnSystemEffect("WAKE", "WAKE WORD DETECTED", detail, ColMagenta, 6.0, inbound: true);
                    LogSystemEvent("WAKE", $"Wake detected ({detail})", ColMagenta);
                    break;
                }

            case "user_speech_start":
                AddEventLine("listening…", FindBrush("Dim"));
                CoreSubStatusText.Text = "AUDIO CAPTURE // STREAMING MICROPHONE";
                SoundEffects.Play("listen_start");
                Core3D.SpawnSystemEffect("VOICE", "VOICE CAPTURE", "Listening to user audio stream", ColGreen, 4.5, inbound: true);
                LogSystemEvent("VOICE", "User speech capture started", ColGreen);
                break;

            case "partial_transcript":
                {
                    var pt = TryStr(d, "text", out var pt0) ? pt0 : "";
                    ShowPartial(pt);
                    CoreSubtitleText.Text = pt;
                    CoreSubStatusText.Text = "LIVE TRANSCRIPTION STREAM";
                    if ((DateTime.UtcNow - _lastPartialEffectAt).TotalSeconds >= 1.4 && !string.IsNullOrWhiteSpace(pt))
                    {
                        _lastPartialEffectAt = DateTime.UtcNow;
                        Core3D.SpawnSystemEffect("STT", "LIVE SPEECH DECODE", pt, ColCyan, 3.6, inbound: true);
                    }
                    break;
                }

            case "stt_transcribing":
                {
                    var ms = TryGet(d, "ms", out var msv) ? msv.ToString() : "?";
                    Core3D.SpawnSystemEffect("STT", "WHISPER STT", $"Transcribing {ms} ms audio", ColCyan, 4.5, inbound: true);
                    LogSystemEvent("STT", $"Transcribing {ms} ms audio", ColCyan);
                    break;
                }

            case "final_transcript":
                {
                    var ft = TryStr(d, "text", out var ft0) ? ft0 : "";
                    ShowFinal(ft);
                    CoreSubtitleText.Text = $"you ▸ {ft}";
                    CoreSubStatusText.Text = "UTTERANCE LOCKED // DISPATCHING TO MIND";
                    SoundEffects.Play("listen_end");
                    Core3D.TriggerShockwave(ColGreen, 320, 1.15);
                    Core3D.SpawnSystemEffect("USER", "USER UTTERANCE", ft, ColGreen, 6.0, inbound: true);
                    LogSystemEvent("USER", ft, ColGreen);
                    break;
                }

            case "barge_in":
                {
                    var why = TryStr(d, "reason", out var br) ? br : "user voice";
                    AddEventLine("interrupted", FindBrush("Amber"));
                    CoreSubStatusText.Text = "BARGE-IN // AUDIO INTERRUPTED";
                    SoundEffects.Play("alert");
                    Core3D.TriggerShockwave(ColAmber, 340, 1.1);
                    Core3D.SpawnSystemEffect("BARGE-IN", "SPEECH INTERRUPTED", $"Reason: {why}", ColAmber, 5.5, inbound: true);
                    LogSystemEvent("VOICE", $"Barge-in ({why})", ColAmber);
                    break;
                }

            case "conscious_thinking":
                {
                    var inp = TryStr(d, "input", out var i0) ? i0 : "";
                    if (inp.Length > 0) ShowFinal(inp);
                    CoreSubStatusText.Text = _consciousReasoning
                        ? "CONSCIOUS REASONING // GENERATING"
                        : "CONSCIOUS DIRECT // GENERATING";
                    Core3D.SpawnSystemEffect("MIND",
                        _consciousReasoning ? "CONSCIOUS REASONING" : "CONSCIOUS RESPONSE",
                        string.IsNullOrEmpty(inp) ? "Evaluating intent" : inp, ColPurple, 5.8);
                    LogSystemEvent("MIND", $"{(_consciousReasoning ? "Thinking" : "Responding")}: {inp}", ColPurple);
                    break;
                }

            case "conscious_reasoning_changed":
                if (d.TryGetProperty("enabled", out var cre))
                {
                    _consciousReasoning = cre.ValueKind == JsonValueKind.True;
                    UpdateReasoningButton();
                }
                break;

            case "conscious_idle":
                CoreSubStatusText.Text = "CONSCIOUS MATRIX // STANDBY";
                break;

            case "assistant_thought":
                {
                    if (!_consciousReasoning) break;
                    var th = TryStr(d, "text", out var th0) ? th0 : "";
                    if (!string.IsNullOrWhiteSpace(th))
                    {
                        UpdateThoughtLine(th);
                        if ((DateTime.UtcNow - _lastThoughtEffectAt).TotalSeconds >= 1.8)
                        {
                            _lastThoughtEffectAt = DateTime.UtcNow;
                            SoundEffects.Play("thought");
                            Core3D.SpawnSystemEffect("THOUGHT", "INTERNAL THOUGHT", th, ColPurple, 5.8);
                            LogSystemEvent("MIND", th, ColPurple);
                        }
                    }
                    break;
                }

            case "conscious_tool":
            case "assistant_tool":
                {
                    var name = TryStr(d, "name", out var tn) ? tn : "?";
                    var args = TryGet(d, "args", out var a) ? a.ToString() : "";
                    AddEventLine($"⚙ {name}", FindBrush("Purple"));
                    CoreSubStatusText.Text = $"EXECUTING TOOL // {name.ToUpperInvariant()}";
                    SoundEffects.Play("tool");
                    Core3D.TriggerShockwave(ColPurple, 300, 1.1);
                    Core3D.SpawnSystemEffect("TOOL", $"TOOL // {name.ToUpperInvariant()}", args, ColMagenta, 6.2);
                    LogSystemEvent("TOOL", $"{name} {args}", ColMagenta);
                    break;
                }

            case "assistant_sentence":
                {
                    var at = TryStr(d, "text", out var at0) ? at0 : "";
                    AddAssistantLine(at, FindBrush("Text"));
                    CoreSubtitleText.Text = $"{_assistantName} ▸ {at}";
                    CoreSubStatusText.Text = $"{_assistantName.ToUpperInvariant()} RESPONSE STREAM";
                    Core3D.SpawnSystemEffect("SPEECH", $"{_assistantName.ToUpperInvariant()} RESPONSE", at, ColCyan, 6.2);
                    LogSystemEvent("SPEECH", at, ColCyan);
                    break;
                }

            case "tts_cache_hit":
                {
                    var kind = TryStr(d, "kind", out var k0) ? k0 : "cache";
                    var text = TryStr(d, "text", out var tx) ? tx : "";
                    CoreSubStatusText.Text = $"CONVERSATION CACHE HIT // {kind.ToUpperInvariant()}";
                    SoundEffects.Play("cache_hit");
                    Core3D.PulseCacheHit();
                    Core3D.SpawnSystemEffect("CACHE ⚡", $"CACHE HIT // {kind.ToUpperInvariant()}", text, ColGold, 6.8);
                    LogSystemEvent("CACHE", $"{kind}: {text}", ColGold);
                    if (TryGet(d, "stats", out var stats) && stats.ValueKind == JsonValueKind.Object)
                        ApplyCacheStatsObject(stats, enabled: true);
                    else
                        _ = RefreshCacheAsync();
                    break;
                }

            case "tts_start":
                {
                    SetVoiceState("speaking");
                    var text = TryStr(d, "text", out var ts) ? ts : "";
                    CoreSubStatusText.Text = "VOICE SYNTHESIS // SPEAKING";
                    Core3D.SpawnSystemEffect("TTS", "VOICE SYNTHESIS", text, ColBlue, 5.2);
                    LogSystemEvent("TTS", $"Speaking: {text}", ColBlue);
                    break;
                }

            case "tts_end":
                break;

            case "tts_timeout":
                {
                    var waited = TryGet(d, "waited", out var wv) ? wv.ToString() : "?";
                    Core3D.SpawnSystemEffect("TTS", "TTS TIMEOUT", $"Waited {waited}s", ColAmber, 5.5);
                    LogSystemEvent("TTS", $"TTS timeout ({waited}s)", ColAmber);
                    break;
                }

            case "task_engine_started":
                {
                    var resumed = TryGet(d, "resumed", out var rv) ? rv.ToString() : "0";
                    Core3D.SpawnSystemEffect("TASK ENG", "SUBCONSCIOUS ONLINE", $"Resumed {resumed} task(s)", ColCyan, 5.5);
                    LogSystemEvent("TASK", $"Engine started (resumed {resumed})", ColCyan);
                    break;
                }

            case "task_created":
                {
                    _ = RefreshTasksAsync();
                    var title = TryStr(d, "title", out var t0) ? t0 : "task";
                    var obj = TryStr(d, "objective", out var o0) ? o0 : "";
                    SoundEffects.Play("task_spawn");
                    Core3D.TriggerShockwave(ColCyan, 350, 1.25);
                    Core3D.SpawnSystemEffect("TASK+", $"SPAWNED // {title}", obj, ColCyan, 6.5);
                    LogSystemEvent("TASK", $"Created: {title}", ColCyan);
                    break;
                }

            case "task_started":
                {
                    _ = RefreshTasksAsync();
                    var title = TryStr(d, "title", out var t0) ? t0 : "task";
                    Core3D.SpawnSystemEffect("TASK", "IBA SESSION ONLINE", title, ColCyan, 5.5);
                    LogSystemEvent("TASK", $"Started IBA session: {title}", ColCyan);
                    break;
                }

            case "task_step_start":
            case "task_step_started":
                {
                    var snum = TryGet(d, "step_number", out var sn) ? sn.ToString() : (TryStr(d, "kind", out var k0) ? k0 : "step");
                    var thought = TryStr(d, "thought", out var th) ? th : (TryStr(d, "title", out var tt) ? tt : "New IBA conversation step");
                    SoundEffects.Play("task_step");
                    Core3D.SpawnSystemEffect("STEP ▸", $"IBA STEP #{snum}", thought, ColBlue, 5.5);
                    LogSystemEvent("STEP", $"Step #{snum}: {thought}", ColBlue);
                    break;
                }

            case "task_thought":
                {
                    var th = TryStr(d, "text", out var th0) ? th0 : "";
                    var delta = TryStr(d, "delta", out var d0) ? d0 : "";
                    var tid = TryStr(d, "task_id", out var tid0) ? tid0 : "";
                    var step = TryGet(d, "step", out var sp) ? sp.ToString() : "";
                    if (!string.IsNullOrWhiteSpace(th) || !string.IsNullOrWhiteSpace(delta))
                    {
                        LogOrUpdateAgentThought(tid, step, th, delta);
                    }
                    break;
                }

            case "task_tool_call":
                {
                    var tool = TryStr(d, "tool", out var tl) ? tl : "tool";
                    var stitle = TryStr(d, "step_title", out var st) ? st : tool;
                    SoundEffects.Play("tool");
                    Core3D.SpawnSystemEffect("ACTION", $"IBA // {tool.ToUpperInvariant()}", stitle, ColMagenta, 5.6);
                    LogSystemEvent("ACTION", $"{tool}: {stitle}", ColMagenta);
                    break;
                }

            case "task_tool_result":
                {
                    var tool = TryStr(d, "tool", out var tl) ? tl : "tool";
                    var preview = TryStr(d, "preview", out var rpv) ? rpv : "";
                    var ok = !TryGet(d, "ok", out var okv) || okv.ValueKind == JsonValueKind.True;
                    Core3D.SpawnSystemEffect("RECEIPT", ok ? $"{tool.ToUpperInvariant()} OK" : $"{tool.ToUpperInvariant()} FAIL", preview, ok ? ColGreen : ColAmber, 5.0);
                    LogSystemEvent("RECEIPT", $"{tool}: {preview}", ok ? ColGreen : ColAmber);
                    break;
                }

            case "task_step_done":
            case "task_step_completed":
                {
                    _ = RefreshTasksAsync();
                    var prog = TryGet(d, "progress", out var p0) ? p0.ToString() : (TryGet(d, "step", out var sp0) ? $"S{sp0}" : "ok");
                    var sum = TryStr(d, "summary", out var s0) ? s0 : "";
                    SoundEffects.Play("task_step");
                    Core3D.SpawnSystemEffect("STEP ✔", $"STEP DONE ({prog})", sum, ColGreen, 5.8);
                    LogSystemEvent("STEP", $"[{prog}] {sum}", ColGreen);
                    break;
                }

            case "task_step_error":
            case "task_step_rejected":
                {
                    var err = TryStr(d, "error", out var e0) ? e0 : (TryStr(d, "reason", out var r0) ? r0 : "retry");
                    SoundEffects.Play("alert");
                    Core3D.SpawnSystemEffect("STEP ⚠", "STEP RETRY", err, ColAmber, 6.0);
                    LogSystemEvent("STEP", $"Retry: {err}", ColAmber);
                    break;
                }

            case "task_asks_user":
            case "task_needs_user":
                {
                    _ = RefreshTasksAsync();
                    var q = TryStr(d, "question", out var q0) ? q0 : "";
                    AddEventLine($"⚠ task needs answer: {q}", FindBrush("Amber"));
                    SoundEffects.Play("alert");
                    Core3D.TriggerShockwave(ColAmber, 360, 1.3);
                    Core3D.SpawnSystemEffect("QUESTION", "TASK NEEDS INPUT", q, ColAmber, 7.5);
                    LogSystemEvent("TASK", $"Needs answer: {q}", ColAmber);
                    break;
                }

            case "task_answered":
                {
                    _ = RefreshTasksAsync();
                    var ans = TryStr(d, "answer", out var a0) ? a0 : "";
                    SoundEffects.Play("task_step");
                    Core3D.SpawnSystemEffect("ANSWER", "USER ANSWERED TASK", ans, ColGreen, 5.5, inbound: true);
                    LogSystemEvent("TASK", $"Answered: {ans}", ColGreen);
                    break;
                }

            case "task_paused":
            case "task_resumed":
            case "task_cancelled":
                {
                    _ = RefreshTasksAsync();
                    var tid = TryStr(d, "task_id", out var id0) ? id0 : "";
                    Core3D.SpawnSystemEffect("TASK", type.Replace('_', ' ').ToUpperInvariant(), $"Task {tid}", ColAmber, 5.0);
                    LogSystemEvent("TASK", $"{type}: {tid}", ColAmber);
                    break;
                }

            case "task_finished":
            case "task_completion_relay":
                {
                    _ = RefreshTasksAsync();
                    var r = TryStr(d, "result", out var r0) ? r0 : "";
                    if (type == "task_finished")
                    {
                        AddEventLine($"✔ task done: {r}", FindBrush("Green"));
                        SoundEffects.Play("task_done");
                        Core3D.TriggerShockwave(ColGreen, 390, 1.4);
                    }
                    Core3D.SpawnSystemEffect("TASK ✔", "TASK COMPLETED", r, ColGreen, 7.0);
                    LogSystemEvent("TASK", $"Done: {r}", ColGreen);
                    break;
                }

            case "task_failed":
            case "task_problem_relay":
                {
                    _ = RefreshTasksAsync();
                    var err = TryStr(d, "error", out var e0) ? e0 : "failed";
                    SoundEffects.Play("alert");
                    Core3D.TriggerShockwave(ColRed, 350, 1.3);
                    Core3D.SpawnSystemEffect("TASK ✖", "TASK FAILED", err, ColRed, 7.0);
                    LogSystemEvent("TASK", $"Failed: {err}", ColRed);
                    break;
                }

            case "task_question_relay":
                {
                    var t2 = TryStr(d, "title", out var t0) ? t0 : "task";
                    var q2 = TryStr(d, "question", out var q0) ? q0 : "";
                    AddEventLine($"⚠ {t2} asks: {q2}", FindBrush("Amber"));
                    SoundEffects.Play("alert");
                    Core3D.TriggerShockwave(ColAmber, 340, 1.2);
                    Core3D.SpawnSystemEffect("QUESTION", $"{t2} ASKS", q2, ColAmber, 7.0);
                    LogSystemEvent("TASK", $"{t2} asks: {q2}", ColAmber);
                    break;
                }

            case "backend_switching":
                {
                    var bt = TryStr(d, "target", out var b0) ? b0 : "?";
                    AddEventLine($"switching backend → {bt}…", FindBrush("Amber"));
                    Core3D.SpawnSystemEffect("LLM", "SWITCHING PROVIDER", $"Target → {bt}", ColAmber, 5.5);
                    LogSystemEvent("LLM", $"Switching → {bt}", ColAmber);
                    break;
                }

            case "backend_switched":
                {
                    var ok = TryGet(d, "ok", out var okv) && okv.ValueKind == JsonValueKind.True;
                    var bb = TryStr(d, "backend", out var b0) ? b0 : "?";
                    var be = TryStr(d, "error", out var e0) ? e0 : "?";
                    AddEventLine(ok ? $"backend now: {bb}" : $"backend switch failed: {be}", ok ? FindBrush("Green") : FindBrush("Red"));
                    Core3D.TriggerShockwave(ok ? ColPurple : ColRed, 350, 1.3);
                    Core3D.SpawnSystemEffect("LLM", ok ? "PROVIDER ONLINE" : "PROVIDER SWITCH FAILED", ok ? bb : be, ok ? ColPurple : ColRed, 6.0);
                    LogSystemEvent("LLM", ok ? $"Switched to {bb}" : $"Switch failed: {be}", ok ? ColPurple : ColRed);
                    _ = RefreshProviderAsync();
                    break;
                }

            case "provider_changed":
                _ = RefreshProviderAsync();
                Core3D.SpawnSystemEffect("LLM", "PROVIDER CHANGED", TryStr(d, "provider", out var pv) ? pv : "updated", ColPurple, 5.0);
                LogSystemEvent("LLM", "Provider updated", ColPurple);
                break;

            case "profile_updated":
            case "onboard_answer":
            case "onboard_extracted":
                {
                    _ = RefreshProviderAsync();
                    var pk = TryStr(d, "key", out var k0) ? k0 : "memory";
                    var pval = TryStr(d, "value", out var v0) ? v0 : "updated";
                    Core3D.SpawnSystemEffect("MEMORY", "PROFILE LEARNED", $"{pk}: {pval}", ColCyan, 5.5);
                    LogSystemEvent("MEMORY", $"Updated {pk}", ColCyan);
                    break;
                }

            case "environment_scanned":
                {
                    _ = RefreshProviderAsync();
                    var roots = TryGet(d, "roots", out var rv) ? rv.ToString() : "?";
                    Core3D.SpawnSystemEffect("SCAN", "ENVIRONMENT SCAN", $"Indexed {roots} workspace roots", ColBlue, 5.5);
                    LogSystemEvent("SCAN", $"Indexed {roots} workspace roots", ColBlue);
                    break;
                }

            case "config_applied":
                Core3D.SpawnSystemEffect("CONFIG", "SETTINGS APPLIED", TryGet(d, "changed", out var ch) ? ch.ToString() : "updated", ColCyan, 5.5);
                LogSystemEvent("CONFIG", "Configuration updated", ColCyan);
                _ = RefreshProviderAsync();
                break;

            case "wakeword_changed":
                Core3D.SpawnSystemEffect("WAKE", "WAKE WORD MATRIX", "Wakeword configuration updated", ColMagenta, 5.0);
                LogSystemEvent("WAKE", "Wakeword updated", ColMagenta);
                break;

            case "conversation_ended":
                CoreSubStatusText.Text = "CONVERSATION STANDBY // WAKE READY";
                Core3D.SpawnSystemEffect("VOICE", "TURN COMPLETE", TryStr(d, "reason", out var cr) ? cr : "standby", ColCyan, 4.5);
                break;

            case "voice_disabled":
                {
                    var ve = TryStr(d, "error", out var e0) ? e0 : "";
                    AddEventLine("voice unavailable: " + ve, FindBrush("Red"));
                    Core3D.SpawnSystemEffect("VOICE ⚠", "VOICE UNAVAILABLE", ve, ColRed, 6.5);
                    LogSystemEvent("ERR", $"Voice unavailable: {ve}", ColRed);
                    break;
                }
        }
    }

    private void ApplySnapshot(JsonElement d)
    {
        if (TryStr(d, "assistant", out var an)) { _assistantName = an; UpdateName(); }
        if (d.TryGetProperty("conscious_reasoning", out var cr))
        {
            _consciousReasoning = cr.ValueKind == JsonValueKind.True;
            UpdateReasoningButton();
        }
        if (d.TryGetProperty("tasks", out var tasks) && tasks.ValueKind == JsonValueKind.Array)
        {
            _tasks.Clear();
            foreach (var t in tasks.EnumerateArray()) _tasks.Add(TaskVm.FromJson(t));
            Core3D.SyncActiveTasks(_tasks);
        }
        if (d.TryGetProperty("profile_progress", out var pp))
            ProfileProgress.Text = $"onboarding: {(int)(pp.GetDouble() * 100)}%";
        if (d.TryGetProperty("voice_state", out var vs))
            SetVoiceState(vs.GetString() ?? "idle");
        if (d.TryGetProperty("voice_engines", out var ve) && ve.ValueKind == JsonValueKind.Object)
        {
            var tts = TryStr(ve, "tts", out var t) ? t : "?";
            var ttsProv = TryStr(ve, "tts_provider", out var tp) ? tp : "";
            var stt = TryStr(ve, "stt", out var s) ? s : "?";
            var wake = ve.TryGetProperty("wake", out var w) && w.ValueKind == JsonValueKind.True;
            EngineText.Text = $"engines: tts={tts}{(string.IsNullOrEmpty(ttsProv) ? "" : $" ({ttsProv})")} stt={stt} wake={(wake ? "ok" : "none")}";
            _engineSummary = string.IsNullOrEmpty(ttsProv) ? $"{tts} • {stt}" : $"{tts} ({ttsProv})";
            if (ve.TryGetProperty("tts_cache", out var tc) && tc.ValueKind == JsonValueKind.Object)
                ApplyCacheStatsObject(tc, enabled: true);
            Core3D.SetEngineTelemetry(_engineSummary, _providerSummary, _cacheSummary);
        }
        _ = RefreshProviderAsync();
    }

    private async Task RefreshProviderAsync()
    {
        var st = await _backend.GetAsync("api/status");
        if (st is JsonElement s && s.TryGetProperty("provider", out var p))
        {
            var eff = TryStr(p, "effective", out var e) ? e : "none";
            var loaded = "";
            if (p.TryGetProperty("lmstudio_loaded", out var lms) && lms.ValueKind == JsonValueKind.Array && lms.GetArrayLength() > 0)
                loaded = " (" + (lms[0].GetString() ?? "") + ")";
            Dispatcher.Invoke(() =>
            {
                ProviderBadge.Text = $"provider: {eff}{loaded}";
                _providerSummary = $"{eff}{loaded}";
                BtnDory.Opacity = eff == "dory" ? 1.0 : 0.55;
                BtnLms.Opacity = eff == "lmstudio" ? 1.0 : 0.55;
                Core3D.SetEngineTelemetry(_engineSummary, _providerSummary, _cacheSummary);
            });
        }
        await RefreshCacheAsync();
    }

    private async Task RefreshCacheAsync()
    {
        var vc = await _backend.GetAsync("api/voice/cache");
        if (vc is JsonElement el && el.ValueKind == JsonValueKind.Object)
        {
            var enabled = el.TryGetProperty("enabled", out var en) && en.ValueKind == JsonValueKind.True;
            Dispatcher.Invoke(() =>
            {
                if (el.TryGetProperty("stats", out var stats) && stats.ValueKind == JsonValueKind.Object)
                    ApplyCacheStatsObject(stats, enabled);
                else
                {
                    _cacheSummary = enabled ? "CACHE: ON" : "CACHE: OFF";
                    CacheBadge.Text = enabled ? "cache: ON" : "cache: OFF";
                    Core3D.SetEngineTelemetry(_engineSummary, _providerSummary, _cacheSummary);
                }
            });
        }
    }

    private void ApplyCacheStatsObject(JsonElement stats, bool enabled)
    {
        int hits = stats.TryGetProperty("hits_total", out var ht) && ht.TryGetInt32(out var h0) ? h0 : 0;
        int sent = stats.TryGetProperty("sentences", out var sv) && sv.TryGetInt32(out var s0) ? s0 : 0;
        int phr = stats.TryGetProperty("phrases", out var pv) && pv.TryGetInt32(out var p0) ? p0 : 0;
        int wrd = stats.TryGetProperty("words", out var wv) && wv.TryGetInt32(out var w0) ? w0 : 0;
        _cacheSummary = enabled
            ? $"CACHE: ON • {hits} HITS ({sent}S/{phr}P/{wrd}W)"
            : $"CACHE: OFF ({sent}S/{phr}P/{wrd}W)";
        CacheBadge.Text = enabled ? $"cache: {hits} hits ({sent}S/{phr}P/{wrd}W)" : "cache: OFF";
        Core3D.SetEngineTelemetry(_engineSummary, _providerSummary, _cacheSummary);
    }

    private async Task RefreshTasksAsync()
    {
        var snap = await _backend.Tasks();
        if (snap is not JsonElement root || root.ValueKind != JsonValueKind.Array) return;
        Dispatcher.Invoke(() =>
        {
            var incoming = root.EnumerateArray().Select(TaskVm.FromJson).ToList();
            var byId = _tasks.ToDictionary(t => t.Id);
            _tasks.Clear();
            foreach (var t in incoming)
            {
                if (byId.TryGetValue(t.Id, out var old))
                {
                    old.Apply(root.EnumerateArray().First(x => (x.TryGetProperty("id", out var id) ? id.GetString() : "") == t.Id));
                    t.AnswerDraft = old.AnswerDraft;
                }
                _tasks.Add(t);
            }
            Core3D.SyncActiveTasks(_tasks);
        });
    }

    private void BuildWave(Canvas canvas, List<Rectangle> bars, Brush brush)
    {
        bars.Clear();
        canvas.Children.Clear();
        var width = Math.Max(canvas.ActualWidth, 140);
        var count = Math.Max(24, (int)(width / 7));
        for (var i = 0; i < count; i++)
        {
            var bar = new Rectangle
            {
                Width = 4,
                Height = 2,
                RadiusX = 2,
                RadiusY = 2,
                Fill = brush,
                Opacity = 0.9,
            };
            Canvas.SetLeft(bar, i * 7);
            Canvas.SetTop(bar, Math.Max(0, canvas.ActualHeight / 2 - 1));
            canvas.Children.Add(bar);
            bars.Add(bar);
        }
    }

    private void WaveTick()
    {
        var userTarget = _userTarget;
        var aiTarget = _voiceState == "speaking" ? 1.0 : 0.0;
        _userLevel = Math.Clamp(_userLevel + (userTarget - _userLevel) * 0.22 + (_waveRng.NextDouble() - 0.5) * 0.015, 0, 1);
        _aiLevel = Math.Clamp(_aiLevel + (aiTarget - _aiLevel) * 0.22 + (_waveRng.NextDouble() - 0.5) * 0.015, 0, 1);
        _userPhase += 0.42;
        _aiPhase += 0.35;
        _ambientLevel = Math.Clamp(_ambientLevel + (_ambientTarget - _ambientLevel) * 0.35, 0, 1);
        _ambientPhase += 0.28;

        DrawWave(UserWave, _userBars, _userLevel, _userPhase);
        DrawWave(AiWave, _aiBars, _aiLevel, _aiPhase);
        DrawWave(AmbientWave, _ambientBars, _ambientLevel, _ambientPhase);
        Core3D.SetAudioLevels(_userLevel, _aiLevel, _ambientLevel);
    }

    private static void DrawWave(Canvas canvas, List<Rectangle> bars, double level, double phase)
    {
        if (bars.Count == 0 || canvas.ActualHeight <= 0) return;
        var mid = canvas.ActualHeight / 2;
        var maxH = Math.Max(4, canvas.ActualHeight - 4);
        for (var i = 0; i < bars.Count; i++)
        {
            var t = i / (double)Math.Max(1, bars.Count - 1);
            var env = Math.Sin(t * Math.PI);
            var wave = Math.Sin(phase + i * 0.55) * Math.Sin(phase * 0.37 + i * 1.71);
            var amp = level * env * (0.2 + 0.8 * Math.Abs(wave));
            var h = Math.Max(2, 2 + amp * maxH);
            var bar = bars[i];
            bar.Height = h;
            bar.Opacity = 0.16 + 0.84 * Math.Min(1.0, level * 1.35);
            Canvas.SetTop(bar, mid - h / 2);
            Canvas.SetLeft(bar, i * 7);
        }
    }

    // ------------------------------------------------------------------ UI helpers

    private string _lastPartial = "";

    private void UpdateName()
    {
        NameText.Text = _assistantName.ToUpperInvariant();
        Core3D.SetAssistantName(_assistantName);
    }

    private void SetVoiceState(string state)
    {
        _voiceState = state;
        if (state == "off")
        {
            _ambientTarget = 0;
            _ambientLevel = 0;
        }
        StateText.Text = state switch
        {
            "idle" => "IDLE — listening for the wake word",
            "listening" => "LISTENING",
            "thinking" => "THINKING",
            "speaking" => "SPEAKING",
            "off" => "MIC OFF",
            _ => state.ToUpperInvariant(),
        };
        Core3D.SetVoiceState(state);
    }

    private void ShowPartial(string text)
    {
        _lastPartial = text;
        if (ConsciousStream.Children.Count > 0 &&
            ConsciousStream.Children[^1] is TextBlock tb && tb.Name == "PartialLine")
        {
            tb.Text = text;
        }
        else
        {
            var p = new TextBlock
            {
                Name = "PartialLine",
                Text = text,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 11.5,
                Foreground = FindBrush("Cyan"),
                Opacity = 0.85,
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 2, 0, 2),
            };
            ConsciousStream.Children.Add(p);
        }
        ConsciousScroll.ScrollToEnd();
    }

    private void ClearActiveThoughtTag()
    {
        for (int i = ConsciousStream.Children.Count - 1; i >= 0; i--)
        {
            if (ConsciousStream.Children[i] is TextBlock tb && tb.Name == "ThoughtLine")
            {
                tb.Name = "";
                break;
            }
        }
    }

    private void ShowFinal(string text)
    {
        if (ConsciousStream.Children.Count > 0 &&
            ConsciousStream.Children[^1] is TextBlock tb && tb.Name == "PartialLine")
            ConsciousStream.Children.Remove(tb);
        _lastPartial = "";
        var clean = (text ?? "").Trim();
        if (clean.Length == 0) return;
        if (string.Equals(clean, _lastShownUserText, StringComparison.OrdinalIgnoreCase) &&
            (DateTime.UtcNow - _lastShownUserAt).TotalSeconds < 8.0)
        {
            return;
        }
        ClearActiveThoughtTag();
        _lastShownUserText = clean;
        _lastShownUserAt = DateTime.UtcNow;
        AddAssistantLine(clean, FindBrush("Cyan"), user: true);
    }

    private void AddAssistantLine(string text, Brush brush, bool user = false)
    {
        if (!user) ClearActiveThoughtTag();
        var tb = new TextBlock
        {
            Text = (user ? "you ▸ " : _assistantName + " ▸ ") + text,
            FontSize = user ? 11.5 : 12.5,
            FontFamily = new FontFamily(user ? "Consolas" : "Segoe UI"),
            Foreground = brush,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 3, 0, 3),
        };
        ConsciousStream.Children.Add(tb);
        ConsciousScroll.ScrollToEnd();
    }

    private void AddEventLine(string text, Brush brush)
    {
        var tb = new TextBlock
        {
            Text = text,
            FontSize = 11,
            FontFamily = new FontFamily("Consolas"),
            Foreground = brush,
            Opacity = 0.78,
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 2, 0, 2),
        };
        ConsciousStream.Children.Add(tb);
        ConsciousScroll.ScrollToEnd();
    }

    private static string CapThoughtForHud(string clean)
    {
        if (clean.Length == 0) return clean;
        int sentences = 0;
        for (int i = 0; i < clean.Length; i++)
        {
            char c = clean[i];
            if ((c == '.' || c == '!' || c == '?') &&
                (i + 1 >= clean.Length || char.IsWhiteSpace(clean[i + 1])))
            {
                sentences++;
                if (sentences >= 2)
                    return clean[..(i + 1)].Trim();
            }
            if (i >= 160)
                break;
        }
        if (clean.Length <= 160) return clean;
        int cut = clean.LastIndexOf(' ', 157);
        if (cut < 60) cut = 157;
        return clean[..cut].TrimEnd() + "…";
    }

    private void UpdateThoughtLine(string text)
    {
        var clean = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        if (clean.Length == 0) return;
        var capped = CapThoughtForHud(clean);
        if (ConsciousStream.Children.Count > 0 &&
            ConsciousStream.Children[^1] is TextBlock tb && tb.Name == "ThoughtLine")
        {
            tb.Text = "◌ " + capped;
        }
        else
        {
            ConsciousStream.Children.Add(new TextBlock
            {
                Name = "ThoughtLine",
                Text = "◌ " + capped,
                FontSize = 10.5,
                FontFamily = new FontFamily("Consolas"),
                Foreground = FindBrush("Dim"),
                Opacity = 0.74,
                TextWrapping = TextWrapping.Wrap,
                Margin = new Thickness(0, 2, 0, 2),
            });
        }
        ConsciousScroll.ScrollToEnd();
    }

    private void CloseActiveThoughtSession()
    {
        _activeThoughtSession = "";
        _activeThoughtTb = null;
    }

    private void LogOrUpdateAgentThought(string taskId, string step, string text, string delta)
    {
        var sessionKey = string.IsNullOrEmpty(step) ? (string.IsNullOrEmpty(taskId) ? "agent" : taskId) : $"{taskId}:{step}";
        var cleanText = (text ?? "").Replace('\r', ' ').Replace('\n', ' ').Trim();
        var cleanDelta = (delta ?? "").Replace('\r', ' ').Replace('\n', ' ').Trim();

        // Check if continuing the active agent reasoning bubble
        if (_activeThoughtSession == sessionKey &&
            _activeThoughtTb != null &&
            SystemEventStream.Children.Count > 0 &&
            SystemEventStream.Children[^1] is Grid lastRow &&
            lastRow.Tag as string == "AGENT_THOUGHT")
        {
            if (!string.IsNullOrEmpty(cleanText) && (cleanText.Length >= _activeThoughtTb.Text.Length || cleanText.StartsWith(_activeThoughtTb.Text[..Math.Min(20, _activeThoughtTb.Text.Length)])))
            {
                _activeThoughtTb.Text = cleanText;
            }
            else
            {
                var chunk = !string.IsNullOrEmpty(cleanDelta) ? cleanDelta : cleanText;
                if (!string.IsNullOrEmpty(chunk))
                {
                    var cur = _activeThoughtTb.Text;
                    _activeThoughtTb.Text = string.IsNullOrEmpty(cur)
                        ? chunk
                        : (cur.EndsWith(" ") || chunk.StartsWith(" ") ? cur + chunk : cur + " " + chunk);
                }
            }
            SystemEventScroll.ScrollToEnd();
            return;
        }

        // New reasoning session bubble
        _activeThoughtSession = sessionKey;

        if ((DateTime.UtcNow - _lastThoughtEffectAtAgent).TotalSeconds >= 4.0)
        {
            _lastThoughtEffectAtAgent = DateTime.UtcNow;
            SoundEffects.Play("thought");
            var preview = !string.IsNullOrEmpty(cleanText) ? cleanText : cleanDelta;
            Core3D.SpawnSystemEffect("AGENT", "AGENT REASONING", preview, ColPurple, 4.8);
        }

        var displayText = !string.IsNullOrEmpty(cleanText) ? cleanText : (!string.IsNullOrEmpty(cleanDelta) ? cleanDelta : "Thinking…");

        while (SystemEventStream.Children.Count >= 120)
            SystemEventStream.Children.RemoveAt(0);

        var row = new Grid
        {
            Margin = new Thickness(0, 1.5, 0, 1.5),
            Tag = "AGENT_THOUGHT"
        };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var chip = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0x30, ColPurple.R, ColPurple.G, ColPurple.B)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0x88, ColPurple.R, ColPurple.G, ColPurple.B)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(3),
            Padding = new Thickness(4, 0, 4, 0),
            Margin = new Thickness(0, 0, 6, 0),
            VerticalAlignment = VerticalAlignment.Top,
            Child = new TextBlock
            {
                Text = "AGENT",
                FontFamily = new FontFamily("Consolas"),
                FontSize = 9.5,
                FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush(ColPurple),
            },
        };
        var msgTb = new TextBlock
        {
            Text = displayText,
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10.5,
            Foreground = new SolidColorBrush(Color.FromRgb(0xBA, 0xCC, 0xE2)),
            TextWrapping = TextWrapping.Wrap,
        };
        Grid.SetColumn(chip, 0);
        Grid.SetColumn(msgTb, 1);
        row.Children.Add(chip);
        row.Children.Add(msgTb);

        _activeThoughtTb = msgTb;

        SystemEventStream.Children.Add(row);
        SystemEventScroll.ScrollToEnd();
    }

    private void LogSystemEvent(string tag, string message, Color color)
    {
        CloseActiveThoughtSession();
        while (SystemEventStream.Children.Count >= 120)
            SystemEventStream.Children.RemoveAt(0);

        var row = new Grid { Margin = new Thickness(0, 1.5, 0, 1.5) };
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

        var chip = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0x30, color.R, color.G, color.B)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0x88, color.R, color.G, color.B)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(3),
            Padding = new Thickness(4, 0, 4, 0),
            Margin = new Thickness(0, 0, 6, 0),
            VerticalAlignment = VerticalAlignment.Top,
            Child = new TextBlock
            {
                Text = tag,
                FontFamily = new FontFamily("Consolas"),
                FontSize = 9.5,
                FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush(color),
            },
        };
        var msgTb = new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(message) ? "—" : message.Replace('\n', ' '),
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10.5,
            Foreground = new SolidColorBrush(Color.FromRgb(0xBA, 0xCC, 0xE2)),
            TextWrapping = TextWrapping.Wrap,
        };
        Grid.SetColumn(chip, 0);
        Grid.SetColumn(msgTb, 1);
        row.Children.Add(chip);
        row.Children.Add(msgTb);

        SystemEventStream.Children.Add(row);
        SystemEventScroll.ScrollToEnd();
    }

    private Brush FindBrush(string key)
        => (Brush)(TryFindResource(key) ?? Brushes.Gray);
    private System.Windows.Media.Effects.Effect? FindEffect(string key)
        => TryFindResource(key) as System.Windows.Media.Effects.Effect;

    private static bool TryStr(JsonElement e, string key, out string value)
    {
        value = "";
        if (e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String)
        {
            value = v.GetString() ?? "";
            return true;
        }
        return false;
    }

    private static bool TryGet(JsonElement e, string key, out JsonElement value)
    {
        value = default;
        return e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out value);
    }

    private static JsonElement EventData(JsonElement ev)
    {
        if (ev.ValueKind != JsonValueKind.Object) return default;
        if (ev.TryGetProperty("data", out var d) && d.ValueKind != JsonValueKind.Undefined)
            return d;
        var fields = new Dictionary<string, JsonElement>();
        foreach (var p in ev.EnumerateObject())
        {
            if (p.Name != "type") fields[p.Name] = p.Value.Clone();
        }
        return JsonSerializer.SerializeToElement(fields);
    }

    // ------------------------------------------------------------------ actions

    private async void OnSend(object sender, RoutedEventArgs e) => await SendChat();
    private async void OnChatKey(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) await SendChat();
    }

    private async Task SendChat()
    {
        var text = ChatInput.Text.Trim();
        if (text.Length == 0) return;
        ChatInput.Clear();
        ShowFinal(text);
        CoreSubtitleText.Text = $"you ▸ {text}";
        CoreSubStatusText.Text = "COMMAND TRANSMITTED // PROCESSING";
        Core3D.TriggerShockwave(ColCyan, 310, 1.1);
        Core3D.SpawnSystemEffect("CMD", "COMMAND SENT", text, ColCyan, 5.5, inbound: true);
        LogSystemEvent("CMD", text, ColCyan);

        var result = await _backend.Say(text);
        var sawSentence = false;
        if (result is JsonElement r && r.TryGetProperty("events", out var evs) && evs.ValueKind == JsonValueKind.Array)
        {
            foreach (var ev in evs.EnumerateArray())
            {
                var type = ev.ValueKind == JsonValueKind.Object && ev.TryGetProperty("type", out var t)
                    ? t.GetString()
                    : null;
                var data = EventData(ev);
                if (type is "sentence" or "done")
                {
                    if (type == "sentence") sawSentence = true;
                    if (type == "done" && !sawSentence &&
                        TryStr(data, "full", out var full) && full.Length > 0)
                    {
                        AddAssistantLine(full, FindBrush("Text"));
                        CoreSubtitleText.Text = $"{_assistantName} ▸ {full}";
                        Core3D.SpawnSystemEffect("SPEECH", $"{_assistantName.ToUpperInvariant()} RESPONSE", full, ColCyan, 6.0);
                    }
                    else if (type == "sentence")
                    {
                        Handle("assistant_sentence", data);
                    }
                }
                else if (type is "tool") Handle("assistant_tool", data);
                else if (type is "thought") Handle("assistant_thought", data);
            }
        }
    }

    private void OnToggleCoreFocus(object sender, RoutedEventArgs e)
    {
        _coreFocusMode = !_coreFocusMode;
        double targetOpacity = _coreFocusMode ? 0.12 : 1.0;
        var anim = new DoubleAnimation(targetOpacity, TimeSpan.FromMilliseconds(260))
        {
            EasingFunction = new CubicEase { EasingMode = EasingMode.EaseOut },
        };
        LeftWing.BeginAnimation(OpacityProperty, anim);
        RightWing.BeginAnimation(OpacityProperty, anim);
        LeftWing.IsHitTestVisible = !_coreFocusMode;
        RightWing.IsHitTestVisible = !_coreFocusMode;
        BtnCoreFocus.Content = _coreFocusMode ? "HUD WINGS" : "CORE VIEW";
        SoundEffects.Play("pulse");
        Core3D.TriggerShockwave(ColCyan, 420, 1.3);
        LogSystemEvent("HUD", _coreFocusMode ? "Core focus mode active" : "HUD wings restored", ColCyan);
    }

    private void OnTriggerFxDemo(object sender, RoutedEventArgs e)
    {
        SoundEffects.Play("pulse");
        Core3D.TriggerDemoCascade();
        LogSystemEvent("FX", "Triggered holographic system telemetry pulse cascade", ColGold);
    }

    private void UpdateReasoningButton()
    {
        BtnReasoning.Content = _consciousReasoning ? "THINK ON" : "THINK OFF";
        BtnReasoning.Opacity = _consciousReasoning ? 1.0 : 0.65;
    }

    private async void OnToggleReasoning(object sender, RoutedEventArgs e)
    {
        _consciousReasoning = !_consciousReasoning;
        UpdateReasoningButton();
        SoundEffects.Play("pulse");
        await _backend.SaveConfig(new Dictionary<string, object>
        {
            ["conscious.reasoning_enabled"] = _consciousReasoning,
        });
        Core3D.SpawnSystemEffect("MIND",
            _consciousReasoning ? "CONSCIOUS REASONING ON" : "CONSCIOUS REASONING OFF",
            _consciousReasoning ? "Foreground chain-of-thought enabled" : "Foreground direct fast-reply mode",
            _consciousReasoning ? ColPurple : ColCyan, 4.8);
        LogSystemEvent("MIND",
            _consciousReasoning ? "Conscious reasoning enabled" : "Conscious reasoning disabled",
            _consciousReasoning ? ColPurple : ColCyan);
    }

    private async void OnToggleCacheClick(object sender, MouseButtonEventArgs e)
    {
        e.Handled = true;
        SoundEffects.Play("pulse");
        await _backend.PostAsync("api/voice/cache", new { action = "toggle" });
        await RefreshCacheAsync();
    }

    private void OnToggleSfx(object sender, RoutedEventArgs e)
    {
        SoundEffects.Enabled = !SoundEffects.Enabled;
        BtnSfx.Content = SoundEffects.Enabled ? "SFX ON" : "SFX OFF";
        if (SoundEffects.Enabled) SoundEffects.Play("pulse");
        Core3D.SpawnSystemEffect("AUDIO", SoundEffects.Enabled ? "HUD AUDIO ARMED" : "HUD AUDIO MUTED", "Futuristic sound cues toggled", SoundEffects.Enabled ? ColCyan : ColAmber, 4.2);
        LogSystemEvent("AUDIO", SoundEffects.Enabled ? "HUD SFX enabled" : "HUD SFX muted", SoundEffects.Enabled ? ColCyan : ColAmber);
    }

    private async void OnTogglePtt(object sender, RoutedEventArgs e)
    {
        _pttActive = !_pttActive;
        BtnPtt.Content = _pttActive ? "STOP PTT" : "PTT VOICE";
        await _backend.Ptt(_pttActive ? "start" : "stop");
        if (_pttActive)
        {
            Core3D.TriggerShockwave(ColMagenta, 330, 1.2);
            Core3D.SpawnSystemEffect("PTT", "PUSH-TO-TALK ACTIVE", "Manual microphone override locked", ColMagenta, 5.5, inbound: true);
        }
        LogSystemEvent("PTT", _pttActive ? "Push-to-talk started" : "Push-to-talk stopped", ColMagenta);
    }

    private async void OnSwitchDory(object sender, RoutedEventArgs e)
        => await _backend.SwitchBackend("dory");
    private async void OnSwitchLms(object sender, RoutedEventArgs e)
        => await _backend.SwitchBackend("lmstudio");

    private void OnToggleMic(object sender, RoutedEventArgs e)
    {
        _micOn = !_micOn;
        BtnMic.Content = _micOn ? "MIC ON" : "MUTED";
        _ = _backend.PostAsync($"api/voice/mute?on={(!_micOn).ToString().ToLowerInvariant()}");
        SetVoiceState(_micOn ? "idle" : "off");
        Core3D.SpawnSystemEffect("MIC", _micOn ? "MICROPHONE ARMED" : "MICROPHONE MUTED", "Voice pipeline toggled", _micOn ? ColGreen : ColRed, 4.8);
        LogSystemEvent("MIC", _micOn ? "Microphone armed" : "Microphone muted", _micOn ? ColGreen : ColRed);
    }

    private void OnOpenSettings(object sender, RoutedEventArgs e)
    {
        var win = new SettingsWindow(_backend) { Owner = this };
        win.ShowDialog();
        _ = RefreshProviderAsync();
    }

    private async void OnSpawnTask(object sender, RoutedEventArgs e)
    {
        var obj = TaskObjective.Text.Trim();
        if (obj.Length == 0) return;
        TaskObjective.Clear();
        Core3D.SpawnSystemEffect("TASK+", "DISPATCHING TASK", obj, ColMagenta, 5.2);
        await _backend.CreateTask(obj);
    }

    private async void OnTaskObjectiveKey(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) OnSpawnTask(sender, e);
    }

    private async void OnAnswerTask(object sender, RoutedEventArgs e)
    {
        if (sender is Button b && b.Tag is string id &&
            b.DataContext is TaskVm vm && vm.AnswerDraft.Trim() is { Length: > 0 } answer)
        {
            await _backend.AnswerTask(id, answer);
            vm.PendingQuestion = "";
            vm.AnswerDraft = "";
        }
    }

    private Task<JsonElement?> Report(string id) => _backend.Say($"Give me a progress report on task {id}.");

    private async void OnTaskReport(object sender, RoutedEventArgs e)
    {
        if (sender is Button b && b.Tag is string id)
        {
            ShowFinal($"progress report on task {id}");
            var r = await Report(id);
            if (r is JsonElement el && el.TryGetProperty("events", out var evs) && evs.ValueKind == JsonValueKind.Array)
                foreach (var ev in evs.EnumerateArray())
                {
                    var type = ev.ValueKind == JsonValueKind.Object && ev.TryGetProperty("type", out var t)
                        ? t.GetString()
                        : null;
                    var data = EventData(ev);
                    if (type == "sentence" && TryStr(data, "text", out var line) && line.Length > 0)
                        AddAssistantLine(line, FindBrush("Text"));
                }
        }
    }

    private async void OnTaskPause(object sender, RoutedEventArgs e)
    { if (sender is Button b && b.Tag is string id) await _backend.TaskAction(id, "pause"); }
    private async void OnTaskResume(object sender, RoutedEventArgs e)
    { if (sender is Button b && b.Tag is string id) await _backend.TaskAction(id, "resume"); }
    private async void OnTaskCancel(object sender, RoutedEventArgs e)
    { if (sender is Button b && b.Tag is string id) await _backend.TaskAction(id, "cancel"); }

    protected override void OnClosed(EventArgs e)
    {
        _backend.Dispose();
        _backendHost.Dispose();
        base.OnClosed(e);
    }
}
