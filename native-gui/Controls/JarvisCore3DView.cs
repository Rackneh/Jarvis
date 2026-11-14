using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Effects;
using System.Windows.Media.Media3D;
using System.Windows.Shapes;
using System.Windows.Threading;
using JarvisDeck.Models;

namespace JarvisDeck.Controls;

/// <summary>
/// Futuristic 3D Jarvis Core (Viewport3D) + full-window holographic 3D perspective background stage
/// where every system action, voice event, cache hit, tool call, and task step materializes cleanly.
/// </summary>
public sealed class JarvisCore3DView : Grid
{
    private const int ParticleCount = 72;
    private const int SpectrumBarCount = 64;
    private const int CompassTickCount = 48;
    private const int MaxActiveNodes = 14;

    private readonly Canvas _bgCanvas;
    private readonly Viewport3D _viewport;
    private readonly Canvas _fgCanvas;
    private readonly Border _coreBadge;
    private readonly TextBlock _badgeTitle;
    private readonly TextBlock _badgeState;
    private readonly TextBlock _badgeTelemetry;
    private readonly TextBlock _badgeCache;

    private readonly DispatcherTimer _timer;
    private DateTime _lastTick = DateTime.UtcNow;
    private double _time;

    // Camera & interactive parallax / drag orbit
    private readonly PerspectiveCamera _camera;
    private double _camYaw;
    private double _camPitch = 0.08;
    private double _targetYaw;
    private double _targetPitch = 0.08;
    private double _mouseNormX;
    private double _mouseNormY;
    private bool _dragging;
    private Point _lastDragPos;

    // State & audio reactivity
    private string _assistantName = "JARVIS";
    private string _voiceState = "idle";
    private string _engineSummary = "jarvis_clone";
    private string _providerSummary = "initializing";
    private string _cacheSummary = "CACHE: STANDBY";

    private double _userLevel;
    private double _aiLevel;
    private double _ambientLevel;
    private double _cacheFlash;
    private Color _currentAccent = Color.FromRgb(0x22, 0xD3, 0xEE);
    private Color _targetAccent = Color.FromRgb(0x22, 0xD3, 0xEE);

    // 3D scene objects & animations
    private readonly ScaleTransform3D _corePulseScale = new(1, 1, 1);
    private readonly ScaleTransform3D _innerPulseScale = new(1, 1, 1);
    private readonly AxisAngleRotation3D _coreRotY = new(new Vector3D(0.2, 1.0, 0.1), 0);
    private readonly AxisAngleRotation3D _cageRot = new(new Vector3D(0.4, -1.0, 0.3), 0);
    private readonly AxisAngleRotation3D _ring1Rot = new(new Vector3D(0.15, 0.9, 0.4), 0);
    private readonly AxisAngleRotation3D _ring2Rot = new(new Vector3D(-0.7, 0.35, 0.6), 0);
    private readonly AxisAngleRotation3D _ring3Rot = new(new Vector3D(0.5, -0.6, 0.8), 0);
    private readonly List<(AxisAngleRotation3D Rot, double Speed)> _shardRots = new();

    private SolidColorBrush _coreDiffuseBrush = new(Color.FromRgb(0x16, 0x9C, 0xC4));
    private SolidColorBrush _coreEmissiveBrush = new(Color.FromRgb(0x22, 0xD3, 0xEE));
    private SolidColorBrush _cageBrush = new(Color.FromArgb(0xB0, 0x67, 0xE8, 0xF9));
    private SolidColorBrush _ring1Brush = new(Color.FromRgb(0x22, 0xD3, 0xEE));
    private SolidColorBrush _ring2Brush = new(Color.FromRgb(0x38, 0xBD, 0xF8));
    private SolidColorBrush _ring3Brush = new(Color.FromRgb(0xA7, 0x8B, 0xFA));
    private SolidColorBrush _nucleusBrush = new(Colors.White);

    // Pooled 2.5D background & reticle elements
    private readonly List<Line> _gridLines = new();
    private readonly List<Particle3D> _particles = new();
    private readonly List<Rectangle> _spectrumBars = new();
    private readonly List<Line> _compassTicks = new();
    private readonly Path _arcOuter;
    private readonly Path _arcInner;
    private readonly Ellipse _haloOuter;
    private readonly Ellipse _haloMid;

    // Dynamic FX collections
    private readonly List<ActivityNode> _nodes = new();
    private readonly List<ShockwaveFx> _shockwaves = new();
    private readonly List<TaskSatellite> _taskSatellites = new();
    private readonly Random _rng = new();
    private int _sectorCursor;

    // Demo cascade timers
    private readonly List<(double TriggerAt, Action Run)> _scheduled = new();

    public JarvisCore3DView()
    {
        Background = Brushes.Transparent;
        ClipToBounds = true;

        _bgCanvas = new Canvas { IsHitTestVisible = false, ClipToBounds = true };
        Children.Add(_bgCanvas);

        _camera = new PerspectiveCamera
        {
            Position = new Point3D(0, 0.1, 6.4),
            LookDirection = new Vector3D(0, -0.02, -1),
            UpDirection = new Vector3D(0, 1, 0),
            FieldOfView = 45,
        };

        _viewport = new Viewport3D
        {
            Width = 480,
            Height = 480,
            HorizontalAlignment = HorizontalAlignment.Center,
            VerticalAlignment = VerticalAlignment.Center,
            IsHitTestVisible = false,
            Camera = _camera,
        };
        Build3DScene();
        Children.Add(_viewport);

        _fgCanvas = new Canvas { IsHitTestVisible = false, ClipToBounds = true };
        Children.Add(_fgCanvas);

        // Reticle halos + rotating arcs
        _haloOuter = new Ellipse
        {
            StrokeThickness = 1.0,
            StrokeDashArray = new DoubleCollection { 6, 6 },
            IsHitTestVisible = false,
        };
        _haloMid = new Ellipse
        {
            StrokeThickness = 1.4,
            StrokeDashArray = new DoubleCollection { 18, 8, 4, 8 },
            IsHitTestVisible = false,
        };
        _arcOuter = new Path { StrokeThickness = 2.4, StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round };
        _arcInner = new Path { StrokeThickness = 1.8, StrokeStartLineCap = PenLineCap.Round, StrokeEndLineCap = PenLineCap.Round };
        _fgCanvas.Children.Add(_haloOuter);
        _fgCanvas.Children.Add(_haloMid);
        _fgCanvas.Children.Add(_arcOuter);
        _fgCanvas.Children.Add(_arcInner);

        // Core HUD Status Badge positioned neatly under the 3D core
        _badgeTitle = new TextBlock
        {
            Text = "JARVIS // CORE V2",
            FontFamily = new FontFamily("Consolas"),
            FontSize = 11,
            FontWeight = FontWeights.Bold,
            Foreground = new SolidColorBrush(Color.FromRgb(0x67, 0xE8, 0xF9)),
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _badgeState = new TextBlock
        {
            Text = "IDLE — STANDBY",
            FontFamily = new FontFamily("Segoe UI"),
            FontSize = 13,
            FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(Color.FromRgb(0xE2, 0xF1, 0xFF)),
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin = new Thickness(0, 1, 0, 2),
        };
        _badgeTelemetry = new TextBlock
        {
            Text = "jarvis_clone • initializing",
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10,
            Foreground = new SolidColorBrush(Color.FromRgb(0x94, 0xA3, 0xB8)),
            HorizontalAlignment = HorizontalAlignment.Center,
        };
        _badgeCache = new TextBlock
        {
            Text = "CACHE: STANDBY",
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10,
            Foreground = new SolidColorBrush(Color.FromRgb(0xFA, 0xCC, 0x15)),
            HorizontalAlignment = HorizontalAlignment.Center,
            Margin = new Thickness(0, 1, 0, 0),
        };

        var badgeStack = new StackPanel { HorizontalAlignment = HorizontalAlignment.Center };
        badgeStack.Children.Add(_badgeTitle);
        badgeStack.Children.Add(_badgeState);
        badgeStack.Children.Add(_badgeTelemetry);
        badgeStack.Children.Add(_badgeCache);

        _coreBadge = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0xB8, 0x08, 0x12, 0x22)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0x66, 0x22, 0xD3, 0xEE)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(10),
            Padding = new Thickness(16, 7, 16, 7),
            Child = badgeStack,
            IsHitTestVisible = false,
        };
        _fgCanvas.Children.Add(_coreBadge);

        InitPools();
        HookMouseInteraction();

        _timer = new DispatcherTimer(DispatcherPriority.Render)
        {
            Interval = TimeSpan.FromMilliseconds(16),
        };
        _timer.Tick += OnFrameTick;
        Loaded += (_, _) =>
        {
            _lastTick = DateTime.UtcNow;
            _timer.Start();
        };
        Unloaded += (_, _) => _timer.Stop();
    }

    // ------------------------------------------------------------------ Public Telemetry API

    public void SetAssistantName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return;
        _assistantName = name.Trim().ToUpperInvariant();
        _badgeTitle.Text = $"{_assistantName} // CORE V2";
    }

    public void SetVoiceState(string state)
    {
        _voiceState = (state ?? "idle").Trim().ToLowerInvariant();
        _targetAccent = _voiceState switch
        {
            "listening" => Color.FromRgb(0xF4, 0x72, 0xB6),
            "thinking" => Color.FromRgb(0xA7, 0x8B, 0xFA),
            "speaking" => Color.FromRgb(0x38, 0xBD, 0xF8),
            "off" => Color.FromRgb(0x64, 0x74, 0x8B),
            _ => Color.FromRgb(0x22, 0xD3, 0xEE),
        };
        _badgeState.Text = _voiceState switch
        {
            "listening" => "LISTENING // AUDIO LOCK",
            "thinking" => "THINKING // NEURAL SYNTHESIS",
            "speaking" => "SPEAKING // VOICE OUTPUT",
            "off" => "OFFLINE // MIC MUTED",
            _ => "IDLE // WAKE MATRIX READY",
        };
        _badgeState.Foreground = new SolidColorBrush(
            _voiceState == "off"
                ? Color.FromRgb(0x94, 0xA3, 0xB8)
                : Color.FromRgb(0xE2, 0xF1, 0xFF));
    }

    public void SetAudioLevels(double userLevel, double aiLevel, double ambientLevel)
    {
        _userLevel = Math.Clamp(userLevel, 0.0, 1.0);
        _aiLevel = Math.Clamp(aiLevel, 0.0, 1.0);
        _ambientLevel = Math.Clamp(ambientLevel, 0.0, 1.0);
    }

    public void SetEngineTelemetry(string engineSummary, string providerSummary, string cacheSummary)
    {
        if (!string.IsNullOrWhiteSpace(engineSummary)) _engineSummary = engineSummary;
        if (!string.IsNullOrWhiteSpace(providerSummary)) _providerSummary = providerSummary;
        if (!string.IsNullOrWhiteSpace(cacheSummary)) _cacheSummary = cacheSummary;
        _badgeTelemetry.Text = $"{_engineSummary}  •  {_providerSummary}";
        _badgeCache.Text = _cacheSummary;
    }

    public void SyncActiveTasks(IEnumerable<TaskVm> tasks)
    {
        var active = tasks.Where(t => t.Status is "running" or "queued" or "waiting_user" or "paused").Take(8).ToList();
        var activeIds = active.Select(t => t.Id).ToHashSet();

        // Remove stale satellites
        for (int i = _taskSatellites.Count - 1; i >= 0; i--)
        {
            if (!activeIds.Contains(_taskSatellites[i].Id))
            {
                _bgCanvas.Children.Remove(_taskSatellites[i].OrbitRing);
                _bgCanvas.Children.Remove(_taskSatellites[i].Badge);
                _taskSatellites.RemoveAt(i);
            }
        }

        // Update or add satellites
        foreach (var t in active)
        {
            var existing = _taskSatellites.FirstOrDefault(s => s.Id == t.Id);
            var col = ParseHexColor(t.StatusColor, Color.FromRgb(0x22, 0xD3, 0xEE));
            if (existing == null)
            {
                var sat = CreateTaskSatellite(t, col);
                _taskSatellites.Add(sat);
            }
            else
            {
                existing.Title = t.Title;
                existing.Progress = t.Progress;
                existing.Status = t.Status;
                existing.Color = col;
                existing.TitleLabel.Text = $"{Truncate(t.Title, 18)}  {t.Progress}%";
                existing.StatusDot.Fill = new SolidColorBrush(col);
                existing.Badge.BorderBrush = new SolidColorBrush(Color.FromArgb(0x99, col.R, col.G, col.B));
            }
        }
    }

    public void PulseCacheHit()
    {
        _cacheFlash = 1.0;
        TriggerShockwave(Color.FromRgb(0xFA, 0xCC, 0x15), 390, 1.35);
    }

    public void TriggerShockwave(Color color, double maxRadius = 360, double durationSec = 1.4)
    {
        if (!Dispatcher.CheckAccess())
        {
            Dispatcher.BeginInvoke(() => TriggerShockwave(color, maxRadius, durationSec));
            return;
        }
        var ellipse = new Ellipse
        {
            Stroke = new SolidColorBrush(color),
            StrokeThickness = 2.4,
            Fill = Brushes.Transparent,
            IsHitTestVisible = false,
        };
        _bgCanvas.Children.Add(ellipse);
        _shockwaves.Add(new ShockwaveFx
        {
            Shape = ellipse,
            Color = color,
            MaxRadius = maxRadius,
            Duration = Math.Max(0.4, durationSec),
            Age = 0,
        });
    }

    public void SpawnSystemEffect(
        string category,
        string title,
        string detail,
        Color accentColor,
        double durationSec = 6.5,
        bool inbound = false)
    {
        if (!Dispatcher.CheckAccess())
        {
            Dispatcher.BeginInvoke(() => SpawnSystemEffect(category, title, detail, accentColor, durationSec, inbound));
            return;
        }

        // Keep frame budget silky smooth by retiring oldest node early if crowded
        var nonClosing = _nodes.Where(n => !n.ForceClosing).ToList();
        if (nonClosing.Count >= MaxActiveNodes)
        {
            var oldest = nonClosing[0];
            oldest.ForceClosing = true;
            oldest.Age = Math.Max(oldest.Age, oldest.Duration - 0.9);
        }

        // Choose radial orbital angle avoiding overlap with recent nodes and staying outside the core
        double[] preferredAngles =
        {
            -2.45, -0.68, 2.45, 0.68,
            -2.85, -0.28, 2.85, 0.28,
            -1.95, -1.18, 1.95, 1.18,
        };
        var baseAngle = preferredAngles[_sectorCursor % preferredAngles.Length];
        _sectorCursor++;
        var angle = baseAngle + (_rng.NextDouble() - 0.5) * 0.22;
        var normDist = 0.58 + _rng.NextDouble() * 0.28; // fraction from core edge to viewport edge
        var zDepth = (_rng.NextDouble() - 0.5) * 0.6;

        // Visual elements on _bgCanvas
        var beamGlow = new Path
        {
            Stroke = new SolidColorBrush(Color.FromArgb(0x3D, accentColor.R, accentColor.G, accentColor.B)),
            StrokeThickness = 5.0,
            StrokeStartLineCap = PenLineCap.Round,
            StrokeEndLineCap = PenLineCap.Round,
            IsHitTestVisible = false,
        };
        var beamCore = new Path
        {
            Stroke = new SolidColorBrush(Color.FromArgb(0xCC, accentColor.R, accentColor.G, accentColor.B)),
            StrokeThickness = 1.6,
            StrokeDashArray = new DoubleCollection { 5, 4 },
            StrokeStartLineCap = PenLineCap.Round,
            StrokeEndLineCap = PenLineCap.Round,
            IsHitTestVisible = false,
        };
        var pulseDot = new Ellipse
        {
            Width = 7,
            Height = 7,
            Fill = new SolidColorBrush(Colors.White),
            Stroke = new SolidColorBrush(accentColor),
            StrokeThickness = 1.5,
            IsHitTestVisible = false,
        };
        var anchorRing = new Ellipse
        {
            Width = 18,
            Height = 18,
            Stroke = new SolidColorBrush(accentColor),
            StrokeThickness = 1.5,
            Fill = new SolidColorBrush(Color.FromArgb(0x28, accentColor.R, accentColor.G, accentColor.B)),
            IsHitTestVisible = false,
        };

        // Floating HUD holographic card
        var catChip = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0x40, accentColor.R, accentColor.G, accentColor.B)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0xAA, accentColor.R, accentColor.G, accentColor.B)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(4),
            Padding = new Thickness(5, 1, 5, 1),
            Margin = new Thickness(0, 0, 6, 0),
            Child = new TextBlock
            {
                Text = category.ToUpperInvariant(),
                FontFamily = new FontFamily("Consolas"),
                FontSize = 9.5,
                FontWeight = FontWeights.Bold,
                Foreground = new SolidColorBrush(accentColor),
            },
        };
        var titleBlock = new TextBlock
        {
            Text = Truncate(title, 34),
            FontFamily = new FontFamily("Segoe UI"),
            FontSize = 11.5,
            FontWeight = FontWeights.SemiBold,
            Foreground = new SolidColorBrush(Color.FromRgb(0xE2, 0xF1, 0xFF)),
            VerticalAlignment = VerticalAlignment.Center,
        };
        var headerRow = new StackPanel { Orientation = Orientation.Horizontal };
        headerRow.Children.Add(catChip);
        headerRow.Children.Add(titleBlock);

        var detailBlock = new TextBlock
        {
            Text = string.IsNullOrWhiteSpace(detail) ? "—" : Truncate(detail.Replace('\n', ' '), 72),
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10,
            Foreground = new SolidColorBrush(Color.FromRgb(0x9C, 0xB3, 0xCC)),
            TextWrapping = TextWrapping.Wrap,
            MaxWidth = 228,
            Margin = new Thickness(0, 4, 0, 4),
        };

        var lifeTrack = new Grid { Height = 2 };
        var lifeBg = new Rectangle { Fill = new SolidColorBrush(Color.FromArgb(0x33, 0x94, 0xA3, 0xB8)), RadiusX = 1, RadiusY = 1 };
        var lifeFill = new Rectangle
        {
            Fill = new SolidColorBrush(accentColor),
            HorizontalAlignment = HorizontalAlignment.Left,
            RadiusX = 1,
            RadiusY = 1,
            Width = 210,
        };
        lifeTrack.Children.Add(lifeBg);
        lifeTrack.Children.Add(lifeFill);

        var cardStack = new StackPanel();
        cardStack.Children.Add(headerRow);
        if (!string.IsNullOrWhiteSpace(detail))
            cardStack.Children.Add(detailBlock);
        cardStack.Children.Add(lifeTrack);

        var scaleTf = new ScaleTransform(0.4, 0.4);
        var card = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0xBF, 0x07, 0x11, 0x20)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0x99, accentColor.R, accentColor.G, accentColor.B)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(9, 7, 9, 6),
            MaxWidth = 250,
            RenderTransformOrigin = new Point(0.5, 0.5),
            RenderTransform = scaleTf,
            Opacity = 0.0,
            IsHitTestVisible = false,
            Child = cardStack,
        };

        // Add in clean z-order behind foreground core
        _bgCanvas.Children.Add(beamGlow);
        _bgCanvas.Children.Add(beamCore);
        _bgCanvas.Children.Add(pulseDot);
        _bgCanvas.Children.Add(anchorRing);
        _bgCanvas.Children.Add(card);

        _nodes.Add(new ActivityNode
        {
            BeamGlow = beamGlow,
            BeamCore = beamCore,
            PulseDot = pulseDot,
            AnchorRing = anchorRing,
            Card = card,
            CardScale = scaleTf,
            LifeBar = lifeFill,
            Angle = angle,
            NormDist = normDist,
            ZDepth = zDepth,
            DriftSpeed = (_rng.NextDouble() - 0.5) * 0.14,
            BobPhase = _rng.NextDouble() * Math.PI * 2,
            Inbound = inbound,
            Duration = Math.Max(2.5, durationSec),
            Age = 0,
            Color = accentColor,
        });
    }

    public void TriggerDemoCascade()
    {
        var cyan = Color.FromRgb(0x22, 0xD3, 0xEE);
        var magenta = Color.FromRgb(0xF4, 0x72, 0xB6);
        var purple = Color.FromRgb(0xA7, 0x8B, 0xFA);
        var gold = Color.FromRgb(0xFA, 0xCC, 0x15);
        var green = Color.FromRgb(0x4A, 0xDE, 0x80);
        var blue = Color.FromRgb(0x38, 0xBD, 0xF8);

        TriggerShockwave(cyan, 410, 1.5);
        SpawnSystemEffect("WAKE", "WAKE MATRIX TRIGGERED", "Acoustic signature locked • beam active", magenta, 6.5, inbound: true);

        ScheduleAfter(0.32, () =>
            SpawnSystemEffect("STT", "WHISPER GPU STREAM", "Decoded speech utterance (340 ms)", cyan, 6.2, inbound: true));
        ScheduleAfter(0.68, () =>
        {
            TriggerShockwave(purple, 320, 1.3);
            SpawnSystemEffect("MIND", "CONSCIOUS SYNTHESIS", "Evaluating multi-tool execution graph", purple, 6.4);
        });
        ScheduleAfter(1.05, () =>
            SpawnSystemEffect("TOOL", "TOOL // TELEMETRY_SWEEP", "Scanning GPU pipeline & background workers", magenta, 6.0));
        ScheduleAfter(1.42, () =>
        {
            PulseCacheHit();
            SpawnSystemEffect("CACHE", "CONVERSATION CACHE HIT", "Reused stitched neural phrase in 10.3 ms", gold, 7.0);
        });
        ScheduleAfter(1.78, () =>
            SpawnSystemEffect("TTS", "DIRECTML VOICE OUTPUT", "jarvis-high.onnx • 0.15x RTF on RX 580", blue, 6.2));
        ScheduleAfter(2.15, () =>
        {
            TriggerShockwave(green, 360, 1.4);
            SpawnSystemEffect("TASK", "SUBCONSCIOUS WORKER", "Autonomous telemetry check complete (100%)", green, 6.5);
        });
    }

    private void ScheduleAfter(double delaySec, Action action)
    {
        _scheduled.Add((_time + delaySec, action));
    }

    // ------------------------------------------------------------------ 3D Core Mesh Construction

    private void Build3DScene()
    {
        var group = new Model3DGroup();

        // Lighting
        group.Children.Add(new AmbientLight(Color.FromRgb(0x16, 0x28, 0x42)));
        group.Children.Add(new DirectionalLight(Color.FromRgb(0xE0, 0xFB, 0xFF), new Vector3D(-0.4, -0.5, -1.0)));
        group.Children.Add(new DirectionalLight(Color.FromRgb(0x7C, 0x3A, 0xED), new Vector3D(0.9, 0.4, -0.35)));
        group.Children.Add(new DirectionalLight(Color.FromRgb(0x22, 0xD3, 0xEE), new Vector3D(-0.9, 0.7, 0.4)));

        // 1. Inner White-Hot Nucleus
        var nucleusMat = MakeDualMaterial(_nucleusBrush, Colors.White, specular: 0.9);
        var nucleusModel = new GeometryModel3D(BuildGeodesicSphere(0.40, 2), nucleusMat)
        {
            BackMaterial = nucleusMat,
            Transform = _innerPulseScale,
        };
        group.Children.Add(nucleusModel);

        // 2. Central Geodesic Plasma Core (faceted icosahedron sphere)
        var coreMats = new MaterialGroup();
        coreMats.Children.Add(new DiffuseMaterial(_coreDiffuseBrush));
        coreMats.Children.Add(new EmissiveMaterial(_coreEmissiveBrush));
        coreMats.Children.Add(new SpecularMaterial(new SolidColorBrush(Color.FromArgb(0xCC, 0xFF, 0xFF, 0xFF)), 70));

        var coreTransform = new Transform3DGroup();
        coreTransform.Children.Add(_corePulseScale);
        coreTransform.Children.Add(new RotateTransform3D(_coreRotY));

        var coreModel = new GeometryModel3D(BuildGeodesicSphere(0.80, 2), coreMats)
        {
            BackMaterial = coreMats,
            Transform = coreTransform,
        };
        group.Children.Add(coreModel);

        // 3. Counter-rotating Lattice Strut Cage
        var cageMat = MakeDualMaterial(_cageBrush, Color.FromRgb(0x67, 0xE8, 0xF9), specular: 0.7);
        var cageModel = new GeometryModel3D(BuildWireCageMesh(1.10, 0.022), cageMat)
        {
            BackMaterial = cageMat,
            Transform = new RotateTransform3D(_cageRot),
        };
        group.Children.Add(cageModel);

        // 4. Three Concentric 3D Gyroscopic Gimbal Rings
        group.Children.Add(MakeGimbalRing(1.42, 0.042, 64, 14, _ring1Brush, _ring1Rot, 26, -18, 0));
        group.Children.Add(MakeGimbalRing(1.76, 0.036, 72, 12, _ring2Brush, _ring2Rot, -48, 32, 15));
        group.Children.Add(MakeGimbalRing(2.10, 0.030, 80, 12, _ring3Brush, _ring3Rot, 62, 54, -22));

        // 5. Orbital 3D Energy Shards (octahedron diamonds)
        var shardMesh = BuildOctahedronMesh(0.11);
        var shardColorList = new[]
        {
            Color.FromRgb(0x22, 0xD3, 0xEE),
            Color.FromRgb(0xF4, 0x72, 0xB6),
            Color.FromRgb(0x38, 0xBD, 0xF8),
            Color.FromRgb(0xA7, 0x8B, 0xFA),
            Color.FromRgb(0x4A, 0xDE, 0x80),
            Color.FromRgb(0xFA, 0xCC, 0x15),
        };
        for (int i = 0; i < 6; i++)
        {
            var sb = new SolidColorBrush(shardColorList[i]);
            var smat = MakeDualMaterial(sb, shardColorList[i], specular: 0.8);
            var rot = new AxisAngleRotation3D(new Vector3D(Math.Sin(i * 1.1), 1.0, Math.Cos(i * 0.9)), i * 60);
            var tg = new Transform3DGroup();
            double radius = 1.55 + (i % 3) * 0.34;
            tg.Children.Add(new TranslateTransform3D(radius, 0, 0));
            tg.Children.Add(new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(1, 0, 0.4), (i - 2.5) * 28)));
            tg.Children.Add(new RotateTransform3D(rot));
            group.Children.Add(new GeometryModel3D(shardMesh, smat) { BackMaterial = smat, Transform = tg });
            _shardRots.Add((rot, 24.0 + i * 7.5));
        }

        _viewport.Children.Add(new ModelVisual3D { Content = group });
    }

    private static MaterialGroup MakeDualMaterial(Brush brush, Color emissiveTint, double specular = 0.6)
    {
        var g = new MaterialGroup();
        g.Children.Add(new DiffuseMaterial(brush));
        g.Children.Add(new EmissiveMaterial(new SolidColorBrush(Color.FromArgb(0x99, emissiveTint.R, emissiveTint.G, emissiveTint.B))));
        g.Children.Add(new SpecularMaterial(new SolidColorBrush(Color.FromArgb((byte)(specular * 255), 255, 255, 255)), 55));
        return g;
    }

    private static GeometryModel3D MakeGimbalRing(
        double majorR,
        double minorR,
        int majorSegs,
        int minorSegs,
        Brush brush,
        AxisAngleRotation3D animRot,
        double tiltX,
        double tiltY,
        double tiltZ)
    {
        var mesh = BuildSegmentedTorus(majorR, minorR, majorSegs, minorSegs);
        var mat = new MaterialGroup();
        mat.Children.Add(new DiffuseMaterial(brush));
        mat.Children.Add(new EmissiveMaterial(brush));
        mat.Children.Add(new SpecularMaterial(Brushes.White, 64));

        var tg = new Transform3DGroup();
        tg.Children.Add(new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(1, 0, 0), tiltX)));
        tg.Children.Add(new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(0, 1, 0), tiltY)));
        tg.Children.Add(new RotateTransform3D(new AxisAngleRotation3D(new Vector3D(0, 0, 1), tiltZ)));
        tg.Children.Add(new RotateTransform3D(animRot));

        return new GeometryModel3D(mesh, mat) { BackMaterial = mat, Transform = tg };
    }

    private static MeshGeometry3D BuildSegmentedTorus(double R, double r, int segU, int segV)
    {
        var mesh = new MeshGeometry3D();
        // Split into 4 tech-arc gaps for futuristic arc-reactor gimbal silhouette
        for (int i = 0; i < segU; i++)
        {
            if (i % (segU / 4) == 0) continue; // subtle technological seam gaps

            double u0 = 2.0 * Math.PI * i / segU;
            double u1 = 2.0 * Math.PI * (i + 1) / segU;
            int baseIdx = mesh.Positions.Count;

            for (int iu = 0; iu <= 1; iu++)
            {
                double u = iu == 0 ? u0 : u1;
                double cu = Math.Cos(u), su = Math.Sin(u);
                for (int j = 0; j < segV; j++)
                {
                    double v = 2.0 * Math.PI * j / segV;
                    double cv = Math.Cos(v), sv = Math.Sin(v);
                    double rad = R + r * cv;
                    mesh.Positions.Add(new Point3D(rad * cu, r * sv, rad * su));
                }
            }

            for (int j = 0; j < segV; j++)
            {
                int jNext = (j + 1) % segV;
                int a = baseIdx + j;
                int b = baseIdx + jNext;
                int c = baseIdx + segV + j;
                int d = baseIdx + segV + jNext;
                mesh.TriangleIndices.Add(a);
                mesh.TriangleIndices.Add(c);
                mesh.TriangleIndices.Add(b);
                mesh.TriangleIndices.Add(b);
                mesh.TriangleIndices.Add(c);
                mesh.TriangleIndices.Add(d);
            }
        }
        return mesh;
    }

    private static MeshGeometry3D BuildGeodesicSphere(double radius, int subdivisions)
    {
        double t = (1.0 + Math.Sqrt(5.0)) / 2.0;
        var pts = new List<Vector3D>
        {
            Normalize(new Vector3D(-1,  t,  0)), Normalize(new Vector3D( 1,  t,  0)),
            Normalize(new Vector3D(-1, -t,  0)), Normalize(new Vector3D( 1, -t,  0)),
            Normalize(new Vector3D( 0, -1,  t)), Normalize(new Vector3D( 0,  1,  t)),
            Normalize(new Vector3D( 0, -1, -t)), Normalize(new Vector3D( 0,  1, -t)),
            Normalize(new Vector3D( t,  0, -1)), Normalize(new Vector3D( t,  0,  1)),
            Normalize(new Vector3D(-t,  0, -1)), Normalize(new Vector3D(-t,  0,  1)),
        };

        var tris = new List<(int A, int B, int C)>
        {
            (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
            (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
            (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
            (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
        };

        for (int s = 0; s < subdivisions; s++)
        {
            var next = new List<(int, int, int)>();
            var cache = new Dictionary<(int, int), int>();
            int Mid(int i1, int i2)
            {
                var key = i1 < i2 ? (i1, i2) : (i2, i1);
                if (cache.TryGetValue(key, out var idx)) return idx;
                var m = Normalize((pts[i1] + pts[i2]) * 0.5);
                pts.Add(m);
                idx = pts.Count - 1;
                cache[key] = idx;
                return idx;
            }
            foreach (var (a, b, c) in tris)
            {
                int ab = Mid(a, b), bc = Mid(b, c), ca = Mid(c, a);
                next.Add((a, ab, ca));
                next.Add((b, bc, ab));
                next.Add((c, ca, bc));
                next.Add((ab, bc, ca));
            }
            tris = next;
        }

        // Duplicate vertices per triangle for crisp faceted holographic crystal look
        var mesh = new MeshGeometry3D();
        foreach (var (a, b, c) in tris)
        {
            int idx = mesh.Positions.Count;
            var pa = pts[a] * radius;
            var pb = pts[b] * radius;
            var pc = pts[c] * radius;
            mesh.Positions.Add(new Point3D(pa.X, pa.Y, pa.Z));
            mesh.Positions.Add(new Point3D(pb.X, pb.Y, pb.Z));
            mesh.Positions.Add(new Point3D(pc.X, pc.Y, pc.Z));
            mesh.TriangleIndices.Add(idx);
            mesh.TriangleIndices.Add(idx + 1);
            mesh.TriangleIndices.Add(idx + 2);
        }
        return mesh;
    }

    private static MeshGeometry3D BuildWireCageMesh(double radius, double strutThickness)
    {
        double t = (1.0 + Math.Sqrt(5.0)) / 2.0;
        var pts = new[]
        {
            Normalize(new Vector3D(-1,  t,  0)) * radius, Normalize(new Vector3D( 1,  t,  0)) * radius,
            Normalize(new Vector3D(-1, -t,  0)) * radius, Normalize(new Vector3D( 1, -t,  0)) * radius,
            Normalize(new Vector3D( 0, -1,  t)) * radius, Normalize(new Vector3D( 0,  1,  t)) * radius,
            Normalize(new Vector3D( 0, -1, -t)) * radius, Normalize(new Vector3D( 0,  1, -t)) * radius,
            Normalize(new Vector3D( t,  0, -1)) * radius, Normalize(new Vector3D( t,  0,  1)) * radius,
            Normalize(new Vector3D(-t,  0, -1)) * radius, Normalize(new Vector3D(-t,  0,  1)) * radius,
        };
        int[,] edges =
        {
            {0,11},{11,5},{5,0},{0,5},{5,1},{1,0},{0,1},{1,7},{7,0},{0,7},{7,10},{10,0},{0,10},{10,11},
            {1,5},{5,9},{9,1},{9,4},{4,5},{4,11},{2,4},{11,2},{10,2},{6,10},{7,6},{8,7},{1,8},{9,8},
            {3,9},{4,3},{3,2},{2,6},{6,3},{3,8},{8,6},
        };
        var mesh = new MeshGeometry3D();
        var seen = new HashSet<(int, int)>();
        for (int e = 0; e < edges.GetLength(0); e++)
        {
            int u = Math.Min(edges[e, 0], edges[e, 1]);
            int v = Math.Max(edges[e, 0], edges[e, 1]);
            if (!seen.Add((u, v))) continue;
            AddStrut(mesh, pts[u], pts[v], strutThickness);
        }
        return mesh;
    }

    private static void AddStrut(MeshGeometry3D mesh, Vector3D a, Vector3D b, double r)
    {
        var dir = b - a;
        if (dir.Length < 1e-5) return;
        dir.Normalize();
        var up = Math.Abs(dir.Y) < 0.9 ? new Vector3D(0, 1, 0) : new Vector3D(1, 0, 0);
        var side1 = Vector3D.CrossProduct(dir, up);
        side1.Normalize();
        var side2 = Vector3D.CrossProduct(dir, side1);
        side2.Normalize();

        Vector3D[] offsets =
        {
            side1 * r, side2 * r, -side1 * r, -side2 * r,
        };
        int baseIdx = mesh.Positions.Count;
        for (int k = 0; k < 4; k++)
        {
            var p = a + offsets[k];
            mesh.Positions.Add(new Point3D(p.X, p.Y, p.Z));
        }
        for (int k = 0; k < 4; k++)
        {
            var p = b + offsets[k];
            mesh.Positions.Add(new Point3D(p.X, p.Y, p.Z));
        }
        for (int k = 0; k < 4; k++)
        {
            int kn = (k + 1) % 4;
            mesh.TriangleIndices.Add(baseIdx + k);
            mesh.TriangleIndices.Add(baseIdx + 4 + k);
            mesh.TriangleIndices.Add(baseIdx + kn);
            mesh.TriangleIndices.Add(baseIdx + kn);
            mesh.TriangleIndices.Add(baseIdx + 4 + k);
            mesh.TriangleIndices.Add(baseIdx + 4 + kn);
        }
    }

    private static MeshGeometry3D BuildOctahedronMesh(double s)
    {
        var mesh = new MeshGeometry3D();
        Point3D[] p =
        {
            new(0, s * 1.4, 0), new(0, -s * 1.4, 0),
            new(s, 0, 0), new(-s, 0, 0), new(0, 0, s), new(0, 0, -s),
        };
        int[,] faces =
        {
            {0,4,2},{0,2,5},{0,5,3},{0,3,4},
            {1,2,4},{1,5,2},{1,3,5},{1,4,3},
        };
        for (int f = 0; f < 8; f++)
        {
            int idx = mesh.Positions.Count;
            mesh.Positions.Add(p[faces[f, 0]]);
            mesh.Positions.Add(p[faces[f, 1]]);
            mesh.Positions.Add(p[faces[f, 2]]);
            mesh.TriangleIndices.Add(idx);
            mesh.TriangleIndices.Add(idx + 1);
            mesh.TriangleIndices.Add(idx + 2);
        }
        return mesh;
    }

    private static Vector3D Normalize(Vector3D v)
    {
        v.Normalize();
        return v;
    }

    // ------------------------------------------------------------------ Pooled 2.5D Layers

    private void InitPools()
    {
        // 3D Holodeck perspective grid lines (16 vertical perspective rays + 12 horizontal depth rings)
        for (int i = 0; i < 28; i++)
        {
            var line = new Line { StrokeThickness = 1.0, IsHitTestVisible = false };
            _bgCanvas.Children.Add(line);
            _gridLines.Add(line);
        }

        // 3D Ambient Starfield / Neural Dust particles
        for (int i = 0; i < ParticleCount; i++)
        {
            var sz = 1.5 + _rng.NextDouble() * 3.0;
            var el = new Ellipse
            {
                Width = sz,
                Height = sz,
                Fill = new SolidColorBrush(Color.FromRgb(0x67, 0xE8, 0xF9)),
                IsHitTestVisible = false,
            };
            _bgCanvas.Children.Add(el);
            _particles.Add(new Particle3D
            {
                Shape = el,
                X = (_rng.NextDouble() - 0.5) * 2.2,
                Y = (_rng.NextDouble() - 0.5) * 1.6,
                Z = 0.25 + _rng.NextDouble() * 1.6,
                SpeedZ = 0.12 + _rng.NextDouble() * 0.28,
                OrbitSpeed = (_rng.NextDouble() - 0.5) * 0.18,
                BaseSize = sz,
            });
        }

        // 64-bar Radial Audio Spectrum Ring
        for (int i = 0; i < SpectrumBarCount; i++)
        {
            var rect = new Rectangle
            {
                Width = 3.0,
                Height = 8.0,
                RadiusX = 1.5,
                RadiusY = 1.5,
                RenderTransformOrigin = new Point(0.5, 1.0),
                IsHitTestVisible = false,
            };
            _fgCanvas.Children.Add(rect);
            _spectrumBars.Add(rect);
        }

        // Precision Degree Compass Ticks
        for (int i = 0; i < CompassTickCount; i++)
        {
            var line = new Line
            {
                StrokeThickness = i % 4 == 0 ? 1.8 : 1.0,
                IsHitTestVisible = false,
            };
            _fgCanvas.Children.Add(line);
            _compassTicks.Add(line);
        }
    }

    private void HookMouseInteraction()
    {
        MouseMove += (_, e) =>
        {
            if (ActualWidth <= 0 || ActualHeight <= 0) return;
            var pt = e.GetPosition(this);
            _mouseNormX = (pt.X / ActualWidth - 0.5) * 2.0;
            _mouseNormY = (pt.Y / ActualHeight - 0.5) * 2.0;
            if (_dragging)
            {
                var dx = pt.X - _lastDragPos.X;
                var dy = pt.Y - _lastDragPos.Y;
                _targetYaw += dx * 0.008;
                _targetPitch = Math.Clamp(_targetPitch + dy * 0.008, -0.75, 0.75);
                _lastDragPos = pt;
            }
        };
        MouseLeftButtonDown += (_, e) =>
        {
            _dragging = true;
            _lastDragPos = e.GetPosition(this);
            CaptureMouse();
        };
        MouseLeftButtonUp += (_, _) =>
        {
            _dragging = false;
            ReleaseMouseCapture();
        };
    }

    // ------------------------------------------------------------------ 60 FPS Frame Loop

    private void OnFrameTick(object? sender, EventArgs e)
    {
        var now = DateTime.UtcNow;
        var dt = Math.Clamp((now - _lastTick).TotalSeconds, 0.001, 0.05);
        _lastTick = now;
        _time += dt;

        // Trigger scheduled demo cascade items
        for (int i = _scheduled.Count - 1; i >= 0; i--)
        {
            if (_time >= _scheduled[i].TriggerAt)
            {
                var act = _scheduled[i].Run;
                _scheduled.RemoveAt(i);
                try { act(); } catch { }
            }
        }

        double w = ActualWidth;
        double h = ActualHeight;
        if (w < 40 || h < 40) return;

        double cx = w * 0.5;
        double cy = h * 0.48;

        // Smooth camera parallax + drag inertia
        if (!_dragging)
        {
            _targetYaw *= 0.96;
            _targetPitch += (0.08 - _targetPitch) * 0.04;
        }
        double goalYaw = _targetYaw + _mouseNormX * 0.22 + Math.Sin(_time * 0.35) * 0.05;
        double goalPitch = _targetPitch - _mouseNormY * 0.14 + Math.Cos(_time * 0.28) * 0.03;
        _camYaw += (goalYaw - _camYaw) * Math.Min(1.0, dt * 8.0);
        _camPitch += (goalPitch - _camPitch) * Math.Min(1.0, dt * 8.0);

        double camDist = 6.35 - _aiLevel * 0.25;
        double camX = camDist * Math.Sin(_camYaw) * Math.Cos(_camPitch);
        double camY = camDist * Math.Sin(_camPitch);
        double camZ = camDist * Math.Cos(_camYaw) * Math.Cos(_camPitch);
        _camera.Position = new Point3D(camX, camY, camZ);
        _camera.LookDirection = new Vector3D(-camX, -camY, -camZ);

        // Smooth accent color transition (+ golden cache flash blend)
        _cacheFlash = Math.Max(0.0, _cacheFlash - dt * 1.4);
        var desiredAccent = LerpColor(_targetAccent, Color.FromRgb(0xFA, 0xCC, 0x15), _cacheFlash * 0.75);
        _currentAccent = LerpColor(_currentAccent, desiredAccent, Math.Min(1.0, dt * 6.0));
        Update3DCore(dt);

        // Render background 3D stage & foreground reticle
        UpdateHolodeckGrid(w, h, cx, cy);
        UpdateParticles(dt, w, h, cx, cy);
        UpdateShockwaves(dt, cx, cy);
        UpdateTaskSatellites(dt, w, h, cx, cy);
        UpdateActivityNodes(dt, w, h, cx, cy);
        UpdateReticle(dt, w, h, cx, cy);
    }

    private void Update3DCore(double dt)
    {
        double speedMult = _voiceState switch
        {
            "thinking" => 2.7,
            "speaking" => 1.7 + _aiLevel * 2.2,
            "listening" => 1.35 + _userLevel * 1.5,
            "off" => 0.25,
            _ => 1.0,
        };

        _coreRotY.Angle = (_coreRotY.Angle + dt * 28.0 * speedMult) % 360.0;
        _cageRot.Angle = (_cageRot.Angle - dt * 21.0 * speedMult) % 360.0;
        _ring1Rot.Angle = (_ring1Rot.Angle + dt * 36.0 * speedMult) % 360.0;
        _ring2Rot.Angle = (_ring2Rot.Angle - dt * 27.0 * speedMult) % 360.0;
        _ring3Rot.Angle = (_ring3Rot.Angle + dt * 19.0 * speedMult) % 360.0;
        foreach (var (rot, sp) in _shardRots)
            rot.Angle = (rot.Angle + dt * sp * speedMult) % 360.0;

        double energy = Math.Max(_aiLevel, _userLevel * 0.85);
        double pulse = 1.0 + 0.30 * _aiLevel + 0.22 * _userLevel + 0.05 * Math.Sin(_time * 2.8);
        _corePulseScale.ScaleX = pulse;
        _corePulseScale.ScaleY = pulse;
        _corePulseScale.ScaleZ = pulse;

        double innerPulse = 1.0 + 0.36 * energy + 0.08 * Math.Sin(_time * 4.2 + 0.8);
        _innerPulseScale.ScaleX = innerPulse;
        _innerPulseScale.ScaleY = innerPulse;
        _innerPulseScale.ScaleZ = innerPulse;

        _coreDiffuseBrush.Color = Color.FromArgb(0xCC, (byte)(_currentAccent.R * 0.72), (byte)(_currentAccent.G * 0.72), (byte)(_currentAccent.B * 0.72));
        _coreEmissiveBrush.Color = Color.FromArgb(0xCC, _currentAccent.R, _currentAccent.G, _currentAccent.B);
        _cageBrush.Color = Color.FromArgb(0xB8, (byte)Math.Min(255, _currentAccent.R + 40), (byte)Math.Min(255, _currentAccent.G + 40), (byte)Math.Min(255, _currentAccent.B + 40));
        _ring1Brush.Color = _currentAccent;
        _ring2Brush.Color = LerpColor(_currentAccent, Color.FromRgb(0x38, 0xBD, 0xF8), 0.45);
        _ring3Brush.Color = LerpColor(_currentAccent, Color.FromRgb(0xA7, 0x8B, 0xFA), 0.55);
        _nucleusBrush.Color = LerpColor(Colors.White, _currentAccent, 0.25);
    }

    private void UpdateHolodeckGrid(double w, double h, double cx, double cy)
    {
        // Floor & Ceiling 3D perspective convergence grid
        double parX = _mouseNormX * 26.0;
        double parY = _mouseNormY * 14.0;
        double vpX = cx - parX * 0.65;
        double vpY = cy - parY * 0.65;

        int idx = 0;
        var gridCol = Color.FromArgb(0x28, _currentAccent.R, _currentAccent.G, _currentAccent.B);
        var gridBrush = new SolidColorBrush(gridCol);

        for (int i = 0; i < 8; i++)
        {
            double frac = (i / 7.0) * 2.0 - 1.0; // -1 .. +1
            double spread = frac * w * 0.72;
            // Floor ray
            var fl = _gridLines[idx++];
            fl.X1 = vpX + frac * 65;
            fl.Y1 = vpY + 55;
            fl.X2 = cx + spread;
            fl.Y2 = h;
            fl.Stroke = gridBrush;
            // Ceiling ray
            var cl = _gridLines[idx++];
            cl.X1 = vpX + frac * 65;
            cl.Y1 = vpY - 55;
            cl.X2 = cx + spread;
            cl.Y2 = 0;
            cl.Stroke = gridBrush;
        }

        // Moving horizontal depth scan lines (6 floor + 6 ceiling)
        for (int d = 0; d < 6; d++)
        {
            double zPhase = (d / 6.0 + _time * 0.16) % 1.0;
            double depth = Math.Pow(zPhase, 1.85);
            double alpha = (byte)(0x0D + 0x2A * depth);
            var hBrush = new SolidColorBrush(Color.FromArgb((byte)alpha, _currentAccent.R, _currentAccent.G, _currentAccent.B));

            double fy = vpY + 55 + depth * (h - (vpY + 55));
            var fl = _gridLines[idx++];
            fl.X1 = 0; fl.X2 = w; fl.Y1 = fy; fl.Y2 = fy;
            fl.Stroke = hBrush;

            double cyLine = (vpY - 55) - depth * (vpY - 55);
            var cl = _gridLines[idx++];
            cl.X1 = 0; cl.X2 = w; cl.Y1 = cyLine; cl.Y2 = cyLine;
            cl.Stroke = hBrush;
        }
    }

    private void UpdateParticles(double dt, double w, double h, double cx, double cy)
    {
        double boost = 1.0 + _aiLevel * 1.4 + _userLevel * 0.8;
        foreach (var p in _particles)
        {
            p.Z -= dt * p.SpeedZ * boost * 0.35;
            if (p.Z <= 0.18)
            {
                p.Z = 1.85;
                p.X = (_rng.NextDouble() - 0.5) * 2.2;
                p.Y = (_rng.NextDouble() - 0.5) * 1.6;
            }
            // Subtle swirl
            double ang = dt * p.OrbitSpeed;
            double ca = Math.Cos(ang), sa = Math.Sin(ang);
            double nx = p.X * ca - p.Y * sa;
            double ny = p.X * sa + p.Y * ca;
            p.X = nx; p.Y = ny;

            double persp = 1.0 / p.Z;
            double sx = cx + (p.X * w * 0.42 - _mouseNormX * 28.0) * persp;
            double sy = cy + (p.Y * h * 0.42 - _mouseNormY * 20.0) * persp;
            double size = p.BaseSize * Math.Clamp(persp * 0.9, 0.5, 2.4);
            double alpha = Math.Clamp((1.85 - p.Z) * 0.55, 0.08, 0.85);

            p.Shape.Width = size;
            p.Shape.Height = size;
            p.Shape.Opacity = alpha;
            Canvas.SetLeft(p.Shape, sx - size * 0.5);
            Canvas.SetTop(p.Shape, sy - size * 0.5);
        }
    }

    private void UpdateShockwaves(double dt, double cx, double cy)
    {
        for (int i = _shockwaves.Count - 1; i >= 0; i--)
        {
            var sw = _shockwaves[i];
            sw.Age += dt;
            double t = sw.Age / sw.Duration;
            if (t >= 1.0)
            {
                _bgCanvas.Children.Remove(sw.Shape);
                _shockwaves.RemoveAt(i);
                continue;
            }
            double eased = 1.0 - Math.Pow(1.0 - t, 3.0);
            double r = 68.0 + eased * sw.MaxRadius;
            // Subtle 3D tilt perspective on shockwave rings
            sw.Shape.Width = r * 2.0;
            sw.Shape.Height = r * 1.68;
            sw.Shape.Opacity = (1.0 - t) * 0.88;
            sw.Shape.StrokeThickness = 2.8 * (1.0 - t * 0.7);
            Canvas.SetLeft(sw.Shape, cx - r);
            Canvas.SetTop(sw.Shape, cy - r * 0.84);
        }
    }

    private TaskSatellite CreateTaskSatellite(TaskVm t, Color col)
    {
        var ring = new Ellipse
        {
            Width = 14,
            Height = 14,
            Stroke = new SolidColorBrush(col),
            StrokeThickness = 1.6,
            Fill = new SolidColorBrush(Color.FromArgb(0x33, col.R, col.G, col.B)),
            IsHitTestVisible = false,
        };
        var dot = new Ellipse
        {
            Width = 7,
            Height = 7,
            Fill = new SolidColorBrush(col),
            Margin = new Thickness(0, 0, 6, 0),
            VerticalAlignment = VerticalAlignment.Center,
        };
        var label = new TextBlock
        {
            Text = $"{Truncate(t.Title, 18)}  {t.Progress}%",
            FontFamily = new FontFamily("Consolas"),
            FontSize = 10,
            Foreground = new SolidColorBrush(Color.FromRgb(0xD8, 0xEA, 0xFA)),
        };
        var row = new StackPanel { Orientation = Orientation.Horizontal };
        row.Children.Add(dot);
        row.Children.Add(label);

        var badge = new Border
        {
            Background = new SolidColorBrush(Color.FromArgb(0xB0, 0x07, 0x11, 0x20)),
            BorderBrush = new SolidColorBrush(Color.FromArgb(0x99, col.R, col.G, col.B)),
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(12),
            Padding = new Thickness(8, 3, 9, 3),
            Child = row,
            IsHitTestVisible = false,
        };
        _bgCanvas.Children.Add(ring);
        _bgCanvas.Children.Add(badge);
        return new TaskSatellite
        {
            Id = t.Id,
            Title = t.Title,
            Progress = t.Progress,
            Status = t.Status,
            Color = col,
            OrbitAngle = _taskSatellites.Count * 1.15,
            OrbitRing = ring,
            Badge = badge,
            StatusDot = dot,
            TitleLabel = label,
        };
    }

    private void UpdateTaskSatellites(double dt, double w, double h, double cx, double cy)
    {
        if (_taskSatellites.Count == 0) return;
        double rx = Math.Min(w * 0.30, 310);
        double ry = Math.Min(h * 0.30, 235);
        for (int i = 0; i < _taskSatellites.Count; i++)
        {
            var s = _taskSatellites[i];
            s.OrbitAngle += dt * (s.Status == "running" ? 0.38 : 0.14);
            double x = cx + Math.Cos(s.OrbitAngle) * rx - _mouseNormX * 14.0;
            double y = cy + Math.Sin(s.OrbitAngle) * ry - _mouseNormY * 10.0;
            double pulse = s.Status == "running" ? 1.0 + 0.2 * Math.Sin(_time * 5.0 + i) : 1.0;

            s.OrbitRing.Width = 14 * pulse;
            s.OrbitRing.Height = 14 * pulse;
            Canvas.SetLeft(s.OrbitRing, x - s.OrbitRing.Width * 0.5);
            Canvas.SetTop(s.OrbitRing, y - s.OrbitRing.Height * 0.5);
            Canvas.SetLeft(s.Badge, x + 12);
            Canvas.SetTop(s.Badge, y - 11);
        }
    }

    private void UpdateActivityNodes(double dt, double w, double h, double cx, double cy)
    {
        double rxMax = Math.Max(215, Math.Min(w * 0.29, 360));
        double ryMax = Math.Max(175, h * 0.34);

        for (int i = _nodes.Count - 1; i >= 0; i--)
        {
            var n = _nodes[i];
            n.Age += dt;
            if (n.Age >= n.Duration)
            {
                _bgCanvas.Children.Remove(n.BeamGlow);
                _bgCanvas.Children.Remove(n.BeamCore);
                _bgCanvas.Children.Remove(n.PulseDot);
                _bgCanvas.Children.Remove(n.AnchorRing);
                _bgCanvas.Children.Remove(n.Card);
                _nodes.RemoveAt(i);
                continue;
            }

            // Smooth appear (0 -> 0.45s) & dissolve (last 1.0s)
            double appear = Math.Clamp(n.Age / 0.45, 0.0, 1.0);
            double appearEase = 1.0 - Math.Pow(1.0 - appear, 3.0);
            double remaining = n.Duration - n.Age;
            double dissolve = remaining < 1.0 ? Math.Clamp(remaining / 1.0, 0.0, 1.0) : 1.0;
            double alpha = appearEase * dissolve;

            double scale = 0.55 + 0.45 * appearEase;
            if (remaining < 1.0) scale *= (0.90 + 0.10 * dissolve);
            n.CardScale.ScaleX = scale;
            n.CardScale.ScaleY = scale;
            n.Card.Opacity = alpha;
            n.BeamGlow.Opacity = alpha * 0.85;
            n.BeamCore.Opacity = alpha;
            n.BeamCore.StrokeDashOffset = -_time * (n.Inbound ? -28.0 : 28.0);
            n.AnchorRing.Opacity = alpha;

            // Gentle orbital hover + 3D parallax
            n.Angle += dt * n.DriftSpeed * 0.16;
            double outwardDrift = remaining < 1.0 ? (1.0 - dissolve) * 26.0 : 0.0;
            double rx = Math.Max(185.0, rxMax * n.NormDist) + outwardDrift;
            double ry = Math.Max(155.0, ryMax * n.NormDist) + outwardDrift * 0.7;

            double nx = cx + Math.Cos(n.Angle) * rx - _mouseNormX * (22.0 + n.ZDepth * 14.0);
            double ny = cy + Math.Sin(n.Angle) * ry + Math.Sin(_time * 1.9 + n.BobPhase) * 6.0 - _mouseNormY * (14.0 + n.ZDepth * 10.0);

            // Clamp node within visible window padding
            nx = Math.Clamp(nx, 145, Math.Max(155, w - 145));
            ny = Math.Clamp(ny, 85, Math.Max(95, h - 85));

            // Origin point on core rim toward node
            double dx = nx - cx, dy = ny - cy;
            double dist = Math.Max(1.0, Math.Sqrt(dx * dx + dy * dy));
            double ux = dx / dist, uy = dy / dist;
            double sx = cx + ux * 112.0;
            double sy = cy + uy * 112.0;

            // Curved synaptic Bezier beam
            double ctrlX = (sx + nx) * 0.5 - uy * 38.0;
            double ctrlY = (sy + ny) * 0.5 + ux * 38.0;
            var geom = new PathGeometry();
            var fig = new PathFigure { StartPoint = new Point(sx, sy) };
            fig.Segments.Add(new QuadraticBezierSegment(new Point(ctrlX, ctrlY), new Point(nx, ny), true));
            geom.Figures.Add(fig);
            n.BeamGlow.Data = geom;
            n.BeamCore.Data = geom;

            // Photon traveling along quadratic Bezier
            double tPulse = (_time * 1.15 + n.BobPhase) % 1.0;
            if (n.Inbound) tPulse = 1.0 - tPulse;
            double omt = 1.0 - tPulse;
            double px = omt * omt * sx + 2 * omt * tPulse * ctrlX + tPulse * tPulse * nx;
            double py = omt * omt * sy + 2 * omt * tPulse * ctrlY + tPulse * tPulse * ny;
            n.PulseDot.Opacity = alpha;
            Canvas.SetLeft(n.PulseDot, px - 3.5);
            Canvas.SetTop(n.PulseDot, py - 3.5);

            // Anchor ring & card placement (offset card away from center so it never overlaps beams)
            Canvas.SetLeft(n.AnchorRing, nx - 9);
            Canvas.SetTop(n.AnchorRing, ny - 9);

            n.Card.Measure(new Size(260, 140));
            double cardW = n.Card.DesiredSize.Width > 0 ? n.Card.DesiredSize.Width : 210;
            double cardH = n.Card.DesiredSize.Height > 0 ? n.Card.DesiredSize.Height : 54;
            double cardX = nx >= cx ? nx + 14 : nx - cardW - 14;
            double cardY = ny - cardH * 0.5;
            cardX = Math.Clamp(cardX, 14, Math.Max(20, w - cardW - 14));
            cardY = Math.Clamp(cardY, 68, Math.Max(74, h - cardH - 68));
            Canvas.SetLeft(n.Card, cardX);
            Canvas.SetTop(n.Card, cardY);

            double lifeFrac = Math.Clamp(1.0 - (n.Age / n.Duration), 0.0, 1.0);
            n.LifeBar.Width = Math.Max(0, (cardW - 22) * lifeFrac);
        }
    }

    private void UpdateReticle(double dt, double w, double h, double cx, double cy)
    {
        double baseR = Math.Min(w, h) * 0.22;
        baseR = Math.Clamp(baseR, 148, 210);

        var accent80 = new SolidColorBrush(Color.FromArgb(0x66, _currentAccent.R, _currentAccent.G, _currentAccent.B));
        var accent40 = new SolidColorBrush(Color.FromArgb(0x38, _currentAccent.R, _currentAccent.G, _currentAccent.B));

        // Halos
        double rOuter = baseR + 28;
        _haloOuter.Width = rOuter * 2;
        _haloOuter.Height = rOuter * 2;
        _haloOuter.Stroke = accent40;
        Canvas.SetLeft(_haloOuter, cx - rOuter);
        Canvas.SetTop(_haloOuter, cy - rOuter);

        double rMid = baseR + 6;
        _haloMid.Width = rMid * 2;
        _haloMid.Height = rMid * 2;
        _haloMid.Stroke = accent80;
        Canvas.SetLeft(_haloMid, cx - rMid);
        Canvas.SetTop(_haloMid, cy - rMid);

        // Rotating HUD Segmented Arcs
        _arcOuter.Stroke = new SolidColorBrush(_currentAccent);
        _arcOuter.Data = BuildArcRingGeometry(cx, cy, baseR + 18, _time * 24.0, 3, 72.0);
        _arcInner.Stroke = new SolidColorBrush(LerpColor(_currentAccent, Colors.White, 0.35));
        _arcInner.Data = BuildArcRingGeometry(cx, cy, baseR - 14, -_time * 36.0, 4, 48.0);

        // 64-Bar Radial Audio Spectrum Ring
        var aiBrush = new SolidColorBrush(_currentAccent);
        var userBrush = new SolidColorBrush(Color.FromRgb(0x4A, 0xDE, 0x80));
        double energy = Math.Max(_aiLevel, _userLevel);

        for (int i = 0; i < SpectrumBarCount; i++)
        {
            double ang = (2.0 * Math.PI * i / SpectrumBarCount) + _time * 0.18;
            double harmonic1 = Math.Abs(Math.Sin(i * 0.45 + _time * 6.2) * Math.Cos(i * 0.22 - _time * 3.1));
            double harmonic2 = Math.Abs(Math.Sin(i * 0.72 - _time * 7.8));
            double mag = _aiLevel * harmonic1 * 38.0 + _userLevel * harmonic2 * 30.0 + _ambientLevel * 8.0 + 3.0 * Math.Abs(Math.Sin(_time * 2.0 + i * 0.3));
            double barH = Math.Clamp(5.0 + mag, 5.0, 44.0);

            var bar = _spectrumBars[i];
            bar.Height = barH;
            bar.Fill = (_userLevel > _aiLevel && i % 2 == 1) ? userBrush : aiBrush;
            bar.Opacity = 0.26 + 0.74 * Math.Min(1.0, energy * 1.4 + 0.15);

            double rStart = baseR + 10.0;
            double x = cx + Math.Cos(ang) * rStart;
            double y = cy + Math.Sin(ang) * rStart;
            Canvas.SetLeft(bar, x - bar.Width * 0.5);
            Canvas.SetTop(bar, y - barH);
            bar.RenderTransform = new RotateTransform(ang * 180.0 / Math.PI + 90.0);
        }

        // Precision Compass Ticks
        for (int i = 0; i < CompassTickCount; i++)
        {
            double ang = (2.0 * Math.PI * i / CompassTickCount) - _time * 0.08;
            bool major = i % 4 == 0;
            double r1 = baseR - (major ? 26 : 22);
            double r2 = baseR - 17;
            var line = _compassTicks[i];
            line.X1 = cx + Math.Cos(ang) * r1;
            line.Y1 = cy + Math.Sin(ang) * r1;
            line.X2 = cx + Math.Cos(ang) * r2;
            line.Y2 = cy + Math.Sin(ang) * r2;
            line.Stroke = major ? accent80 : accent40;
        }

        // Position Core Badge cleanly underneath the 3D reticle
        _coreBadge.BorderBrush = accent80;
        _coreBadge.Measure(new Size(380, 120));
        double bw = _coreBadge.DesiredSize.Width;
        double badgeY = Math.Min(h - 135, cy + baseR + 42);
        Canvas.SetLeft(_coreBadge, cx - bw * 0.5);
        Canvas.SetTop(_coreBadge, badgeY);
    }

    private static Geometry BuildArcRingGeometry(double cx, double cy, double r, double angleOffsetDeg, int count, double sweepDeg)
    {
        var g = new StreamGeometry();
        using var ctx = g.Open();
        double step = 360.0 / count;
        for (int i = 0; i < count; i++)
        {
            double a0 = (angleOffsetDeg + i * step) * Math.PI / 180.0;
            double a1 = (angleOffsetDeg + i * step + sweepDeg) * Math.PI / 180.0;
            var p0 = new Point(cx + Math.Cos(a0) * r, cy + Math.Sin(a0) * r);
            var p1 = new Point(cx + Math.Cos(a1) * r, cy + Math.Sin(a1) * r);
            ctx.BeginFigure(p0, false, false);
            ctx.ArcTo(p1, new Size(r, r), 0, sweepDeg > 180, SweepDirection.Clockwise, true, false);
        }
        g.Freeze();
        return g;
    }

    private static Color LerpColor(Color a, Color b, double t)
    {
        t = Math.Clamp(t, 0.0, 1.0);
        return Color.FromArgb(
            (byte)(a.A + (b.A - a.A) * t),
            (byte)(a.R + (b.R - a.R) * t),
            (byte)(a.G + (b.G - a.G) * t),
            (byte)(a.B + (b.B - a.B) * t));
    }

    private static Color ParseHexColor(string? hex, Color fallback)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(hex) && ColorConverter.ConvertFromString(hex) is Color c)
                return c;
        }
        catch { }
        return fallback;
    }

    private static string Truncate(string s, int max)
        => s.Length <= max ? s : s[..(max - 1)] + "…";

    // ------------------------------------------------------------------ FX Records

    private sealed class Particle3D
    {
        public Ellipse Shape = null!;
        public double X, Y, Z, SpeedZ, OrbitSpeed, BaseSize;
    }

    private sealed class ShockwaveFx
    {
        public Ellipse Shape = null!;
        public Color Color;
        public double MaxRadius, Duration, Age;
    }

    private sealed class TaskSatellite
    {
        public string Id = "";
        public string Title = "";
        public string Status = "";
        public int Progress;
        public Color Color;
        public double OrbitAngle;
        public Ellipse OrbitRing = null!;
        public Border Badge = null!;
        public Ellipse StatusDot = null!;
        public TextBlock TitleLabel = null!;
    }

    private sealed class ActivityNode
    {
        public Path BeamGlow = null!;
        public Path BeamCore = null!;
        public Ellipse PulseDot = null!;
        public Ellipse AnchorRing = null!;
        public Border Card = null!;
        public ScaleTransform CardScale = null!;
        public Rectangle LifeBar = null!;
        public double Angle, NormDist, ZDepth, DriftSpeed, BobPhase, Duration, Age;
        public bool Inbound, ForceClosing;
        public Color Color;
    }
}
