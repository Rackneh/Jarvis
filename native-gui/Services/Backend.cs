using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace JarvisDeck.Services;

public record JarvisEvent(string Type, JsonElement Data);

/// <summary>
/// Client for the Jarvis backend: one WebSocket for events, REST for actions.
/// </summary>
public sealed class Backend : IDisposable
{
    private readonly HttpClient _http;
    private ClientWebSocket? _ws;
    private CancellationTokenSource? _cts;
    private string _baseUrl;
    private volatile bool _disposed;

    public event Action<JarvisEvent>? EventReceived;
    public event Action<bool>? ConnectionChanged;

    public Backend(string baseUrl = "http://127.0.0.1:8010")
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _http = new HttpClient { BaseAddress = new Uri(_baseUrl + "/") };
    }

    public bool Connected { get; private set; }

    public async Task ConnectAsync(CancellationToken ct = default)
    {
        if (_disposed)
            return;
        _cts = _cts is { IsCancellationRequested: false } ? _cts : new CancellationTokenSource();
        var ws = new ClientWebSocket();
        var uri = new Uri(_baseUrl.Replace("http", "ws") + "/ws");
        await ws.ConnectAsync(uri, _cts.Token);
        _ws = ws;
        Connected = true;
        ConnectionChanged?.Invoke(true);
        _ = Task.Run(() => ReadLoop(ws, _cts.Token));
    }

    public async Task ReconnectForeverAsync()
    {
        while (!_disposed)
        {
            try
            {
                if (_ws == null || _ws.State != WebSocketState.Open)
                {
                    await ConnectAsync();
                }
            }
            catch { Connected = false; ConnectionChanged?.Invoke(false); }
            await Task.Delay(3000);
        }
    }

    private async Task ReadLoop(ClientWebSocket ws, CancellationToken ct)
    {
        var buf = new byte[64 * 1024];
        try
        {
            while (ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var sb = new StringBuilder();
                WebSocketReceiveResult result;
                do
                {
                    result = await ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                    sb.Append(Encoding.UTF8.GetString(buf, 0, result.Count));
                } while (!result.EndOfMessage);

                if (result.MessageType == WebSocketMessageType.Close) break;

                try
                {
                    var doc = JsonDocument.Parse(sb.ToString());
                    var root = doc.RootElement;
                    var type = root.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
                    var data = root.TryGetProperty("data", out var d)
                        ? d.Clone()
                        : (root.TryGetProperty("snapshot", out var s) ? s.Clone() : JsonSerializer.SerializeToElement(root));
                    EventReceived?.Invoke(new JarvisEvent(type, data));
                }
                catch (JsonException) { /* skip malformed */ }
            }
        }
        catch (OperationCanceledException) { }
        catch { /* dropped */ }
        finally
        {
            Connected = false;
            ConnectionChanged?.Invoke(false);
        }
    }

    // ------------------------------------------------------------------ REST

    public async Task<JsonElement?> PostAsync(string path, object? body = null)
    {
        try
        {
            var content = body is null
                ? new StringContent("{}", Encoding.UTF8, "application/json")
                : new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
            var resp = await _http.PostAsync(path, content);
            var text = await resp.Content.ReadAsStringAsync();
            return text.Length > 0 ? JsonSerializer.Deserialize<JsonElement>(text) : null;
        }
        catch { return null; }
    }

    public async Task<JsonElement?> GetAsync(string path)
    {
        try
        {
            var resp = await _http.GetAsync(path);
            var text = await resp.Content.ReadAsStringAsync();
            return text.Length > 0 ? JsonSerializer.Deserialize<JsonElement>(text) : null;
        }
        catch { return null; }
    }

    public Task<JsonElement?> Say(string text) => PostAsync("api/say", new { text });
    public Task<JsonElement?> Status() => GetAsync("api/status");
    public Task<JsonElement?> Tasks() => GetAsync("api/tasks");
    public Task<JsonElement?> CreateTask(string objective, string? title = null)
        => PostAsync("api/tasks", new { objective, title });
    public Task<JsonElement?> AnswerTask(string taskId, string answer)
        => PostAsync("api/tasks/answer", new { task_id = taskId, answer });
    public Task<JsonElement?> TaskAction(string taskId, string action)
        => PostAsync($"api/tasks/{taskId}/{action}");
    public Task<JsonElement?> SwitchBackend(string target) => PostAsync("api/provider", new { target });
    public Task<JsonElement?> Rename(string name) => PostAsync("api/rename", new { name });
    public Task<JsonElement?> Ptt(string action) => PostAsync("api/voice/ptt", new { action });
    public Task<JsonElement?> Config() => GetAsync("api/config");
    public Task<JsonElement?> VoiceDevices() => GetAsync("api/voice/devices");
    public Task<JsonElement?> SaveConfig(Dictionary<string, object> settings)
        => PostAsync("api/config", new { settings });

    public void Dispose()
    {
        _disposed = true;
        _cts?.Cancel();
        _ws?.Dispose();
        _http.Dispose();
    }
}
