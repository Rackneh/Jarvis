using System.Collections.Generic;
using System.ComponentModel;
using System.Text.Json;

namespace JarvisDeck.Models;

public sealed class TaskVm : INotifyPropertyChanged
{
    public string Id { get; set; } = "";
    private string _title = "";
    public string Title { get => _title; set { _title = value; On(nameof(Title), value); } }

    private string _status = "queued";
    public string Status { get => _status; set { _status = value; On(nameof(Status), value); On(nameof(StatusColor), value); On(nameof(StatusGlow), value); } }

    private int _progress;
    public int Progress { get => _progress; set { _progress = value; On(nameof(Progress), value); } }

    private string _lastStep = "";
    public string LastStep { get => _lastStep; set { _lastStep = value; On(nameof(LastStep), value); } }

    private string _pendingQuestion = "";
    public string PendingQuestion { get => _pendingQuestion; set { _pendingQuestion = value; On(nameof(PendingQuestion), value); On(nameof(HasQuestion), value); } }

    public bool HasQuestion => !string.IsNullOrEmpty(PendingQuestion);
    public string AnswerDraft { get; set; } = "";

    public string StatusColor => Status switch
    {
        "running" => "#22D3EE",
        "queued" => "#A78BFA",
        "waiting_user" => "#FBBF24",
        "paused" => "#FBBF24",
        "done" => "#4ADE80",
        "failed" => "#F87171",
        "cancelled" => "#64748B",
        _ => "#5B6B85",
    };

    public string StatusGlow => Status switch
    {
        "running" => "#22D3EE",
        "waiting_user" or "paused" => "#FBBF24",
        "done" => "#4ADE80",
        "failed" => "#F87171",
        _ => "#334155",
    };

    public event PropertyChangedEventHandler? PropertyChanged;

    private void On(string name, object value)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    public static TaskVm FromJson(JsonElement e)
    {
        string S(JsonElement el, string k, string def = "")
            => el.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? def : def;
        int I(JsonElement el, string k, int def = 0)
            => el.TryGetProperty(k, out var v) && v.TryGetInt32(out var i) ? i : def;
        return new TaskVm
        {
            Id = S(e, "id"),
            Title = S(e, "title", "task"),
            Status = S(e, "status", "queued"),
            Progress = I(e, "progress"),
            LastStep = S(e, "last_step") is { Length: > 0 } ls ? ls : S(e, "result"),
            PendingQuestion = S(e, "pending_question"),
        };
    }

    public void Apply(JsonElement e)
    {
        string S(string k, string def = "")
            => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() ?? def : def;
        if (S("status") is { Length: > 0 } st) Status = st;
        if (e.TryGetProperty("progress", out var p) && p.TryGetInt32(out var pr)) Progress = pr;
        if (S("last_step") is { Length: > 0 } ls) LastStep = ls;
        if (S("result") is { Length: > 0 } r) LastStep = r;
        if (S("pending_question") is { Length: > 0 } q) PendingQuestion = q;
        else if (Status != "waiting_user") PendingQuestion = "";
    }
}
