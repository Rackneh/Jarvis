using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using JarvisDeck.Services;

namespace JarvisDeck.Views;

public partial class SettingsWindow : Window
{
    private readonly Backend _backend;

    public SettingsWindow(Backend backend)
    {
        InitializeComponent();
        _backend = backend;
        Loaded += async (_, _) => await LoadAsync();
    }

    private async Task LoadAsync()
    {
        var cfg = await _backend.Config();
        if (!cfg.HasValue)
        {
            StatusText.Text = "backend not reachable";
            return;
        }
        var root = cfg.Value;
        var assistant = Sec(root, "assistant");
        var providers = Sec(root, "providers");
        var voice = Sec(root, "voice");
        var tasks = Sec(root, "tasks");
        var conscious = Sec(root, "conscious");

        SetText(AssistantNameBox, assistant, "name");
        SetText(AssistantPersonaBox, assistant, "persona");
        SetCombo(ProviderDefaultBox, providers, "default", "lmstudio");
        SetText(LmsModelBox, providers, "lmstudio_model");
        SetText(DoryModelBox, providers, "dory_model");

        SetCombo(ActivationModeBox, voice, "activation_mode", "wake");
        SetText(ConvIdleBox, voice, "conversation_idle_seconds");
        SetText(ConvExitBox, voice, "conversation_exit_phrases");
        SetCombo(WakeWordBox, voice, "wake_word", "");
        SetText(WakeThresholdBox, voice, "wake_threshold");
        SetText(WakeCooldownBox, voice, "wake_cooldown_seconds");
        SetText(VadThresholdBox, voice, "vad_threshold");
        SetText(SilenceEndBox, voice, "silence_end_ms");
        SetText(MinSpeechBox, voice, "min_speech_ms");
        SetText(MaxUtteranceBox, voice, "max_utterance_ms");
        SetCombo(SttEngineBox, voice, "stt_engine", "auto");
        SetText(SttModelBox, voice, "stt_model");
        SetCombo(SttMoonshineBox, voice, "stt_moonshine_size", "tiny");
        SetText(SttPartialBox, voice, "stt_partial_interval_ms");
        SetCombo(TtsEngineBox, voice, "tts_engine", "jarvis_clone");
        SetCheck(CachedConvBox, voice, "cached_conversation", true);
        SetText(TtsVoiceBox, voice, "tts_voice");
        SetText(TtsSpeedBox, voice, "tts_speed");
        SetText(TtsCloneRefBox, voice, "tts_clone_reference");
        SetText(BreezeServerBox, voice, "breeze_server_url");
        SetText(BreezeModelBox, voice, "breeze_model");
        SetText(BreezeVoiceBox, voice, "breeze_voice_id");
        SetCheck(EchoGuardBox, voice, "echo_guard", true);
        SetText(PttKeyBox, voice, "push_to_talk_key");

        SetText(MaxParallelBox, tasks, "max_parallel");
        SetText(MaxStepsBox, tasks, "max_steps_default");
        SetText(HistoryBox, conscious, "history_messages");
        SetText(TokenBudgetBox, conscious, "response_token_budget");
        SetCheck(ConsciousReasoningBox, conscious, "reasoning_enabled", false);
        await LoadDeviceBoxesAsync(voice);
        StatusText.Text = "";
    }

    private async void OnSave(object sender, RoutedEventArgs e)
    {
        var settings = new Dictionary<string, object>
        {
            ["assistant.name"] = AssistantNameBox.Text.Trim(),
            ["assistant.persona"] = AssistantPersonaBox.Text,
            ["providers.default"] = Selected(ProviderDefaultBox, "lmstudio"),
            ["providers.lmstudio.model"] = LmsModelBox.Text.Trim(),
            ["providers.dory.model"] = string.IsNullOrWhiteSpace(DoryModelBox.Text) ? "dory" : DoryModelBox.Text.Trim(),
            ["voice.activation_mode"] = Selected(ActivationModeBox, "wake"),
            ["voice.conversation_idle_seconds"] = ConvIdleBox.Text.Trim(),
            ["voice.conversation_exit_phrases"] = ConvExitBox.Text.Trim(),
            ["voice.wake_word"] = WakeWordBox.Text.Trim(),
            ["voice.wake_threshold"] = WakeThresholdBox.Text.Trim(),
            ["voice.wake_cooldown_seconds"] = WakeCooldownBox.Text.Trim(),
            ["voice.stt_engine"] = Selected(SttEngineBox, "auto"),
            ["voice.stt_model"] = SttModelBox.Text.Trim(),
            ["voice.stt_moonshine_size"] = Selected(SttMoonshineBox, "tiny"),
            ["voice.stt_partial_interval_ms"] = SttPartialBox.Text.Trim(),
            ["voice.vad_threshold"] = VadThresholdBox.Text.Trim(),
            ["voice.silence_end_ms"] = SilenceEndBox.Text.Trim(),
            ["voice.min_speech_ms"] = MinSpeechBox.Text.Trim(),
            ["voice.max_utterance_ms"] = MaxUtteranceBox.Text.Trim(),
            ["voice.tts_engine"] = Selected(TtsEngineBox, "jarvis_clone"),
            ["voice.cached_conversation"] = CachedConvBox.IsChecked == true,
            ["voice.tts_voice"] = TtsVoiceBox.Text.Trim(),
            ["voice.tts_speed"] = TtsSpeedBox.Text.Trim(),
            ["voice.tts_clone_reference"] = TtsCloneRefBox.Text.Trim(),
            ["voice.breeze_server_url"] = BreezeServerBox.Text.Trim(),
            ["voice.breeze_model"] = BreezeModelBox.Text.Trim(),
            ["voice.breeze_voice_id"] = BreezeVoiceBox.Text.Trim(),
            ["voice.echo_guard"] = EchoGuardBox.IsChecked == true,
            ["voice.push_to_talk_key"] = PttKeyBox.Text.Trim(),
            ["voice.input_device"] = DeviceValue(InputDeviceBox),
            ["voice.output_device"] = DeviceValue(OutputDeviceBox),
            ["tasks.max_parallel"] = MaxParallelBox.Text.Trim(),
            ["tasks.max_steps_default"] = MaxStepsBox.Text.Trim(),
            ["conscious.history_messages"] = HistoryBox.Text.Trim(),
            ["conscious.response_token_budget"] = TokenBudgetBox.Text.Trim(),
            ["conscious.reasoning_enabled"] = ConsciousReasoningBox.IsChecked == true,
        };
        StatusText.Text = "saving…";
        var resp = await _backend.SaveConfig(settings);
        if (resp.HasValue && resp.Value.TryGetProperty("ok", out var ok) && ok.ValueKind == JsonValueKind.True)
        {
            StatusText.Text = "saved";
            DialogResult = true;
            Close();
        }
        else
        {
            StatusText.Text = "save failed";
        }
    }

    private void OnCancel(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private async Task LoadDeviceBoxesAsync(JsonElement voice)
    {
        var devices = await _backend.VoiceDevices();
        PopulateDeviceBox(InputDeviceBox, devices, "input");
        PopulateDeviceBox(OutputDeviceBox, devices, "output");
        SetDeviceBox(InputDeviceBox, voice, "input_device");
        SetDeviceBox(OutputDeviceBox, voice, "output_device");
    }

    private static void PopulateDeviceBox(ComboBox box, JsonElement? devices, string kind)
    {
        box.Items.Clear();
        box.Items.Add(new ComboBoxItem { Content = "default" });
        if (devices is JsonElement d && d.ValueKind == JsonValueKind.Object &&
            d.TryGetProperty(kind, out var arr) && arr.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in arr.EnumerateArray())
            {
                var index = item.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
                    ? i.GetInt32()
                    : -1;
                var name = item.TryGetProperty("name", out var n) ? n.GetString() ?? $"device {index}" : $"device {index}";
                box.Items.Add(new ComboBoxItem { Content = $"{index}: {name}", Tag = index });
            }
        }
        box.SelectedIndex = 0;
    }

    private static void SetDeviceBox(ComboBox box, JsonElement parent, string key)
    {
        int? value = null;
        if (parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(key, out var v) &&
            v.ValueKind == JsonValueKind.Number)
        {
            value = v.GetInt32();
        }
        var match = box.Items.Cast<ComboBoxItem>().FirstOrDefault(i => Equals(i.Tag, value));
        box.SelectedItem = match ?? box.Items[0];
    }

    private static int DeviceValue(ComboBox box)
        => box.SelectedItem is ComboBoxItem { Tag: int index } ? index : -1;

    private static string Selected(ComboBox box, string fallback)
        => box.SelectedItem is ComboBoxItem item ? item.Content?.ToString() ?? fallback : fallback;

    private static JsonElement Sec(JsonElement root, string name)
        => root.ValueKind == JsonValueKind.Object && root.TryGetProperty(name, out var s) ? s : default;

    private static string Str(JsonElement parent, string key, string def = "")
    {
        if (parent.ValueKind != JsonValueKind.Object || !parent.TryGetProperty(key, out var v))
            return def;
        return v.ValueKind switch
        {
            JsonValueKind.String => v.GetString() ?? def,
            JsonValueKind.Number => v.GetRawText(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => def,
        };
    }

    private static void SetText(TextBox box, JsonElement parent, string key, string def = "")
        => box.Text = Str(parent, key, def);

    private static void SetCombo(ComboBox box, JsonElement parent, string key, string def = "")
    {
        var value = Str(parent, key, def);
        if (box.IsEditable)
        {
            box.Text = value;
            return;
        }
        var match = box.Items.Cast<ComboBoxItem>()
            .FirstOrDefault(i => string.Equals(i.Content?.ToString(), value, StringComparison.OrdinalIgnoreCase));
        box.SelectedItem = match ?? (box.Items.Count > 0 ? box.Items[0] : null);
    }

    private static void SetCheck(CheckBox box, JsonElement parent, string key, bool def)
    {
        if (parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(key, out var v))
            box.IsChecked = v.ValueKind == JsonValueKind.True;
        else
            box.IsChecked = def;
    }
}
