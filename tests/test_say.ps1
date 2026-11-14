$body = @{ text = "Hello Jarvis, what are you? Keep it to two sentences." } | ConvertTo-Json
$r = Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/say" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900
$thoughts = 0
foreach ($ev in $r.events) {
    if ($ev.type -eq "thought") { $thoughts++ }
    else { $ev | ConvertTo-Json -Compress -Depth 3 }
}
Write-Output ("(thought deltas: {0})" -f $thoughts)
