param([int]$Seconds = 65)

$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'native-gui\bin\Debug\net9.0-windows\JarvisDeck.exe'
if (-not (Get-Process JarvisDeck -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $exe -WorkingDirectory $root
}

$names = @('JarvisDeck', 'python', 'LM Studio', 'dotnet')
$logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$last = @{}
$rows = New-Object System.Collections.Generic.List[object]

for ($i = 0; $i -lt $Seconds; $i++) {
    $now = Get-Date
    $cur = @{}
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $names -contains $_.ProcessName } |
        Group-Object ProcessName |
        ForEach-Object {
            $cpu = ($_.Group | Measure-Object -Property CPU -Sum).Sum
            $ws = ($_.Group | Measure-Object -Property WorkingSet64 -Sum).Sum
            $cur[$_.Name] = @{ cpu = $cpu; ws = $ws }
        }

    $sys = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average

    foreach ($n in $names) {
        if (-not $cur.ContainsKey($n)) {
            $last.Remove($n)
            continue
        }
        $c = $cur[$n]
        $delta = 0.0
        $pct = 0.0
        if ($last.ContainsKey($n)) {
            $dt = ($now - $last[$n].time).TotalSeconds
            if ($dt -gt 0) {
                $delta = ($c.cpu - $last[$n].cpu) / $dt
                $pct = $delta / $logical * 100
            }
        }
        $rows.Add([pscustomobject]@{
            t = $i
            sys = [math]::Round($sys, 1)
            name = $n
            cpuDelta = [math]::Round($delta, 2)
            cpuPctTotal = [math]::Round($pct, 1)
            wsMB = [math]::Round($c.ws / 1MB, 0)
        })
        $last[$n] = @{ cpu = $c.cpu; ws = $c.ws; time = $now }
    }
    Start-Sleep -Milliseconds 1000
}

'--- samples with activity or high system load ---'
$rows | Where-Object { $_.name -eq 'JarvisDeck' -or $_.cpuDelta -gt 1 -or $_.sys -gt 85 } |
    Format-Table -AutoSize

'--- summary ---'
$rows | Group-Object name | ForEach-Object {
    $g = $_.Group
    [pscustomobject]@{
        name = $_.Name
        samples = $g.Count
        maxCpuDelta = ($g | Measure-Object -Property cpuDelta -Maximum).Maximum
        avgCpuDelta = [math]::Round(($g | Measure-Object -Property cpuDelta -Average).Average, 2)
        maxWsMB = ($g | Measure-Object -Property wsMB -Maximum).Maximum
        maxSys = ($g | Measure-Object -Property sys -Maximum).Maximum
    }
} | Format-Table -AutoSize
