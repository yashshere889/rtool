<#
.SYNOPSIS
    Open the Research Tool UI that runs on Barkla in this Windows computer's browser.

.DESCRIPTION
    rtool-ui.ps1            start the UI job (or reuse the running one), tunnel, open the browser
    rtool-ui.ps1 status     is a UI job queued or running, and where?
    rtool-ui.ps1 stop       end the UI job

    The Windows counterpart of scripts/rtool-ui (macOS, Linux). It needs the OpenSSH client that
    comes with Windows 10 and 11, an SSH key for Barkla, and "barkla" in your SSH config
    (%USERPROFILE%\.ssh\config). On campus, or off campus with the University VPN on.

    Ctrl-C closes the tunnel only: the job keeps running until its hours are used up or you stop
    it, and running this again reconnects. Runs keep going without the UI either way.

    Security: the UI job listens on a Unix socket in a folder only your Barkla account can open,
    and the tunnel ends on 127.0.0.1 here. The access token is read over ssh and handed to the
    browser in the address's #fragment, which browsers never send to a server; it is not printed.
    The compute node's host key is read through the login node into
    %USERPROFILE%\.config\rtool\barkla_known_hosts (your own known_hosts is not changed).

.PARAMETER Hours
    How long a new UI job may run (default 8).
.PARAMETER Port
    The port on this computer (default 8780, or RTOOL_UI_PORT).
.PARAMETER Bind
    socket (default) or cluster: a TCP port on the cluster network, token-protected, for when
    ssh to compute nodes does not work. Applies to a new job: stop the running one first.
.PARAMETER SshHost
    The ssh name of the Barkla login node (default barkla).
.PARAMETER NoBrowser
    Do not open the browser.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\rtool-ui.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\rtool-ui.ps1 stop
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet("start", "status", "stop")][string]$Command = "start",
    [ValidateRange(0.25, 72)][double]$Hours = 8,
    [ValidateRange(1, 65535)][int]$Port = $(if ($env:RTOOL_UI_PORT) { [int]$env:RTOOL_UI_PORT } else { 8780 }),
    [ValidateSet("socket", "cluster")][string]$Bind = "socket",
    [ValidatePattern('^[A-Za-z0-9_.@-]+$')][string]$SshHost = "barkla",
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
# No double quotes in remote commands: Windows PowerShell 5.1 mangles them for native programs.
$Remote = 'bash -l ~/Research\ tool/scripts/barkla_rtool.sh'
$KnownHosts = Join-Path $HOME ".config\rtool\barkla_known_hosts"

function Invoke-Barkla([string]$Arguments) {
    $out = & ssh $SshHost "$Remote $Arguments"
    if ($LASTEXITCODE -ne 0) { throw "rtool on $SshHost failed (exit code $LASTEXITCODE)" }
    return ($out -join "`n")
}

function Assert-Plain([string]$Value) {
    # refuse anything the remote shell could read as more than a name or a path
    if ($Value -notmatch '^[A-Za-z0-9_./-]+$') { throw "unexpected value from Barkla: $Value" }
}

function Test-Listening([int]$P) {
    $client = New-Object System.Net.Sockets.TcpClient
    try { return $client.ConnectAsync("127.0.0.1", $P).Wait(1500) -and $client.Connected }
    catch { return $false }
    finally { $client.Close() }
}

function Test-UI {
    try {
        $request = [System.Net.WebRequest]::Create("http://127.0.0.1:$Port/api/health")
        $request.Proxy = $null  # never through a proxy: it is this computer
        $request.Timeout = 10000
        $request.GetResponse().Close()
        return $true
    }
    catch { return $false }
}

if ($Command -ne "start") {
    & ssh $SshHost "$Remote ui --$Command"
    exit $LASTEXITCODE
}

if (Test-Listening $Port) {
    throw "something already listens on 127.0.0.1:$Port (an earlier tunnel?); close it or use -Port"
}

Write-Host "starting or finding the UI job on $SshHost..."
$hoursText = $Hours.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$job = Invoke-Barkla "ui --json --wait 60 --hours $hoursText --bind $Bind" | ConvertFrom-Json
$said = ""
while (-not ($job -and $job.serving)) {
    if (-not $job) { throw "the UI job is gone; its log is in the ui folder next to your runs on Barkla" }
    $line = "UI job $($job.job_id): $($job.state.ToLower()) $($job.reason)"
    if ($line -ne $said) { Write-Host "$line (waiting; Ctrl-C stops waiting, the job stays queued)" }
    $said = $line
    Start-Sleep -Seconds 15
    $job = Invoke-Barkla "ui --status --json" | ConvertFrom-Json
}

$jobId = [string]$job.job_id
$node = [string]$job.node
$dir = [string]$job.directory
foreach ($value in @($jobId, $node, [string]$job.user, $dir)) { Assert-Plain $value }
$token = ((& ssh $SshHost "cat $dir/token") -join "").Trim()
if ($LASTEXITCODE -ne 0 -or -not $token) { throw "could not read the UI's token from $SshHost" }

$sshArgs = @("-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30")
if ($job.bind -eq "socket") {
    $sock = [string]$job.socket
    Assert-Plain $sock
    $key = @(& ssh $SshHost "ssh-keyscan -t ed25519 $node 2>/dev/null") | Where-Object { $_ -like "$node *" }
    if (-not $key) { throw "could not read $node's host key through $SshHost" }
    New-Item -ItemType Directory -Force -Path (Split-Path $KnownHosts) | Out-Null
    $kept = @()
    if (Test-Path $KnownHosts) { $kept = @(Get-Content $KnownHosts | Where-Object { $_ -and -not $_.StartsWith("$node ") }) }
    Set-Content -Path $KnownHosts -Value ($kept + @($key)) -Encoding Ascii
    # ssh expands ~ itself, which keeps a home folder with spaces out of the argument
    $sshArgs += @("-o", "BatchMode=yes", "-o", "UserKnownHostsFile=~/.config/rtool/barkla_known_hosts",
        "-o", "StrictHostKeyChecking=yes", "-J", $SshHost, "-L", "127.0.0.1:${Port}:$sock", "$($job.user)@$node")
}
else {
    $remotePort = [string]$job.port
    if ($remotePort -notmatch '^[0-9]+$') { throw "unexpected port from Barkla: $remotePort" }
    $sshArgs += @("-L", "127.0.0.1:${Port}:${node}:$remotePort", $SshHost)
}

$tunnel = Start-Process -FilePath "ssh" -ArgumentList $sshArgs -NoNewWindow -PassThru
$stopped = $false
try {
    $ready = $false
    for ($i = 0; $i -lt 30 -and -not $ready; $i++) {
        if ($tunnel.HasExited) {
            $hint = if ($job.bind -eq "socket") { " (if ssh to the node keeps failing: rtool-ui.ps1 stop, then rtool-ui.ps1 -Bind cluster)" } else { "" }
            throw "the tunnel to $node failed$hint"
        }
        $ready = Test-UI
        if (-not $ready) { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw "the UI did not answer" }

    $url = "http://127.0.0.1:$Port/"
    if ($NoBrowser) {
        Write-Host "open $url and sign in with the token: ssh $SshHost cat $dir/token"
    }
    else {
        # other users of this computer cannot read the browser's command line (only administrators can)
        Start-Process "$url#token=$token"
    }
    Write-Host "Research Tool UI: $url (job $jobId on $node). Ctrl-C closes the tunnel; the job keeps running."

    $misses = 0
    while (-not $tunnel.HasExited) {  # until the tunnel breaks or the job ends
        Start-Sleep -Seconds 30
        if (Test-UI) { $misses = 0 } else { $misses++ }
        if ($misses -ge 2) {
            Write-Warning "the UI stopped answering (job ended, or the connection dropped); run rtool-ui.ps1 to start or reconnect"
            $stopped = $true
            exit 1
        }
    }
    Write-Warning "the tunnel closed; run rtool-ui.ps1 again to reconnect"
    $stopped = $true
}
finally {
    if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue }
    if (-not $stopped) { Write-Host "tunnel closed; UI job $jobId keeps running (rtool-ui.ps1 reconnects, rtool-ui.ps1 stop ends it)" }
}
