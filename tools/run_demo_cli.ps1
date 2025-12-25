param(
  [Parameter(Mandatory=$true)][string]$Query,
  [string]$RuntimeYaml = 'configs\aihub_retrieval_runtime.yaml',
  [string]$JsonOut = 'experiments\retrieval_tuning\out.json',
  [int]$DomainId = 0,
  [string]$IndexDir = ''
)

# Make PowerShell output UTF-8 (helps if your console is cp949)
try {
  $utf8 = [System.Text.UTF8Encoding]::new($false)
  [Console]::OutputEncoding = $utf8
  $OutputEncoding = $utf8
} catch {}

# Ensure output folder exists
$dir = Split-Path -Parent $JsonOut
if ($dir -and !(Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

$cmd = @('python','-m','retrieval.aihub_flat.demo_cli','--query',$Query,'--runtime_yaml',$RuntimeYaml,'--show_cfg','--json_out',$JsonOut)
if ($DomainId -gt 0) { $cmd += @('--domain_id', "$DomainId") }
if ($IndexDir -ne '') { $cmd += @('--index_dir', $IndexDir) }

Write-Host ('> ' + ($cmd -join ' '))
& $cmd[0] $cmd[1..($cmd.Length-1)]
