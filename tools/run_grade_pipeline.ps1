param(
  [Parameter(Mandatory=$true)][string]$Evalset,
  [string]$Run       = "experiments\\eval_runs\\run.jsonl",
  [string]$Rubric    = "configs\\eval_rubric.yaml",
  [string]$Out       = "experiments\\eval_runs\\grades.jsonl",
  [string]$ModelMode = "default",
  [switch]$NoLLMJudge,
  [switch]$DebugAgent
)

$RepoRoot = Split-Path -Parent $PSScriptRoot

# Ensure repo modules (agent/, retrieval/, tools/) are importable when running tool scripts.
$env:PYTHONPATH = $RepoRoot

# Force UTF-8 to avoid cp949 UnicodeEncodeError in Windows consoles.
$env:PYTHONUTF8 = "1"

$argsList = @(
  "-X", "utf8",
  (Join-Path $RepoRoot "tools\\grade_run.py"),
  "--pipeline",
  "--evalset", $Evalset,
  "--run", $Run,
  "--rubric", $Rubric,
  "--out", $Out,
  "--model_mode", $ModelMode
)

if ($NoLLMJudge) { $argsList += "--no_llm_judge" }
if ($DebugAgent) { $argsList += "--debug_agent" }

Write-Host "> python $($argsList -join ' ')"
& python @argsList
exit $LASTEXITCODE
