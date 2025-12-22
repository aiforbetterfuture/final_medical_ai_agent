$ErrorActionPreference = "Stop"

# Repo root = two levels up from this script (tools/aihub_indexing)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

Write-Host "Repo root:" $repoRoot

# Create a dedicated venv for indexing (recommended: Python 3.11 on Windows)
Write-Host "Creating venv: .venv_indexing (Python 3.11 recommended)..."

# If you have Python Launcher, this will target 3.11. If not, adjust manually.
py -3.11 -m venv .venv_indexing

& .\.venv_indexing\Scripts\python.exe -m pip install -U pip
& .\.venv_indexing\Scripts\python.exe -m pip install -r tools\aihub_indexing\requirements_indexing.txt

Write-Host ""
Write-Host "DONE."
Write-Host "Activate: .\.venv_indexing\Scripts\activate"
Write-Host "If you want HNSW (hnswlib): .\.venv_indexing\Scripts\python.exe -m pip install -r tools\aihub_indexing\requirements_hnsw_optional.txt"
