<#
  clean_pycache.ps1
  Recursively remove __pycache__ directories and .pyc files in the repo.
  Use this before running `git pull --rebase` or other operations if Git
  reports it can't delete cache directories (common on Windows w/ OneDrive).

  Usage (PowerShell):
    # from the repo root
    .\scripts\clean_pycache.ps1

  Notes:
    - Close editors and pause OneDrive syncing before running if you get
      'access denied' errors.
    - This script only deletes generated cache files, not source files.
#>

Param()

Write-Host "Cleaning __pycache__ directories and *.pyc files..." -ForegroundColor Cyan

try {
    # Find and remove __pycache__ directories
    $pycacheDirs = Get-ChildItem -Path . -Directory -Filter "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
    if ($pycacheDirs) {
        foreach ($d in $pycacheDirs) {
            Write-Host "Removing: $($d.FullName)" -ForegroundColor Yellow
            Remove-Item -LiteralPath $d.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "No __pycache__ directories found." -ForegroundColor Green
    }

    # Remove .pyc and other bytecode files
    $pycFiles = Get-ChildItem -Path . -Include *.pyc,*.pyo,*.pyc -Recurse -Force -ErrorAction SilentlyContinue
    if ($pycFiles) {
        foreach ($f in $pycFiles) {
            Write-Host "Removing file: $($f.FullName)" -ForegroundColor Yellow
            Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "No .pyc/.pyo files found." -ForegroundColor Green
    }

    Write-Host "Cleaning complete." -ForegroundColor Cyan
    exit 0
} catch {
    Write-Host "Error while cleaning: $_" -ForegroundColor Red
    Write-Host "If files are locked, pause OneDrive and close editors, then re-run this script." -ForegroundColor Yellow
    exit 1
}
