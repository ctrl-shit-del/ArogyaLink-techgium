# Synera 2.0 — Windows PowerShell Dev Runner (no Docker)
param([string]$Command = "help")
$Root = $PSScriptRoot
$env:PYTHONPATH = $Root

switch ($Command) {
    "env" {
        if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "[OK] .env created" -ForegroundColor Green }
        else { Write-Host "[SKIP] .env exists" -ForegroundColor Yellow }
    }
    "install" {
        pip install -r requirements.txt
        Write-Host "[OK] Done" -ForegroundColor Green
    }
    "setup-db" {
        Write-Host "Run supabase/schema.sql in Supabase SQL Editor" -ForegroundColor Cyan
    }
    "seed" {
        python scripts\setup\seed_data.py
        Write-Host "[OK] Seeded" -ForegroundColor Green
    }
    "ingest" {
        python -m rag.knowledge_base.processed.indexer --collection all
        Write-Host "[OK] pgvector seeded" -ForegroundColor Green
    }
    "start" {
        Write-Host "Starting backend..." -ForegroundColor Cyan
        python run.py
    }
    "simulate" {
        Start-Process cmd -ArgumentList "/k", "set PYTHONPATH=$Root && python scripts\data_gen\mock_simulator.py"
    }
    "test" {
        python -m pytest backend\tests\unit\ -v --tb=short
    }
    "health" {
        try { (Invoke-RestMethod "http://localhost:8000/api/v1/health") | ConvertTo-Json -Depth 5 }
        catch { Write-Host "Backend not reachable" -ForegroundColor Red }
    }
    "demo" { python scripts\testing\demo_scenarios.py }
    default {
        Write-Host "Synera 2.0 — No Docker" -ForegroundColor Cyan
        Write-Host "  .\run-windows.ps1 env | install | seed | ingest | start | simulate | test | health"
    }
}
