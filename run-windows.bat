@echo off
SET ROOT=%~dp0
SET PYTHONPATH=%ROOT%

IF "%1"=="env"      GOTO env
IF "%1"=="install"  GOTO install
IF "%1"=="setup-db" GOTO setup_db
IF "%1"=="seed"     GOTO seed
IF "%1"=="ingest"   GOTO ingest
IF "%1"=="start"    GOTO start
IF "%1"=="simulate" GOTO simulate
IF "%1"=="test"     GOTO test
IF "%1"=="health"   GOTO health
IF "%1"=="demo"     GOTO demo
GOTO help

:env
IF NOT EXIST .env ( copy .env.example .env && echo [OK] .env created ) ELSE ( echo [SKIP] .env exists )
GOTO end

:install
pip install -r requirements.txt
echo [OK] Dependencies installed
GOTO end

:setup_db
echo [INFO] Run supabase/schema.sql manually in Supabase SQL Editor
echo [INFO] Then run: run-windows.bat seed
GOTO end

:seed
set PYTHONPATH=%ROOT%
python scripts\setup\seed_data.py
echo [OK] 5 mock patients seeded into Supabase
GOTO end

:ingest
set PYTHONPATH=%ROOT%
python -m rag.knowledge_base.processed.indexer --collection all
echo [OK] Vector store seeded in Supabase pgvector
GOTO end

:start
echo [INFO] Starting Synera backend (Ctrl+C to stop)...
set PYTHONPATH=%ROOT%
python run.py
GOTO end

:simulate
echo [INFO] Starting simulator (run in a second terminal after start)...
start "Synera Simulator" cmd /k "set PYTHONPATH=%ROOT% && python scripts\data_gen\mock_simulator.py"
GOTO end

:test
set PYTHONPATH=%ROOT%
python -m pytest backend\tests\unit\ -v --tb=short
GOTO end

:health
curl -s http://localhost:8000/api/v1/health
GOTO end

:demo
set PYTHONPATH=%ROOT%
python scripts\testing\demo_scenarios.py
GOTO end

:help
echo.
echo  FIRST TIME SETUP:
echo    1. run-windows.bat env        - create .env, fill Supabase + Groq keys
echo    2. run-windows.bat install     - pip install
echo    3. Run supabase/schema.sql in Supabase SQL Editor
echo    4. run-windows.bat seed        - seed 5 patients
echo    5. run-windows.bat ingest      - seed pgvector
echo    6. run-windows.bat start       - start backend
echo    7. run-windows.bat simulate    - start simulator (new window)
echo    8. run-windows.bat health      - verify
echo.
echo  TESTING: run-windows.bat test
echo.

:end
