@echo off
echo 🚀 Starting smoke test for codechat...

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

REM Create a temporary directory for the test
set TEST_STORE=%TEMP%\codechat_smoke_test
if exist "%TEST_STORE%" rmdir /s /q "%TEST_STORE%"

echo 🔄 Testing ingestion...
REM Test ingestion with a small, reliable repository
python -m codechat ingest --repo-url https://github.com/psf/requests --out-store "%TEST_STORE%"
if %ERRORLEVEL% neq 0 (
    echo ❌ Ingestion failed
    exit /b 1
)

echo 🔍 Testing store info...
python -m codechat info --store "%TEST_STORE%"
if %ERRORLEVEL% neq 0 (
    echo ❌ Store info failed
    exit /b 1
)

echo 💬 Testing chat functionality...
REM Test chat with a predefined question (non-interactive mode)
echo Where is authentication handled? | python -m codechat chat --store "%TEST_STORE%" --no-interactive
if %ERRORLEVEL% neq 0 (
    echo ❌ Chat functionality failed
    exit /b 1
)

echo 🧹 Cleaning up...
if exist "%TEST_STORE%" rmdir /s /q "%TEST_STORE%"

echo ✅ ALL CHECKS PASSED
