@echo off
echo 🔥 Starting smoke test...

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Test basic imports
echo 🔍 Testing imports...
python -c "import codechat; print('✅ codechat import successful')"

REM Test ingestion with a small public repo
echo 🚀 Testing ingestion...
python -m codechat ingest --repo-url https://github.com/psf/requests --out-store temp_req_idx

REM Test chat with a simple question
echo 💬 Testing chat...
echo Where is authentication handled? | python -m codechat chat --store temp_req_idx --no-interactive

echo ✅ ALL CHECKS PASSED
