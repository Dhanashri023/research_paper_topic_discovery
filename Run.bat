\
    @echo off
    title Topic Discovery — Run Web App
    cd /d %~dp0
    echo Installing requirements (if missing)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo Starting Flask app...
    python web/app.py
    pause
