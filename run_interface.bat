@echo off
cd /d C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction
echo Starting Plant Disease and Yield Prediction interface...
echo.
echo Open this URL in your browser:
echo http://127.0.0.1:8501
echo.
echo Keep this window open while using the interface.
echo Press Ctrl+C in this window to stop it.
echo.
venv39\Scripts\python.exe src\yield_prediction.py
pause
