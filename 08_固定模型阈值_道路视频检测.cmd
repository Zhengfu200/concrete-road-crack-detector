@echo off
chcp 65001 >nul
pushd "%~dp0"
"%USERPROFILE%\miniconda3\envs\concrete-yolov10\python.exe" run_frozen_video.py
popd
pause
