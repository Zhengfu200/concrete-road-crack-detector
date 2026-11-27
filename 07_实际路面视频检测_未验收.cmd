@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; & (Join-Path $env:USERPROFILE 'miniconda3/envs/concrete-yolov10/python.exe') preview_target_video.py"
pause
