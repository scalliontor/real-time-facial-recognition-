@echo off
REM Add cuDNN to PATH so ONNXRuntime can find cudnn64_9.dll
set PATH=C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64;%PATH%

call "%~dp0venv\Scripts\activate.bat"
python "%~dp0benchmark.py" > "%~dp0benchmark_results.txt"
pause
