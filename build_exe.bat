@echo off
setlocal

cd /d "%~dp0"

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --name "DeadAsDiscoAutoMapper" ^
  --collect-all librosa ^
  --collect-all pyqtgraph ^
  --hidden-import PySide6.QtMultimedia ^
  --hidden-import scipy ^
  --hidden-import sklearn ^
  --add-binary "C:\Users\V1Sta\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe;." ^
  --paths . ^
  run_mapper.py

echo.
echo Build complete: dist\DeadAsDiscoAutoMapper\DeadAsDiscoAutoMapper.exe
endlocal
