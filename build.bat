@echo off
REM ===========================================================================
REM  Isolate — Windows build script (PyInstaller)
REM  Run from an activated Python 3.10 virtual environment that has all
REM  packages from requirements.txt installed:
REM      .venv\Scripts\activate && build.bat
REM  Output: dist\Isolate\Isolate.exe  (one-dir bundle; TensorFlow is far
REM  too large and fragile for --onefile).
REM ===========================================================================
setlocal

where pyinstaller >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pyinstaller not found. Activate the venv and run:
    echo     pip install -r requirements.txt
    exit /b 1
)

REM Bundle ffmpeg AND ffprobe from the LGPL build ONLY (licensing: the gyan.dev
REM build on PATH is GPLv3 and must NOT ship inside a proprietary bundle).
REM Get the LGPL build from https://github.com/BtbN/FFmpeg-Builds (win64-lgpl)
REM and extract it under %LOCALAPPDATA%\Isolate\ffmpeg-lgpl\.
set FFDIR=
for /d %%d in ("%LOCALAPPDATA%\Isolate\ffmpeg-lgpl\ffmpeg-*") do (
    if exist "%%d\bin\ffmpeg.exe" set "FFDIR=%%d"
)
if not defined FFDIR (
    echo [ERROR] LGPL ffmpeg build not found under %LOCALAPPDATA%\Isolate\ffmpeg-lgpl\.
    echo Download ffmpeg-nX.Y-latest-win64-lgpl-X.Y.zip from
    echo     https://github.com/BtbN/FFmpeg-Builds/releases
    echo and extract it there. Refusing to bundle a GPL build.
    exit /b 1
)
echo Bundling LGPL ffmpeg from: %FFDIR%
set "FFMPEG_OPT=--add-binary=%FFDIR%\bin\ffmpeg.exe;."
set "FFPROBE_OPT=--add-binary=%FFDIR%\bin\ffprobe.exe;."

pyinstaller ^
  --noconfirm --clean ^
  --name Isolate ^
  --windowed ^
  --icon isolate.ico ^
  --add-data isolate.ico;. ^
  --collect-all spleeter ^
  --collect-all tensorflow ^
  --collect-all norbert ^
  --collect-all customtkinter ^
  --collect-all tkinterdnd2 ^
  --collect-all yt_dlp ^
  --collect-binaries sounddevice ^
  --collect-binaries soundfile ^
  --collect-data soundfile ^
  --hidden-import=scipy.signal ^
  --hidden-import=scipy.spatial.transform._rotation_groups ^
  --hidden-import=tensorflow.python.keras.engine.base_layer_v1 ^
  %FFMPEG_OPT% ^
  %FFPROBE_OPT% ^
  main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    exit /b 1
)

REM --- License compliance: third-party license texts shipped with the app ---
mkdir "dist\Isolate\licenses" 2>nul

REM ffmpeg (LGPLv3 build): license text + source offer (LGPL requirement)
copy /y "%FFDIR%\LICENSE.txt" "dist\Isolate\licenses\FFMPEG-LICENSE.txt" >nul
(
    echo Isolate bundles unmodified ffmpeg.exe and ffprobe.exe binaries, invoked
    echo as separate command-line programs. They are licensed under the GNU
    echo Lesser General Public License v3 ^(see FFMPEG-LICENSE.txt^) and can be
    echo replaced by the user with any compatible ffmpeg build.
    echo.
    echo Binary build: BtbN FFmpeg-Builds, variant win64-lgpl
    echo     https://github.com/BtbN/FFmpeg-Builds/releases
    echo Corresponding source code:
    echo     https://ffmpeg.org/download.html
    echo Build scripts ^(exact configuration^):
    echo     https://github.com/BtbN/FFmpeg-Builds
) > "dist\Isolate\licenses\FFMPEG-SOURCE.txt"

REM Python dependencies: full license texts via pip-licenses
where pip-licenses >nul 2>nul
if errorlevel 1 (
    echo [WARN] pip-licenses not installed - THIRD-PARTY-LICENSES.txt NOT generated.
    echo        Run: pip install pip-licenses
) else (
    pip-licenses --with-license-file --no-license-path --format=plain-vertical ^
        --output-file="dist\Isolate\licenses\THIRD-PARTY-LICENSES.txt"
    echo Wrote dist\Isolate\licenses\THIRD-PARTY-LICENSES.txt
)

echo.
echo ===========================================================================
echo  Build complete: dist\Isolate\Isolate.exe
echo  Distribute the WHOLE dist\Isolate folder (or zip it / wrap it with an
echo  installer such as Inno Setup). On first run the app downloads the
echo  Spleeter pretrained models to %%LOCALAPPDATA%%\Isolate\pretrained_models.
echo ===========================================================================
endlocal
