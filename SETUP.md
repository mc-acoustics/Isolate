# Isolate — Setup & Build Instructions (Windows 10/11)

## 0. Prerequisite: Python 3.10 (mandatory)

Spleeter depends on TensorFlow 2.x builds that **do not support Python 3.12+**
(your system Python 3.13 will not work for this project). Install Python 3.10
side-by-side — it does not interfere with your existing Python:

```bat
winget install Python.Python.3.10
```

## 1. Install ffmpeg (required by Spleeter, yt-dlp, and MP3 export)

```bat
winget install Gyan.FFmpeg
```

Close and reopen the terminal afterwards so `ffmpeg` is on `PATH`.
Verify with `ffmpeg -version`.

## 2. Create the environment and install dependencies

From the project folder (`Z:\claude\audio\Isolate`):

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the app locally

```bat
.venv\Scripts\activate
python main.py
```

Notes for the first run:

* Clicking **Separate Tracks** for the first time downloads the selected
  pretrained model (~75–200 MB) to `%LOCALAPPDATA%\Isolate\pretrained_models`.
  The status bar reports progress; subsequent runs are instant.
* Separation runs on CPU; a 4-minute song takes roughly 1–4 minutes
  depending on the machine.

## 4. Build the Windows executable

```bat
.venv\Scripts\activate
build.bat
```

The result is a **one-dir bundle** at `dist\Isolate\Isolate.exe`
(TensorFlow is too large/fragile for `--onefile`). If `ffmpeg.exe` was on
`PATH` at build time it is copied next to the exe, making the bundle fully
self-contained. Distribute the whole `dist\Isolate` folder — zip it, or wrap
it with [Inno Setup](https://jrsoftware.org/isinfo.php) to produce a
single-file installer.

## 5. Quick usage recap

1. Drag & drop a `.wav`/`.mp3`/`.m4a`/`.mp4` onto the landing zone, **or**
   paste a YouTube URL and click **Download & Load**.
2. Pick 2 / 4 / 5 stems and click **Separate Tracks**.
3. Mix in real time with the per-track volume sliders, **M**ute and **S**olo
   (standard DAW solo matrix: any active solo silences all non-soloed tracks).
4. Choose the output device in the top bar; use Play / Pause / Stop and the
   timeline slider to navigate.
5. **Export Mix** renders the current fader/mute/solo state to
   WAV (PCM 16-bit / 44.1 kHz) or MP3 (320 kbps CBR).
