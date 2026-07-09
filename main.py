"""
Isolate — Local high-fidelity audio source separation & multi-track mixer.

A standalone Windows desktop application:
  * Loads local audio files (.wav, .mp3, .m4a, .mp4) or downloads audio
    from YouTube via yt-dlp at maximum quality.
  * Separates the audio into 2 / 4 / 5 stems using Spleeter's
    high-frequency (16 kHz bandwidth) pretrained models.
  * Plays all stems through a single sample-accurate sounddevice
    OutputStream (phase-locked, real-time gain / mute / solo mixing).
  * Exports the current mix to WAV (PCM 16-bit / 44.1 kHz) or
    MP3 (320 kbps CBR).

Heavy work (downloading, Spleeter inference, exporting) always runs in
background threads; the UI thread is only ever touched via `after()`.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Environment setup (must happen before TensorFlow / Spleeter are imported)
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence TF chatter

_APP_NAME = "Isolate"
_APPDATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / _APP_NAME
_MODELS_DIR = _APPDATA_DIR / "pretrained_models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
# Spleeter reads the MODEL_PATH env var to decide where pretrained models live.
os.environ.setdefault("MODEL_PATH", str(_MODELS_DIR))

# Under pythonw / --windowed builds there is no console: sys.stdout and
# sys.stderr are None, and anything that writes to them (tqdm progress
# bars inside Spleeter's model download, TensorFlow banners, logging)
# raises "'NoneType' object has no attribute 'write'". Route them to a
# log file so background libraries can always write safely.
if sys.stdout is None or sys.stderr is None:
    _console_log = open(_APPDATA_DIR / "isolate.log", "a",
                        encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _console_log
    if sys.stderr is None:
        sys.stderr = _console_log

import numpy as np
import sounddevice as sd
import soundfile as sf
import customtkinter as ctk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:            # drag & drop becomes optional, app still works
    _HAS_DND = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(_APP_NAME)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4"}
NATIVE_SF_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

STEM_MODELS = {
    "2 Stems (Vocals / Accompaniment)": ("spleeter:2stems-16kHz",
                                         ["vocals", "accompaniment"]),
    "4 Stems (Vocals / Drums / Bass / Other)": ("spleeter:4stems-16kHz",
                                                ["vocals", "drums",
                                                 "bass", "other"]),
    "5 Stems (Vocals / Drums / Bass / Piano / Other)": ("spleeter:5stems-16kHz",
                                                        ["vocals", "drums",
                                                         "bass", "piano",
                                                         "other"]),
}

BLOCKSIZE = 1024          # frames per audio callback (~23 ms @ 44.1 kHz)
UI_POLL_MS = 66           # transport / status / VU-meter poll interval
METER_FLOOR_DB = -60.0    # VU meter display floor ("-inf" end of the scale)

# ---------------------------------------------------------------------------
# Visual theme — tema-isolate.md is the visual source of truth.
# Only colors/radii/fonts/spacing live here; layout and logic stay unchanged.
# ---------------------------------------------------------------------------
COL_BG = "#0c0c0e"          # window background
COL_PANEL = "#18181c"       # panels / cards
COL_ELEV = "#1d1d22"        # elevated surface (MASTER row, inputs)
COL_TROUGH = "#0e0e11"      # slider trough / VU background
COL_BORDER = "#26262b"      # default panel border
COL_TEXT = "#ECEAE6"
COL_TEXT_2 = "#8f8d88"
COL_TEXT_DIM = "#7c7973"
AMBER = "#E5A54B"           # exclusive accent: Key/BPM, master, active Solo
AMBER_HOVER = "#F0B562"
AMBER_DIM = "#63512e"       # amber @35%: empty Key/BPM values
CHIP_BG = "#241e15"         # amber @8% over panel
CHIP_BORDER = "#5c4a2b"     # amber @30%
VU_GREEN = "#6fae7c"
VU_AMBER = AMBER
VU_RED = "#d96b4a"
BTN_PRI_BG = "#ECEAE6"      # primary buttons (Play, Separar, Baixar, Exportar)
BTN_PRI_TX = "#161613"
BTN_PRI_HOV = "#ffffff"
BTN_GHOST_BG = "#232328"
BTN_GHOST_BRD = "#2c2c31"
BTN_GHOST_HOV = "#2f2f35"
OK_GREEN = "#6fae7c"        # "Pronto." / file-loaded indicator
MASTER_BORDER = "#4a3d26"   # amber @25%
RADIO_RING = "#55534e"

# Set to Outfit / Spline Sans Mono at startup when installed (tema §3);
# otherwise the spec's fallbacks below are kept.
UI_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"

STEM_LABELS_PT = {"vocals": "Vocais", "drums": "Bateria", "bass": "Baixo",
                  "piano": "Piano", "other": "Outros",
                  "accompaniment": "Acompanhamento"}


def key_short(key: str | None) -> str | None:
    """'A minor' -> 'Am', 'F# major' -> 'F#' (tema §3: letter notation)."""
    if not key:
        return None
    note, _, mode = key.partition(" ")
    return note + ("m" if mode == "minor" else "")


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def find_ffmpeg() -> str | None:
    """Locate ffmpeg: next to a frozen exe, on PATH, or in the app dir."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        for base in (exe_dir, Path(getattr(sys, "_MEIPASS", exe_dir))):
            bundled = base / "ffmpeg.exe"
            if bundled.exists():
                return str(bundled)
    found = shutil.which("ffmpeg")
    if found:
        return found
    # fallback: ffmpeg unpacked under %LOCALAPPDATA%\Isolate\ffmpeg\<build>\bin
    for candidate in sorted((_APPDATA_DIR / "ffmpeg").glob("*/bin/ffmpeg.exe")):
        return str(candidate)
    return None


def _ensure_ffmpeg_on_path() -> None:
    """
    Spleeter and yt-dlp resolve ffmpeg/ffprobe through PATH themselves.
    If ffmpeg is only available via our fallback locations, prepend its
    directory to this process's PATH so those libraries find it too.
    """
    if shutil.which("ffmpeg"):
        return
    ff = find_ffmpeg()
    if ff:
        os.environ["PATH"] = (str(Path(ff).parent) + os.pathsep
                              + os.environ.get("PATH", ""))


_ensure_ffmpeg_on_path()


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------

class Track:
    """One mixer channel: immutable audio data + live mix parameters."""

    __slots__ = ("name", "data", "gain", "mute", "solo")

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data          # float32, shape (n_frames, 2)
        self.gain = 1.0           # linear amplitude multiplier (0.0 .. 1.0)
        self.mute = False
        self.solo = False


class AudioEngine:
    """
    Sample-accurate multi-track playback engine.

    All tracks share a single position pointer and are summed inside one
    sounddevice.OutputStream callback, which guarantees phase-locked
    playback and instantaneous gain changes.
    """

    def __init__(self):
        self.tracks: list[Track] = []
        self.samplerate = 44100
        self.n_frames = 0
        self.device: int | None = None       # None -> system default
        self.master_gain = 1.0               # master fader (0.0 .. 1.0)
        self.levels = np.zeros(0)            # per-track post-fader peak
        self.master_level = 0.0              # post-master peak (pre-clip)
        self._stream: sd.OutputStream | None = None
        self._position = 0
        self._playing = False
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def position_seconds(self) -> float:
        return self._position / self.samplerate if self.samplerate else 0.0

    @property
    def duration_seconds(self) -> float:
        return self.n_frames / self.samplerate if self.samplerate else 0.0

    # -- track management ----------------------------------------------------

    @staticmethod
    def _to_stereo_f32(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        return np.ascontiguousarray(data)

    def set_tracks(self, named_arrays: list[tuple[str, np.ndarray]],
                   samplerate: int) -> None:
        """Replace the whole track set (stops playback, rewinds to zero)."""
        arrays = [self._to_stereo_f32(a) for _, a in named_arrays]
        n = max((len(a) for a in arrays), default=0)
        tracks = []
        for (name, _), a in zip(named_arrays, arrays):
            if len(a) < n:  # pad shorter stems so all share one timeline
                a = np.vstack([a, np.zeros((n - len(a), 2), np.float32)])
            tracks.append(Track(name, a))

        rate_changed = samplerate != self.samplerate
        with self._lock:
            self._playing = False
            self._position = 0
            self.tracks = tracks
            self.n_frames = n
            self.samplerate = samplerate
            self.levels = np.zeros(len(tracks))
            self.master_level = 0.0
        if rate_changed:
            self._close_stream()

    # -- device / stream -----------------------------------------------------

    def set_device(self, device: int | None) -> None:
        self.device = device
        self._close_stream()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            device=self.device,
            channels=2,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()

    # -- transport -------------------------------------------------------------

    def play(self) -> None:
        if not self.tracks:
            return
        if self._position >= self.n_frames:
            self._position = 0
        self._ensure_stream()
        self._playing = True

    def pause(self) -> None:
        self._playing = False
        self.levels[:] = 0.0
        self.master_level = 0.0

    def stop(self) -> None:
        self._playing = False
        self._position = 0
        self.levels[:] = 0.0
        self.master_level = 0.0

    def seek_fraction(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        self._position = int(fraction * self.n_frames)

    def shutdown(self) -> None:
        self._playing = False
        self._close_stream()

    # -- real-time callback ------------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info,
                  status) -> None:
        if status:
            log.debug("stream status: %s", status)
        if not self._playing or not self.tracks:
            outdata.fill(0.0)
            return

        with self._lock:
            pos = self._position
            end = pos + frames
            mix = np.zeros((frames, 2), dtype=np.float32)
            any_solo = any(t.solo for t in self.tracks)
            for i, t in enumerate(self.tracks):
                if t.mute or (any_solo and not t.solo):
                    if i < len(self.levels):
                        self.levels[i] = 0.0
                    continue
                chunk = t.data[pos:end]
                if len(chunk):
                    gained = chunk * t.gain
                    mix[:len(gained)] += gained
                    if i < len(self.levels):    # post-fader peak for VU
                        self.levels[i] = float(np.max(np.abs(gained)))
                elif i < len(self.levels):
                    self.levels[i] = 0.0
            mix *= self.master_gain
            self.master_level = float(np.max(np.abs(mix)))
            # clipping prevention: hard ceiling at full scale
            np.clip(mix, -1.0, 1.0, out=mix)
            outdata[:] = mix
            self._position = min(end, self.n_frames)
            if self._position >= self.n_frames:
                self._playing = False      # reached end: auto-stop

    # -- offline mixdown --------------------------------------------------------

    def render_mix(self) -> np.ndarray:
        """Full-length stereo mixdown honouring gain / mute / solo / master."""
        with self._lock:
            mix = np.zeros((self.n_frames, 2), dtype=np.float32)
            any_solo = any(t.solo for t in self.tracks)
            for t in self.tracks:
                if t.mute or (any_solo and not t.solo):
                    continue
                mix += t.data * t.gain
            mix *= self.master_gain
        np.clip(mix, -1.0, 1.0, out=mix)
        return mix


# ---------------------------------------------------------------------------
# Media helpers (decoding, downloading, separation, encoding)
# ---------------------------------------------------------------------------

class MediaError(RuntimeError):
    """User-facing media processing failure."""


def decode_to_array(path: str, temp_dir: str) -> tuple[np.ndarray, int, str]:
    """
    Decode any supported media file to a float32 array at its native
    sample rate. Returns (data, samplerate, wav_path) where wav_path is a
    WAV file usable as Spleeter input.
    """
    p = Path(path)
    if not p.exists():
        raise MediaError(f"File not found: {path}")

    if p.suffix.lower() in NATIVE_SF_EXTENSIONS:
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            return data, sr, str(p)
        except Exception:
            pass  # fall through to ffmpeg (e.g. exotic WAV codecs)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(
            "ffmpeg was not found on PATH. Install it (e.g. "
            "`winget install Gyan.FFmpeg`) and restart Isolate."
        )
    out_wav = os.path.join(temp_dir, f"decoded_{uuid.uuid4().hex[:12]}.wav")
    cmd = [ffmpeg, "-y", "-i", str(p), "-vn",
           "-acodec", "pcm_f32le", out_wav]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=getattr(subprocess,
                                                "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0 or not os.path.exists(out_wav):
        raise MediaError(
            f"ffmpeg could not decode '{p.name}':\n{proc.stderr[-400:]}"
        )
    data, sr = sf.read(out_wav, dtype="float32", always_2d=True)
    return data, sr, out_wav


def download_youtube(url: str, temp_dir: str,
                     progress) -> tuple[str, str]:
    """
    Download the best available audio stream from a YouTube URL and
    convert it to WAV. Returns (wav_path, title).
    `progress(text)` is called with human-readable status updates.
    """
    from yt_dlp import YoutubeDL   # local import: heavy module

    def hook(d):
        if d.get("status") == "downloading":
            pct = (d.get("_percent_str") or "").strip()
            progress(f"Downloading audio... {pct}")
        elif d.get("status") == "finished":
            progress("Converting download to WAV...")

    base_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "yt_%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 30,
        "overwrites": True,
    }
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        base_opts["ffmpeg_location"] = str(Path(ffmpeg).parent)

    # YouTube intermittently returns HTTP 403 (PO-token / SABR enforcement)
    # for some player clients; retry the whole download with alternative
    # clients before giving up.
    client_attempts: list[dict] = [
        {},                                                     # yt-dlp default
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"extractor_args": {"youtube": {"player_client": ["tv"]}}},
    ]
    info = None
    last_exc: Exception | None = None
    for n, extra in enumerate(client_attempts):
        if n:
            progress(f"Retrying download (attempt {n + 1}/"
                     f"{len(client_attempts)})...")
        try:
            with YoutubeDL({**base_opts, **extra}) as ydl:
                info = ydl.extract_info(url, download=True)
            if info is not None:
                break
        except Exception as exc:
            last_exc = exc
            log.warning("yt-dlp attempt %d failed: %s", n + 1, exc)
    if info is None:
        raise MediaError(
            f"YouTube download failed after {len(client_attempts)} "
            f"attempts: {last_exc}") from last_exc
    if "entries" in info:                       # playlist -> first entry
        info = info["entries"][0]
    wav_path = os.path.join(temp_dir, f"yt_{info['id']}.wav")
    if not os.path.exists(wav_path):
        raise MediaError("Downloaded audio file was not produced.")
    return wav_path, info.get("title") or "YouTube Audio"


_SEPARATOR_CACHE: dict[str, object] = {}


def separate_stems(source_wav: str, model_spec: str, stem_order: list[str],
                   duration_s: float, temp_dir: str,
                   progress) -> list[tuple[str, np.ndarray]]:
    """
    Run Spleeter on `source_wav` and return the stems, in `stem_order`,
    as (display_name, float32 array) pairs at 44.1 kHz.
    """
    progress("Loading separation engine (TensorFlow)...")
    from spleeter.separator import Separator   # heavy import, keep lazy

    sep = _SEPARATOR_CACHE.get(model_spec)
    if sep is None:
        # Guard against a partially-downloaded model: Spleeter skips the
        # download whenever the model directory exists, and TensorFlow then
        # silently runs with UNTRAINED weights (stems come out as the full
        # mix at -6 dB). A completed download always contains ".probe".
        model_dir = _MODELS_DIR / model_spec.split(":", 1)[1].split("-")[0]
        if model_dir.exists() and not (model_dir / ".probe").exists():
            log.warning("Removing broken model directory: %s", model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
        if not (model_dir / ".probe").exists():
            progress("Downloading pretrained model (first run only)...")
        sep = Separator(model_spec, multiprocess=False)
        _SEPARATOR_CACHE[model_spec] = sep

    progress("Separating audio... this can take a few minutes.")
    out_dir = os.path.join(temp_dir, "stems")
    os.makedirs(out_dir, exist_ok=True)
    # default duration kwarg truncates at 600 s — pass the real length
    sep.separate_to_file(source_wav, out_dir, codec="wav",
                         duration=duration_s + 1.0,
                         filename_format="{filename}/{instrument}.{codec}",
                         synchronous=True)

    base = Path(source_wav).stem
    stems = []
    for instrument in stem_order:
        stem_path = os.path.join(out_dir, base, f"{instrument}.wav")
        if not os.path.exists(stem_path):
            raise MediaError(f"Spleeter did not produce '{instrument}.wav'.")
        data, _sr = sf.read(stem_path, dtype="float32", always_2d=True)
        stems.append((STEM_LABELS_PT.get(instrument,
                                         instrument.capitalize()), data))
    return stems


def export_mix(mix: np.ndarray, samplerate: int, out_path: str,
               fmt: str, temp_dir: str) -> None:
    """
    Write the stereo mixdown to disk.
      fmt == "wav": PCM 16-bit, 44.1 kHz.
      fmt == "mp3": 320 kbps CBR (libmp3lame), 44.1 kHz.
    """
    if fmt == "wav" and samplerate == 44100:
        sf.write(out_path, mix, samplerate, subtype="PCM_16")
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        if fmt == "wav":
            # fallback: correct bit depth, native rate (no resampler on hand)
            sf.write(out_path, mix, samplerate, subtype="PCM_16")
            return
        raise MediaError("MP3 export requires ffmpeg on PATH.")

    tmp_wav = os.path.join(temp_dir, "export_master_f32.wav")
    sf.write(tmp_wav, mix, samplerate, subtype="FLOAT")
    if fmt == "wav":
        cmd = [ffmpeg, "-y", "-i", tmp_wav, "-ar", "44100",
               "-c:a", "pcm_s16le", out_path]
    else:
        cmd = [ffmpeg, "-y", "-i", tmp_wav, "-ar", "44100",
               "-c:a", "libmp3lame", "-b:a", "320k", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=getattr(subprocess,
                                                "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0:
        raise MediaError(f"Export failed:\n{proc.stderr[-400:]}")


# ---------------------------------------------------------------------------
# Musical analysis: key & BPM detection
# ---------------------------------------------------------------------------

# Krumhansl-Kessler key profiles (perceptual pitch-class weights, index 0 =
# tonic). Reference: Krumhansl, "Cognitive Foundations of Musical Pitch",
# Oxford University Press, 1990.
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


def _stft_mag(mono: np.ndarray, sr: int, n_fft: int,
              hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude STFT. Returns (frames x bins) and the bin frequencies."""
    n = (len(mono) - n_fft) // hop + 1
    if n <= 0:
        return (np.zeros((0, n_fft // 2 + 1), np.float32),
                np.fft.rfftfreq(n_fft, 1.0 / sr))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx] * np.hanning(n_fft).astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1)), np.fft.rfftfreq(n_fft, 1.0 / sr)


def detect_bpm(mono: np.ndarray, sr: int) -> float | None:
    """
    Tempo estimate in BPM via autocorrelation of the spectral-flux onset
    envelope, searched over 60-200 BPM (folded into 70-180 when ambiguous).
    """
    hop = 512
    mag, _ = _stft_mag(mono, sr, n_fft=1024, hop=hop)
    if len(mag) < 64:
        return None
    flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
    flux -= flux.mean()
    if not np.any(flux):
        return None
    fps = sr / hop                                  # envelope frame rate
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    lag_min = max(2, int(round(60.0 * fps / 200.0)))     # 200 BPM
    lag_max = min(int(round(60.0 * fps / 60.0)), len(ac) - 2)  # 60 BPM
    if lag_max <= lag_min + 2:
        return None
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1]))
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]     # parabolic refinement
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    bpm = 60.0 * fps / (lag + float(np.clip(delta, -0.5, 0.5)))
    if bpm < 70.0:                                  # prefer the double tempo
        half_lag = int(round(lag / 2))
        if half_lag > lag_min and ac[half_lag] >= 0.7 * b:
            bpm *= 2.0
    return round(float(bpm), 1)


def detect_key(mono: np.ndarray, sr: int) -> str | None:
    """
    Musical key estimate ("A minor", "F# major", ...) by correlating the
    average chromagram with the 24 Krumhansl-Kessler key profiles.
    """
    mag, freqs = _stft_mag(mono, sr, n_fft=4096, hop=2048)
    if len(mag) == 0:
        return None
    band = (freqs >= 55.0) & (freqs <= 2000.0)
    pcs = np.round(69.0 + 12.0 * np.log2(freqs[band] / 440.0)).astype(int) % 12
    energy = (mag[:, band] ** 2).mean(axis=0).astype(np.float64)
    chroma = np.bincount(pcs, weights=energy, minlength=12)
    if chroma.sum() <= 0.0 or np.ptp(chroma) == 0.0:
        return None
    best_r, best_name = -2.0, None
    for tonic in range(12):
        for profile, mode in ((KRUMHANSL_MAJOR, "major"),
                              (KRUMHANSL_MINOR, "minor")):
            r = float(np.corrcoef(np.roll(profile, tonic), chroma)[0, 1])
            if r > best_r:
                best_r, best_name = r, f"{PITCH_NAMES[tonic]} {mode}"
    return best_name


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------

class VUMeter(tk.Canvas):
    """
    Classic LED-segment peak meter. Scale: -inf (METER_FLOOR_DB) .. 0 dBFS.
    Green below -9 dBFS, yellow -9..-3, red above -3 (near clipping).
    Peak-hold ballistics: instant attack, smooth release.
    """

    SEGMENTS = 18
    RELEASE_DB_PER_TICK = 3.0     # ~45 dB/s fall time at the UI poll rate

    def __init__(self, master, width: int = 150, height: int = 12):
        super().__init__(master, width=width, height=height,
                         bg=COL_TROUGH, highlightthickness=1,
                         highlightbackground=COL_BORDER)
        self._meter_w, self._meter_h = width, height
        self._disp_db = METER_FLOOR_DB
        self._drawn = -1                     # lit-segment count on canvas
        self._draw(0)

    @staticmethod
    def _seg_color(seg_db: float, lit: bool) -> str:
        if seg_db >= -3.0:
            return VU_RED if lit else "#33201b"        # red / dim red
        if seg_db >= -9.0:
            return VU_AMBER if lit else "#332a18"      # amber / dim amber
        return VU_GREEN if lit else "#1d2b20"          # green / dim green

    def _draw(self, lit_count: int) -> None:
        if lit_count == self._drawn:
            return
        self._drawn = lit_count
        self.delete("all")
        gap = 2
        seg_w = (self._meter_w - gap) / self.SEGMENTS
        for i in range(self.SEGMENTS):
            # dB value this segment represents (left = floor, right = 0 dBFS)
            seg_db = METER_FLOOR_DB * (1.0 - (i + 1) / self.SEGMENTS)
            x0 = 2 + i * seg_w
            self.create_rectangle(
                x0, 2, x0 + seg_w - gap, self._meter_h,
                fill=self._seg_color(seg_db, i < lit_count), width=0)

    def set_level(self, linear_peak: float) -> None:
        db = (20.0 * np.log10(linear_peak)
              if linear_peak > 1e-9 else METER_FLOOR_DB)
        db = min(0.0, max(METER_FLOOR_DB, db))
        self._disp_db = max(db, self._disp_db - self.RELEASE_DB_PER_TICK)
        frac = 1.0 - self._disp_db / METER_FLOOR_DB   # 0 at floor, 1 at 0 dB
        self._draw(int(round(frac * self.SEGMENTS)))


class TrackRow(ctk.CTkFrame):
    """One mixer channel strip: name, VU meter, volume slider, Mute, Solo."""

    def __init__(self, master, track: Track, on_change):
        super().__init__(master, fg_color=COL_ELEV, corner_radius=18)
        self.track = track
        self._on_change = on_change

        self.grid_columnconfigure(2, weight=1)

        self.name_label = ctk.CTkLabel(
            self, text=track.name, width=110, anchor="w",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=14, weight="bold"))
        self.name_label.grid(row=0, column=0, padx=(22, 8), pady=13,
                             sticky="w")

        self.meter = VUMeter(self)
        self.meter.grid(row=0, column=1, padx=(0, 8))

        self.slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=100,
            fg_color=COL_TROUGH, progress_color=COL_TEXT,
            button_color=COL_TEXT, button_hover_color=BTN_PRI_HOV,
            corner_radius=3, height=16,
            command=self._on_slider)
        self.slider.set(track.gain * 100.0)
        self.slider.grid(row=0, column=2, padx=8, pady=13, sticky="ew")

        self.value_label = ctk.CTkLabel(
            self, text="100%", width=52, anchor="e",
            text_color=COL_TEXT_2,
            font=ctk.CTkFont(family=MONO_FAMILY, size=13))
        self.value_label.grid(row=0, column=3, padx=(0, 10))

        self.mute_btn = ctk.CTkButton(
            self, text="M", width=28, height=28, corner_radius=14,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            hover_color=BTN_GHOST_HOV, command=self._toggle_mute,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.mute_btn.grid(row=0, column=4, padx=4, pady=13)

        self.solo_btn = ctk.CTkButton(
            self, text="S", width=28, height=28, corner_radius=14,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            hover_color=BTN_GHOST_HOV, command=self._toggle_solo,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.solo_btn.grid(row=0, column=5, padx=(4, 22), pady=13)

        self._refresh_value_label()

    def _on_slider(self, value: float) -> None:
        self.track.gain = float(value) / 100.0     # linear amplitude map
        self._refresh_value_label()

    def _refresh_value_label(self) -> None:
        self.value_label.configure(text=f"{int(round(self.track.gain * 100))}%")

    def _toggle_mute(self) -> None:
        self.track.mute = not self.track.mute
        # tema §5: active M = warm-white bg with dark text
        self.mute_btn.configure(
            fg_color=BTN_PRI_BG if self.track.mute else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.mute else COL_TEXT_2)
        self._on_change()

    def _toggle_solo(self) -> None:
        self.track.solo = not self.track.solo
        # tema §5: active S = amber bg with dark text
        self.solo_btn.configure(
            fg_color=AMBER if self.track.solo else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.solo else COL_TEXT_2)
        self._on_change()


class MasterRow(ctk.CTkFrame):
    """
    Master bus strip — the sum of every stem, like the master fader of an
    analog console. Always present; only the stem rows below it change
    with the selected separation model.
    """

    def __init__(self, master, engine: AudioEngine):
        super().__init__(master, fg_color=COL_ELEV, corner_radius=18,
                         border_width=1, border_color=MASTER_BORDER)
        self.engine = engine
        self.grid_columnconfigure(2, weight=1)

        self.name_label = ctk.CTkLabel(
            self, text="MASTER", width=110, anchor="w",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=15, weight="bold"))
        self.name_label.grid(row=0, column=0, padx=(22, 8), pady=13,
                             sticky="w")

        self.meter = VUMeter(self, width=150, height=14)
        self.meter.grid(row=0, column=1, padx=(0, 8))

        self.slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=100,
            fg_color=COL_TROUGH, progress_color=AMBER,
            button_color=AMBER, button_hover_color=AMBER_HOVER,
            corner_radius=3, height=16,
            command=self._on_slider)
        self.slider.set(100.0)
        self.slider.grid(row=0, column=2, padx=8, pady=13, sticky="ew")

        self.value_label = ctk.CTkLabel(
            self, text="100%", width=52, anchor="e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=MONO_FAMILY, size=13, weight="bold"))
        self.value_label.grid(row=0, column=3, padx=(0, 22))

    def _on_slider(self, value: float) -> None:
        self.engine.master_gain = float(value) / 100.0
        self.value_label.configure(text=f"{int(round(float(value)))}%")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

if _HAS_DND:
    class _Root(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class _Root(ctk.CTk):
        pass


class IsolateApp(_Root):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # tema §3: Outfit / Spline Sans Mono when installed, else fallbacks
        global UI_FAMILY, MONO_FAMILY
        try:
            import tkinter.font as tkfont
            families = set(tkfont.families(self))
            if "Outfit" in families:
                UI_FAMILY = "Outfit"
            if "Spline Sans Mono" in families:
                MONO_FAMILY = "Spline Sans Mono"
        except Exception:
            pass

        self.title("Isolate — Stem Splitter & Multi-Track Mixer")
        self.geometry("980x720")
        self.minsize(860, 600)
        self.configure(fg_color=COL_BG)
        try:                                   # tema §6: Venn logo icon
            base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
            icon = base / "isolate.ico"
            if icon.exists():
                self.iconbitmap(str(icon))
        except Exception:
            pass

        self.engine = AudioEngine()
        self.temp_dir = tempfile.mkdtemp(prefix="isolate_")
        atexit.register(self._cleanup_temp)

        self.source_wav: str | None = None      # Spleeter input file
        self.source_title = ""
        self._busy = False
        self._seeking = False
        self._analyzing = False
        self._status_queue: queue.Queue[str] = queue.Queue()
        self.track_rows: list[TrackRow] = []

        self._build_ui()
        self._populate_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<space>", self._on_space)   # space = play/pause
        self.after(UI_POLL_MS, self._poll)

        if not find_ffmpeg():
            self._set_status(
                "Warning: ffmpeg not found on PATH — MP3/M4A/MP4 loading, "
                "YouTube download and MP3 export will not work.")

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # ---------- Top bar ----------
        top = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                           border_width=1, border_color=COL_BORDER)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

        # Drag & drop landing zone (tema §5: solid 1.5px border, radius 22)
        drop_text = ("Arraste um arquivo de áudio aqui\n.wav  .mp3  .m4a  .mp4"
                     if _HAS_DND else
                     "Clique para escolher um arquivo de áudio\n"
                     ".wav  .mp3  .m4a  .mp4")
        self.drop_zone = ctk.CTkFrame(
            top, corner_radius=22, fg_color=COL_TROUGH,
            border_width=2, border_color=BTN_GHOST_BRD)
        self.drop_zone.grid(row=0, column=0, sticky="ew",
                            padx=(10, 6), pady=10)
        self.drop_label = ctk.CTkLabel(
            self.drop_zone, text=drop_text, height=64,
            text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=13))
        self.drop_label.pack(fill="both", expand=True, padx=6, pady=2)
        for w in (self.drop_zone, self.drop_label):
            w.bind("<Button-1>", lambda e: self._browse_file())
            if _HAS_DND:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop)

        # YouTube URL input
        url_frame = ctk.CTkFrame(top, fg_color="transparent")
        url_frame.grid(row=0, column=1, sticky="ew", padx=(6, 10), pady=10)
        url_frame.grid_columnconfigure(0, weight=1)
        self.url_entry = ctk.CTkEntry(
            url_frame, placeholder_text="URL do YouTube...", height=36,
            corner_radius=999, fg_color=COL_ELEV,
            border_width=1, border_color=BTN_GHOST_BRD,
            text_color=COL_TEXT, placeholder_text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=13))
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.download_btn = ctk.CTkButton(
            url_frame, text="Baixar & Carregar", width=140, height=36,
            corner_radius=999, fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_download)
        self.download_btn.grid(row=0, column=1)

        # Device selector + transport
        row2 = ctk.CTkFrame(top, fg_color="transparent")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew",
                  padx=10, pady=(0, 10))
        row2.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(row2, text="Saída:", text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12)
                     ).grid(row=0, column=0, padx=(0, 6))
        self.device_menu = ctk.CTkOptionMenu(
            row2, values=["System default"], width=300, corner_radius=999,
            dynamic_resizing=False,     # long device names must NOT widen
                                        # the menu and push the transport
                                        # buttons out of the window
            fg_color=COL_ELEV, button_color=COL_ELEV,
            button_hover_color=BTN_GHOST_HOV, text_color=COL_TEXT,
            dropdown_fg_color=COL_ELEV, dropdown_text_color=COL_TEXT,
            dropdown_hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=12),
            command=self._on_device_selected)
        self.device_menu.grid(row=0, column=1, padx=(0, 16))

        self.play_btn = ctk.CTkButton(
            row2, text="▶", width=46, height=46, corner_radius=23,
            fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=16, weight="bold"),
            command=self._on_play)
        self.play_btn.grid(row=0, column=2, padx=3)
        self.pause_btn = ctk.CTkButton(
            row2, text="⏸", width=40, height=40, corner_radius=20,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=14),
            command=self._on_pause)
        self.pause_btn.grid(row=0, column=3, padx=3)
        self.stop_btn = ctk.CTkButton(
            row2, text="■", width=40, height=40, corner_radius=20,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=14),
            command=self._on_stop)
        self.stop_btn.grid(row=0, column=4, padx=3)

        self.time_label = ctk.CTkLabel(row2, text="00:00 / 00:00",
                                       width=110, anchor="e",
                                       text_color=COL_TEXT,
                                       font=ctk.CTkFont(size=14,
                                                        family=MONO_FAMILY))
        self.time_label.grid(row=0, column=6, padx=(8, 0))

        # Timeline slider (full width)
        self.timeline = ctk.CTkSlider(top, from_=0.0, to=1.0,
                                      number_of_steps=2000,
                                      fg_color=COL_TROUGH,
                                      progress_color=COL_TEXT,
                                      button_color=COL_TEXT,
                                      button_hover_color=BTN_PRI_HOV,
                                      corner_radius=3, height=16,
                                      command=self._on_timeline_drag)
        self.timeline.set(0.0)
        self.timeline.grid(row=2, column=0, columnspan=2, sticky="ew",
                           padx=10, pady=(0, 12))
        self.timeline.bind("<Button-1>", self._on_timeline_press)
        self.timeline.bind("<ButtonRelease-1>", self._on_timeline_release)

        # ---------- Configuration panel ----------
        cfg = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                           border_width=1, border_color=COL_BORDER)
        cfg.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        cfg.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(cfg, text="M O D O   D E   S E P A R A Ç Ã O",
                     text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                      weight="bold")
                     ).grid(row=0, column=0, rowspan=3,
                            padx=(24, 18), pady=10, sticky="w")

        self.stem_var = ctk.StringVar(
            value="4 Stems (Vocals / Drums / Bass / Other)")
        for i, label in enumerate(STEM_MODELS):
            ctk.CTkRadioButton(cfg, text=label, value=label,
                               variable=self.stem_var,
                               fg_color=COL_TEXT, hover_color=COL_TEXT,
                               border_color=RADIO_RING,
                               text_color=COL_TEXT,
                               font=ctk.CTkFont(family=UI_FAMILY, size=13)
                               ).grid(row=i, column=1, sticky="w",
                                      padx=4, pady=3)

        self.separate_btn = ctk.CTkButton(
            cfg, text="Separar Faixas", height=48, width=190,
            corner_radius=999, fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=15, weight="bold"),
            command=self._on_separate)
        self.separate_btn.grid(row=0, column=3, rowspan=3,
                               padx=24, pady=10, sticky="e")

        # ---------- Musical analysis panel ----------
        ana = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                           border_width=1, border_color=CHIP_BORDER)
        ana.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        ana.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(ana, text="A N Á L I S E   M U S I C A L",
                     text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=(24, 16), pady=8)

        # tema §5: two chips (Tom | BPM) — small amber caption, big value
        key_chip = ctk.CTkFrame(ana, fg_color=CHIP_BG, corner_radius=18,
                                border_width=1, border_color=CHIP_BORDER)
        key_chip.grid(row=0, column=1, padx=(0, 12), pady=8)
        ctk.CTkLabel(key_chip, text="TOM", text_color=CHIP_BORDER,
                     font=ctk.CTkFont(family=UI_FAMILY, size=10,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=(16, 16), pady=(5, 0))
        self.key_label = ctk.CTkLabel(
            key_chip, text="—", width=64, text_color=AMBER_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=21, weight="bold"))
        self.key_label.grid(row=1, column=0, padx=(16, 16), pady=(0, 6))

        bpm_chip = ctk.CTkFrame(ana, fg_color=CHIP_BG, corner_radius=18,
                                border_width=1, border_color=CHIP_BORDER)
        bpm_chip.grid(row=0, column=2, padx=(0, 20), pady=8)
        ctk.CTkLabel(bpm_chip, text="BPM", text_color=CHIP_BORDER,
                     font=ctk.CTkFont(family=UI_FAMILY, size=10,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=(16, 16), pady=(5, 0))
        self.bpm_label = ctk.CTkLabel(
            bpm_chip, text="—", width=64, text_color=AMBER_DIM,
            font=ctk.CTkFont(family=MONO_FAMILY, size=21, weight="bold"))
        self.bpm_label.grid(row=1, column=0, padx=(16, 16), pady=(0, 6))

        self.analyze_btn = ctk.CTkButton(
            ana, text="Detectar Tom & BPM", width=160, height=32,
            corner_radius=999, fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_analyze)
        self.analyze_btn.grid(row=0, column=4, padx=24, pady=8, sticky="e")

        # ---------- Mixer ----------
        self.mixer = ctk.CTkScrollableFrame(
            self, corner_radius=22, fg_color=COL_PANEL,
            label_text="M I X E R", label_fg_color="transparent",
            label_text_color=COL_TEXT_2,
            label_font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                   weight="bold"))
        self.mixer.grid(row=3, column=0, sticky="nsew", padx=12, pady=6)
        self.mixer.grid_columnconfigure(0, weight=1)

        self.master_row = MasterRow(self.mixer, self.engine)
        self.master_row.grid(row=0, column=0, sticky="ew", padx=6,
                             pady=(4, 10))

        self.mixer_hint = ctk.CTkLabel(
            self.mixer, text="Carregue um arquivo de áudio para começar.",
            text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=13))
        self.mixer_hint.grid(row=1, column=0, pady=30)

        # ---------- Bottom bar ----------
        bottom = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                              border_width=1, border_color=COL_BORDER)
        bottom.grid(row=4, column=0, sticky="ew", padx=12, pady=(6, 12))
        bottom.grid_columnconfigure(2, weight=1)

        self.export_btn = ctk.CTkButton(
            bottom, text="Exportar Mix", width=130, height=36,
            corner_radius=999, fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_export)
        self.export_btn.grid(row=0, column=0, padx=(14, 8), pady=10)

        self.format_toggle = ctk.CTkSegmentedButton(
            bottom, values=["WAV  16-bit / 44.1 kHz", "MP3  320 kbps CBR"],
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=MONO_FAMILY, size=12))
        self.format_toggle.set("WAV  16-bit / 44.1 kHz")
        self.format_toggle.grid(row=0, column=1, padx=8, pady=10)

        self.status_label = ctk.CTkLabel(
            bottom, text="●  Pronto.", anchor="e", text_color=OK_GREEN,
            font=ctk.CTkFont(family=UI_FAMILY, size=12))
        self.status_label.grid(row=0, column=2, sticky="e",
                               padx=(8, 18), pady=10)

        # tema §5: mandatory educational note — own row so it never
        # collides with the status text; everything else stays put
        ctk.CTkLabel(
            bottom,
            text="Ferramenta educacional para separação de instrumentos "
                 "e análise musical. Distribuição gratuita.",
            text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=12)
        ).grid(row=1, column=0, columnspan=3, pady=(0, 8))

    # ------------------------------------------------------ device handling --

    def _populate_devices(self) -> None:
        self._device_map: dict[str, int | None] = {"System default": None}
        labels = ["System default"]
        try:
            hostapis = sd.query_hostapis()
            for idx, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] < 2:
                    continue
                api = hostapis[dev["hostapi"]]["name"]
                name = dev["name"][:40]     # keep labels dropdown-friendly
                label = (f"{idx}: {name} — {api} "
                         f"({dev['default_samplerate']:.0f} Hz)")
                self._device_map[label] = idx
                labels.append(label)
        except Exception as exc:
            log.warning("Could not enumerate audio devices: %s", exc)
        self.device_menu.configure(values=labels)
        self.device_menu.set("System default")

    def _on_device_selected(self, label: str) -> None:
        was_playing = self.engine.playing
        self.engine.pause()
        self.engine.set_device(self._device_map.get(label))
        if was_playing:
            try:
                self.engine.play()
            except Exception as exc:
                self._set_status(f"Device error: {exc}")

    # ------------------------------------------------------------ transport --

    def _on_play(self) -> None:
        """Main transport button: toggles between play and pause."""
        if self.engine.playing:
            self.engine.pause()
            return
        try:
            self.engine.play()
        except Exception as exc:
            self._set_status(f"Playback error: {exc}")
            self.engine.set_device(None)

    def _on_space(self, event):
        """Space bar = play/pause, except while typing in a text field."""
        widget = self.focus_get()
        if isinstance(widget, tk.Entry):
            return None
        self._on_play()
        return "break"

    def _on_pause(self) -> None:
        self.engine.pause()

    def _on_stop(self) -> None:
        self.engine.stop()

    def _on_timeline_press(self, _event) -> None:
        self._seeking = True

    def _on_timeline_drag(self, value: float) -> None:
        if self._seeking:
            pos = float(value) * self.engine.duration_seconds
            self.time_label.configure(
                text=f"{format_time(pos)} / "
                     f"{format_time(self.engine.duration_seconds)}")

    def _on_timeline_release(self, _event) -> None:
        self.engine.seek_fraction(float(self.timeline.get()))
        self._seeking = False

    # ------------------------------------------------------------- loading --

    def _on_drop(self, event) -> None:
        paths = self.tk.splitlist(event.data)
        if paths:
            self._load_file(paths[0])

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open audio file",
            filetypes=[("Audio / video", "*.wav *.mp3 *.m4a *.mp4"),
                       ("All files", "*.*")])
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        if self._busy:
            return
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            messagebox.showwarning(
                _APP_NAME,
                f"Unsupported file type '{ext}'. "
                "Supported: .wav, .mp3, .m4a, .mp4")
            return
        self._run_async(self._task_load_file, path)

    def _task_load_file(self, path: str) -> None:
        self._status_async(f"Loading '{Path(path).name}'...")
        data, sr, wav_path = decode_to_array(path, self.temp_dir)
        self.source_wav = wav_path
        self.source_title = Path(path).stem
        self.after(0, lambda: self._install_tracks(
            [("Áudio Original", data)], sr,
            f"Loaded '{Path(path).name}' — {sr} Hz, "
            f"{format_time(len(data) / sr)}."))

    def _on_download(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self._set_status("Paste a YouTube URL first.")
            return
        if self._busy:
            return
        self._run_async(self._task_download, url)

    def _task_download(self, url: str) -> None:
        wav_path, title = download_youtube(url, self.temp_dir,
                                           self._status_async)
        self._status_async("Loading downloaded audio...")
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        self.source_wav = wav_path
        self.source_title = title
        self.after(0, lambda: self._install_tracks(
            [("Áudio Original", data)], sr,
            f"Loaded '{title}' — {sr} Hz, {format_time(len(data) / sr)}."))

    # ----------------------------------------------------------- separation --

    def _on_separate(self) -> None:
        if self._busy:
            return
        if not self.source_wav:
            self._set_status("Load an audio file or YouTube URL first.")
            return
        model_spec, stem_order = STEM_MODELS[self.stem_var.get()]
        self._run_async(self._task_separate, model_spec, stem_order)

    def _task_separate(self, model_spec: str, stem_order: list[str]) -> None:
        duration = self.engine.duration_seconds
        stems = separate_stems(self.source_wav, model_spec, stem_order,
                               duration, self.temp_dir, self._status_async)
        sr = 44100    # Spleeter models always render at 44.1 kHz
        self.after(0, lambda: self._install_tracks(
            stems, sr,
            f"Separation complete — {len(stems)} stems ready."))

    # -------------------------------------------------------------- export --

    def _on_export(self) -> None:
        if self._busy:
            return
        if not self.engine.tracks:
            self._set_status("Nothing to export — load audio first.")
            return
        fmt = "mp3" if self.format_toggle.get().startswith("MP3") else "wav"
        default_name = (self.source_title or "isolate_mix") + f"_mix.{fmt}"
        out_path = filedialog.asksaveasfilename(
            title="Export mix",
            initialfile=default_name,
            defaultextension=f".{fmt}",
            filetypes=[("WAV file", "*.wav")] if fmt == "wav"
                      else [("MP3 file", "*.mp3")])
        if not out_path:
            return
        self._run_async(self._task_export, fmt, out_path)

    def _task_export(self, fmt: str, out_path: str) -> None:
        self._status_async("Rendering mixdown...")
        mix = self.engine.render_mix()
        self._status_async(f"Encoding {fmt.upper()}...")
        export_mix(mix, self.engine.samplerate, out_path, fmt,
                   self.temp_dir)
        self._status_async(f"Exported: {out_path}")

    # ------------------------------------------------------------ mixer UI --

    def _install_tracks(self, named_arrays: list[tuple[str, np.ndarray]],
                        samplerate: int, status: str) -> None:
        self.engine.set_tracks(named_arrays, samplerate)
        for row in self.track_rows:
            row.destroy()
        self.track_rows.clear()
        self.mixer_hint.grid_remove()
        # master strip stays fixed at row 0; stem rows rebuild below it
        for i, track in enumerate(self.engine.tracks):
            row = TrackRow(self.mixer, track, on_change=lambda: None)
            row.grid(row=i + 2, column=0, sticky="ew", padx=6, pady=4)
            self.track_rows.append(row)
        self.timeline.set(0.0)
        self._set_status(status)
        self._start_analysis()

    # ------------------------------------------------------ music analysis --

    def _on_analyze(self) -> None:
        if not self.engine.tracks:
            self._set_status("Load an audio file first.")
            return
        self._start_analysis()

    def _start_analysis(self) -> None:
        """Detect key & BPM of the loaded material on a background thread."""
        if self._analyzing or not self.engine.tracks:
            return
        self._analyzing = True
        self.key_label.configure(text="…", text_color=AMBER_DIM)
        self.bpm_label.configure(text="…", text_color=AMBER_DIM)

        sr = self.engine.samplerate
        n = self.engine.n_frames
        tracks = list(self.engine.tracks)

        def work():
            try:
                # analyse up to 60 s from the middle of the FULL mix
                # (stems summed back together, faders ignored on purpose)
                span = min(n, 60 * sr)
                start = max(0, (n - span) // 2)
                mono = np.zeros(span, dtype=np.float32)
                for t in tracks:
                    mono += t.data[start:start + span].mean(axis=1)
                bpm = detect_bpm(mono, sr)
                key = detect_key(mono, sr)
                key_txt = key_short(key) or "—"     # tema §3: Am, C, F#m...
                bpm_txt = f"{bpm:.0f}" if bpm else "—"
                self.after(0, lambda: (
                    self.key_label.configure(
                        text=key_txt,
                        text_color=AMBER if key else AMBER_DIM),
                    self.bpm_label.configure(
                        text=bpm_txt,
                        text_color=AMBER if bpm else AMBER_DIM)))
                if key or bpm:
                    self._status_async(
                        f"Analysis: {key or '?'}, {bpm or '?'} BPM.")
            except Exception:
                log.error("Analysis failed:\n%s", traceback.format_exc())
                self.after(0, lambda: (
                    self.key_label.configure(text="—",
                                             text_color=AMBER_DIM),
                    self.bpm_label.configure(text="—",
                                             text_color=AMBER_DIM)))
            finally:
                self._analyzing = False

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------- async plumbing --

    def _run_async(self, fn, *args) -> None:
        """Run `fn(*args)` on a worker thread with busy-state handling."""
        self._set_busy(True)

        def wrapper():
            try:
                fn(*args)
            except MediaError as exc:
                self._status_async(f"Error: {exc}")
            except Exception as exc:
                log.error("Worker failed:\n%s", traceback.format_exc())
                self._status_async(f"Unexpected error: {exc}")
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=wrapper, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self.separate_btn, self.download_btn, self.export_btn):
            btn.configure(state=state)

    def _status_async(self, text: str) -> None:
        """Thread-safe status update (called from worker threads)."""
        self._status_queue.put(text)

    def _set_status(self, text: str) -> None:
        ok = text.startswith(("Loaded", "Exported", "Separation complete",
                              "Analysis", "●"))
        self.status_label.configure(
            text=text, text_color=OK_GREEN if ok else COL_TEXT_2)
        log.info(text)

    # ------------------------------------------------------------- polling --

    def _poll(self) -> None:
        # drain worker status messages
        try:
            while True:
                self._set_status(self._status_queue.get_nowait())
        except queue.Empty:
            pass

        # transport readout
        dur = self.engine.duration_seconds
        pos = self.engine.position_seconds
        if not self._seeking:
            self.time_label.configure(
                text=f"{format_time(pos)} / {format_time(dur)}")
            self.timeline.set(pos / dur if dur > 0 else 0.0)

        # VU meters (post-fader per track, post-master on the master bus)
        playing = self.engine.playing
        levels = self.engine.levels
        for i, row in enumerate(self.track_rows):
            row.meter.set_level(
                float(levels[i]) if playing and i < len(levels) else 0.0)
        self.master_row.meter.set_level(
            self.engine.master_level if playing else 0.0)

        # main transport button doubles as play/pause toggle
        self.play_btn.configure(
            text="⏸" if playing else "▶",
            fg_color=BTN_PRI_HOV if playing else BTN_PRI_BG)
        self.after(UI_POLL_MS, self._poll)

    # ------------------------------------------------------------- shutdown --

    def _cleanup_temp(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _on_close(self) -> None:
        self.engine.shutdown()
        self._cleanup_temp()
        self.destroy()


def main() -> None:
    app = IsolateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
