# Role & Objective
You are an expert Senior Audio Software Engineer, DSP Architect, and Systems Specialist. Your objective is to create a complete, production-ready desktop application named "Isolate" for Windows 10/11. The core utility is local high-fidelity audio source separation (stem splitting) with a multi-track DAW-like playback interface and real-time volume mixing designed specifically for professional musical use.

/goal Deliver a single, standalone, production-ready Windows executable (.exe) installer or package for Windows 10/11 that runs locally without external dependencies (except for python/ffmpeg during build), ensuring a complete code generation process with zero placeholders or unfulfilled functions.

---

## 1. Technical Stack & Architecture
* **Frontend/GUI:** Python with `customtkinter` (for a modern, dark, high-end DAW/plugin aesthetics) or PySide6.
* **Audio Separation Engine:** `spleeter` (Deezer) via its Python API configured explicitly for professional high-quality output (using high-sample-rate configurations, 44.1kHz models, and uncompressed intermediate processing). Support for 2stems, 4stems, and 5stems models.
* **Audio Downloader:** `yt-dlp` integrated natively to handle YouTube URL inputs, extracting the highest available audio bitrate.
* **Multi-Track Audio Engine:** `sounddevice` + `numpy` (crucial for real-time, sample-accurate synchronized multi-track playback, ultra-low latency, and real-time linear gain/volume adjustment).
* **Packaging:** `PyInstaller` tailored for Windows packaging, handling complex TensorFlow binaries.

---

## 2. Core Features & UI Requirements

### UI Layout (Professional DAW Style)
* **Top Bar:** 
    * Drag & Drop landing zone for audio files.
    * URL Input field for YouTube links + `[Download & Load]` button.
    * Audio Output Device selector (Dropdown populating active system audio outputs and sample rates using `sounddevice.query_devices()`).
    * Global Transport Controls: `[Play]`, `[Pause]`, `[Stop]`, and a visual Time/Position Timeline Slider.
* **Configuration Panel:** Radio buttons to select the separation mode:
    * 2 Stems: Vocals / Accompaniment
    * 4 Stems: Vocals / Drums / Bass / Other
    * 5 Stems: Vocals / Drums / Bass / Piano / Other
    * A prominent `[Separate Tracks]` action button.
* **Main Mixer Section (Dynamic Tracks):**
    * Before separation: Displays a single master track (Original Audio Track).
    * After separation: Dynamically updates to show N tracks (depending on the chosen stem model).
    * Each track row must contain:
        * Track Name label (e.g., "Vocals", "Drums") with clear typography.
        * A precise horizontal or vertical Volume Slider (0% to 100% linear gain map).
        * `[Mute]` and `[Solo]` buttons that function exactly like a standard DAW matrix.
* **Bottom Bar:** `[Export Mix]` button with format toggle (`.wav` [Uncompressed 16-bit 44.1kHz] / `.mp3` [High Quality 320kbps CBR]) and status bar indicator.

---

## 3. Technical Specifications & High-Fidelity Audio Pipeline

### A. High-Quality Default & File Ingestion
* **High-Fidelity Standard:** The application must default to studio-quality audio pipeline constraints. All sample rate operations must respect a baseline of 44.1kHz or 48kHz matching the native source.
* **Files:** Accept `.wav`, `.mp3`, `.m4a`, and `.mp4` (extract high-bitrate audio stream).
* **YouTube:** Use `yt-dlp` to download the audio stream directly at maximum quality (bestaudio) into a temporary directory in uncompressed `.wav` format, then load it into the app memory.

### B. Professional Spleeter Integration
* Programmatically call `spleeter.separator.Separator` with the selected configuration, enforcing high-bitrate/high-frequency models (e.g., ensuring 16kHz or 22kHz spectral bandwidth configurations rather than heavily compressed low-passed variants).
* Run the separation inside a isolated background `threading.Thread` or `QThread` to prevent the UI from freezing. Update the status bar dynamically (e.g., "Downloading model...", "Separating audio...").

### C. Synchronized Multi-Track Playback (The DAW Engine)
* **The Problem:** Playing multiple audio files simultaneously using standard media players causes phase drifting and hardware desynchronization.
* **The Solution:** Load all separated stem audio files into memory as floating-point `numpy` arrays (`float32`). 
* Create a single unified stereo audio output stream using `sounddevice.OutputStream` operating at the native hardware sample rate (e.g., 44100Hz).
* Inside the real-time audio stream callback function, multiply each track's frame data by its corresponding GUI volume slider value (converted to a linear amplitude multiplier), sum the arrays together, apply a soft limiter or clipping prevention mechanism (`np.clip` between -1.0 and 1.0), and feed the combined buffer to the output device. This ensures 100% phase-accurate synchronization and instantaneous, real-time volume mixing.

### D. Professional Exporting
* When the user clicks `[Export Mix]`, the app must process the numpy arrays of each track, apply the current volume/gain, mute, and solo levels set by the sliders, mix them into a single stereo master array, and save it to the user's chosen location using `soundfile` or `pydub`.
* Export formats must maintain musical fidelity: `.wav` (PCM 16-bit/44.1kHz) or `.mp3` (320kbps Constant Bitrate).

---

## 4. Robustness & Threading
* All heavy computations (Downloading, Spleeter inference, File Exporting) must run asynchronously.
* Ensure proper cleanup of temporary audio files on application close.
* Gracefully handle cases where the Spleeter backend downloads its pre-trained models on the first run, informing the user with an explicit progress status.

---

## 5. Delivery Requirements (Long-Horizon Execution)

Please output the complete project structure:
1.  `requirements.txt` listing all explicit dependencies (including `spleeter`, `yt-dlp`, `sounddevice`, `numpy`, `soundfile`, UI libraries).
2.  `main.py` containing the entire, fully-implemented, non-placeholded Python application source code.
3.  A `build.bat` script using `PyInstaller` that includes all hidden imports and binary dependencies required by `spleeter` (TensorFlow backend dependencies) and audio I/O libraries.

Provide concise step-by-step instructions on how to set up the environment, install ffmpeg (required by Spleeter/yt-dlp), run the app locally, and compile it into the final Windows Executable. Do not use `# TODO` or omit any audio callback logic.