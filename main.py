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
import json
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
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Environment setup (must happen before TensorFlow / Spleeter are imported)
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence TF chatter

_APP_NAME = "Isolate"
APP_VERSION = "2.3"
_APPDATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / _APP_NAME
_MODELS_DIR = _APPDATA_DIR / "pretrained_models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
# Spleeter reads the MODEL_PATH env var to decide where pretrained models live.
os.environ.setdefault("MODEL_PATH", str(_MODELS_DIR))
# PyTorch/Demucs cache (hub checkpoints) also stays under our app folder.
_TORCH_DIR = _APPDATA_DIR / "torch"
os.environ.setdefault("TORCH_HOME", str(_TORCH_DIR))

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

# ---------------------------------------------------------------------------
# Internationalization (PT-BR / EN) — user-selectable, persisted in
# %LOCALAPPDATA%\Isolate\settings.json, applied on startup.
# ---------------------------------------------------------------------------

_SETTINGS_FILE = _APPDATA_DIR / "settings.json"


def _load_settings() -> dict:
    try:
        import json
        return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(settings: dict) -> None:
    try:
        import json
        _SETTINGS_FILE.write_text(json.dumps(settings, indent=2),
                                  encoding="utf-8")
    except Exception:
        log.warning("Could not save settings:\n%s", traceback.format_exc())


_SETTINGS = _load_settings()
LANG = _SETTINGS.get("language", "pt")
if LANG not in ("pt", "en"):
    LANG = "pt"

I18N: dict[str, dict[str, str]] = {
    "pt": {
        "drop_dnd": "Arraste um arquivo de áudio aqui\n.wav  .mp3  .m4a  .mp4",
        "drop_click": "Clique para escolher um arquivo de áudio\n"
                      ".wav  .mp3  .m4a  .mp4",
        "url_placeholder": "URL do YouTube...",
        "btn_download": "Baixar & Carregar",
        "lbl_output": "Saída:",
        "device_default": "Padrão do sistema",
        "lbl_sep_mode": "M O D O   D E   S E P A R A Ç Ã O",
        "stems2": "2 Stems — Demucs (Vocais / Acompanhamento)",
        "stems4": "4 Stems (Vocais / Bateria / Baixo / Outros)",
        "stems5": "5 Stems (Vocais / Bateria / Baixo / Piano / Outros)",
        "stems6": "6 Stems — Demucs (com Guitarra; mais lento)",
        # compact mode picker (v2.1.1): short label on the pill, full stem
        # list on the caption line beside it
        "stems2_short": "2 Stems",
        "stems4_short": "4 Stems",
        "stems5_short": "5 Stems",
        "stems6_short": "6 Stems",
        "stems2_desc": "Vocais / Acompanhamento — voz mais limpa, "
                       "mais lento",
        "stems4_desc": "Vocais / Bateria / Baixo / Outros",
        "stems5_desc": "Vocais / Bateria / Baixo / Piano / Outros",
        "stems6_desc": "Demucs: Vocais / Bateria / Baixo / Guitarra / "
                       "Piano / Outros — mais lento",
        "btn_separate": "Separar Faixas",
        "lbl_analysis": "A N Á L I S E   M U S I C A L",
        "chip_key": "TOM",
        "btn_analyze": "Detectar Tom & BPM",
        "mixer_hint": "Carregue um arquivo de áudio para começar.",
        "mixer_title": "M I X E R",
        "layout_rows": "Linhas",
        "layout_strips": "Canais",
        "contact_title": "Contato",
        "contact_head": "Dúvidas e sugestões",
        "contact_body": "Escreva para o criador do Isolate — retorno de "
                        "bugs, ideias e pedidos de recurso são bem-vindos.",
        "contact_write": "Escrever e-mail",
        "contact_copy": "Copiar e-mail",
        "contact_copied": "E-mail copiado para a área de transferência.",
        "contact_close": "Fechar",
        "contact_version": "Versão {v}",
        "btn_export": "Exportar Mix",
        "status_ready": "●  Pronto.",
        "footer": "Ferramenta educacional para separação de instrumentos "
                  "e análise musical. Distribuição gratuita.",
        "track_original": "Áudio Original",
        "stem_vocals": "Vocais", "stem_drums": "Bateria",
        "stem_bass": "Baixo", "stem_piano": "Piano",
        "stem_other": "Outros", "stem_accompaniment": "Acompanhamento",
        "stem_guitar": "Guitarra",
        "st_engine_demucs": "Carregando o motor de separação "
                            "(Demucs/PyTorch)...",
        "dlg_open_title": "Abrir arquivo de áudio",
        "dlg_export_title": "Exportar mix",
        "ft_audio": "Áudio / vídeo", "ft_all": "Todos os arquivos",
        "ft_wav": "Arquivo WAV", "ft_mp3": "Arquivo MP3",
        "msg_unsupported": "Tipo de arquivo não suportado '{ext}'. "
                           "Suportados: .wav, .mp3, .m4a, .mp4",
        "st_loading": "Carregando '{name}'...",
        "st_loaded": "Carregado: '{name}' — {sr} Hz, {dur}.",
        "st_downloading": "Baixando áudio... {pct}",
        "st_converting": "Convertendo download para WAV...",
        "st_retry": "Tentando baixar de novo (tentativa {n}/{total})...",
        "st_loading_dl": "Carregando áudio baixado...",
        "st_engine": "Carregando o motor de separação (TensorFlow)...",
        "st_model_dl": "Baixando modelo pré-treinado (só na primeira vez)...",
        "st_separating": "Separando o áudio... isso pode levar alguns minutos.",
        "st_sep_chunk": "As faixas estão sendo separadas ({pct}%)... isso "
                        "pode levar alguns minutos — vá fazer um café ☕",
        "st_model_retry": "Modelo corrompido detectado — baixando de novo...",
        "st_sep_done": "Separação concluída — {n} stems prontos.",
        "st_render": "Renderizando o mixdown...",
        "st_encoding": "Codificando {fmt}...",
        "st_exported": "Exportado: {path}",
        "st_error": "Erro: {exc}",
        "st_unexpected": "Erro inesperado: {exc}",
        "st_paste_url": "Cole uma URL do YouTube primeiro.",
        "st_load_first_sep": "Carregue um arquivo ou URL do YouTube primeiro.",
        "st_nothing_export": "Nada para exportar — carregue um áudio primeiro.",
        "st_load_first": "Carregue um arquivo de áudio primeiro.",
        "st_analysis": "Análise: {key}, {bpm} BPM.",
        "metro_label": "METRÔNOMO",
        "st_metro_unavailable": "O pulso ainda não foi detectado — "
                                "carregue e analise uma faixa primeiro.",
        "st_exported_metro": "Exportado: {path} (+ metrônomo: {metro})",
        "lbl_speed": "Velocidade:",
        "st_speed_working": "Ajustando o andamento para {pct}%... "
                            "só um instante ☕",
        "st_speed_done": "● Velocidade {pct}% — mesmo tom, mais tempo "
                         "para tocar junto.",
        "st_speed_done_bpm": "● Velocidade {pct}%: {bpm0} → {bpm} BPM, "
                             "no mesmo tom.",
        "st_speed_normal": "● Velocidade original (100%).",
        "err_speed_ffmpeg": "Mudar a velocidade exige o ffmpeg.",
        "err_speed": "Não foi possível mudar a velocidade:\n{err}",
        "btn_library": "Biblioteca",
        "btn_save_lib": "Salvar na biblioteca",
        "lib_title": "Biblioteca",
        "lib_count": "{n} músicas",
        "lib_count_1": "1 música",
        "lib_search": "Buscar...",
        "lib_empty": "Nenhuma música salva ainda.\nSepare uma faixa e clique em "
                     "“Salvar na biblioteca” para guardá-la aqui.",
        "lib_no_match": "Nada encontrado para “{q}”.",
        "lib_open": "Abrir",
        "lib_open_folder": "Abrir pasta",
        "lib_change_folder": "Alterar pasta...",
        "lib_choose_folder": "Escolha a pasta da biblioteca",
        "lib_close": "Fechar",
        "lib_mode_original": "Original",
        "lib_delete_q": "Mover “{name}” para a Lixeira?",
        "lib_exists": "Já existe “{name}” na biblioteca.\n\n"
                      "Sim — substituir pela versão atual\n"
                      "Não — salvar como uma cópia nova\n"
                      "Cancelar — não salvar",
        "st_saving": "Salvando na biblioteca ({pct}%)... só um instante ☕",
        "st_saved": "● Salvo na biblioteca: {name}",
        "st_opening": "Abrindo “{name}” da biblioteca...",
        "st_opened": "● Aberto da biblioteca: {name} — {n} faixas, offline.",
        "st_nothing_save": "Nada para salvar — carregue ou separe uma "
                           "música primeiro.",
        "err_save": "Não foi possível salvar na biblioteca:\n{err}",
        "err_session_bad": "“{name}” não é uma sessão válida do Isolate "
                           "(arquivos faltando ou corrompidos).",
        "err_delete": "Não foi possível apagar:\n{err}",
        "st_playback_err": "Erro de reprodução: {exc}",
        "st_device_err": "Erro no dispositivo: {exc}",
        "st_no_ffmpeg": "Aviso: ffmpeg não encontrado no PATH — carregar "
                        "MP3/M4A/MP4, baixar do YouTube e exportar MP3 "
                        "não vão funcionar.",
        "err_no_ffmpeg": "O ffmpeg não foi encontrado no PATH. Instale-o "
                         "e reinicie o Isolate.",
        "err_decode": "O ffmpeg não conseguiu decodificar '{name}':\n{err}",
        "err_not_found": "Arquivo não encontrado: {path}",
        "err_yt": "O download do YouTube falhou após {n} tentativas: {exc}",
        "err_yt_nofile": "O áudio baixado não foi gerado.",
        "err_stem_missing": "O Spleeter não produziu o stem '{name}'.",
        "err_oom": "Memória RAM insuficiente para separar essa faixa. "
                   "Feche outros programas (navegador etc.) e tente de "
                   "novo — ou use um modo com menos stems.",
        "err_model_corrupt": "Os arquivos do modelo estão corrompidos e o "
                             "novo download também falhou. Verifique sua "
                             "conexão e tente de novo.",
        "err_mp3_ffmpeg": "Exportar MP3 exige o ffmpeg no PATH.",
        "err_export": "A exportação falhou:\n{err}",
        "lang_restart": "Reiniciar o Isolate agora para aplicar o idioma?\n"
                        "Restart Isolate now to apply the language?",
    },
    "en": {
        "drop_dnd": "Drop an audio file here\n.wav  .mp3  .m4a  .mp4",
        "drop_click": "Click to choose an audio file\n"
                      ".wav  .mp3  .m4a  .mp4",
        "url_placeholder": "YouTube URL...",
        "btn_download": "Download & Load",
        "lbl_output": "Output:",
        "device_default": "System default",
        "lbl_sep_mode": "S E P A R A T I O N   M O D E",
        "stems2": "2 Stems — Demucs (Vocals / Accompaniment)",
        "stems4": "4 Stems (Vocals / Drums / Bass / Other)",
        "stems5": "5 Stems (Vocals / Drums / Bass / Piano / Other)",
        "stems6": "6 Stems — Demucs (with Guitar; slower)",
        "stems2_short": "2 Stems",
        "stems4_short": "4 Stems",
        "stems5_short": "5 Stems",
        "stems6_short": "6 Stems",
        "stems2_desc": "Vocals / Accompaniment — cleaner vocals, slower",
        "stems4_desc": "Vocals / Drums / Bass / Other",
        "stems5_desc": "Vocals / Drums / Bass / Piano / Other",
        "stems6_desc": "Demucs: Vocals / Drums / Bass / Guitar / Piano / "
                       "Other — slower",
        "btn_separate": "Separate Tracks",
        "lbl_analysis": "M U S I C A L   A N A L Y S I S",
        "chip_key": "KEY",
        "btn_analyze": "Detect Key & BPM",
        "mixer_hint": "Load an audio file to get started.",
        "mixer_title": "M I X E R",
        "layout_rows": "Rows",
        "layout_strips": "Strips",
        "contact_title": "Contact",
        "contact_head": "Questions & suggestions",
        "contact_body": "Write to the creator of Isolate — bug reports, "
                        "ideas and feature requests are welcome.",
        "contact_write": "Write e-mail",
        "contact_copy": "Copy e-mail",
        "contact_copied": "E-mail copied to the clipboard.",
        "contact_close": "Close",
        "contact_version": "Version {v}",
        "btn_export": "Export Mix",
        "status_ready": "●  Ready.",
        "footer": "Educational tool for instrument separation and musical "
                  "analysis. Free distribution.",
        "track_original": "Original Audio",
        "stem_vocals": "Vocals", "stem_drums": "Drums",
        "stem_bass": "Bass", "stem_piano": "Piano",
        "stem_other": "Other", "stem_accompaniment": "Accompaniment",
        "stem_guitar": "Guitar",
        "st_engine_demucs": "Loading separation engine "
                            "(Demucs/PyTorch)...",
        "dlg_open_title": "Open audio file",
        "dlg_export_title": "Export mix",
        "ft_audio": "Audio / video", "ft_all": "All files",
        "ft_wav": "WAV file", "ft_mp3": "MP3 file",
        "msg_unsupported": "Unsupported file type '{ext}'. "
                           "Supported: .wav, .mp3, .m4a, .mp4",
        "st_loading": "Loading '{name}'...",
        "st_loaded": "Loaded '{name}' — {sr} Hz, {dur}.",
        "st_downloading": "Downloading audio... {pct}",
        "st_converting": "Converting download to WAV...",
        "st_retry": "Retrying download (attempt {n}/{total})...",
        "st_loading_dl": "Loading downloaded audio...",
        "st_engine": "Loading separation engine (TensorFlow)...",
        "st_model_dl": "Downloading pretrained model (first run only)...",
        "st_separating": "Separating audio... this can take a few minutes.",
        "st_sep_chunk": "Your tracks are being separated ({pct}%)... this "
                        "can take a few minutes — go grab a coffee ☕",
        "st_model_retry": "Corrupted model detected — downloading it again...",
        "st_sep_done": "Separation complete — {n} stems ready.",
        "st_render": "Rendering mixdown...",
        "st_encoding": "Encoding {fmt}...",
        "st_exported": "Exported: {path}",
        "st_error": "Error: {exc}",
        "st_unexpected": "Unexpected error: {exc}",
        "st_paste_url": "Paste a YouTube URL first.",
        "st_load_first_sep": "Load an audio file or YouTube URL first.",
        "st_nothing_export": "Nothing to export — load audio first.",
        "st_load_first": "Load an audio file first.",
        "st_analysis": "Analysis: {key}, {bpm} BPM.",
        "metro_label": "METRONOME",
        "st_metro_unavailable": "Beat grid not detected yet — load and "
                                "analyze a track first.",
        "st_exported_metro": "Exported: {path} (+ metronome: {metro})",
        "lbl_speed": "Speed:",
        "st_speed_working": "Setting the tempo to {pct}%... just a "
                            "moment ☕",
        "st_speed_done": "● Speed {pct}% — same key, more time to play "
                         "along.",
        "st_speed_done_bpm": "● Speed {pct}%: {bpm0} → {bpm} BPM, same key.",
        "st_speed_normal": "● Original speed (100%).",
        "err_speed_ffmpeg": "Changing the speed requires ffmpeg.",
        "err_speed": "Could not change the speed:\n{err}",
        "btn_library": "Library",
        "btn_save_lib": "Save to library",
        "lib_title": "Library",
        "lib_count": "{n} songs",
        "lib_count_1": "1 song",
        "lib_search": "Search...",
        "lib_empty": "No songs saved yet.\nSeparate a track and click "
                     "“Save to library” to keep it here.",
        "lib_no_match": "Nothing found for “{q}”.",
        "lib_open": "Open",
        "lib_open_folder": "Open folder",
        "lib_change_folder": "Change folder...",
        "lib_choose_folder": "Choose the library folder",
        "lib_close": "Close",
        "lib_mode_original": "Original",
        "lib_delete_q": "Move “{name}” to the Recycle Bin?",
        "lib_exists": "“{name}” is already in the library.\n\n"
                      "Yes — replace it with the current version\n"
                      "No — save it as a new copy\n"
                      "Cancel — don't save",
        "st_saving": "Saving to the library ({pct}%)... just a moment ☕",
        "st_saved": "● Saved to the library: {name}",
        "st_opening": "Opening “{name}” from the library...",
        "st_opened": "● Opened from the library: {name} — {n} tracks, "
                     "offline.",
        "st_nothing_save": "Nothing to save — load or separate a song first.",
        "err_save": "Could not save to the library:\n{err}",
        "err_session_bad": "“{name}” is not a valid Isolate session "
                           "(missing or corrupted files).",
        "err_delete": "Could not delete:\n{err}",
        "st_playback_err": "Playback error: {exc}",
        "st_device_err": "Device error: {exc}",
        "st_no_ffmpeg": "Warning: ffmpeg not found on PATH — MP3/M4A/MP4 "
                        "loading, YouTube download and MP3 export will "
                        "not work.",
        "err_no_ffmpeg": "ffmpeg was not found on PATH. Install it and "
                         "restart Isolate.",
        "err_decode": "ffmpeg could not decode '{name}':\n{err}",
        "err_not_found": "File not found: {path}",
        "err_yt": "YouTube download failed after {n} attempts: {exc}",
        "err_yt_nofile": "Downloaded audio file was not produced.",
        "err_stem_missing": "Spleeter did not produce the '{name}' stem.",
        "err_oom": "Not enough RAM to separate this track. Close other "
                   "programs (browser etc.) and try again — or pick a "
                   "mode with fewer stems.",
        "err_model_corrupt": "The model files are corrupted and the fresh "
                             "download failed too. Check your connection "
                             "and try again.",
        "err_mp3_ffmpeg": "MP3 export requires ffmpeg on PATH.",
        "err_export": "Export failed:\n{err}",
        "lang_restart": "Restart Isolate now to apply the language?\n"
                        "Reiniciar o Isolate agora para aplicar o idioma?",
    },
}


def L(key: str, /, **kw) -> str:
    """Localized string for `key` in the active language (PT fallback EN)."""
    s = I18N[LANG].get(key) or I18N["en"].get(key) or key
    return s.format(**kw) if kw else s


STEM_MODELS = {
    # v2.2 — Demucs with every non-vocal source summed back into one
    # "accompaniment" channel (what Demucs itself does for `--two-stems`).
    # Model picked by an A/B over 30 MUSDB18 tracks with ground truth
    # (medians, SDR of vocals / accompaniment): spleeter 6.64/10.88,
    # htdemucs_6s 8.66/11.88, htdemucs 9.15/12.32, htdemucs_ft 9.38/12.43.
    # htdemucs base beats _6s on both stems; _ft only buys +0.36 dB of
    # vocals (its accompaniment ties, p=0.95) for 4x the time and 320 MB.
    # It replaced Spleeter's 2stems outright, which lost by ~2.5 dB of
    # vocals in 29 of the 30 tracks.
    L("stems2"): ("demucs:htdemucs:vocals", ["vocals", "accompaniment"]),
    L("stems4"): ("spleeter:4stems-16kHz", ["vocals", "drums",
                                            "bass", "other"]),
    L("stems5"): ("spleeter:5stems-16kHz", ["vocals", "drums",
                                            "bass", "piano", "other"]),
    L("stems6"): ("demucs:htdemucs_6s", ["vocals", "drums", "bass",
                                         "guitar", "piano", "other"]),
}

# v2.1.1 — the separation mode used to be a 4-row radio stack, which ate
# ~110 px of vertical space and squeezed the mixer to ~1 visible strip on
# 768 px laptop screens. It is now one pill row (short labels) plus a
# caption line with the full stem list of the selected mode.
STEM_SHORT: list[str] = [L("stems2_short"), L("stems4_short"),
                         L("stems5_short"), L("stems6_short")]
_SHORT_TO_FULL: dict[str, str] = dict(zip(STEM_SHORT, STEM_MODELS.keys()))
STEM_DESC: dict[str, str] = {
    L("stems2_short"): L("stems2_desc"),
    L("stems4_short"): L("stems4_desc"),
    L("stems5_short"): L("stems5_desc"),
    L("stems6_short"): L("stems6_desc"),
}

# Mixer layout: "rows" = horizontal channel rows (classic Isolate look),
# "strips" = vertical console strips side by side (fits every stem on a
# short screen without scrolling). Persisted, applied without a restart.
MIXER_LAYOUT = _SETTINGS.get("mixer_layout", "rows")
if MIXER_LAYOUT not in ("rows", "strips"):
    MIXER_LAYOUT = "rows"

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
RADIO_RING = "#55534e"      # legacy radio ring (tema §5); kept for reference

# Set to Outfit / Spline Sans Mono at startup when installed (tema §3);
# otherwise the spec's fallbacks below are kept.
UI_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"

STEM_LABELS = {inst: L(f"stem_{inst}")
               for inst in ("vocals", "drums", "bass", "piano",
                            "other", "accompaniment", "guitar")}
# saved sessions store the stem id, not the (localized) channel name, so a
# song saved in PT reopens with EN names after a language switch
_LABEL_TO_STEM = {v: k for k, v in STEM_LABELS.items()}
_LABEL_TO_STEM[L("track_original")] = "original"
_SPEC_TO_SHORT = {STEM_MODELS[full][0]: short
                  for short, full in _SHORT_TO_FULL.items()}


def stem_label(stem_id: str) -> str:
    """Channel name for a saved stem id ("drums" -> "Bateria" / "Drums")."""
    if stem_id == "original":
        return L("track_original")
    return STEM_LABELS.get(stem_id, stem_id)


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

    __slots__ = ("name", "orig", "data", "gain", "mute", "solo")

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.orig = data          # float32, shape (base_frames, 2), 100% speed
        self.data = data          # playback buffer: orig, or time-stretched
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
        self.n_frames = 0                    # playback length (stretched)
        self.base_frames = 0                 # length at the original speed
        self.speed = 1.0                     # playback speed (pitch kept)
        self.device: int | None = None       # None -> system default
        self.master_gain = 1.0               # master fader (0.0 .. 1.0)
        self.levels = np.zeros(0)            # per-track post-fader peak
        self.master_level = 0.0              # post-master peak (pre-clip)
        self.click_track: np.ndarray | None = None   # mono metronome buffer
        self.beat_times: np.ndarray | None = None    # beat grid (seconds)
        self.metronome_on = False
        self.metronome_gain = 0.8            # independent of master_gain
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
            self.base_frames = n
            self.speed = 1.0
            self.samplerate = samplerate
            self.levels = np.zeros(len(tracks))
            self.master_level = 0.0
            self.click_track = None       # stale beat grid: new material
            self.beat_times = None
            self.metronome_on = False
        if rate_changed:
            self._close_stream()

    def set_click_track(self, click: np.ndarray | None,
                        beat_times: np.ndarray | None = None) -> None:
        """Swap the metronome buffer + beat grid (thread-safe). `None`
        means the beat grid could not be detected, which also forces the
        metronome off."""
        with self._lock:
            self.click_track = click
            self.beat_times = beat_times if click is not None else None
            if click is None:
                self.metronome_on = False

    def set_playback_speed(self, tracks: list[Track], speed: float,
                           arrays: list[np.ndarray],
                           click: np.ndarray | None,
                           beat_times: np.ndarray | None) -> bool:
        """Swap in the playback buffers for `speed` (one per track, equal
        length) together with the matching click, atomically, keeping the
        playhead on the same musical spot. Returns False — and changes
        nothing — if the track set was replaced while they were rendered."""
        with self._lock:
            if tracks is not self.tracks:
                return False
            n = len(arrays[0]) if arrays else self.n_frames
            pos = int(round(self._position * self.speed / speed))
            for t, a in zip(self.tracks, arrays):
                t.data = a
            self.n_frames = n
            self.speed = speed
            self._position = min(pos, n)
            self.click_track = click
            self.beat_times = beat_times if click is not None else None
            if click is None:
                self.metronome_on = False
        return True

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
            # metronome click: injected POST-master (control-room style),
            # so the master fader never silences the practice click; the
            # master VU keeps showing the music mix only
            if self.metronome_on and self.click_track is not None:
                c = self.click_track[pos:end]
                if len(c):
                    mix[:len(c)] += (c * self.metronome_gain)[:, None]
            # clipping prevention: hard ceiling at full scale
            np.clip(mix, -1.0, 1.0, out=mix)
            outdata[:] = mix
            self._position = min(end, self.n_frames)
            if self._position >= self.n_frames:
                self._playing = False      # reached end: auto-stop

    # -- offline mixdown --------------------------------------------------------

    def render_mix(self) -> np.ndarray:
        """Full-length stereo mixdown honouring gain / mute / solo / master,
        always at the ORIGINAL speed (the speed control is for practice)."""
        with self._lock:
            mix = np.zeros((self.base_frames, 2), dtype=np.float32)
            any_solo = any(t.solo for t in self.tracks)
            for t in self.tracks:
                if t.mute or (any_solo and not t.solo):
                    continue
                mix += t.orig * t.gain
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
        raise MediaError(L("err_not_found", path=path))

    if p.suffix.lower() in NATIVE_SF_EXTENSIONS:
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            return data, sr, str(p)
        except Exception:
            pass  # fall through to ffmpeg (e.g. exotic WAV codecs)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(L("err_no_ffmpeg"))
    out_wav = os.path.join(temp_dir, f"decoded_{uuid.uuid4().hex[:12]}.wav")
    cmd = [ffmpeg, "-y", "-i", str(p), "-vn",
           "-acodec", "pcm_f32le", out_wav]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=getattr(subprocess,
                                                "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0 or not os.path.exists(out_wav):
        raise MediaError(L("err_decode", name=p.name,
                           err=proc.stderr[-400:]))
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
            progress(L("st_downloading", pct=pct))
        elif d.get("status") == "finished":
            progress(L("st_converting"))

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
            progress(L("st_retry", n=n + 1, total=len(client_attempts)))
        try:
            with YoutubeDL({**base_opts, **extra}) as ydl:
                info = ydl.extract_info(url, download=True)
            if info is not None:
                break
        except Exception as exc:
            last_exc = exc
            log.warning("yt-dlp attempt %d failed: %s", n + 1, exc)
    if info is None:
        raise MediaError(L("err_yt", n=len(client_attempts),
                           exc=last_exc)) from last_exc
    if "entries" in info:                       # playlist -> first entry
        info = info["entries"][0]
    wav_path = os.path.join(temp_dir, f"yt_{info['id']}.wav")
    if not os.path.exists(wav_path):
        raise MediaError(L("err_yt_nofile"))
    return wav_path, info.get("title") or "YouTube Audio"


_SEPARATOR_CACHE: dict[str, object] = {}
# Loaded torch nets, keyed by model name: several presets (6 stems and
# 2 Stems+) share one net and must not load it twice.
_DEMUCS_MODEL_CACHE: dict[str, object] = {}

# Chunked separation keeps TensorFlow's peak memory flat regardless of track
# length: separating a whole 3.5 min song at once peaks at ~10 GB of commit
# in 5-stems mode, which kills 8 GB machines with "Graph execution error".
_CHUNK_S = 30.0          # seconds of audio fed to the model per prediction
_XFADE_S = 2.0           # crossfaded overlap between consecutive chunks

# A completed model download contains ".probe" plus these checkpoint files;
# anything less and TensorFlow either crashes mid-graph or silently runs
# with UNTRAINED weights (stems come out as the full mix at -6 dB).
_MODEL_FILES = (".probe", "checkpoint", "model.data-00000-of-00001",
                "model.index", "model.meta")

_OOM_MARKERS = ("oom when allocating", "resource exhausted",
                "resourceexhausted", "failed to allocate", "out of memory",
                "not enough memory", "bad_alloc", "defaultcpuallocator")
_CORRUPT_MODEL_MARKERS = ("data loss", "datalosserror", "checksum",
                          "corrupt", "unable to open table", "truncated",
                          "failed to find any matching files", "restor",
                          "pytorchstreamreader", "invalid load key",
                          "unexpected eof", "central directory")


def _demucs_parts(model_spec: str) -> tuple[str, str | None]:
    """Split a Demucs spec into (model name, solo source).
    "demucs:htdemucs_6s" -> ("htdemucs_6s", None)
    "demucs:htdemucs_6s:vocals" -> ("htdemucs_6s", "vocals")  # 2 Stems+
    """
    parts = model_spec.split(":")
    return parts[1], (parts[2] if len(parts) > 2 else None)


def _model_dir(model_spec: str) -> Path:
    return _MODELS_DIR / model_spec.split(":", 1)[1].split("-")[0]


def _model_is_broken(model_dir: Path) -> bool:
    """True when the model folder is missing checkpoint files or has
    empty ones (interrupted download/extraction)."""
    for name in _MODEL_FILES:
        f = model_dir / name
        if not f.exists() or f.stat().st_size == 0:
            return True
    return False


def _purge_model(model_spec: str) -> None:
    """Drop the cached separator and delete the model files so the next
    attempt re-downloads them from scratch."""
    _SEPARATOR_CACHE.pop(model_spec, None)
    if model_spec.startswith("demucs:"):
        # every preset built on this net has to be rebuilt after the purge
        _DEMUCS_MODEL_CACHE.pop(_demucs_parts(model_spec)[0], None)
        for spec in [s for s in _SEPARATOR_CACHE if s.startswith("demucs:")]:
            _SEPARATOR_CACHE.pop(spec, None)
        shutil.rmtree(_TORCH_DIR / "hub" / "checkpoints", ignore_errors=True)
    else:
        shutil.rmtree(_model_dir(model_spec), ignore_errors=True)


def _classify_separation_error(exc: Exception) -> str | None:
    """Map a TensorFlow/Spleeter failure to "oom", "corrupt" or None.
    TF wraps everything in a generic "Graph execution error", so the real
    cause has to be sniffed from the message text."""
    if isinstance(exc, MemoryError):
        return "oom"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(m in text for m in _OOM_MARKERS):
        return "oom"
    if any(m in text for m in _CORRUPT_MODEL_MARKERS):
        return "corrupt"
    return None


def _prepare_separation_input(source_wav: str, temp_dir: str) -> str:
    """Return a 44.1 kHz stereo WAV for `source_wav`, converting through
    ffmpeg only when the source is at another rate / channel count."""
    info = sf.info(source_wav)
    if info.samplerate == 44100 and info.channels == 2:
        return source_wav
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(L("err_no_ffmpeg"))
    out_wav = os.path.join(temp_dir, f"sep_input_{uuid.uuid4().hex[:12]}.wav")
    cmd = [ffmpeg, "-y", "-i", source_wav, "-vn", "-ar", "44100", "-ac", "2",
           "-acodec", "pcm_f32le", out_wav]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          creationflags=getattr(subprocess,
                                                "CREATE_NO_WINDOW", 0))
    if proc.returncode != 0 or not os.path.exists(out_wav):
        raise MediaError(L("err_decode", name=Path(source_wav).name,
                           err=proc.stderr[-400:]))
    return out_wav


class _DemucsSeparator:
    """Adapter exposing Spleeter's `.separate(waveform) -> dict` interface
    for a Demucs model, so `_separate_chunked` drives both engines. CPU
    only; apply_model already splits each call into ~8 s segments with
    overlap, keeping memory bounded.

    With `solo` set, only that source is kept and every other one is summed
    into a single "accompaniment" channel — the two-stem mode of Demucs
    itself (`--two-stems`), used by the v2.2 "2 Stems+" preset.
    """

    def __init__(self, model_name: str, solo: str | None = None):
        import torch                          # heavy imports, keep lazy
        from demucs.pretrained import get_model
        self._torch = torch
        # Shared across presets: the 6-stem and the 2-stem+ modes run the
        # same net, so they must not hold two copies of it in memory.
        model = _DEMUCS_MODEL_CACHE.get(model_name)
        if model is None:
            model = get_model(model_name)
            model.cpu()
            model.eval()
            _DEMUCS_MODEL_CACHE[model_name] = model
        self._model = model
        self.sources = list(model.sources)
        if solo is not None and solo not in self.sources:
            raise MediaError(L("err_stem_missing", name=solo))
        self._solo = solo
        self.outputs = ([solo, "accompaniment"] if solo else self.sources)

    def separate(self, waveform: np.ndarray) -> dict[str, np.ndarray]:
        from demucs.apply import apply_model
        torch = self._torch
        wav = torch.from_numpy(
            np.ascontiguousarray(waveform.T)).float()      # (2, n)
        # Demucs expects input normalized to zero mean / unit std
        # (same as demucs.separate does), undone on the way out.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std
        with torch.no_grad():
            out = apply_model(self._model, wav[None], device="cpu",
                              shifts=0, split=True, overlap=0.25,
                              progress=False)[0]
        out = out * std + mean                # (n_sources, 2, n)
        arr = out.cpu().numpy()
        if self._solo is None:
            return {src: np.ascontiguousarray(arr[i].T)
                    for i, src in enumerate(self.sources)}
        i = self.sources.index(self._solo)
        rest = arr.sum(axis=0) - arr[i]       # (2, n)
        return {self._solo: np.ascontiguousarray(arr[i].T),
                "accompaniment": np.ascontiguousarray(rest.T)}


def _separate_chunked(sep, wav_path: str, instruments: list[str],
                      progress) -> dict[str, np.ndarray]:
    """
    Separate `wav_path` in _CHUNK_S-second chunks with an _XFADE_S linear
    crossfade at the seams. Returns {instrument: float32 array of shape
    (n_frames, 2)} at 44.1 kHz, same length as the input.
    """
    info = sf.info(wav_path)
    total, sr = info.frames, info.samplerate
    chunk = int(_CHUNK_S * sr)
    fade = int(_XFADE_S * sr)
    step = chunk - fade
    n_chunks = (1 + (total - chunk + step - 1) // step if total > chunk
                else 1)

    out = {inst: np.zeros((total, 2), np.float32) for inst in instruments}
    for ci in range(n_chunks):
        progress(L("st_sep_chunk", pct=int(100 * ci / n_chunks)))
        start = ci * step
        stop = min(start + chunk, total)
        seg, _sr = sf.read(wav_path, start=start, stop=stop,
                           dtype="float32", always_2d=True)
        result = sep.separate(seg)
        for inst in instruments:
            if inst not in result:
                raise MediaError(L("err_stem_missing", name=inst))
            arr = np.asarray(result[inst], dtype=np.float32)
            n = min(len(arr), total - start)
            buf = out[inst]
            if ci == 0:
                buf[:n] = arr[:n]
                continue
            f = min(fade, n)
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
            buf[start:start + f] *= 1.0 - ramp
            buf[start:start + f] += arr[:f] * ramp
            buf[start + f:start + n] = arr[f:n]
    return out


def separate_stems(source_wav: str, model_spec: str, stem_order: list[str],
                   temp_dir: str, progress,
                   allow_retry: bool = True) -> list[tuple[str, np.ndarray]]:
    """
    Run Spleeter on `source_wav` and return the stems, in `stem_order`,
    as (display_name, float32 array) pairs at 44.1 kHz.
    """
    is_demucs = model_spec.startswith("demucs:")
    progress(L("st_engine_demucs" if is_demucs else "st_engine"))

    sep = _SEPARATOR_CACHE.get(model_spec)
    if sep is None:
        if is_demucs:
            # torch.hub downloads atomically and demucs verifies the
            # checksum itself; a corrupted file surfaces as an exception
            # handled below (purge + retry).
            ckpt_dir = _TORCH_DIR / "hub" / "checkpoints"
            if not (ckpt_dir.exists() and any(ckpt_dir.glob("*.th"))):
                progress(L("st_model_dl"))
            sep = _DemucsSeparator(*_demucs_parts(model_spec))
        else:
            from spleeter.separator import Separator   # heavy, keep lazy
            # Guard against a partially-downloaded model: Spleeter skips
            # the download whenever the model directory exists (see
            # _MODEL_FILES).
            model_dir = _model_dir(model_spec)
            if model_dir.exists() and _model_is_broken(model_dir):
                log.warning("Removing broken model directory: %s", model_dir)
                shutil.rmtree(model_dir, ignore_errors=True)
            if not model_dir.exists():
                progress(L("st_model_dl"))
            sep = Separator(model_spec, multiprocess=False)
        _SEPARATOR_CACHE[model_spec] = sep

    prepared = _prepare_separation_input(source_wav, temp_dir)
    try:
        separated = _separate_chunked(sep, prepared, stem_order, progress)
    except MediaError:
        raise
    except Exception as exc:
        kind = _classify_separation_error(exc)
        if kind == "oom":
            raise MediaError(L("err_oom")) from exc
        if kind == "corrupt":
            log.warning("Model %s looks corrupted (%s); purging.",
                        model_spec, exc)
            _purge_model(model_spec)
            if allow_retry:
                progress(L("st_model_retry"))
                return separate_stems(source_wav, model_spec, stem_order,
                                      temp_dir, progress, allow_retry=False)
            raise MediaError(L("err_model_corrupt")) from exc
        raise

    return [(STEM_LABELS.get(inst, inst.capitalize()), separated[inst])
            for inst in stem_order]


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
        raise MediaError(L("err_mp3_ffmpeg"))

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
        raise MediaError(L("err_export", err=proc.stderr[-400:]))


# ---------------------------------------------------------------------------
# Song library (v2.3) — separated songs kept on disk and reopened offline
# ---------------------------------------------------------------------------
# One folder per song under the library root:
#   <root>/<title>/session.json      mode, key, BPM, beat grid, mixer, speed
#   <root>/<title>/01_vocals.flac …  one file per mixer channel (100% speed)
#   <root>/<title>/original.flac     the source, so it can be re-separated
# FLAC 24-bit is lossless at the mixer's resolution and about half the size
# of a WAV: an archive of lesson songs must not eat the disk.

SESSION_FILE = "session.json"
SESSION_FORMAT = 1
_SESSION_SUBTYPE = "PCM_24"
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}


def default_library_dir() -> Path:
    """Music\\Isolate, wherever Windows keeps the user's Music folder.
    Not Documents: Windows 11 often moves Documents into OneDrive, and a
    library of stems (100-300 MB per song) would fill the free 5 GB."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(260)
        # CSIDL_MYMUSIC = 13
        if (ctypes.windll.shell32.SHGetFolderPathW(None, 13, None, 0, buf)
                == 0 and buf.value):
            return Path(buf.value) / _APP_NAME
    except Exception:
        pass
    return Path.home() / "Music" / _APP_NAME


def library_dir() -> Path:
    """Library root: the folder chosen by the user, else the default."""
    custom = _SETTINGS.get("library_dir")
    return Path(custom) if custom else default_library_dir()


def safe_folder_name(title: str) -> str:
    """A Windows-safe folder / file name for a song title."""
    name = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c
                   for c in title)
    name = " ".join(name.split()).strip(" .")[:80].strip(" .")
    if not name:
        name = _APP_NAME
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        name = "_" + name
    return name


def unique_folder(target: Path) -> Path:
    """`target`, or "<target> (2)", "(3)"… — the first one not taken."""
    base, n, cand = target.name, 2, target
    while cand.exists():
        cand = target.with_name(f"{base} ({n})")
        n += 1
    return cand


def _write_audio(path: Path, data: np.ndarray, samplerate: int) -> float:
    """Write a FLAC 24-bit file; returns the scale to multiply back on
    load. A stem can overshoot 0 dBFS (the float mix clips only at the
    output) and integer FLAC cannot hold that, so such a stem is stored
    normalized to its peak instead of clipped."""
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    scale = peak if peak > 1.0 else 1.0
    sf.write(str(path), data / scale if scale != 1.0 else data, samplerate,
             subtype=_SESSION_SUBTYPE)
    return scale


def _read_audio(path: Path, scale: float) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if scale != 1.0:
        data *= np.float32(scale)
    return data, sr


def save_session(target: Path, title: str,
                 tracks: list[tuple[str, np.ndarray]], samplerate: int,
                 source_wav: str | None, state: dict, progress) -> Path:
    """
    Save a song to the folder `target` (replaced if it exists): one FLAC
    per mixer channel, given as (stem id, float32 array at 100% speed),
    the source audio, and session.json with `state` (mode, key, BPM, beat
    grid, mixer, speed, metronome). Everything is written to a sibling
    folder first and swapped in at the end, so a crash or a full disk
    never leaves half a song in place of a good one.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.saving-{uuid.uuid4().hex[:6]}")
    tmp.mkdir()
    try:
        n_files = len(tracks) + 1
        stems = []
        for i, (stem_id, data) in enumerate(tracks):
            progress(L("st_saving", pct=int(100 * i / n_files)))
            fname = f"{i + 1:02d}_{safe_folder_name(stem_id)}.flac"
            scale = _write_audio(tmp / fname, data, samplerate)
            stems.append({"id": stem_id, "file": fname, "scale": scale})
        source = None
        if len(stems) == 1 and stems[0]["id"] == "original":
            # not separated yet: the only channel IS the source
            source = {"file": stems[0]["file"], "scale": stems[0]["scale"]}
        elif source_wav and os.path.exists(source_wav):
            progress(L("st_saving", pct=int(100 * len(tracks) / n_files)))
            data, sr = sf.read(source_wav, dtype="float32", always_2d=True)
            source = {"file": "original.flac",
                      "scale": _write_audio(tmp / "original.flac", data, sr)}
        manifest = {
            "format": SESSION_FORMAT,
            "app_version": APP_VERSION,
            "title": title,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "samplerate": samplerate,
            "frames": int(max((len(d) for _, d in tracks), default=0)),
            "stems": stems,
            "source": source,
            **state,
        }
        (tmp / SESSION_FILE).write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False),
            encoding="utf-8")
        if target.exists():
            old = target.with_name(f".{target.name}.old-{uuid.uuid4().hex[:6]}")
            target.rename(old)
            tmp.rename(target)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def read_manifest(folder: Path) -> dict:
    """session.json of a saved song, checked against the files on disk.
    Raises MediaError when the folder is not a usable session."""
    folder = Path(folder)
    try:
        m = json.loads((folder / SESSION_FILE).read_text(encoding="utf-8"))
        ok = (isinstance(m, dict)
              and int(m.get("format", 0)) <= SESSION_FORMAT
              and m.get("stems")
              and all((folder / s["file"]).is_file() for s in m["stems"]))
    except Exception:
        ok = False
    if not ok:
        raise MediaError(L("err_session_bad", name=folder.name))
    return m


def list_library(root: Path) -> list[tuple[Path, dict]]:
    """Every valid song under `root`, most recently saved first. Hidden
    folders (half-written saves, see save_session) are skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    songs = []
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            songs.append((folder, read_manifest(folder)))
        except MediaError:
            log.warning("Skipping invalid library folder: %s", folder)
    songs.sort(key=lambda fm: fm[1].get("saved", ""), reverse=True)
    return songs


def load_session(folder: Path, temp_dir: str, progress
                 ) -> tuple[dict, list[tuple[str, np.ndarray]], int,
                            str | None]:
    """
    Read a saved song back: (manifest, [(stem id, array)], samplerate,
    source_wav). The source is copied to a float WAV in `temp_dir`, so
    re-separating it later never depends on the library folder (which the
    user may delete while the song is open).
    """
    folder = Path(folder)
    m = read_manifest(folder)
    sr0 = int(m.get("samplerate") or 44100)
    tracks = []
    for i, s in enumerate(m["stems"]):
        data, sr = _read_audio(folder / s["file"], float(s.get("scale", 1.0)))
        if sr != sr0:
            raise MediaError(L("err_session_bad", name=folder.name))
        tracks.append((s["id"], data))
    source_wav = None
    src = m.get("source")
    if src and (folder / src["file"]).is_file():
        data, sr = _read_audio(folder / src["file"],
                               float(src.get("scale", 1.0)))
        source_wav = os.path.join(temp_dir,
                                  f"session_src_{uuid.uuid4().hex[:12]}.wav")
        sf.write(source_wav, data, sr, subtype="FLOAT")
    return m, tracks, sr0, source_wav


def send_to_recycle_bin(path: Path) -> None:
    """Delete a song folder through the Windows Recycle Bin, so a song
    removed by mistake can still come back; plain delete elsewhere."""
    if sys.platform != "win32":
        shutil.rmtree(path)
        return
    import ctypes
    from ctypes import wintypes

    class _SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort),
                    ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p),
                    ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE = 3
    flags = 0x40 | 0x10 | 0x400 | 0x4   # ALLOWUNDO|NOCONFIRMATION|NOERRORUI|SILENT
    # pFrom is a double-NUL-terminated list
    op = _SHFILEOPSTRUCTW(None, FO_DELETE, str(Path(path).resolve()) + "\0",
                          None, flags, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc != 0 or op.fAnyOperationsAborted or Path(path).exists():
        raise OSError(f"SHFileOperationW failed (0x{rc:x})")


# Playback speeds offered for practice (fraction of the original tempo).
# 50% and 60% were dropped after use: too slow to be practical to play
# along. ffmpeg's atempo would go down to 0.5 in a single stage anyway.
SPEED_STEPS = (0.7, 0.8, 0.9, 1.0)
_STRETCH_BLOCK = 1 << 16          # frames per pipe write


def time_stretch_tracks(arrays: list[np.ndarray], samplerate: int,
                        speed: float) -> list[np.ndarray]:
    """
    Change the tempo of every track by `speed` (SPEED_STEPS) keeping the
    pitch, through ffmpeg's `atempo` (WSOLA, which keeps the attacks of
    drums intact better than a phase vocoder).

    All tracks go through ONE multichannel stream on purpose: atempo picks
    its splice points from a downmix of every channel, so the stems get
    identical splices and stay phase-locked to each other. Stretched one
    by one, each stem would splice somewhere else and their sum would smear.

    Input: float32 (n, 2) arrays of equal length. Output: views of one
    (round(n / speed), 2 * len(arrays)) buffer, one stereo pair per track.
    Audio is streamed through pipes, so memory stays at the output size.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise MediaError(L("err_speed_ffmpeg"))
    n = len(arrays[0])
    ch = 2 * len(arrays)
    n_out = int(round(n / speed))
    out = np.zeros((n_out, ch), dtype=np.float32)
    cmd = [ffmpeg, "-v", "error", "-f", "f32le", "-ar", str(samplerate),
           "-ac", str(ch), "-i", "pipe:0", "-af", f"atempo={speed:.4f}",
           "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess,
                                                  "CREATE_NO_WINDOW", 0))
    err: list[bytes] = []

    def feed() -> None:
        try:
            for i in range(0, n, _STRETCH_BLOCK):
                block = np.concatenate(
                    [a[i:i + _STRETCH_BLOCK] for a in arrays], axis=1)
                proc.stdin.write(block.astype(np.float32, copy=False)
                                 .tobytes())
        except (BrokenPipeError, OSError):
            pass                      # ffmpeg died: reported below
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def drain_err() -> None:
        err.append(proc.stderr.read())

    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=drain_err, daemon=True)]
    for t in threads:
        t.start()
    # read straight into the output buffer; whole frames, no reassembly
    buf = memoryview(out).cast("B")
    got = 0
    while got < len(buf):
        k = proc.stdout.readinto(buf[got:got + (1 << 22)])
        if not k:
            break
        got += k
    while proc.stdout.read(1 << 20):  # atempo may overshoot by a few frames
        pass
    proc.wait()
    for t in threads:
        t.join()
    if proc.returncode != 0:
        msg = (err[0] if err else b"").decode("utf-8", "replace")
        raise MediaError(L("err_speed", err=msg[-400:]))
    return [out[:, 2 * k:2 * k + 2] for k in range(len(arrays))]


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


# Onset-envelope resolution. The hop is the quantization step of every
# beat time, so it caps how tightly the click can sit on the kick:
# 256 samples = 5.8 ms at 44.1 kHz (512 was 11.6 ms and audibly loose).
_ONSET_NFFT = 1024
_ONSET_HOP = 256
# Frame -> time correction. A frame spans [k*hop, k*hop + n_fft) and its
# Hann window peaks at the middle, so the flux of a transient starting at
# sample s peaks on the frame with k*hop + n_fft/2 ~ s: the onset is
# n_fft/(2*hop) frames LATER than the frame's start index (hence the
# negative sign below). Measured against synthetic kicks in the tests.
_ONSET_LAG_FRAMES = -(_ONSET_NFFT / (2.0 * _ONSET_HOP))
# Tempo prior (Ellis 2007): most music is heard around 120 BPM; the width
# is generous (octaves) so genuinely slow or fast tracks still win when
# the evidence is there.
_TEMPO_PRIOR_BPM = 120.0
_TEMPO_PRIOR_OCT = 1.2


def _percussive_weights(freqs: np.ndarray) -> np.ndarray:
    """
    Per-bin weights that emphasize the percussive part of the spectrum:
    the kick band and the snare/hat transient band count double, while
    250 Hz - 2 kHz — where sustained vocals and harmonic instruments live
    — is halved. Sub-band onset weighting is standard practice (Klapuri
    1999); the vocal de-emphasis follows Zapata & Gómez (2013), who show
    predominant vocals mislead the onset envelope of a full mix.
    """
    w = np.ones_like(freqs, dtype=np.float32)
    w[freqs < 150.0] = 2.0                                # kick
    w[(freqs >= 250.0) & (freqs < 2000.0)] = 0.5          # voice / harmony
    w[(freqs >= 2000.0) & (freqs < 10000.0)] = 2.0        # snare / hats
    w[freqs >= 12000.0] = 0.7                             # hiss
    return w


def _onset_envelope(mono: np.ndarray, sr: int,
                    n_fft: int = _ONSET_NFFT, hop: int = _ONSET_HOP,
                    percussive: bool = True) -> np.ndarray:
    """
    Spectral-flux onset envelope of the WHOLE signal, computed in ~30 s
    chunks so the strided STFT never materializes a huge frame matrix.
    Envelope frame k covers samples [k*hop, k*hop + n_fft); see
    _ONSET_LAG_FRAMES for the frame -> time convention.

    With `percussive` (the default) the flux is taken on log-compressed
    magnitudes and weighted per band: log compression keeps a quiet
    hi-hat from being buried under a loud chorus, and the band weights
    favour drums over voice. Set it False for a plain, unweighted flux.
    """
    step = max(hop, (int(30.0 * sr) // hop) * hop)
    parts: list[np.ndarray] = []
    prev_last: np.ndarray | None = None
    weights: np.ndarray | None = None
    ref = 0.0                        # log-compression reference level
    for start in range(0, max(1, len(mono) - n_fft + 1), step):
        seg = mono[start:start + step + n_fft - hop]
        mag, freqs = _stft_mag(seg, sr, n_fft=n_fft, hop=hop)
        if len(mag) == 0:
            break
        if percussive:
            if weights is None:
                weights = _percussive_weights(freqs)
                pos = mag[mag > 0.0]
                # one reference for the whole track, taken from the first
                # chunk, so the compression is consistent across chunks
                ref = float(np.median(pos)) if len(pos) else 1.0
                ref = ref if ref > 1e-12 else 1.0
            mag = np.log1p(mag / ref) * weights
        prev = mag[:1] if prev_last is None else prev_last
        flux = np.maximum(mag - np.vstack([prev, mag[:-1]]), 0.0).sum(axis=1)
        if prev_last is None:
            flux[0] = 0.0            # no predecessor for the first frame
        prev_last = mag[-1:]
        parts.append(flux.astype(np.float32))
    if not parts:
        return np.zeros(0, np.float32)
    return np.concatenate(parts)


def _bpm_from_flux(flux: np.ndarray,
                   fps: float) -> tuple[float, float] | None:
    """
    (bpm, period_frames) from the autocorrelation of the onset envelope,
    searched over 60-200 BPM (folded into 70-180 when ambiguous). FFT
    autocorrelation keeps full-track envelopes O(n log n).
    """
    if len(flux) < 64:
        return None
    x = flux - flux.mean()
    if not np.any(x):
        return None
    nfft = 1 << int(np.ceil(np.log2(2 * len(x))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, nfft)) ** 2)[:len(x)]
    lag_min = max(2, int(round(60.0 * fps / 200.0)))     # 200 BPM
    lag_max = min(int(round(60.0 * fps / 60.0)), len(ac) - 2)  # 60 BPM
    if lag_max <= lag_min + 2:
        return None
    # Tempo prior: the raw autocorrelation of a drum pattern peaks at
    # every metrical level (half beat, beat, bar) and the longer lags
    # often win, which is how a 100 BPM loop used to read as 66.7. A
    # log-Gaussian weighting centred on 120 BPM picks the perceptually
    # dominant level instead (Ellis 2007, "Beat Tracking by Dynamic
    # Programming").
    lags = np.arange(lag_min, lag_max + 1)
    prior = np.exp(-0.5 * (np.log2((60.0 * fps / lags) / _TEMPO_PRIOR_BPM)
                           / _TEMPO_PRIOR_OCT) ** 2)
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1] * prior))
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]     # parabolic refinement
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    period = lag + float(np.clip(delta, -0.5, 0.5))
    bpm = 60.0 * fps / period
    return float(bpm), float(period)


def detect_bpm(mono: np.ndarray, sr: int) -> float | None:
    """Tempo estimate in BPM (scalar convenience wrapper)."""
    hop = _ONSET_HOP
    res = _bpm_from_flux(_onset_envelope(mono, sr, hop=hop), sr / hop)
    return round(res[0], 1) if res else None


def detect_beats(mono: np.ndarray,
                 sr: int) -> tuple[float, np.ndarray] | None:
    """
    Beat grid for the whole signal: (bpm, beat times in seconds).

    Global tempo comes from the autocorrelation of the spectral-flux
    envelope; the beat PHASE is chosen by comb alignment over the WHOLE
    envelope; each subsequent beat is predicted one period ahead and then
    snapped onto the strongest nearby flux peak (sub-frame accurate),
    which is what makes the click sit on the kick and still follow gentle
    tempo drift. Percussive input gives the most reliable grid (Gkiokas
    et al. 2012); the caller should prefer a drums stem when one is
    audible, and otherwise keep vocals out of the source signal (Zapata &
    Gómez 2013).
    """
    hop = _ONSET_HOP
    fps = sr / hop
    flux = _onset_envelope(mono, sr, hop=hop)
    res = _bpm_from_flux(flux, fps)
    if res is None:
        return None
    bpm, period = res                    # period in envelope frames
    if period < 4.0:
        return None
    # phase: comb over the ENTIRE envelope (an earlier version only used
    # the first 16 beats, so a quiet or rubato intro skewed the whole grid)
    best_o = _comb_phase(flux, period)
    half = max(1, int(round(0.12 * period)))
    pos_flux = flux[flux > 0]
    med = float(np.median(pos_flux)) if len(pos_flux) else 0.0

    # Note on the metrical level: with hats on the eighths, both the beat
    # and the eighth are valid pulses, and the tempo prior settles near
    # 120 BPM. Measured strength of the two levels does not separate a
    # right from a wrong choice (ratios overlap), so no heuristic guesses
    # here — the UI offers a ÷2 / ×2 switch and the musician decides.

    # predictive walk, snapping onto the onset peak within ±12% of the
    # period (a narrow window: it can only lock onto the beat's own
    # transient, not onto a syncopation)
    beats: list[float] = []
    t = float(best_o)
    while t < len(flux):
        c = int(round(t))
        lo, hi = max(0, c - half), min(len(flux), c + half + 1)
        if hi > lo:
            pk = lo + int(np.argmax(flux[lo:hi]))
            if flux[pk] > 1.5 * med:     # only snap to a real onset
                t = _parabolic_peak(flux, pk)
        beats.append(t)
        t += period
    if len(beats) < 2:
        return None
    # frame -> time: frame k spans [k*hop, k*hop + n_fft), and the flux
    # of a transient peaks on the frame that already contains it, so the
    # onset sits _ONSET_LAG_FRAMES frames before the frame index. The
    # constant is measured against synthetic clicks (see the beat tests).
    times = ((np.asarray(beats, dtype=np.float64) - _ONSET_LAG_FRAMES)
             * hop / sr)
    return round(float(bpm), 1), np.maximum(times, 0.0)


def _comb_phase(flux: np.ndarray, period: float) -> int:
    """Beat offset (in frames) whose comb collects the most onset energy."""
    best_o, best_s = 0, -1.0
    for o in range(max(1, int(round(period)))):
        idx = np.round(np.arange(o, len(flux) - 0.5, period)).astype(int)
        s = float(flux[idx].sum())
        if s > best_s:
            best_s, best_o = s, o
    return best_o


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position around index i by parabolic fit."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    den = a - 2.0 * b + c
    if den == 0.0:
        return float(i)
    return i + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


_LAG_WINDOW_S = 90.0     # original-time excerpt used to measure the lag
_LAG_MAX_S = 0.060       # search range, either side
_LAG_HOP = 128           # 2.9 ms envelope frames (sub-frame by parabola)


def measure_stretch_lag(orig: np.ndarray, stretched: np.ndarray, sr: int,
                        speed: float) -> float:
    """
    Where the time-stretched audio really is, relative to t / speed.

    atempo does not hand events back exactly at t / speed: its splices
    move them by ~2..16 ms, and HOW much depends on the material (the
    splice search runs on the content, so sustained tones and drums pull
    it differently — measured on 7 synthetic mixes). A fixed constant
    cannot fix that, so the offset is measured on every stretch: the
    onset envelope of the original, mapped to the stretched timeline, is
    cross-correlated with the envelope of the stretched audio itself.

    Returns seconds (positive = the stretched audio is late); 0.0 when the
    excerpt has no clear onsets to lock to (silence, pads).
    """
    n_o = min(len(orig), int(_LAG_WINDOW_S * sr))
    n_s = min(len(stretched), int(round(n_o / speed)))
    e_o = _onset_envelope(orig[:n_o], sr, hop=_LAG_HOP)
    e_s = _onset_envelope(stretched[:n_s], sr, hop=_LAG_HOP)
    if len(e_o) < 8 or len(e_s) < 8:
        return 0.0
    # stretched frame k is centred on sample k*hop + n_fft/2, which shows
    # original sample (k*hop + n_fft/2) * speed; map through the frame
    # CENTRES — scaling the frame index alone would also scale the half
    # window and bias the lag by n_fft/2 * (1/speed - 1)
    half = _ONSET_NFFT / 2.0
    src = ((np.arange(len(e_s)) * _LAG_HOP + half) * speed - half) / _LAG_HOP
    e_m = np.interp(src, np.arange(len(e_o)), e_o)
    e_m = e_m - e_m.mean()
    e_s = e_s - e_s.mean()
    max_lag = int(round(_LAG_MAX_S * sr / _LAG_HOP))
    if len(e_s) <= 2 * max_lag + 8:
        return 0.0
    core = slice(max_lag, len(e_s) - max_lag)
    ref = e_s[core]
    xc = np.array([float(np.dot(ref, e_m[max_lag - L:len(e_s) - max_lag - L]))
                   for L in range(-max_lag, max_lag + 1)])
    i = int(np.argmax(xc))
    norm = float(np.linalg.norm(ref) * np.linalg.norm(e_m[core])) or 1.0
    if xc[i] / norm < 0.3:                 # nothing to lock onto
        return 0.0
    return (_parabolic_peak(xc, i) - max_lag) * _LAG_HOP / sr


def scale_beat_grid(beats: np.ndarray, mult: float) -> np.ndarray:
    """
    Same grid at another metrical level, keeping the phase locked:
    mult=2 adds the midpoints (eighths), mult=0.5 keeps every other beat.
    Tempo tracking cannot know which level the musician wants to hear —
    both are musically valid — so this is a UI control, not a guess.
    """
    if beats is None or len(beats) < 2 or mult == 1.0:
        return beats
    if mult == 2.0:
        mid = 0.5 * (beats[:-1] + beats[1:])
        out = np.empty(len(beats) + len(mid), dtype=beats.dtype)
        out[0::2], out[1::2] = beats, mid
        return out
    if mult == 0.5:
        return beats[::2]
    return beats


def render_click_track(beat_times: np.ndarray, n_frames: int,
                       sr: int) -> np.ndarray:
    """
    Mono float32 metronome buffer on the tracks' timeline: a short 1 kHz
    decaying sine burst on every beat (classic click), full-scale 0.9.
    """
    click = np.zeros(n_frames, dtype=np.float32)
    dur = max(8, int(0.030 * sr))
    t = np.arange(dur, dtype=np.float32) / sr
    burst = (0.9 * np.sin(2.0 * np.pi * 1000.0 * t)
             * np.exp(-t / 0.008)).astype(np.float32)
    for bt in beat_times:
        i = int(round(bt * sr))
        if 0 <= i < n_frames:
            m = min(dur, n_frames - i)
            click[i:i + m] += burst[:m]
    np.clip(click, -1.0, 1.0, out=click)
    return click


def _pick_beat_source(audible: list, n_frames: int) -> np.ndarray | None:
    """
    Choose what the beat tracker listens to, from the audible stems:

      1. the DRUMS stem, when it is audible and not silent — percussive
         material carries the pulse (Gkiokas et al. 2012; Chiu 2021);
      2. otherwise the audible mix WITHOUT the vocals stem — predominant
         vocals push the onset envelope around and make the tempo read
         high (Zapata & Gómez 2013), which is exactly what showed up in
         testing when the voice was up;
      3. otherwise None, and the caller falls back to the full audible mix.

    Returns a mono, post-fader buffer.
    """
    def mono_of(track) -> np.ndarray:
        return track.orig.mean(axis=1) * track.gain   # original speed

    def loud(x: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(x * x))) > 1e-4

    for t in audible:
        if t.name == STEM_LABELS["drums"]:
            d = mono_of(t)
            if loud(d):
                return d
            break

    rest = [t for t in audible if t.name != STEM_LABELS["vocals"]]
    if not rest or len(rest) == len(audible):
        return None                      # nothing to drop: use the mix
    out = np.zeros(n_frames, dtype=np.float32)
    for t in rest:
        out += mono_of(t)
    return out if loud(out) else None


def detect_key(mono: np.ndarray, sr: int) -> str | None:
    """
    Musical key estimate ("A minor", "F# major", ...) by correlating the
    WHOLE-signal average chromagram with the 24 Krumhansl-Kessler key
    profiles. Full-track accumulation (in ~60 s STFT chunks, memory-flat)
    follows the industry convention — KeyFinder, Essentia and madmom all
    report one global key computed over the entire track.
    """
    n_fft, hop = 4096, 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 55.0) & (freqs <= 2000.0)
    pcs = np.round(69.0 + 12.0 * np.log2(freqs[band] / 440.0)).astype(int) % 12
    chroma = np.zeros(12, dtype=np.float64)
    step = max(hop, (int(60.0 * sr) // hop) * hop)
    for start in range(0, max(1, len(mono) - n_fft + 1), step):
        seg = mono[start:start + step + n_fft - hop]
        mag, _ = _stft_mag(seg, sr, n_fft=n_fft, hop=hop)
        if len(mag) == 0:
            break
        energy = (mag[:, band] ** 2).sum(axis=0).astype(np.float64)
        chroma += np.bincount(pcs, weights=energy, minlength=12)
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

    def __init__(self, master, width: int = 150, height: int = 12,
                 orient: str = "h"):
        super().__init__(master, width=width, height=height,
                         bg=COL_TROUGH, highlightthickness=1,
                         highlightbackground=COL_BORDER)
        self._meter_w, self._meter_h = width, height
        self._orient = orient          # "h": left→right, "v": bottom→top
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
        if self._orient == "v":
            seg_h = (self._meter_h - gap) / self.SEGMENTS
            for i in range(self.SEGMENTS):
                # bottom = floor, top = 0 dBFS
                seg_db = METER_FLOOR_DB * (1.0 - (i + 1) / self.SEGMENTS)
                y1 = self._meter_h - 2 - i * seg_h
                self.create_rectangle(
                    2, y1 - seg_h + gap, self._meter_w, y1,
                    fill=self._seg_color(seg_db, i < lit_count), width=0)
            return
        seg_w = (self._meter_w - gap) / self.SEGMENTS
        for i in range(self.SEGMENTS):
            # dB value this segment represents (left = floor, right = 0 dBFS)
            seg_db = METER_FLOOR_DB * (1.0 - (i + 1) / self.SEGMENTS)
            x0 = 2 + i * seg_w
            self.create_rectangle(
                x0, 2, x0 + seg_w - gap, self._meter_h,
                fill=self._seg_color(seg_db, i < lit_count), width=0)

    def set_height(self, height: int) -> None:
        """Resize a vertical meter to follow its fader, and redraw."""
        if height == self._meter_h:
            return
        self._meter_h = height
        self.configure(height=height)
        self._drawn = -1
        self._draw(0)

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
        self.name_label.grid(row=0, column=0, padx=(22, 8), pady=5,
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
        self.slider.grid(row=0, column=2, padx=8, pady=5, sticky="ew")

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
        self.mute_btn.grid(row=0, column=4, padx=4, pady=5)

        self.solo_btn = ctk.CTkButton(
            self, text="S", width=28, height=28, corner_radius=14,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            hover_color=BTN_GHOST_HOV, command=self._toggle_solo,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.solo_btn.grid(row=0, column=5, padx=(4, 22), pady=5)

        self._refresh_value_label()
        self._refresh_toggles()   # layout switches rebuild rows: keep M/S state

    def _on_slider(self, value: float) -> None:
        self.track.gain = float(value) / 100.0     # linear amplitude map
        self._refresh_value_label()
        self._on_change()

    def _refresh_value_label(self) -> None:
        self.value_label.configure(text=f"{int(round(self.track.gain * 100))}%")

    def _refresh_toggles(self) -> None:
        # tema §5: active M = warm-white bg, active S = amber, dark text
        self.mute_btn.configure(
            fg_color=BTN_PRI_BG if self.track.mute else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.mute else COL_TEXT_2)
        self.solo_btn.configure(
            fg_color=AMBER if self.track.solo else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.solo else COL_TEXT_2)

    def _toggle_mute(self) -> None:
        self.track.mute = not self.track.mute
        self._refresh_toggles()
        self._on_change()

    def _toggle_solo(self) -> None:
        self.track.solo = not self.track.solo
        self._refresh_toggles()
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
        self.name_label.grid(row=0, column=0, padx=(22, 8), pady=5,
                             sticky="w")

        self.meter = VUMeter(self, width=150, height=14)
        self.meter.grid(row=0, column=1, padx=(0, 8))

        self.slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=100,
            fg_color=COL_TROUGH, progress_color=AMBER,
            button_color=AMBER, button_hover_color=AMBER_HOVER,
            corner_radius=3, height=16,
            command=self._on_slider)
        self.slider.set(engine.master_gain * 100.0)
        self.slider.grid(row=0, column=2, padx=8, pady=5, sticky="ew")

        self.value_label = ctk.CTkLabel(
            self, text=f"{int(round(engine.master_gain * 100))}%",
            width=52, anchor="e", text_color=COL_TEXT,
            font=ctk.CTkFont(family=MONO_FAMILY, size=13, weight="bold"))
        self.value_label.grid(row=0, column=3, padx=(0, 22))

    def _on_slider(self, value: float) -> None:
        self.engine.master_gain = float(value) / 100.0
        self.value_label.configure(text=f"{int(round(float(value)))}%")


# --- vertical console strips (mixer layout "strips", v2.1.1) ---------------
# Same widgets as the rows above, stacked vertically and placed side by
# side, so 6 stems + master fit on a 768 px laptop screen with no scrolling.

_STRIP_FADER_H = 128        # default fader travel (resized to fit at runtime)
_STRIP_W = 104              # channel width (7 strips ≈ 760 px)
_STRIP_CHROME_H = 128       # name + value + M/S + paddings + scrollbar
_STRIP_CHROME_MIN = 108     # same, with the % label hidden (short windows)
_STRIP_COMPACT_AT = 186     # viewport height (logical px) that triggers it


def _px(widget, value: float) -> int:
    """
    CustomTkinter scales its widgets by the display scaling factor, but a
    raw tk.Canvas (the VU meter) does not — convert to real pixels so the
    meter always matches the fader beside it.
    """
    try:
        return int(round(value * ctk.ScalingTracker.get_widget_scaling(widget)))
    except Exception:
        return int(value)


class TrackStrip(ctk.CTkFrame):
    """One vertical mixer channel: name, VU + fader, value, Mute, Solo."""

    def __init__(self, master, track: Track, on_change):
        super().__init__(master, fg_color=COL_ELEV, corner_radius=18)
        self.track = track
        self._on_change = on_change

        self.name_label = ctk.CTkLabel(
            self, text=track.name, width=_STRIP_W - 16, height=18,
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.name_label.grid(row=0, column=0, columnspan=2,
                             padx=8, pady=(8, 4))

        self.meter = VUMeter(self, width=_px(self, 12),
                             height=_px(self, _STRIP_FADER_H), orient="v")
        self.meter.grid(row=1, column=0, padx=(14, 4), pady=2)

        self.slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=100,
            orientation="vertical", height=_STRIP_FADER_H, width=16,
            fg_color=COL_TROUGH, progress_color=COL_TEXT,
            button_color=COL_TEXT, button_hover_color=BTN_PRI_HOV,
            corner_radius=3, command=self._on_slider)
        self.slider.set(track.gain * 100.0)
        self.slider.grid(row=1, column=1, padx=(4, 14), pady=2)

        self.value_label = ctk.CTkLabel(
            self, text="100%", height=16, text_color=COL_TEXT_2,
            font=ctk.CTkFont(family=MONO_FAMILY, size=12))
        self.value_label.grid(row=2, column=0, columnspan=2, pady=(2, 2))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.grid(row=3, column=0, columnspan=2, pady=(0, 8))
        self.mute_btn = ctk.CTkButton(
            btns, text="M", width=28, height=26, corner_radius=13,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            hover_color=BTN_GHOST_HOV, command=self._toggle_mute,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.mute_btn.grid(row=0, column=0, padx=3)
        self.solo_btn = ctk.CTkButton(
            btns, text="S", width=28, height=26, corner_radius=13,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            hover_color=BTN_GHOST_HOV, command=self._toggle_solo,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.solo_btn.grid(row=0, column=1, padx=3)

        self._refresh_value_label()
        self._refresh_toggles()

    def set_fader_height(self, h: int, compact: bool = False) -> None:
        """Fit the fader (and its meter) to the mixer viewport; in a very
        short window drop the % readout so M / S stay reachable."""
        self.slider.configure(height=h)
        self.meter.set_height(_px(self, h))
        if compact:
            self.value_label.grid_remove()
        else:
            self.value_label.grid()

    # the four methods below mirror TrackRow's, on the same Track object
    def _on_slider(self, value: float) -> None:
        self.track.gain = float(value) / 100.0
        self._refresh_value_label()
        self._on_change()

    def _refresh_value_label(self) -> None:
        self.value_label.configure(text=f"{int(round(self.track.gain * 100))}%")

    def _refresh_toggles(self) -> None:
        self.mute_btn.configure(
            fg_color=BTN_PRI_BG if self.track.mute else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.mute else COL_TEXT_2)
        self.solo_btn.configure(
            fg_color=AMBER if self.track.solo else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if self.track.solo else COL_TEXT_2)

    def _toggle_mute(self) -> None:
        self.track.mute = not self.track.mute
        self._refresh_toggles()
        self._on_change()

    def _toggle_solo(self) -> None:
        self.track.solo = not self.track.solo
        self._refresh_toggles()
        self._on_change()


CONTACT_EMAIL = "mscanabarro@gmail.com"


class ContactDialog(ctk.CTkToplevel):
    """
    The "?" dialog: who made this and how to reach them, plus the exact
    version. The app circulates hand to hand, so every copy has to carry
    a way back to its author and say which build it is.
    """

    def __init__(self, master):
        super().__init__(master, fg_color=COL_BG)
        self.title(L("contact_title"))
        self.resizable(False, False)
        self.transient(master)
        try:
            base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
            if (base / "isolate.ico").exists():
                self.after(200, lambda: self.iconbitmap(
                    str(base / "isolate.ico")))
        except Exception:
            pass

        card = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                            border_width=1, border_color=COL_BORDER)
        card.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(card, text="Isolate", text_color=COL_TEXT,
                     font=ctk.CTkFont(family=UI_FAMILY, size=22,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=26, pady=(20, 0),
                            sticky="w")
        ctk.CTkLabel(card, text=L("contact_version", v=APP_VERSION),
                     text_color=AMBER,
                     font=ctk.CTkFont(family=MONO_FAMILY, size=13,
                                      weight="bold")
                     ).grid(row=1, column=0, padx=26, pady=(0, 14),
                            sticky="w")

        ctk.CTkLabel(card, text=L("contact_head"), text_color=COL_TEXT,
                     font=ctk.CTkFont(family=UI_FAMILY, size=14,
                                      weight="bold")
                     ).grid(row=2, column=0, padx=26, sticky="w")
        ctk.CTkLabel(card, text=L("contact_body"), text_color=COL_TEXT_2,
                     justify="left", wraplength=330,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12)
                     ).grid(row=3, column=0, padx=26, pady=(4, 12),
                            sticky="w")

        mail_chip = ctk.CTkFrame(card, fg_color=CHIP_BG, corner_radius=999,
                                 border_width=1, border_color=CHIP_BORDER)
        mail_chip.grid(row=4, column=0, padx=26, sticky="w")
        ctk.CTkLabel(mail_chip, text=CONTACT_EMAIL, text_color=AMBER,
                     font=ctk.CTkFont(family=MONO_FAMILY, size=13)
                     ).grid(row=0, column=0, padx=18, pady=7)

        self.note = ctk.CTkLabel(card, text="", text_color=OK_GREEN,
                                 font=ctk.CTkFont(family=UI_FAMILY, size=11))
        self.note.grid(row=5, column=0, padx=26, pady=(6, 0), sticky="w")

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.grid(row=6, column=0, padx=26, pady=(10, 20), sticky="e")
        ctk.CTkButton(btns, text=L("contact_write"), width=130, height=32,
                      corner_radius=999, fg_color=BTN_PRI_BG,
                      text_color=BTN_PRI_TX, hover_color=BTN_PRI_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                       weight="bold"),
                      command=self._write).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(btns, text=L("contact_copy"), width=120, height=32,
                      corner_radius=999, fg_color=BTN_GHOST_BG,
                      text_color=COL_TEXT, border_width=1,
                      border_color=BTN_GHOST_BRD,
                      hover_color=BTN_GHOST_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12),
                      command=self._copy).grid(row=0, column=1, padx=(0, 8))
        ctk.CTkButton(btns, text=L("contact_close"), width=90, height=32,
                      corner_radius=999, fg_color=BTN_GHOST_BG,
                      text_color=COL_TEXT_2, border_width=1,
                      border_color=BTN_GHOST_BRD,
                      hover_color=BTN_GHOST_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12),
                      command=self.destroy).grid(row=0, column=2)

        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        # centred on the main window
        x = master.winfo_rootx() + (master.winfo_width()
                                    - self.winfo_width()) // 2
        y = master.winfo_rooty() + (master.winfo_height()
                                    - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.after(120, self.grab_set)          # modal once it is mapped

    def _write(self) -> None:
        import webbrowser
        webbrowser.open(f"mailto:{CONTACT_EMAIL}"
                        f"?subject=Isolate%20v{APP_VERSION}")

    def _copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(CONTACT_EMAIL)
        self.note.configure(text=L("contact_copied"))


class LibraryDialog(ctk.CTkToplevel):
    """
    The song library: every song saved with "Save to library", newest
    first, reopened in one click — stems, key, BPM, beat grid, mixer and
    speed as they were left, offline, with no separation to wait for.
    (Asked for by a drummer building an archive of songs for his lessons.)
    """

    def __init__(self, app: "IsolateApp"):
        super().__init__(app, fg_color=COL_BG)
        self.app = app
        self.title(L("lib_title"))
        self.geometry("660x540")
        self.minsize(540, 380)
        self.transient(app)
        try:
            base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
            if (base / "isolate.ico").exists():
                self.after(200, lambda: self.iconbitmap(
                    str(base / "isolate.ico")))
        except Exception:
            pass

        card = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                            border_width=1, border_color=COL_BORDER)
        card.pack(fill="both", expand=True, padx=14, pady=14)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 2))
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text=L("lib_title"), text_color=COL_TEXT,
                     font=ctk.CTkFont(family=UI_FAMILY, size=22,
                                      weight="bold")
                     ).grid(row=0, column=0, sticky="w")
        self.count_label = ctk.CTkLabel(
            head, text="", text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=12))
        self.count_label.grid(row=0, column=1, sticky="w", padx=(12, 0),
                              pady=(6, 0))
        # placeholder text only shows on a CTkEntry without textvariable,
        # so the filter reads the entry on every key release instead
        self.search = ctk.CTkEntry(
            head, placeholder_text=L("lib_search"), width=210, height=32,
            corner_radius=999, fg_color=COL_ELEV, border_width=1,
            border_color=BTN_GHOST_BRD, text_color=COL_TEXT,
            placeholder_text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=12))
        self.search.grid(row=0, column=2, sticky="e")
        self.search.bind("<KeyRelease>", lambda e: self._render())

        where = ctk.CTkFrame(card, fg_color="transparent")
        where.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 8))
        where.grid_columnconfigure(0, weight=1)
        self.path_label = ctk.CTkLabel(
            where, text="", anchor="w", text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=MONO_FAMILY, size=11))
        self.path_label.grid(row=0, column=0, sticky="w")
        for col, (key, cmd) in enumerate((("lib_open_folder", self._open_folder),
                                          ("lib_change_folder",
                                           self._change_folder)), start=1):
            ctk.CTkButton(where, text=L(key), height=26, width=110,
                          corner_radius=999, fg_color=BTN_GHOST_BG,
                          text_color=COL_TEXT_2, border_width=1,
                          border_color=BTN_GHOST_BRD,
                          hover_color=BTN_GHOST_HOV,
                          font=ctk.CTkFont(family=UI_FAMILY, size=11),
                          command=cmd).grid(row=0, column=col, padx=(6, 0))

        self.list = ctk.CTkScrollableFrame(card, corner_radius=16,
                                           fg_color="transparent")
        self.list.grid(row=2, column=0, sticky="nsew", padx=10)
        self.list.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(card, text=L("lib_close"), width=90, height=32,
                      corner_radius=999, fg_color=BTN_GHOST_BG,
                      text_color=COL_TEXT_2, border_width=1,
                      border_color=BTN_GHOST_BRD, hover_color=BTN_GHOST_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12),
                      command=self.destroy
                      ).grid(row=3, column=0, padx=22, pady=(8, 16),
                             sticky="e")

        self._root_dir = library_dir()
        self._songs: list[tuple[Path, dict]] = []
        self._reload()

        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        x = app.winfo_rootx() + (app.winfo_width() - self.winfo_width()) // 2
        y = app.winfo_rooty() + (app.winfo_height()
                                 - self.winfo_height()) // 3
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.after(120, self.grab_set)          # modal once it is mapped

    # -- content -----------------------------------------------------------

    def _reload(self) -> None:
        self._root_dir = library_dir()
        path = str(self._root_dir)
        if len(path) > 44:              # keep the tail, the meaningful part
            path = "…" + path[-43:]
        self.path_label.configure(text=path)
        self._songs = list_library(self._root_dir)
        self._render()

    def _render(self) -> None:
        for child in self.list.winfo_children():
            child.destroy()
        q = self.search.get().strip().lower()
        songs = [(f, m) for f, m in self._songs
                 if not q or q in str(m.get("title", f.name)).lower()]
        n = len(self._songs)
        self.count_label.configure(
            text=L("lib_count_1") if n == 1 else L("lib_count", n=n))
        if not songs:
            ctk.CTkLabel(self.list,
                         text=L("lib_no_match", q=q) if q else L("lib_empty"),
                         text_color=COL_TEXT_DIM, justify="center",
                         font=ctk.CTkFont(family=UI_FAMILY, size=13)
                         ).grid(row=0, column=0, pady=40)
            return
        for i, (folder, m) in enumerate(songs):
            self._song_row(i, folder, m)

    def _song_row(self, i: int, folder: Path, m: dict) -> None:
        row = ctk.CTkFrame(self.list, fg_color=COL_ELEV, corner_radius=18)
        row.grid(row=i, column=0, sticky="ew", padx=4, pady=3)
        row.grid_columnconfigure(0, weight=1)

        title = str(m.get("title") or folder.name)
        shown = title if len(title) <= 52 else title[:51] + "…"
        name = ctk.CTkLabel(row, text=shown, anchor="w", text_color=COL_TEXT,
                            font=ctk.CTkFont(family=UI_FAMILY, size=14,
                                             weight="bold"))
        name.grid(row=0, column=0, sticky="w", padx=(18, 8), pady=(8, 0))

        mode = m.get("mode")
        mode_txt = (_SPEC_TO_SHORT.get(mode) or f"{len(m['stems'])} stems"
                    if mode else L("lib_mode_original"))
        sr = m.get("samplerate") or 44100
        dur = format_time(float(m.get("frames", 0)) / sr)
        try:
            saved = datetime.fromisoformat(m.get("saved", ""))
            saved_txt = saved.strftime("%d/%m/%Y" if LANG == "pt"
                                       else "%Y-%m-%d")
        except ValueError:
            saved_txt = ""
        meta = ctk.CTkFrame(row, fg_color="transparent")
        meta.grid(row=1, column=0, sticky="w", padx=(18, 8), pady=(0, 8))
        # tema §5: amber belongs to Key / BPM
        musical = " · ".join(t for t in (
            key_short(m.get("key")),
            f"{m['bpm']:.0f} BPM" if m.get("bpm") else None) if t)
        info = "  ·  ".join(t for t in (mode_txt, dur, saved_txt) if t)
        ctk.CTkLabel(meta, text=info + ("  ·" if musical else ""),
                     text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12)
                     ).grid(row=0, column=0)
        if musical:
            ctk.CTkLabel(meta, text=musical, text_color=AMBER,
                         font=ctk.CTkFont(family=MONO_FAMILY, size=12)
                         ).grid(row=0, column=1, padx=(6, 0))

        ctk.CTkButton(row, text=L("lib_open"), width=78, height=30,
                      corner_radius=999, fg_color=BTN_PRI_BG,
                      text_color=BTN_PRI_TX, hover_color=BTN_PRI_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                       weight="bold"),
                      command=lambda: self._open(folder)
                      ).grid(row=0, column=1, rowspan=2, padx=(0, 6))
        ctk.CTkButton(row, text="✕", width=30, height=30, corner_radius=15,
                      fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
                      hover_color=BTN_GHOST_HOV,
                      font=ctk.CTkFont(family=UI_FAMILY, size=12),
                      command=lambda: self._delete(folder, title)
                      ).grid(row=0, column=2, rowspan=2, padx=(0, 14))
        for w in (row, name, meta):
            w.bind("<Double-Button-1>", lambda e: self._open(folder))

    # -- actions -----------------------------------------------------------

    def _open(self, folder: Path) -> None:
        self.destroy()
        self.app._open_session(folder)

    def _delete(self, folder: Path, title: str) -> None:
        if not messagebox.askyesno(_APP_NAME, L("lib_delete_q", name=title),
                                   parent=self):
            return
        try:
            send_to_recycle_bin(folder)
        except Exception as exc:
            log.error("Delete failed:\n%s", traceback.format_exc())
            messagebox.showerror(_APP_NAME, L("err_delete", err=exc),
                                 parent=self)
        if self.app._session_dir == folder:
            self.app._session_dir = None    # next save makes a new entry
        self._reload()

    def _open_folder(self) -> None:
        try:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(self._root_dir))
        except Exception as exc:
            log.warning("Could not open the library folder: %s", exc)

    def _change_folder(self) -> None:
        start = self._root_dir if self._root_dir.is_dir() else Path.home()
        path = filedialog.askdirectory(parent=self, initialdir=str(start),
                                       title=L("lib_choose_folder"))
        if not path:
            return
        _SETTINGS["library_dir"] = str(Path(path))
        _save_settings(_SETTINGS)
        self._reload()


class MasterStrip(ctk.CTkFrame):
    """Vertical master bus strip — amber, always the leftmost channel."""

    def __init__(self, master, engine: AudioEngine):
        super().__init__(master, fg_color=COL_ELEV, corner_radius=18,
                         border_width=1, border_color=MASTER_BORDER)
        self.engine = engine

        self.name_label = ctk.CTkLabel(
            self, text="MASTER", width=_STRIP_W - 16, height=18,
            text_color=AMBER,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"))
        self.name_label.grid(row=0, column=0, columnspan=2,
                             padx=8, pady=(8, 4))

        self.meter = VUMeter(self, width=_px(self, 12),
                             height=_px(self, _STRIP_FADER_H), orient="v")
        self.meter.grid(row=1, column=0, padx=(14, 4), pady=2)

        self.slider = ctk.CTkSlider(
            self, from_=0, to=100, number_of_steps=100,
            orientation="vertical", height=_STRIP_FADER_H, width=16,
            fg_color=COL_TROUGH, progress_color=AMBER,
            button_color=AMBER, button_hover_color=AMBER_HOVER,
            corner_radius=3, command=self._on_slider)
        self.slider.set(engine.master_gain * 100.0)
        self.slider.grid(row=1, column=1, padx=(4, 14), pady=2)

        self.value_label = ctk.CTkLabel(
            self, text=f"{int(round(engine.master_gain * 100))}%",
            height=16, text_color=COL_TEXT,
            font=ctk.CTkFont(family=MONO_FAMILY, size=12, weight="bold"))
        self.value_label.grid(row=2, column=0, columnspan=2, pady=(2, 2))

        # keeps the master aligned with the stems' M/S button row
        ctk.CTkFrame(self, fg_color="transparent", height=26,
                     width=10).grid(row=3, column=0, columnspan=2,
                                    pady=(0, 8))

    def set_fader_height(self, h: int, compact: bool = False) -> None:
        self.slider.configure(height=h)
        self.meter.set_height(_px(self, h))
        if compact:
            self.value_label.grid_remove()
        else:
            self.value_label.grid()

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

        self.title(f"Isolate v{APP_VERSION} — Stem Splitter & "
                   "Multi-Track Mixer")
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
        self._analysis_dirty = False
        self._analysis_dirty_key = False
        self._key_last: str | None = None
        self._bpm_global: float | None = None
        self._bpm_live_txt: str | None = None
        self._mix_change_job: str | None = None
        self._status_queue: queue.Queue[str] = queue.Queue()
        self.track_rows: list[TrackRow | TrackStrip] = []
        self._strip_fader_h = _STRIP_FADER_H
        self._strip_compact = False
        self._beats_base: np.ndarray | None = None   # detected grid, 1x
        self._metro_mult = 1.0                       # ÷2 / 1x / ×2
        self._speed_lag = 0.0       # measured offset of the stretched audio
        self._mode_spec: str | None = None    # separation run, None = original
        self._source_url: str | None = None   # YouTube URL of the material
        self._session_dir: Path | None = None  # library folder it lives in

        self._build_ui()
        self._populate_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind_all("<space>", self._on_space)   # space = play/pause
        self.after(UI_POLL_MS, self._poll)

        if not find_ffmpeg():
            self._set_status(L("st_no_ffmpeg"))

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)      # the mixer takes the slack

        # ---------- Top bar ----------
        top = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                           border_width=1, border_color=COL_BORDER)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

        # Drag & drop landing zone (tema §5: solid 1.5px border, radius 22)
        drop_text = L("drop_dnd") if _HAS_DND else L("drop_click")
        self.drop_zone = ctk.CTkFrame(
            top, corner_radius=22, fg_color=COL_TROUGH,
            border_width=2, border_color=BTN_GHOST_BRD)
        self.drop_zone.grid(row=0, column=0, sticky="ew",
                            padx=(10, 6), pady=10)
        self.drop_label = ctk.CTkLabel(
            self.drop_zone, text=drop_text, height=40,
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
            url_frame, placeholder_text=L("url_placeholder"), height=36,
            corner_radius=999, fg_color=COL_ELEV,
            border_width=1, border_color=BTN_GHOST_BRD,
            text_color=COL_TEXT, placeholder_text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=13))
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.download_btn = ctk.CTkButton(
            url_frame, text=L("btn_download"), width=140, height=36,
            corner_radius=999, fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_download)
        self.download_btn.grid(row=0, column=1)
        # the library is a third source of material, next to the file drop
        # and YouTube: songs already separated, reopened offline
        self.library_btn = ctk.CTkButton(
            url_frame, text=L("btn_library"), width=104, height=36,
            corner_radius=999, fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_library)
        self.library_btn.grid(row=0, column=2, padx=(6, 0))

        # Device selector + transport
        row2 = ctk.CTkFrame(top, fg_color="transparent")
        row2.grid(row=1, column=0, columnspan=2, sticky="ew",
                  padx=10, pady=(0, 10))
        row2.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(row2, text=L("lbl_output"), text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12)
                     ).grid(row=0, column=0, padx=(0, 6))
        self.device_menu = ctk.CTkOptionMenu(
            row2, values=[L("device_default")], width=240, corner_radius=999,
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

        # practice speed (pitch kept): lives with the transport because it
        # changes how the music plays, not how it is mixed
        speed_box = ctk.CTkFrame(row2, fg_color="transparent")
        speed_box.grid(row=0, column=5, sticky="w", padx=(14, 6))
        # collapsed by default: one pill with the current speed; the six
        # options unfold beside it on click and fold back once one is picked
        self._speed_open = False
        self.speed_btn = ctk.CTkButton(
            speed_box, text="", width=150, height=32, corner_radius=999,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._toggle_speed_menu)
        self.speed_btn.grid(row=0, column=0, padx=(0, 6))
        self.speed_picker = ctk.CTkSegmentedButton(
            speed_box, values=[f"{round(s * 100)}%" for s in SPEED_STEPS],
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._on_speed)
        self.speed_picker.set("100%")
        self.speed_picker.grid(row=0, column=1)
        self.speed_picker.grid_remove()
        self._refresh_speed_btn(1.0)

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
                           padx=10, pady=(0, 8))
        self.timeline.bind("<Button-1>", self._on_timeline_press)
        self.timeline.bind("<ButtonRelease-1>", self._on_timeline_release)

        # ---------- Configuration panel ----------
        cfg = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                           border_width=1, border_color=COL_BORDER)
        cfg.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        cfg.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(cfg, text=L("lbl_sep_mode"),
                     text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                      weight="bold")
                     ).grid(row=0, column=0, rowspan=2,
                            padx=(24, 18), pady=8, sticky="w")

        # one pill row + a caption with the stems of the selected mode
        # (was a 4-row radio stack: ~110 px of the mixer's height)
        self.stem_var = ctk.StringVar(value=L("stems4_short"))
        self.stem_picker = ctk.CTkSegmentedButton(
            cfg, values=STEM_SHORT, variable=self.stem_var,
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_stem_mode)
        self.stem_picker.grid(row=0, column=1, columnspan=2, sticky="w",
                              padx=(0, 16), pady=(9, 0))

        self.stem_desc = ctk.CTkLabel(
            cfg, text=STEM_DESC[self.stem_var.get()], anchor="w",
            text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=12))
        self.stem_desc.grid(row=1, column=1, columnspan=2, sticky="w",
                            padx=(2, 16), pady=(1, 8))

        self.separate_btn = ctk.CTkButton(
            cfg, text=L("btn_separate"), height=42, width=190,
            corner_radius=999, fg_color=BTN_PRI_BG, text_color=BTN_PRI_TX,
            hover_color=BTN_PRI_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=15, weight="bold"),
            command=self._on_separate)
        self.separate_btn.grid(row=0, column=4, rowspan=2,
                               padx=(0, 24), pady=8, sticky="e")
        # what the separation produced goes to the library from right here
        self.save_lib_btn = ctk.CTkButton(
            cfg, text=L("btn_save_lib"), height=42, width=170,
            corner_radius=999, fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=13, weight="bold"),
            command=self._on_save_session)
        self.save_lib_btn.grid(row=0, column=3, rowspan=2,
                               padx=(0, 10), pady=8, sticky="e")

        # ---------- Mixer (with the musical analysis in its header) ----------
        mix_wrap = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                                border_width=1, border_color=COL_BORDER)
        mix_wrap.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        mix_wrap.grid_columnconfigure(0, weight=1)
        mix_wrap.grid_rowconfigure(1, weight=1)

        # v2.1.2: the analysis used to be a panel of its own. On a 768 px
        # laptop that row alone cost the mixer a whole channel, and the
        # analysis belongs next to the mixer anyway (the BPM follows the
        # audible mix and the click plays with it), so it now lives in the
        # mixer's header line.
        head = ctk.CTkFrame(mix_wrap, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=14, pady=(6, 0))
        head.grid_columnconfigure(4, weight=1)
        ctk.CTkLabel(head, text=L("mixer_title"), text_color=COL_TEXT_2,
                     font=ctk.CTkFont(family=UI_FAMILY, size=12,
                                      weight="bold")
                     ).grid(row=0, column=0, sticky="w", padx=(2, 14))

        # tema §5: two chips (Tom | BPM), caption and value on one line
        key_chip = ctk.CTkFrame(head, fg_color=CHIP_BG, corner_radius=999,
                                border_width=1, border_color=CHIP_BORDER)
        key_chip.grid(row=0, column=1, padx=(0, 8), pady=4)
        ctk.CTkLabel(key_chip, text=L("chip_key"), text_color=CHIP_BORDER,
                     font=ctk.CTkFont(family=UI_FAMILY, size=10,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=(14, 6), pady=5)
        self.key_label = ctk.CTkLabel(
            key_chip, text="—", width=48, anchor="w", text_color=AMBER_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=17, weight="bold"))
        self.key_label.grid(row=0, column=1, padx=(0, 14), pady=5)

        bpm_chip = ctk.CTkFrame(head, fg_color=CHIP_BG, corner_radius=999,
                                border_width=1, border_color=CHIP_BORDER)
        bpm_chip.grid(row=0, column=2, padx=(0, 12), pady=4)
        ctk.CTkLabel(bpm_chip, text="BPM", text_color=CHIP_BORDER,
                     font=ctk.CTkFont(family=UI_FAMILY, size=10,
                                      weight="bold")
                     ).grid(row=0, column=0, padx=(14, 6), pady=5)
        self.bpm_label = ctk.CTkLabel(
            bpm_chip, text="—", width=48, anchor="w", text_color=AMBER_DIM,
            font=ctk.CTkFont(family=MONO_FAMILY, size=17, weight="bold"))
        self.bpm_label.grid(row=0, column=1, padx=(0, 14), pady=5)

        # metronome: the audible form of the BPM chip — not a stem, so it
        # stays out of the channel list; the click is injected post-master
        # in the engine, with its own volume
        metro = ctk.CTkFrame(head, fg_color="transparent")
        metro.grid(row=0, column=3, padx=(0, 12), pady=4, sticky="w")
        self.metro_btn = ctk.CTkButton(
            metro, text="⏱  " + L("metro_label"), width=132, height=30,
            corner_radius=999, fg_color=BTN_GHOST_BG, text_color=COL_TEXT_2,
            border_width=1, border_color=CHIP_BORDER,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._on_metronome)
        self.metro_btn.grid(row=0, column=0, padx=(0, 8))
        self.metro_slider = ctk.CTkSlider(
            metro, from_=0, to=100, number_of_steps=100, width=76,
            fg_color=COL_TROUGH, progress_color=AMBER,
            button_color=AMBER, button_hover_color=AMBER_HOVER,
            corner_radius=3, height=14,
            command=self._on_metro_gain)
        self.metro_slider.set(80.0)
        self.metro_slider.grid(row=0, column=1, padx=(0, 8))

        # metrical level: both the beat and its eighths are valid pulses
        # (see detect_beats), so the musician picks — the phase stays
        # locked to the detected grid either way
        self.metro_mult_btn = ctk.CTkSegmentedButton(
            metro, values=["÷2", "1×", "×2"], width=104,
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._on_metro_mult)
        self.metro_mult_btn.set("1×")
        self.metro_mult_btn.grid(row=0, column=2)

        self.analyze_btn = ctk.CTkButton(
            head, text=L("btn_analyze"), width=156, height=30,
            corner_radius=999, fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._on_analyze)
        self.analyze_btn.grid(row=0, column=5, padx=(8, 10), pady=4,
                              sticky="e")

        # layout switch: classic rows vs. vertical console strips
        self.layout_toggle = ctk.CTkSegmentedButton(
            head, values=[L("layout_rows"), L("layout_strips")], width=160,
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=11, weight="bold"),
            command=self._on_layout)
        self.layout_toggle.set(L("layout_strips") if MIXER_LAYOUT == "strips"
                               else L("layout_rows"))
        self.layout_toggle.grid(row=0, column=6, sticky="e")

        self.mixer_host = ctk.CTkFrame(mix_wrap, fg_color="transparent")
        self.mixer_host.grid(row=1, column=0, sticky="nsew",
                             padx=6, pady=(2, 6))
        self.mixer_host.grid_columnconfigure(0, weight=1)
        self.mixer_host.grid_rowconfigure(0, weight=1)
        self.mixer_host.bind("<Configure>", self._fit_strips)
        self._build_mixer_body()

        # ---------- Bottom bar ----------
        bottom = ctk.CTkFrame(self, corner_radius=22, fg_color=COL_PANEL,
                              border_width=1, border_color=COL_BORDER)
        bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 8))
        bottom.grid_columnconfigure(2, weight=1)

        self.export_btn = ctk.CTkButton(
            bottom, text=L("btn_export"), width=130, height=36,
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
            bottom, text=L("status_ready"), anchor="e", text_color=OK_GREEN,
            font=ctk.CTkFont(family=UI_FAMILY, size=12))
        self.status_label.grid(row=0, column=2, sticky="e",
                               padx=(8, 8), pady=10)

        # language selector (PT-BR / EN) — applied after an app restart
        self.lang_toggle = ctk.CTkSegmentedButton(
            bottom, values=["PT", "EN"], width=90,
            corner_radius=999, fg_color=COL_TROUGH,
            selected_color=BTN_GHOST_BG, selected_hover_color=BTN_GHOST_HOV,
            unselected_color=COL_TROUGH, unselected_hover_color="#1a1a1e",
            text_color=COL_TEXT,
            font=ctk.CTkFont(family=UI_FAMILY, size=12, weight="bold"),
            command=self._on_language)
        self.lang_toggle.set(LANG.upper())
        self.lang_toggle.grid(row=0, column=3, padx=(0, 10), pady=10)

        # version + "?" contact: copies of the app travel hand to hand,
        # so each one shows which build it is and how to reach the author
        ctk.CTkLabel(
            bottom, text=f"v{APP_VERSION}", text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=MONO_FAMILY, size=11)
        ).grid(row=0, column=4, padx=(0, 6), pady=10)

        self.help_btn = ctk.CTkButton(
            bottom, text="?", width=30, height=30, corner_radius=15,
            fg_color=BTN_GHOST_BG, text_color=COL_TEXT,
            border_width=1, border_color=BTN_GHOST_BRD,
            hover_color=BTN_GHOST_HOV,
            font=ctk.CTkFont(family=UI_FAMILY, size=14, weight="bold"),
            command=self._on_contact)
        self.help_btn.grid(row=0, column=5, padx=(0, 14), pady=10)

        # tema §5: mandatory educational note — own row so it never
        # collides with the status text; everything else stays put
        ctk.CTkLabel(
            bottom,
            text=L("footer"),
            text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=12)
        ).grid(row=1, column=0, columnspan=6, pady=(0, 8))

    # --------------------------------------------------------- mixer layout --

    def _build_mixer_body(self) -> None:
        """
        (Re)create the mixer surface for the active layout. Called once at
        startup and again whenever the user flips the layout switch; all
        state lives in the Track objects, so rebuilding is lossless.
        """
        for child in self.mixer_host.winfo_children():
            child.destroy()
        strips = MIXER_LAYOUT == "strips"

        self.mixer = ctk.CTkScrollableFrame(
            self.mixer_host, corner_radius=16, fg_color="transparent",
            orientation="horizontal" if strips else "vertical")
        self.mixer.grid(row=0, column=0, sticky="nsew")

        if strips:
            self.mixer.grid_rowconfigure(0, weight=1)
            self.master_row = MasterStrip(self.mixer, self.engine)
            self.master_row.grid(row=0, column=0, padx=(4, 10), pady=4,
                                 sticky="ns")
        else:
            self.mixer.grid_columnconfigure(0, weight=1)
            self.master_row = MasterRow(self.mixer, self.engine)
            self.master_row.grid(row=0, column=0, sticky="ew", padx=6,
                                 pady=(2, 8))

        self.mixer_hint = ctk.CTkLabel(
            self.mixer, text=L("mixer_hint"), text_color=COL_TEXT_DIM,
            font=ctk.CTkFont(family=UI_FAMILY, size=13))
        self.track_rows = []
        self._populate_mixer()

    def _populate_mixer(self) -> None:
        """Rebuild the stem channels below/beside the master strip."""
        for row in self.track_rows:
            row.destroy()
        self.track_rows.clear()
        strips = MIXER_LAYOUT == "strips"

        if not self.engine.tracks:
            if strips:
                self.mixer_hint.grid(row=0, column=1, padx=30, pady=30)
            else:
                self.mixer_hint.grid(row=1, column=0, pady=24)
            return
        self.mixer_hint.grid_remove()

        for i, track in enumerate(self.engine.tracks):
            if strips:
                row = TrackStrip(self.mixer, track,
                                 on_change=self._on_mix_change)
                row.grid(row=0, column=i + 1, padx=4, pady=4, sticky="ns")
            else:
                row = TrackRow(self.mixer, track,
                               on_change=self._on_mix_change)
                row.grid(row=i + 2, column=0, sticky="ew", padx=6, pady=3)
            self.track_rows.append(row)
        self._fit_strips()

    def _fit_strips(self, event=None) -> None:
        """
        Vertical layout only: size every fader to the mixer viewport, so
        the name, the value and the M/S buttons never get clipped — the
        whole point of the strips layout is fitting a short screen.
        """
        if MIXER_LAYOUT != "strips" or not self.track_rows:
            return
        try:
            scale = ctk.ScalingTracker.get_widget_scaling(self)
        except Exception:
            scale = 1.0
        avail = self.mixer_host.winfo_height() / max(scale, 0.1)
        compact = avail < _STRIP_COMPACT_AT
        chrome = _STRIP_CHROME_MIN if compact else _STRIP_CHROME_H
        h = int(max(56, min(220, avail - chrome)))
        # hysteresis: ignore the sub-5 px jitter of resize events
        if abs(h - self._strip_fader_h) < 5 and compact == self._strip_compact:
            return
        self._strip_fader_h, self._strip_compact = h, compact
        for strip in [self.master_row, *self.track_rows]:
            strip.set_fader_height(h, compact)

    def _on_layout(self, value: str) -> None:
        global MIXER_LAYOUT
        new = "strips" if value == L("layout_strips") else "rows"
        if new == MIXER_LAYOUT:
            return
        MIXER_LAYOUT = new
        _SETTINGS["mixer_layout"] = new
        _save_settings(_SETTINGS)
        self._build_mixer_body()

    def _on_contact(self) -> None:
        """The one item behind the "?": who to write to, and which build."""
        if getattr(self, "_contact_win", None) is not None:
            try:
                if self._contact_win.winfo_exists():
                    self._contact_win.focus_force()
                    return
            except Exception:
                pass
        self._contact_win = ContactDialog(self)

    def _on_stem_mode(self, value: str) -> None:
        """Pill picker changed: show the stems of the selected mode."""
        self.stem_desc.configure(text=STEM_DESC.get(value, ""))

    # ------------------------------------------------------------ language --

    def _on_language(self, value: str) -> None:
        """Persist the chosen UI language; offer to restart to apply it."""
        new_lang = "en" if value == "EN" else "pt"
        if new_lang == LANG:
            return
        _SETTINGS["language"] = new_lang
        _save_settings(_SETTINGS)
        if messagebox.askyesno(_APP_NAME, L("lang_restart")):
            self.engine.shutdown()
            self._cleanup_temp()
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable])
            else:
                subprocess.Popen([sys.executable,
                                  os.path.abspath(sys.argv[0])])
            self.destroy()

    # ------------------------------------------------------ device handling --

    def _populate_devices(self) -> None:
        self._device_map: dict[str, int | None] = {L("device_default"): None}
        labels = [L("device_default")]
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
        self.device_menu.set(L("device_default"))

    def _on_device_selected(self, label: str) -> None:
        was_playing = self.engine.playing
        self.engine.pause()
        self.engine.set_device(self._device_map.get(label))
        if was_playing:
            try:
                self.engine.play()
            except Exception as exc:
                self._set_status(L("st_device_err", exc=exc))

    # ------------------------------------------------------------ transport --

    def _on_play(self) -> None:
        """Main transport button: toggles between play and pause."""
        if self.engine.playing:
            self.engine.pause()
            return
        try:
            self.engine.play()
        except Exception as exc:
            self._set_status(L("st_playback_err", exc=exc))
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
            title=L("dlg_open_title"),
            filetypes=[(L("ft_audio"), "*.wav *.mp3 *.m4a *.mp4"),
                       (L("ft_all"), "*.*")])
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        if self._busy:
            return
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            messagebox.showwarning(_APP_NAME, L("msg_unsupported", ext=ext))
            return
        self._run_async(self._task_load_file, path)

    def _task_load_file(self, path: str) -> None:
        self._status_async(L("st_loading", name=Path(path).name))
        data, sr, wav_path = decode_to_array(path, self.temp_dir)
        self.source_wav = wav_path
        self.source_title = Path(path).stem
        self._new_material(None)
        self.after(0, lambda: self._install_tracks(
            [(L("track_original"), data)], sr,
            L("st_loaded", name=Path(path).name, sr=sr,
              dur=format_time(len(data) / sr))))

    def _on_download(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            self._set_status(L("st_paste_url"))
            return
        if self._busy:
            return
        self._run_async(self._task_download, url)

    def _task_download(self, url: str) -> None:
        wav_path, title = download_youtube(url, self.temp_dir,
                                           self._status_async)
        self._status_async(L("st_loading_dl"))
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        self.source_wav = wav_path
        self.source_title = title
        self._new_material(url)
        self.after(0, lambda: self._install_tracks(
            [(L("track_original"), data)], sr,
            L("st_loaded", name=title, sr=sr,
              dur=format_time(len(data) / sr))))

    # ----------------------------------------------------------- separation --

    def _on_separate(self) -> None:
        if self._busy:
            return
        if not self.source_wav:
            self._set_status(L("st_load_first_sep"))
            return
        # the picker holds the short label; STEM_MODELS is keyed by the full one
        model_spec, stem_order = STEM_MODELS[_SHORT_TO_FULL[
            self.stem_var.get()]]
        self._run_async(self._task_separate, model_spec, stem_order)

    def _task_separate(self, model_spec: str, stem_order: list[str]) -> None:
        stems = separate_stems(self.source_wav, model_spec, stem_order,
                               self.temp_dir, self._status_async)
        sr = 44100    # Spleeter models always render at 44.1 kHz
        # same song, new stems: a library entry it came from stays its home
        self._mode_spec = model_spec
        self.after(0, lambda: self._install_tracks(
            stems, sr, L("st_sep_done", n=len(stems))))

    def _new_material(self, url: str | None) -> None:
        """A different song came in: it is not separated and not in the
        library yet."""
        self._mode_spec = None
        self._source_url = url
        self._session_dir = None

    # ------------------------------------------------------------- library --

    def _on_library(self) -> None:
        if self._busy:
            return
        win = getattr(self, "_library_win", None)
        try:
            if win is not None and win.winfo_exists():
                win.focus_force()
                return
        except Exception:
            pass
        self._library_win = LibraryDialog(self)

    def _on_save_session(self) -> None:
        if self._busy:
            return
        if not self.engine.tracks:
            self._set_status(L("st_nothing_save"))
            return
        root = library_dir()
        if self._session_dir is not None and self._session_dir.parent == root:
            target = self._session_dir          # opened/saved before: update
        else:
            target = root / safe_folder_name(self.source_title or _APP_NAME)
            if target.exists():
                answer = messagebox.askyesnocancel(
                    _APP_NAME, L("lib_exists", name=target.name))
                if answer is None:
                    return
                if not answer:
                    target = unique_folder(target)
        self._run_async(self._task_save_session, target,
                        self._session_snapshot())

    def _session_snapshot(self) -> dict:
        """Everything a saved song needs, read on the UI thread so it is
        one consistent moment of the mixer (the arrays are never mutated
        in place, so the worker can write them while the user plays on)."""
        e = self.engine
        beats = self._beats_base
        return {
            "title": self.source_title or _APP_NAME,
            "tracks": [(_LABEL_TO_STEM.get(t.name, t.name), t.orig)
                       for t in e.tracks],
            "samplerate": e.samplerate,
            "source_wav": self.source_wav,
            "state": {
                "mode": self._mode_spec,
                "source_url": self._source_url,
                "key": self._key_last,
                "bpm": self._bpm_global,
                "beats": (None if beats is None
                          else [round(float(b), 5) for b in beats]),
                "mixer": {
                    "master": e.master_gain,
                    "tracks": [{"gain": t.gain, "mute": t.mute,
                                "solo": t.solo} for t in e.tracks],
                },
                "speed": e.speed,
                "metronome": {"on": bool(e.metronome_on),
                              "gain": e.metronome_gain,
                              "mult": self._metro_mult},
            },
        }

    def _task_save_session(self, target: Path, snap: dict) -> None:
        try:
            folder = save_session(target, snap["title"], snap["tracks"],
                                  snap["samplerate"], snap["source_wav"],
                                  snap["state"], self._status_async)
        except MediaError:
            raise
        except Exception as exc:
            log.error("Save failed:\n%s", traceback.format_exc())
            raise MediaError(L("err_save", err=exc)) from exc
        tracks = self.engine.tracks

        def done() -> None:
            if self.engine.tracks is tracks:   # still the song that was saved
                self._session_dir = folder
        self.after(0, done)
        self._status_async(L("st_saved", name=folder.name))

    def _open_session(self, folder: Path) -> None:
        if self._busy:
            return
        self._run_async(self._task_open_session, Path(folder))

    def _task_open_session(self, folder: Path) -> None:
        self._status_async(L("st_opening", name=folder.name))
        m, stems, sr, source_wav = load_session(folder, self.temp_dir,
                                                self._status_async)
        named = [(stem_label(sid), data) for sid, data in stems]

        def install() -> None:
            self.source_wav = source_wav
            self.source_title = str(m.get("title") or folder.name)
            self._mode_spec = m.get("mode")
            self._source_url = m.get("source_url")
            self._session_dir = folder
            self._install_tracks(named, sr,
                                 L("st_opened", name=self.source_title,
                                   n=len(named)), session=m)
        self.after(0, install)

    # -------------------------------------------------------------- export --

    def _on_export(self) -> None:
        if self._busy:
            return
        if not self.engine.tracks:
            self._set_status(L("st_nothing_export"))
            return
        fmt = "mp3" if self.format_toggle.get().startswith("MP3") else "wav"
        default_name = (self.source_title or "isolate_mix") + f"_mix.{fmt}"
        out_path = filedialog.asksaveasfilename(
            title=L("dlg_export_title"),
            initialfile=default_name,
            defaultextension=f".{fmt}",
            filetypes=[(L("ft_wav"), "*.wav")] if fmt == "wav"
                      else [(L("ft_mp3"), "*.mp3")])
        if not out_path:
            return
        self._run_async(self._task_export, fmt, out_path)

    def _task_export(self, fmt: str, out_path: str) -> None:
        self._status_async(L("st_render"))
        mix = self.engine.render_mix()
        self._status_async(L("st_encoding", fmt=fmt.upper()))
        export_mix(mix, self.engine.samplerate, out_path, fmt,
                   self.temp_dir)
        # companion click track (always WAV, full click level): the same
        # timeline as the mix — original speed, like the mix — so it drops
        # into any DAW next to it
        click, _ = self._click_for(self._beats_base, 1.0,
                                   self.engine.base_frames)
        if click is not None:
            stem = Path(out_path).stem
            base = stem[:-4] if stem.endswith("_mix") else stem
            metro_path = str(Path(out_path).with_name(
                base + "_metronome.wav"))
            export_mix(np.repeat(click[:, None], 2, axis=1),
                       self.engine.samplerate, metro_path, "wav",
                       self.temp_dir)
            self._status_async(L("st_exported_metro", path=out_path,
                                 metro=Path(metro_path).name))
        else:
            self._status_async(L("st_exported", path=out_path))

    # ------------------------------------------------------------ mixer UI --

    def _install_tracks(self, named_arrays: list[tuple[str, np.ndarray]],
                        samplerate: int, status: str,
                        session: dict | None = None) -> None:
        """Put new material in the mixer. `session` is the manifest of a
        song reopened from the library: its mixer, analysis, metronome and
        speed come back as they were saved."""
        self.engine.set_tracks(named_arrays, samplerate)
        self._beats_base = None          # new material: stale beat grid
        self._bpm_global = None
        self._speed_lag = 0.0
        self._sync_speed_picker()        # new material plays at 100%
        if session is not None:
            self._restore_mixer(session)
            # the master widget reads the gain only when built: rebuild it too
            self._build_mixer_body()
        else:
            # master strip stays fixed first; stem channels rebuild after
            # it, in whichever layout (rows / vertical strips) is active
            self._populate_mixer()
        self.timeline.set(0.0)
        self._set_status(status)
        self._key_last = None            # new material: stale key
        if session is None or not self._restore_analysis(session):
            self._start_analysis(refresh_key=True)
        if session is not None:
            self._restore_speed(float(session.get("speed") or 1.0),
                                self.engine.tracks)

    def _restore_mixer(self, m: dict) -> None:
        """Faders, mute/solo, master, metronome level and the separation
        mode of a saved song (before the mixer widgets are built, which
        read their state from the Track objects)."""
        mixer = m.get("mixer") or {}
        try:
            self.engine.master_gain = float(mixer.get("master", 1.0))
            for t, s in zip(self.engine.tracks, mixer.get("tracks") or []):
                t.gain = float(s.get("gain", 1.0))
                t.mute = bool(s.get("mute", False))
                t.solo = bool(s.get("solo", False))
        except (TypeError, ValueError, AttributeError):
            log.warning("Saved mixer state ignored (malformed)")
        metro = m.get("metronome") or {}
        try:
            gain = float(metro.get("gain", self.engine.metronome_gain))
            self.engine.metronome_gain = gain
            self.metro_slider.set(gain * 100.0)
            mult = float(metro.get("mult", 1.0))
            label = {0.5: "÷2", 1.0: "1×", 2.0: "×2"}.get(mult, "1×")
            self._metro_mult = {"÷2": 0.5, "1×": 1.0, "×2": 2.0}[label]
            self.metro_mult_btn.set(label)
        except (TypeError, ValueError, AttributeError):
            log.warning("Saved metronome state ignored (malformed)")
        short = _SPEC_TO_SHORT.get(m.get("mode"))
        if short:
            self.stem_var.set(short)
            self._on_stem_mode(short)

    def _restore_analysis(self, m: dict) -> bool:
        """Key, BPM and beat grid saved with the song: the metronome is
        ready at once, with the very grid the user practised with. False
        when the song was saved before its analysis finished."""
        beats = m.get("beats")
        if not beats:
            return False
        try:
            base = np.asarray(beats, dtype=np.float64)
            bpm = float(m["bpm"]) if m.get("bpm") else None
        except (TypeError, ValueError):
            return False
        key = m.get("key") or None
        self._key_last = key
        self._bpm_global = bpm
        self._beats_base = base
        self.engine.set_click_track(*self._click_for(
            base, self.engine.speed, self.engine.n_frames))
        self.engine.metronome_on = bool((m.get("metronome") or {}).get("on"))
        self.key_label.configure(text=key_short(key) or "—",
                                 text_color=AMBER if key else AMBER_DIM)
        self.bpm_label.configure(text=self._bpm_text(),
                                 text_color=AMBER if bpm else AMBER_DIM)
        self._refresh_metro_button()
        return True

    def _restore_speed(self, speed: float, tracks: list[Track]) -> None:
        """Re-apply the practice speed a song was saved at, once the
        worker that opened it has released the busy state."""
        if self.engine.tracks is not tracks or abs(speed - 1.0) < 1e-6:
            return
        if not any(abs(speed - s) < 1e-6 for s in SPEED_STEPS):
            return
        if self._busy:
            self.after(100, lambda: self._restore_speed(speed, tracks))
            return
        value = f"{round(speed * 100)}%"
        self.speed_picker.set(value)
        self._on_speed(value)

    # ------------------------------------------------------ music analysis --

    def _on_analyze(self) -> None:
        if not self.engine.tracks:
            self._set_status(L("st_load_first"))
            return
        self._start_analysis()

    def _start_analysis(self, refresh_key: bool = True) -> None:
        """
        Detect BPM + beat grid of the AUDIBLE material (post-fader) on a
        background thread; the grid also (re)builds the metronome click.
        Mixer changes re-trigger it (debounced) with refresh_key=False:
        the BPM follows what is audible, but the KEY is fixed per track —
        computed once over the whole full mix, faders ignored — matching
        the industry convention (Mixed In Key, Essentia, madmom,
        KeyFinder all report one static global key per track).
        """
        if not self.engine.tracks:
            return
        if self._analyzing:                 # queue a rerun instead of racing
            self._analysis_dirty = True
            self._analysis_dirty_key = self._analysis_dirty_key or refresh_key
            return
        self._analyzing = True
        if refresh_key:
            self.key_label.configure(text="…", text_color=AMBER_DIM)
        self.bpm_label.configure(text="…", text_color=AMBER_DIM)
        self._refresh_metro_button()

        # analysis always runs on the ORIGINAL-speed audio; the grid is
        # mapped to the playback speed when the click is built
        sr = self.engine.samplerate
        n = self.engine.base_frames
        track_set = self.engine.tracks       # identity: was it replaced?
        tracks = list(track_set)
        any_solo = any(t.solo for t in tracks)
        audible = [t for t in tracks
                   if not t.mute and (not any_solo or t.solo)
                   and t.gain > 0.02]
        if not audible:                     # everything muted: analyse all
            audible = tracks

        def work():
            try:
                # audible mix (post-fader, pre-master)
                mono = np.zeros(n, dtype=np.float32)
                for t in audible:
                    mono += t.orig.mean(axis=1) * t.gain
                # beat source: prefer the drums stem when audible and
                # non-silent — percussive material carries the pulse
                # (Gkiokas 2012; Chiu 2021), while predominant vocals and
                # harmonic content mislead the onset envelope (Zapata &
                # Gómez 2013). Muting the drums re-routes to the mix.
                beat_src = _pick_beat_source(audible, n)
                if beat_src is None:
                    beat_src = mono
                res = detect_beats(beat_src, sr)
                bpm, beat_times = None, None
                if res is not None:
                    bpm, beat_times = res
                key = self._key_last
                if refresh_key:
                    # key: whole track, FULL mix, faders ignored — the
                    # song's key does not change when a stem is muted
                    key_mono = np.zeros(n, dtype=np.float32)
                    for t in tracks:
                        key_mono += t.orig.mean(axis=1)
                    key = detect_key(key_mono, sr)
                    self._key_last = key
                key_txt = key_short(key) or "—"     # tema §3: Am, C, F#m...

                def apply() -> None:
                    # on the UI thread: the click is rendered for the speed
                    # in effect NOW (a speed change may have landed meanwhile)
                    if self.engine.tracks is not track_set:
                        return                      # material replaced
                    self._bpm_global = bpm
                    self._beats_base = beat_times
                    self.engine.set_click_track(
                        *self._click_for(beat_times, self.engine.speed,
                                         self.engine.n_frames))
                    self.key_label.configure(
                        text=key_txt, text_color=AMBER if key else AMBER_DIM)
                    self.bpm_label.configure(
                        text=self._bpm_text(),
                        text_color=AMBER if bpm else AMBER_DIM)
                    self._refresh_metro_button()
                self.after(0, apply)
                if key or bpm:
                    self._status_async(L("st_analysis", key=key or "?",
                                         bpm=bpm or "?"))
            except Exception:
                log.error("Analysis failed:\n%s", traceback.format_exc())
                self._bpm_global = None
                self._beats_base = None
                self.engine.set_click_track(None)
                self.after(0, lambda: (
                    self.key_label.configure(text="—",
                                             text_color=AMBER_DIM),
                    self.bpm_label.configure(text="—",
                                             text_color=AMBER_DIM),
                    self._refresh_metro_button()))
            finally:
                self._analyzing = False
                if self._analysis_dirty:
                    self._analysis_dirty = False
                    rerun_key = self._analysis_dirty_key
                    self._analysis_dirty_key = False
                    self.after(0, lambda: self._start_analysis(rerun_key))

        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------- metronome --

    def _on_metronome(self) -> None:
        """Toggle the click track layered over the master output."""
        if self.engine.click_track is None:
            self._set_status(L("st_metro_unavailable"))
            return
        self.engine.metronome_on = not self.engine.metronome_on
        self._refresh_metro_button()

    def _refresh_metro_button(self) -> None:
        # tema §5: active state = amber bg with dark text (like Solo)
        on = self.engine.metronome_on and self.engine.click_track is not None
        self.metro_btn.configure(
            fg_color=AMBER if on else BTN_GHOST_BG,
            text_color=BTN_PRI_TX if on else COL_TEXT_2)

    def _on_metro_gain(self, value: float) -> None:
        self.engine.metronome_gain = float(value) / 100.0

    def _on_metro_mult(self, value: str) -> None:
        """Halve / double the click rate, keeping the detected phase."""
        self._metro_mult = {"÷2": 0.5, "1×": 1.0, "×2": 2.0}.get(value, 1.0)
        base = self._beats_base
        if base is None or self.engine.n_frames <= 0:
            return
        was_on = self.engine.metronome_on
        self.engine.set_click_track(*self._click_for(
            base, self.engine.speed, self.engine.n_frames))
        self.engine.metronome_on = was_on
        self._refresh_metro_button()

    def _click_for(self, base: np.ndarray | None, speed: float,
                   n_frames: int) -> tuple[np.ndarray | None,
                                           np.ndarray | None]:
        """Click buffer + beat grid in PLAYBACK time: the detected grid
        (original time) at the chosen metrical level, divided by the speed.
        The click itself is re-rendered, never stretched, so it stays dry,
        and it is shifted by the lag measured on the stretched audio (see
        measure_stretch_lag). The returned grid keeps the beat (not the
        ÷2/×2 level), so the BPM chip shows the song's own pulse."""
        if base is None or n_frames <= 0:
            return None, None
        lag = 0.0 if abs(speed - 1.0) < 1e-6 else self._speed_lag
        click = render_click_track(
            scale_beat_grid(base, self._metro_mult) / speed + lag, n_frames,
            self.engine.samplerate)
        return click, base / speed + lag

    def _bpm_text(self) -> str:
        """Global BPM as heard at the current playback speed."""
        if not self._bpm_global:
            return "—"
        return f"{self._bpm_global * self.engine.speed:.0f}"

    # -------------------------------------------------------------- speed --

    def _on_speed(self, value: str) -> None:
        """Practice speed (70..100%): re-render the stems time-stretched,
        pitch unchanged, off the UI thread."""
        speed = int(value.rstrip("%")) / 100.0
        self._set_speed_menu(False)
        if not self.engine.tracks or self._busy:
            self._sync_speed_picker()
            if not self.engine.tracks:
                self._set_status(L("st_load_first"))
            return
        if abs(speed - self.engine.speed) < 1e-6:
            return
        self._refresh_speed_btn(speed)   # shows the pick while it renders
        if not self.engine.playing and abs(self.engine.speed - 1.0) > 1e-6:
            # stopped: drop the old stretched buffers BEFORE rendering the
            # new ones (6 stems at 70% are ~0.6 GB for a 3.5 min song, and
            # holding two speeds at once hurts on 4 GB machines)
            e = self.engine
            self._speed_lag = 0.0
            e.set_playback_speed(e.tracks, 1.0, [t.orig for t in e.tracks],
                                 *self._click_for(self._beats_base, 1.0,
                                                  e.base_frames))
        self._run_async(self._task_speed, speed)

    def _task_speed(self, speed: float) -> None:
        tracks = self.engine.tracks
        sr = self.engine.samplerate
        pct = round(speed * 100)
        lag = 0.0
        if abs(speed - 1.0) < 1e-6:
            arrays = [t.orig for t in tracks]
        else:
            self._status_async(L("st_speed_working", pct=pct))
            arrays = time_stretch_tracks([t.orig for t in tracks], sr, speed)
            # where atempo actually put the hits, so the click follows them
            orig_mono = sum(t.orig.mean(axis=1) for t in tracks)
            str_mono = sum(a.mean(axis=1) for a in arrays)
            lag = measure_stretch_lag(orig_mono, str_mono, sr, speed)
            log.info("speed %.2f: stretch lag %+.1f ms", speed, lag * 1000)
        self.after(0, lambda: self._apply_speed(tracks, speed, arrays, lag))

    def _apply_speed(self, tracks: list[Track], speed: float,
                     arrays: list[np.ndarray], lag: float) -> None:
        if self.engine.tracks is not tracks:
            return                      # new material arrived meanwhile
        was_on = self.engine.metronome_on
        self._speed_lag = lag
        click, beats = self._click_for(self._beats_base, speed,
                                       len(arrays[0]))
        if not self.engine.set_playback_speed(tracks, speed, arrays,
                                              click, beats):
            return                      # new material arrived meanwhile
        self.engine.metronome_on = was_on and click is not None
        self._refresh_metro_button()
        self._bpm_live_txt = None
        pct = round(speed * 100)
        if self._bpm_global and not self._analyzing:
            self.bpm_label.configure(text=self._bpm_text(), text_color=AMBER)
        if abs(speed - 1.0) < 1e-6:
            self._set_status(L("st_speed_normal"))
        elif self._bpm_global:
            self._set_status(L("st_speed_done_bpm", pct=pct,
                               bpm0=f"{self._bpm_global:.0f}",
                               bpm=self._bpm_text()))
        else:
            self._set_status(L("st_speed_done", pct=pct))

    def _sync_speed_picker(self) -> None:
        self.speed_picker.set(f"{round(self.engine.speed * 100)}%")
        self._refresh_speed_btn(self.engine.speed)

    def _toggle_speed_menu(self) -> None:
        self._set_speed_menu(not self._speed_open)

    def _set_speed_menu(self, open_: bool) -> None:
        """Unfold / fold the speed options beside the speed pill."""
        self._speed_open = open_
        if open_:
            self.speed_picker.grid()
        else:
            self.speed_picker.grid_remove()
        self._refresh_speed_btn(None)

    def _refresh_speed_btn(self, speed: float | None) -> None:
        """Pill text: "Velocidade 70% ▸". Brighter while slowed down, so a
        reduced speed is never forgotten with the options folded away."""
        if speed is not None:
            self._speed_shown = speed
        speed = getattr(self, "_speed_shown", 1.0)
        slowed = abs(speed - 1.0) > 1e-6
        # open: the pill shrinks to a fold-back arrow, or the 100% option
        # would slide under the time readout on a 1366 px laptop
        text = ("◂" if self._speed_open else
                f"{L('lbl_speed').rstrip(':')}  {round(speed * 100)}%  ▸")
        self.speed_btn.configure(
            text=text, width=36 if self._speed_open else 150,
            text_color=COL_TEXT if slowed else COL_TEXT_2,
            fg_color=BTN_GHOST_HOV if slowed else BTN_GHOST_BG)

    def _on_mix_change(self) -> None:
        """Mixer changed (gain/mute/solo): re-analyse the audible material
        so Key/BPM and the metronome grid follow it (debounced)."""
        if self._mix_change_job is not None:
            try:
                self.after_cancel(self._mix_change_job)
            except Exception:
                pass
        self._mix_change_job = self.after(900, self._mix_change_fire)

    def _mix_change_fire(self) -> None:
        self._mix_change_job = None
        self._start_analysis(refresh_key=False)

    # ------------------------------------------------------- async plumbing --

    def _run_async(self, fn, *args) -> None:
        """Run `fn(*args)` on a worker thread with busy-state handling."""
        self._set_busy(True)

        def wrapper():
            try:
                fn(*args)
            except MediaError as exc:
                self._status_async(L("st_error", exc=exc))
            except Exception as exc:
                log.error("Worker failed:\n%s", traceback.format_exc())
                self._status_async(L("st_unexpected", exc=exc))
            finally:
                self.after(0, lambda: self._set_busy(False))

        threading.Thread(target=wrapper, daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self.separate_btn, self.download_btn, self.export_btn,
                    self.speed_picker, self.speed_btn, self.save_lib_btn,
                    self.library_btn):
            btn.configure(state=state)
        if not busy:
            self._sync_speed_picker()    # a failed change snaps back
        # native Windows spinner while a worker is busy ("watch" maps to
        # the OS loading cursor; "" restores the default arrow)
        try:
            self.configure(cursor="watch" if busy else "")
        except tk.TclError:
            pass

    def _status_async(self, text: str) -> None:
        """Thread-safe status update (called from worker threads)."""
        self._status_queue.put(text)

    def _set_status(self, text: str) -> None:
        ok = text.startswith(("Loaded", "Exported", "Separation complete",
                              "Analysis", "Carregado", "Exportado",
                              "Separação concluída", "Análise", "●"))
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

        # BPM chip: instantaneous tempo while playing (median interval of
        # the beat grid around the playhead — follows tempo drift, like a
        # dynamic DJ beatgrid); reverts to the global tempo when stopped
        playing = self.engine.playing
        bt = self.engine.beat_times
        if playing and bt is not None and len(bt) >= 5:
            i = int(np.searchsorted(bt, self.engine.position_seconds))
            lo, hi = max(0, i - 3), min(len(bt), i + 3)
            iv = np.diff(bt[lo:hi])
            if len(iv) >= 2:
                med = float(np.median(iv))
                if med > 0:
                    txt = f"{60.0 / med:.0f}"
                    if txt != self._bpm_live_txt:
                        self._bpm_live_txt = txt
                        self.bpm_label.configure(text=txt,
                                                 text_color=AMBER)
        elif self._bpm_live_txt is not None:
            self._bpm_live_txt = None
            if self._bpm_global is not None and not self._analyzing:
                self.bpm_label.configure(text=self._bpm_text(),
                                         text_color=AMBER)

        # VU meters (post-fader per track, post-master on the master bus)
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


def _selftest(argv: list[str]) -> None:
    """Headless separation check (`Isolate.exe --selftest song.wav
    [model_spec]`), used to validate frozen builds; output lands in
    isolate.log under --windowed."""
    wav = argv[0]
    # default: the first registered mode (2 Stems); Spleeter's 2stems is
    # gone since v2.2, so hardcoding a spec here would break the default.
    spec = argv[1] if len(argv) > 1 else next(iter(STEM_MODELS.values()))[0]
    order = next(o for s, o in STEM_MODELS.values() if s == spec)
    def echo(msg: object) -> None:
        """Print without ever aborting the run on encoding: a console at
        the Windows default codepage (cp1252) cannot encode the ☕ of the
        progress message, and the raised UnicodeEncodeError used to kill
        the selftest mid-separation."""
        try:
            print(msg)
        except UnicodeEncodeError:
            print(str(msg).encode("ascii", "replace").decode("ascii"))

    tmp = tempfile.mkdtemp(prefix="isolate_selftest_")
    try:
        stems = separate_stems(wav, spec, order, tmp, progress=echo)
        echo(f"SELFTEST OK: {[(n, a.shape) for n, a in stems]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        _selftest(sys.argv[2:])
        sys.exit(0)
    main()
