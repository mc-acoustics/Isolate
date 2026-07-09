# Isolate — Stem Splitter & Multi-Track Mixer

App desktop para Windows 10/11, **100% local e gratuito**, que separa uma
música em stems (vocais, bateria, baixo, piano, outros) e abre tudo num
mixer multi-track estilo DAW, com playback sincronizado em tempo real,
detecção automática de tom e BPM, e export do seu mix.

![Isolate em uso](docs/screenshot.png)

## Recursos

- **Separação de stems** com [Spleeter](https://github.com/deezer/spleeter)
  (Deezer): 2, 4 ou 5 stems, modelos de alta fidelidade (16 kHz).
- **Mixer multi-track**: fader de ganho, mute e solo por stem, canal MASTER,
  VU meters LED com peak-hold — tudo em tempo real, sample-accurate.
- **Análise musical**: tom (perfis de Krumhansl-Kessler) e BPM
  (autocorrelação de spectral flux) detectados automaticamente, em NumPy puro.
- **Entrada flexível**: arraste um arquivo (`.wav .mp3 .m4a .mp4`), navegue,
  ou cole uma URL do YouTube (via yt-dlp).
- **Export**: WAV 16-bit/44.1 kHz ou MP3 320 kbps CBR do mix atual.
- **Offline e privado**: nada sai da sua máquina; os modelos são baixados
  uma única vez no primeiro uso.

## Instalação e uso

Veja o passo a passo em [SETUP.md](SETUP.md). Resumo:

```bat
REM Requer Python 3.10 (TensorFlow/Spleeter não suportam 3.12+)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

O ffmpeg precisa estar no PATH (para MP3/M4A/MP4, YouTube e export MP3).

## Build do executável

```bat
.venv\Scripts\activate
build.bat
```

Gera `dist\Isolate\Isolate.exe` (one-dir). O script embute **exclusivamente
a build LGPL do ffmpeg** ([BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds),
extraída em `%LOCALAPPDATA%\Isolate\ffmpeg-lgpl\`) e gera a pasta
`licenses\` com os textos de licença de todas as dependências.

A especificação original completa do projeto está em [SPEC.md](SPEC.md).

## Licença

[GPLv3](LICENSE). Dependências principais: Spleeter (MIT), TensorFlow
(Apache-2.0), CustomTkinter (MIT), yt-dlp (Unlicense), ffmpeg (LGPLv3,
binário separado e substituível).

## Avisos

- Ferramenta **educacional**, para prática musical, estudo de mixagem e
  análise. Separar material protegido por direitos autorais exige
  autorização dos detentores; o resultado é responsabilidade do usuário.
- Baixar conteúdo do YouTube pode violar os Termos de Serviço da
  plataforma. Use o recurso apenas com conteúdo próprio ou licenciado.
