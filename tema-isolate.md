# Tema Isolate — Especificação visual

> Especificação para aplicar o redesign visual ao `main.py` (Python 3.10 + customtkinter).
> Referência visual: `Isolate Redesign.dc.html` neste projeto.
> Regra geral: **não alterar a disposição do layout** — apenas cores, raios, fontes e espaçamentos.

---

## 1. Paleta

| Uso | Hex |
|---|---|
| Fundo da janela | `#0c0c0e` |
| Painéis / cartões (transporte, separação, análise, mixer) | `#18181c` |
| Superfície elevada (linha MASTER, input, select) | `#1d1d22` |
| Trilha de slider / fundo do VU | `#0e0e11` |
| Borda padrão de painel | `#26262b` (equivale a branco 4,5%) |
| Texto principal | `#ECEAE6` |
| Texto secundário / labels | `#8f8d88` |
| Texto apagado / placeholder | `#7c7973` |
| **Acento âmbar** (exclusivo: valores de Tom e BPM, borda do cartão de análise e do MASTER, fader do master, botão Solo ativo) | `#E5A54B` |
| Âmbar hover | `#F0B562` |
| Fundo dos chips Tom/BPM | âmbar a 8% sobre o painel ≈ `#241e15` |
| Borda dos chips Tom/BPM | âmbar a 30% ≈ `#5c4a2b` |
| VU verde | `#6fae7c` |
| VU âmbar (a partir de −9 dB) | `#E5A54B` |
| VU vermelho (a partir de −3 dB) | `#d96b4a` |
| Botão primário (Play, Separar Faixas, Baixar & Carregar, Exportar Mix): fundo | `#ECEAE6` |
| Botão primário: texto | `#161613` |
| Botão primário: hover | `#ffffff` |
| Botão fantasma / secundário: fundo | `#232328` com borda `#2c2c31` |
| Indicador "Pronto." / arquivo carregado | `#6fae7c` |

**Regra do âmbar:** ele é a cor de destaque de Tom e BPM. Não usar em botões genéricos — só nos pontos listados acima.

## 2. Curvas (corner_radius)

- Janela/painéis principais: `corner_radius=22`
- Cartões internos (chips Tom/BPM, linhas de canal, MASTER): `corner_radius=18`
- Botões e inputs: formato pílula → `corner_radius=999` (customtkinter trunca para altura/2)
- Botões M / S: círculos de 28×28 px, `corner_radius=14`
- Botão Play: círculo 46×46 px; Pause/Stop: círculos 40×40 px
- Sliders: `corner_radius=3` na trilha (6 px de altura), thumb circular ~17 px
- Radio buttons: círculo 18 px com ponto interno 8 px

## 3. Tipografia

Instalar no Windows (baixar do Google Fonts → clique-direito → Instalar; instalar "para todos os usuários" se o exe for distribuído):

- **Outfit** — toda a UI. Pesos: 400 (texto), 500 (nomes de canal), 600 (botões, títulos), 700 (MASTER)
- **Spline Sans Mono** — valores numéricos: tempo (01:24 / 03:52), BPM, percentuais dos faders, chips de formato

Tamanhos (px):
- Título da janela "Isolate": 17 / 600
- Subtítulo "Separador de stems & mixer multipista": 11,5 / 400, cor secundária
- Labels de seção ("MODO DE SEPARAÇÃO", "ANÁLISE MUSICAL", "MIXER"): 12 / 600, CAIXA ALTA, letter-spacing largo, cor secundária
- Valores Tom/BPM: 21 / 600, âmbar. **Tom em notação de letra: Am, C, F#m…**
- Corpo/opções de rádio: 13,5
- Botões: 13,5–14,5 / 600
- Tempo (mono): 14
- Percentuais (mono): 13,5

Fallback se a fonte não estiver instalada: `Segoe UI` (UI) e `Consolas` (mono).

## 4. Espaçamentos

- Padding externo do conteúdo: 26–30 px
- Gap entre painéis: 20 px
- Padding interno de painéis: 20 × 24 px
- Gap entre linhas de canal do mixer: 8 px
- Linha de canal: padding 13 × 22 px; colunas: nome 110 px · M/S · VU 150 px · fader flex · % 52 px alinhado à direita

## 5. Componentes — notas de aplicação

- **Drop zone:** customtkinter não tem borda tracejada; usar frame com borda sólida 1,5 px `#2c2c31`, radius 22, texto centralizado ("Arraste um arquivo de áudio aqui" + extensões em mono, cor apagada).
- **VU meters (canvas existente):** manter LEDs segmentados; trocar cores para verde/âmbar/vermelho acima; fundo `#0e0e11`; segmentos com gap de 2 px; cantos arredondados no canvas se possível.
- **MASTER:** fundo `#1d1d22`, borda 1 px âmbar a ~25% (`#4a3d26`), fader preenchido em âmbar com thumb âmbar.
- **Análise musical:** dois chips lado a lado (Tom | BPM), label pequeno em caixa alta âmbar apagado, valor grande âmbar. Sem arquivo: valor "—" em âmbar a 35%.
- **Radio buttons:** ponto interno branco-quente `#ECEAE6` quando selecionado; anel `#55534e`.
- **M ativo:** fundo `#ECEAE6`, texto escuro. **S ativo:** fundo `#E5A54B`, texto escuro. Inativos: fundo `#232328`, texto `#8f8d88`.
- **Barra de título:** se possível janela sem moldura nativa (`overrideredirect`) com barra própria: logo + nome à esquerda, controles em círculos de 30 px à direita. Se complicar, manter a barra nativa do Windows e aplicar o resto.
- **Rodapé:** Exportar Mix (primário) + chips de formato (mono, borda sutil) à esquerda; ao centro a mensagem obrigatória: *"Ferramenta educacional para separação de instrumentos e análise musical. Distribuição gratuita."* (12 px, cor apagada); status "Pronto." com ponto verde à direita.

## 6. Logotipo

Diagrama de Venn incompleto: dois círculos sobrepostos — o da esquerda é um **arco aberto** (incompleto, cor `#77746E`), o da direita é pleno (contorno `#ECEAE6`), e a **interseção é preenchida em âmbar** `#E5A54B` (o "stem isolado").

- SVG de referência na barra de título de `Isolate Redesign.dc.html` (34 px) e no estado vazio do mixer (72 px).
- Para o app: gerar `isolate.ico` a partir do SVG (fundo `#0c0c0e` ou transparente) e usar no `iconbitmap` da janela e no PyInstaller (`--icon`).
