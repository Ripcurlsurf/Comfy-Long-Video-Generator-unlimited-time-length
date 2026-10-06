# 🎬 ComfyUI Long Movie Local Generator
> **Zero install. 100% offline.** Generate production-grade, multi-scene ComfyUI workflows straight from your browser.

[![Status](https://img.shields.io/badge/Status-Community-blue?style=flat-square)](#disclaimer)
[![Platform](https://img.shields.io/badge/Platform-Browser_%7C_ComfyUI-orange?style=flat-square)](#quick-setup)
[![Engines](https://img.shields.io/badge/Engines-MiniMax_H3_%7C_Wan_%7C_LTX-green?style=flat-square)](#engine--continuity)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

### ⚡ Quick Setup & AI Storyboarding
1. **Open** `minimax_h3_comfyui_local_generator_v2.html` in any browser (or host on GitHub Pages).
2. **Storyboard with AI:** Click **Generate Template** to export a pre-structured JSON schema ➔ paste your story into ChatGPT/Claude to fill it in ➔ click **Upload Filled Template** to instantly load your entire film.
3. **Export & Run:** Click **Generate Workflow** ➔ drop the JSON into ComfyUI ➔ put image assets in `ComfyUI/input/` ➔ **Queue**.

---

### 🎥 Engine & Continuity Matrix

| Category | Options & Specifications |
| :--- | :--- |
| **Video Engines** | MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3, LTX 2.5 *(auto-hides incompatible UI controls)* |
| **Image Engines** | Qwen Image 2.1, Flux 2, Z-Image Turbo, MiniMax H3 Stills *(supports LoRA stacks: `file \| weight`)* |
| **Resolutions** | 16:9, 9:16, 1:1, 4:3, 3:4, 21:9 (all $\times 32$ steps) + custom sizing with auto-crop/pad protection |
| **Last-Frame Chain** | Frame $N_{\text{last}} \rightarrow (N+1)_{\text{first}}$. Smooth action; best with sharpen/refresh passes to cut compression drift. |
| **Keyframe Mode** | Interpolates between `keyframe_{n}.png` pairs. Zero cumulative drift; best for planned blocking. |
| **Ref-to-Video** | Injects character/style sheets via `<Picture N>` tokens into `ref2va`. Preserves traits across angles. |

---

### ⚙️ Production & Scaling Features

* **AI Template Roundtrip:** Export a schema, let an AI populate scenes/consistency cards, and import it back with one click.
* **Anti-Degradation:** Optional sharpen pass, upscale roundtrip, and img2img refresh at scene handoffs.
* **Subgraph Packing:** Encapsulates each scene into a subgraph node to keep ComfyUI fast with 20+ scenes.
* **FFmpeg Pipeline:** Use **Skip in-graph join** + generated `assemble_long_video.sh` to prevent RAM bottlenecks, trim keyframe seams, and mix TTS tracks from `dialogue/scene_NN.wav`.
* **Consistency Bible:** Auto-injects global lighting, character tags, and sound design across every scene prompt.
* **Strict Scene Order:** Optional dependency locks to ensure sequential generation without parallel races.

---

<details>
<summary><b>🛠️ Requirements, FAQ & Limitations (Click to expand)</b></summary>

- **Requirements:** ComfyUI (subgraph support), MiniMax nodes (`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`), standard samplers (`SamplerCustomAdvanced`), and model weights. FFmpeg required for the assembly script.
- **Red Nodes in ComfyUI:** Check model filenames on the HTML page against your local drive before export.
- **Button Does Nothing:** Inspect browser alerts for validation errors (e.g., missing scene prompts).
- **Limitations:** MiniMax voice consistency varies; route dialogue to **External TTS** for stability. Pure first/last frame models do not accept reference images (handled strictly via `ref2va`).

</details>

---

<sub>*Community project. Not affiliated with, endorsed by, or maintained by MiniMax, ComfyUI, or Qwen.*</sub>
