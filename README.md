# 🎬 ComfyUI Long Movie Local Generator

> **Zero installation. 100% offline.** Generate production-grade, multi-scene ComfyUI workflow JSONs directly in your browser.

[![Status: Community](https://img.shields.io/badge/Status-Community_Tool-blue.svg)](#disclaimer)
[![Platform: Browser + ComfyUI](https://img.shields.io/badge/Platform-Browser_%7C_ComfyUI-orange.svg)](#requirements)
[![Engines: MiniMax • Wan • LTX](https://img.shields.io/badge/Engines-MiniMax_%7C_Wan_%7C_LTX-green.svg)](#core-pipeline)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

### ⚡ 3-Step Quick Start
1. **Open** `minimax_h3_comfyui_local_generator_v2.html` in any web browser *(or host on GitHub Pages)*.
2. **Configure** scenes, consistency bible, and references *(or upload an AI-filled JSON template)*.
3. **Export & Queue:** Click **Generate Workflow** ➔ download the JSON ➔ drop into ComfyUI ➔ **Queue Prompt**.

---

## 🛠️ Core Pipeline

| Continuity Mode | Mechanism | Drift Profile | Best Fit |
| :--- | :--- | :--- | :--- |
| **Last-Frame Chain** | Frame $N_{\text{last}} \rightarrow \text{Frame } (N+1)_{\text{first}}$ | Minor drift over long runs | Fast continuous movement & rapid cuts |
| **First + Last Keyframes** | Direct interpolation between 2 keyframe images | Zero cumulative drift | Scripted cinematography & fixed storyboards |
| **Reference-to-Video** | Injects character/style sheets via `<Picture N>` | Consistent traits, free angles | Dynamic camera angles across distinct scenes |

---

## 🌟 Feature Matrix

| 🎥 Engine & Resolution | 🧠 Continuity & Quality | 📦 Scale & Long Renders |
| :--- | :--- | :--- |
| • **Video:** MiniMax H3, Wan 2.1/2.2, LTX 2.3/2.5<br>• **Images:** Qwen 2.1, Flux 2, Z-Turbo, H3 Stills<br>• **LoRAs:** Stack custom `file \| strength` lines<br>• **Aspect:** Multiples of 32 (16:9, 9:16, 21:9, 1:1) with auto-crop/pad | • **Anti-Degradation:** Sharpen, upscale, and img2img refresh passes<br>• **Hybrid Anchoring:** Re-anchor every $N$ scenes to fresh reference frames<br>• **Consistency Bible:** Global style, character, and audio design injection | • **Subgraph Packing:** Collapses scenes into subgraphs for fluid UI response<br>• **Skip In-Graph Join:** Offloads memory-heavy concat to FFmpeg<br>• **FFmpeg Assembler:** Script stitches, deduplicates keyframe seams, and mixes TTS |

---

## 🤖 AI Storyboarding Pipeline
