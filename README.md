ComfyUI Local Long Video Workflow Generator

[![ComfyUI Compatible](https://img.shields.io/badge/ComfyUI-Compatible-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Single-File HTML](https://img.shields.io/badge/Setup-Zero--Install%20HTML-brightgreen.svg)](#quick-start)
[![Engines Supported](https://img.shields.io/badge/Engines-MiniMax%20H3%20%7C%20Wan%202.1%2F2.2%20%7C%20LTX%20Video-orange.svg)](#supported-engines--models)

> **Zero-install, browser-based ComfyUI workflow generator for long AI video generation, multi-scene continuity, multi-angle character references, and prompt consistency.** Built natively for **MiniMax H3**, **Wan 2.1 / Wan 2.2**, **LTX 2.3 / LTX 2.5**, **Qwen Image 2.1**, and **Flux 2**.

---

## Overview

The **MiniMax H3 ComfyUI Local Generator** is a lightweight, offline HTML utility that outputs fully wired **ComfyUI workflow JSON files** for long-form, multi-scene AI video productions.

Eliminate the manual wiring overhead of ComfyUI. Define your shots, attach character references, configure scene transitions, and export modular workflow graphs packed into clean subgraphs ready to drag and drop straight onto your ComfyUI canvas.

---

## Key Features

### 🎬 Multi-Engine Video & Diffusion Support
- **Cross-Engine Support:** Native workflow nodes for **MiniMax H3** (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`), **Wan 2.1 / Wan 2.2**, and **LTX 2.3 / LTX 2.5**.
- **Dynamic Context UI:** Inputs automatically adapt to your chosen engine; unsupported samplers, steps, and options are hidden automatically.
- **Engine-Aware File Naming:** Output media and configuration files adapt automatically to your video engine (e.g., `minimax_h3_final_video.mp4`, `wan22_final_video.mp4`, `ltx25_final_video.mp4`).
- **Aspect Ratio & Megapixel Presets:** Native multiples of 32 for 16:9, 9:16, 1:1, 4:3, 3:4, and 21:9 ultrawide with automated aspect-aware center cropping and padding.

### 👤 Global Reference Engine & Character Consistency
- **Single Master Switch ("Activate Reference Images"):** One master toggle configures your entire reference pipeline.
  - **MiniMax H3:** Activates `ref2va` Reference-to-Video mode, wiring every reference as a `<Picture N>` token.
  - **Wan & LTX:** Automatically configures *Generate Scene First Frames* with single-scene re-anchoring, prompting Qwen Image 2.1 or Flux 2 to construct each shot's opening frame using your references and the prior scene's ending frame.
- **Per-Scene Reference Linking:** Choose *Automatic* (links only references mentioned by name or dialogue in that scene), *Manual Selection* (per-scene checkboxes), or *All References*.
- **Canvas Stitching (Exceeding 9 References):** Automatically stitches references side-by-side into composite sheets to bypass MiniMax's native 9-image limit.
- **Multi-Angle Generation:** Generates Front, Left, Right, Back, and Three-Quarter views directly from a single reference asset using Qwen Image 2.1 edit mode, Flux 2, or MiniMax H3 reference stills.
- **Auto-Camera Matching:** Scenes automatically parse shot prompts (e.g., *"seen from behind"*, *"profile view"*, *"reverse angle"*) and feed the corresponding perspective asset.

### 🛡️ Anti-Drift Continuity & Scene Guard
- **Scene Guard Node (`sceneguard.py`):** Monitors generated clips for unwanted model drift, scene hallucinations, or sudden cuts, automatically trimming or holding the final good frame to keep actions locked to your prompt.
- **Continuity Blending for First Frames:** Scene 2 onward blends the previous scene's final frame with your core character references and the current scene prompt. Control visual carryover with the *Continuity Strength* slider (0.2–0.9).
- **Long-Sequence Quality Refresh:** Counter cumulative compression degradation with automated sharpen passes, upscale round-trips, and periodic img2img re-anchoring.
- **Hard Cuts:** Sever frame carryovers instantly for scene transitions while keeping global style and character identity locked.

### 🎨 Finishing Pipeline & Custom Nodes (`ripcurlsurf_comfy`)
- **Photographic Finish (`RefPhotoFinish`):** Eliminates plastic, oversaturated diffusion artifacts. Restores micro-contrast, matches tonal balance against original reference photos, and layers organic film grain.
- **Optional Asset Fallback (`Load Image Optional`):** Prevents ComfyUI workflow stalls by substituting missing reference angles with the primary reference image.
- **Production Stems & FFmpeg Assembly:** Generates an automated `assemble_long_video.sh` bash script with per-segment loudness normalization, TTS dialogue mixing (`dialogue/scene_NN.wav`), and lossless ProRes/FFV1/H.264/H.265 master encoding.
- **Audio Workflows:** Generates full-length background scores using ACE-Step in a dedicated audio graph and exports dialogue CSV cue sheets.

### ⚡ Performance & Scalability
- **Subgraph Packing:** Encapsulates each scene inside its own subgraph node to keep large, multi-scene compositions responsive in ComfyUI.
- **Hardware Profile Presets:** Apply balanced, speed, or high-fidelity settings based on local system RAM and GPU VRAM.
- **Strict Scene Ordering:** Hidden dependency locks force sequential rendering across Wan, LTX, and MiniMax pipelines to prevent VRAM spikes.
- **Update Checker:** Integrated non-intrusive version bar alerts you whenever updates are pushed to GitHub without breaking offline workflow functionality.

---

## Supported Engines & Models

| Component | Supported Architectures | Primary ComfyUI Nodes |
| :--- | :--- | :--- |
| **Video Engine** | MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3, LTX 2.5 | `MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo` |
| **Image / Ref Engine** | Qwen Image 2.1, Flux 2, Z-Image Turbo, MiniMax H3 Stills | `TextEncodeQwenImage21`, `QwenImage21Cache`, `LoraLoader` |
| **Custom Helpers** | `ripcurlsurf_comfy` Node Pack | `RefPhotoFinish`, `Load Image (optional)`, `Scene Guard` |
| **Sound & Voice** | ACE-Step Music, External TTS Integration | Standalone audio subgraphs, FFmpeg stem ducks |

---

## Continuity Modes

| Mode | Visual Consistency Mechanism | Recommended Use Case |
| :--- | :--- | :--- |
| **Last-Frame Chain** | The final frame of scene *N* becomes the starting frame of scene *N+1*. | Smooth, continuous camera movements and uninterrupted real-time takes. |
| **First + Last Keyframes** | Clips are generated between pairs of anchor keyframe images. | Sequence shots where starting and ending compositions are predetermined. |
| **Reference-to-Video (`ref2va`)** | Reference sheets are injected directly into MiniMax via `<Picture N>` tokens. | Preserving complex character wardrobes, facial identities, and locations across cuts. |

---

## Custom Nodes Installation

The optional helper nodes improve asset fallback, image quality, and generation stability:

```bash
cd ComfyUI/custom_nodes/
# Unpack the included ripcurlsurf_comfy package
unzip ripcurlsurf_comfy.zip

This project is open-source under the MIT License.
