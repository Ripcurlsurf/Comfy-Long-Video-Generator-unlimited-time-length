# ComfyUI Local Generator: Long Multi-Scene Video Workflow Builder

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: ComfyUI](https://img.shields.io/badge/ComfyUI-Workflow%20Generator-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Offline First](https://img.shields.io/badge/Offline-100%25%20Local-brightgreen.svg)]()
[![Engines: MiniMax%20H3%20%7C%20Wan%202.2%20%7C%20LTX%202.5](https://img.shields.io/badge/Models-MiniMax%20H3%20%7C%20Wan%20%7C%20LTX-purple.svg)]()

A lightweight, zero-install, single-file offline HTML application that generates production-grade **ComfyUI JSON workflows** for multi-scene cinematic AI video generation.

Designed specifically for **MiniMax H3**, **Wan 2.1 / 2.2**, and **LTX 2.3 / 2.5**, this tool solves character drift, scene continuity, aspect-ratio stretching, and video-model freezing without requiring a running server, Node.js, or external API keys.

---

## Key Highlights & Capabilities

* **Zero-Setup & 100% Local**: Single self-contained HTML file. No build steps, dependencies, or telemetry. Runs completely in any modern web browser.
* **Multi-Engine Video Generation**: Target **MiniMax H3**, **Wan 2.1**, **Wan 2.2**, **LTX 2.3**, or **LTX 2.5** with dynamic UI adapting to each engine's native nodes, samplers, and schedulers.
* **Integrated Reference Generation**: Generate character and environment turnaround references directly inside the workflow using **Qwen Image 2.1**, **Flux 2**, **Z-Image Turbo**, or native MiniMax stills.
* **Automated Multi-Angle Camera Turnarounds**: Preserves character and location consistency by synthesizing front, side, back, and 3/4 angles directly from a single reference image via Qwen/Flux edit graphs.
* **Native Subgraph Packing**: Automatically collapses complex modular multi-node scene graphs into ComfyUI subgraphs, maintaining high canvas UI performance even across 50+ scene films.
* **Drift-Free Scene Continuity**: Choose between Last-Frame Chaining (with auto-sharpening/upscaling repair passes), Keyframe Anchoring, or Reference-to-Video modes.
* **Smart Scene Guard & Auto-Splitting**: Auto-splits long scenes by sentence boundaries (~5s per action) to bypass MiniMax context limits, with integrated post-generation anomaly detection (cuts, freezes, replays).

---

## Supported Architecture & Compatibility

| Component | Target Models & Custom Nodes |
|---|---|
| **Video Models** | MiniMax H3 (`MiniMaxH3ReferenceToVideo`, `MiniMaxH3ImageToVideo`), Wan 2.1, Wan 2.2, LTX 2.3, LTX 2.5 |
| **Image Models** | Qwen Image 2.1 (`TextEncodeQwenImage21`, `QwenImage21Cache`), Flux 2, Z-Image Turbo |
| **Audio & Music** | ACE-Step standalone music pipeline, external dialogue sheet / TTS sync |
| **Custom Helpers** | `ripcurlsurf_comfy` suite: `Scene Guard`, `RefPhotoFinish`, `Load Image (optional)` |
| **Assembly** | Lossless bash script generation (`ffmpeg` multi-track concat, loudness normalizer, LUT grading) |

---

## Continuity & Generation Modes

| Mode | Best For | Technical Mechanism |
|---|---|---|
| **Long Consistent Movie (Preset)** | Feature-length films, dramatic narrative continuity | Uses an image model (Qwen 2.1 / Flux 2) to build a consistent first frame from character references and previous pose text, then runs image-to-video for $\le 5\text{ s}$. Prevents reference flashing. |
| **Reference-to-Video (`ref2va`)** | Fast multi-subject generation | Feeds character/setting sheets directly into MiniMax `<Picture N>` tokens. Handles $>9$ images via auto side-by-side sheet stitching. |
| **Last-Frame Chaining** | Continuous motion shots | Passes Frame $N_{last} \to N+1_{first}$ through an optional sharpening, upscale, and img2img de-artifacting filter. |
| **First + Last Keyframes** | Tightly planned VFX & complex blocking | Renders scenes bounded by predefined image-model keyframes (`keyframe_{n}.png`). Zero accumulated drift. |

---

## Quick Start Guide

### 1. Installation & Environment Setup
Clone the repository:
```bash
git clone [https://github.com/](https://github.com/)<your-username>/minimax-h3-comfyui-generator.git
cd minimax-h3-comfyui-generator
