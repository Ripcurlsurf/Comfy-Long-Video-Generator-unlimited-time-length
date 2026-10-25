# ComfyUI Film Generator: Long, Multi-Scene AI Films

[![ComfyUI Version](https://img.shields.io/badge/ComfyUI-October%202026%2B-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Custom Nodes](https://img.shields.io/badge/Nodes-ripcurlsurf__comfy-green.svg)]()
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

A standalone browser-based tool that compiles a complete, production-ready ComfyUI workflow for multi-scene AI films of any length. Generates and anchors reference images, scene stills, video shots, transitions, native dialogue, Foley/music, and automated post-processing scripts—locking character likeness, scene geometry, and grain consistency across cuts.

Runs locally directly from disk. **Zero servers, zero Node.js dependencies, zero API keys, and 100% offline.** Accompanied by a lightweight node suite (`ripcurlsurf_comfy`) that enforces runtime quality gates and handoffs inside ComfyUI.

---

## Repository Structure

| File / Path | Description |
| :--- | :--- |
| `comfyui_local_generator_v2.html` | The generator interface. Open directly in any modern browser. |
| `ripcurlsurf_comfy/` | Custom nodes package (6 standalone `.py` modules). |
| `ripcurlsurf_comfy/ripcurlsurf_comfy.zip` | Bundled archive of the custom node suite (downloadable via the HTML UI). |
| `setup_ripcurlsurf.bat` / `.ps1` | Automated Windows installer for ComfyUI Desktop/standalone environments. |

---

## Quick Start

### 1. Requirements
* **ComfyUI:** Current build (October 2026 or later; `MOSS-Image` requires `CLIPLoader` type `moss_image`). Compatible with ComfyUI Desktop, portable zip installs, and manual git clones.
* **Dependencies:** Custom nodes require only standard `numpy` and `Pillow` (already present in ComfyUI runtimes).

### 2. Install Nodes
Place `ripcurlsurf_comfy` directly into your `custom_nodes/` directory so the entry point exists at:
```text
ComfyUI/custom_nodes/ripcurlsurf_comfy/__init__.py
