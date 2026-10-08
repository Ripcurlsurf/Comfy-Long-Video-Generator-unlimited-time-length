ComfyUI Local Generator

A single-file, offline HTML tool that generates ComfyUI workflow JSON for long, multi-scene videos. Describe your scenes, configure your settings, and download a ready-to-use workflow you can drag directly into ComfyUI.

There is nothing to install and no server to run. Simply open the HTML file in any modern web browser, configure your project, and export.

## Features

- **Video Engine Selector:** Choose between MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3, or LTX 2.5. Settings that do not apply to the selected engine are automatically hidden, and engine defaults (resolution, frame rate, sampler, and model paths) update dynamically.
- **Image Model Selector:** Choose Qwen Image 2.1, Flux 2, Z-Image Turbo, or *"Same as video model"* (MiniMax H3 stills; MiniMax only). Automatically configures model files, LoRAs, and node pipelines for generated references and first frames.
- **Dedicated Image Sampler and Scheduler:** Separate controls from video generation:
  - **Qwen Image 2.1:** `euler` with `simple` (40 steps).
  - **Flux 2:** `euler` with `Flux 2` scheduler (20 steps, guidance 4).
  - **Z-Image Turbo:** `res_multistep` with `simple` (8 steps, CFG 1).
  - *"Same as video model"* stills share the video sampler, scheduler, and step count (Turbo mode can cause soft outputs; a dedicated image model is recommended for references).
- **Photorealistic Reference Prompts:** Reference and first-frame prompts prepend `"The image is a photorealistic photograph of..."` and append an editable *Image style* line (optimized for Qwen Image 2.1 conventions). A warning displays if a description is under 20 words to prevent stylized or illustrative drift. If the style mentions painting, illustration, or 3D, the photographic opening is omitted automatically.
- **MiniMax Aspect and Resolution Presets:** Supports official 16:9 sizes alongside 9:16, 1:1, 4:3, 3:4, and 21:9 at matching megapixel increments (all multiples of 32), plus custom width and height inputs. Reference images are automatically padded or center-cropped to the target aspect ratio to avoid stretching.
- **Long-Film Detail Preservation:** Mitigates compression and softening across chained scenes using three optional cleanup stages:
  1. High-frequency sharpening pass.
  2. Upscale-model round-trip.
  3. Image-to-image refresh that preserves active references.
- **Suggested Settings Presets:** Choose a target (Fast Preview, Balanced, Best Quality, Long Film) and hardware profile (RAM/VRAM) to auto-configure resolutions, steps, caching, join behaviors, and quality filters. Includes a "Detect GPU" button to detect your graphics card and prefill estimated VRAM.
- **Reference Sheets and Angle Expansion:** 
  - Exceeds the native 9-reference limit for MiniMax by tiling additional reference inputs into composite reference sheets.
  - Generates multi-angle turnarounds (Character: front, left, right, back, three-quarter; Location: main, left, right, reverse, high corner) directly from a primary reference image to maintain subject consistency.
- **Production Pipeline Utilities:** Includes scene range exports, lossless PNG frame sequence output, dependency locking for strictly sequential execution, subgraph packing for responsive node graphs, and an automated `ffmpeg` stitching script.

---

## Core Architecture & Custom Nodes

### Activate Reference Images Switch
Located at the top of the **Reference images** section (disabled by default):
- **Off:** Hides reference lists, filenames, angle configurations, aspect options, and scene-linking controls. No reference images are passed into the workflow.
- **On (MiniMax):** Activates Reference-to-Video mode (`MiniMaxH3ReferenceToVideo`), routing references as `<Picture N>` inputs.
- **On (Wan / LTX):** Because Wan and LTX lack native multi-reference inputs, enabling this switch automatically turns on *Generate scene first frames* (with re-anchoring every scene), directing Qwen or Flux 2 to construct each scene's initial frame from the references, the previous last frame, and the scene prompt.

### Output Naming Convention
Default output filenames adapt dynamically based on the active video engine (e.g., `wan22_final_video.mp4`, `ltx25_final_video.mp4`, `minimax_h3_final_video.mp4`). Scene clips, first frames, reference images, production plans, and workflow exports reflect this prefix unless manually overwritten.

### Photographic Finish Custom Node (`RefPhotoFinish`)
To prevent flat or overly smooth, plastic AI textures without running extra diffusion passes:
1. Check **Photographic finish** under *Video and image engines*.
2. Click **Download the custom nodes** to download `ripcurlsurf_comfy.zip`.
3. Extract the archive into your ComfyUI directory:  
   `ComfyUI/custom_nodes/ripcurlsurf_comfy/` (containing `__init__.py` and `refimagetools.py`).
4. Restart ComfyUI.

This node applies procedural color and tone matching (for generated angle consistency), restores high-frequency texture, and applies fine film grain (default strengths: color `0.7`, detail `0.6`, grain `0.3`).

### Fallback Node Handling (`Load Image (optional)`)
Included within `refimagetools.py`. If a designated manual angle file is missing from `ComfyUI/input/`, the node passes the original reference through rather than terminating the queue with a missing-file error.

Note: ripcurlsurf_comfy is a custom node to be placed in the comfy custom node folder

---

## Requirements

- **ComfyUI:** A recent release with native subgraph support.
- **Video Nodes:** `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, `SamplerCustomAdvanced`, `CreateVideo`, `SaveVideo`, and associated standard nodes.
- **Model Files:**
  - MiniMax H3 first/last-frame and/or `ref2va` checkpoints, text encoder, video VAE, and audio VAE.
  - Image generation models (e.g., Qwen Image 2.1 with `TextEncodeQwenImage21` and `QwenImage21Cache`, Flux 2, or Z-Image Turbo).
- **Video Assembly (Optional):** `ffmpeg` and a Bash environment (Linux, macOS, WSL, or Git Bash for Windows).

> *Note: Default model filenames in the generator can be edited to match your local `ComfyUI/models/` setup.*
> *Note: if you update the ripcurlsurf_comfy node you have to restart comfy or it will error

---

## Quick Start

1. Open `minimax_h3_comfyui_local_generator_v2.html` in your browser (or rename it to `index.html` to host via GitHub Pages).
2. Set your desired **Resolution**, **FPS**, and **Scene Duration**.
3. Add scenes manually with specific prompts, or load an AI-assisted plan via **Upload Filled Template**.
4. Select your **Continuity Mode**.
5. Click **Generate Workflow**, then click **Download Workflow JSON**.
6. Drag the downloaded JSON file into ComfyUI. Place any referenced source images into `ComfyUI/input/`.
7. Queue the workflow. Output video segments are saved to `ComfyUI/output/video/`.

---

## Continuity Modes

| Mode | Consistency Mechanism | Notes |
| :--- | :--- | :--- |
| **Last-frame chain** | The last frame of Scene $N$ serves as the first frame of Scene $N+1$. | Simple and smooth; minor compression artifacts can accumulate over long sequences without cleanup passes. |
| **First + last keyframes** | Each scene generates between defined starting and ending keyframe images. | You provide or generate $N+1$ keyframes using an image model. Frames re-anchor to primary references to eliminate long-term drift. |
| **Reference-to-video mode** | Master reference images pass directly into each scene as `<Picture N>` inputs. | Uses the MiniMax `ref2va` model. First and last frame inputs are bypassed; visual continuity is maintained via reference conditioning. |

---

## Configuration Options

- **First + Last Keyframes:** Connects first and last frames into H3 using filename patterns such as `keyframe_{n}.png`. Enabling "Hard cut" on a scene starts a fresh keyframe pair.
- **Consistency Bible:** Automatically injects global visual style, matched character descriptions, and sound design directives into every scene prompt.
- **External TTS:** Instructs H3 to generate ambient environmental audio and sound effects only, reserving vocal space for external TTS dialogue.
- **Film-Wide Music:** Generates continuous background music using ACE-Step from a single prompt, suppresses per-scene H3 music generation, and configures the `ffmpeg` assembly script to mix music under the video.
- **Skip In-Graph Join:** Saves each scene as an independent MP4 file, delegating concatenation to `ffmpeg`. Strongly recommended for long sequences to prevent Out-Of-Memory (OOM) errors during VRAM decoding.
- **Force Scene Order:** Injects hidden execution dependency links so scenes generate sequentially rather than in arbitrary worker order.
- **Subgraph Packing:** Consolidates each scene pipeline into an individual subgraph node, keeping the ComfyUI canvas organized and responsive.
- **Scene Ranges:** Renders specific subsets of scenes (e.g., `3, 5, 10–14`) while preserving global scene numbering, context links, and random seeds.

---

## Reference Images & Multi-Angle Pipelines

### Managing References
Each entry in the **Reference images** section defines a character, location, or visual style sheet:
- **Name:** Unique identifier used in scene prompts (e.g., `Maya`, `Harbor town`).
- **Filename:** An existing image file inside `ComfyUI/input/`, or the target export filename if generating inside the workflow.
- **Description:** Physical traits, clothing, palette, and lighting. If the filename is left empty in reference-to-video mode, this field serves as the text-to-image generation prompt.

In reference mode, scenes automatically receive prompt bindings:
```text
Reference images: <Picture 1> = Maya; <Picture 2> = Harbor town.

Multi-Angle Consistency

Enable Create several angles of every reference to expand single viewpoint images:

    Character Types: Front, left, right, back, and three-quarter angles.

    Location Types: Main, left, right, reverse, and high-corner views.

Angles are generated strictly from the base reference image—never sequentially from subsequent angles—to prevent identity drift.

Each scene card includes a Camera view of the subjects setting (Auto, Front, Left side, Right side, Back, Three-quarter). When set to Auto, the generator infers the perspective from descriptive keywords in your scene prompt (e.g., "seen from behind", "walks away", "in profile", "over the shoulder").
Long-Film Production Workflow

For extended productions:

    Draft Pass: Set Render pass to Draft. Scenes render at a fraction of full resolution (default: 50%) with reduced sampling steps and upscaling disabled, providing a fast preview for pacing edits.

    Selective Rerendering: Enter specific scene numbers in Scenes to render (e.g., 1–4, 9, 14) to iterate only on approved or revised shots.

    Takes & Finals: Switch to Final. Incrementing Take number alters seeds across all scenes without mutating base reference seeds. Use Download all takes to export multiple workflows simultaneously.

    Lossless Frame Output (Optional): Enable Save each scene as lossless PNG frames (output/lossless/<scene>/f_#####_.png) for precision post-production grading and frame interpolation.

    Finishing Script Assembly: Run assemble_long_video.sh via terminal:

        Detects all generated scene takes and clips.

        Cleans duplicate transition frames when using keyframes.

        Normalizes audio loudness across scenes.

        Mixes external voice tracks from dialogue/scene_NN.wav.

        Exports the master edit via high-bitrate H.264, H.265, ProRes, or FFV1.

AI Template Workflow

Accelerate writing long-form video prompts using an external LLM:

    Click Generate Template to export a project JSON structure containing schema instructions and your current settings.

    Provide the JSON file along with your story outline or treatment to an LLM, instructing it to complete the JSON according to the built-in guidelines.

    Save the output and click Upload Filled Template to populate scene blocks, continuity directives, and reference definitions automatically.

    Make any necessary fine-tuning adjustments on the page, then export your final workflow JSON.

Troubleshooting

    Generate Workflow does not respond: Check for modal alerts indicating validation errors (such as missing scene prompts or invalid frame dimensions).

    Red Nodes in ComfyUI: Verify that custom node packages (MiniMax, Qwen Image 2.1) are up to date and that model filenames in the generator match your local file names.

    ComfyUI Stalls on Complex Graphs: Enable Subgraph packing and Skip in-graph join. Render long productions in batches of 15–20 scenes using Scene range.

    Visual Drift Across Consecutive Shots: Enable Multi-angle references, assign explicit Camera view angles rather than Auto, and decrease the re-anchoring interval to 1–2 scenes.

    Image Files Not Found: Ensure referenced assets exist in ComfyUI/input/. Images generated in-graph save to ComfyUI/output/ref/ and must be moved to input/ to be referenced as static files in future runs.

Contributing

Pull requests, node template updates, and issue reports are welcome. When filing an issue, please include your ComfyUI version, active generator options, and relevant console error outputs.
License

This project is open-source under the MIT License.
