ComfyUI Local Workflow Generator
Build production-ready, multi-scene ComfyUI workflows for long AI video generation. A standalone, zero-install local HTML generator optimized for MiniMax H3, Wan 2.1/2.2, LTX Video, Qwen Image 2.1, and Flux 2.

Overview
The MiniMax H3, LTX and Wan - ComfyUI Local Generator is an offline, client-side web application that automates the creation of complex ComfyUI workflow JSON files. It solves the primary bottlenecks in generative AI filmmaking: character drift across scenes, canvas clutter with massive node counts, VRAM limits during long-sequence rendering, and multi-angle asset conditioning.

Generate multi-scene sequences, maintain strict character consistency using image conditioning or keyframing, configure audio tracks, and export modular workflow JSON files ready to drag and drop straight into ComfyUI.

Key Features
🎬 Multi-Model Video Architecture
Video Engine Selection: Native nodes for MiniMax H3 (MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo), Wan 2.1 / Wan 2.2, and LTX 2.3 / LTX 2.5. UI inputs automatically adapt, showing only valid samplers, steps, resolutions, and parameter fields for the selected engine.

Resolution Presets: Multiples of 32 tailored for generative diffusion engines: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9 ultrawide, and custom manual pixel controls. Automatic center-cropping and padding prevent image distortion.

👤 Identity Preservation & Reference Generation
Integrated Reference Engines: Generate conditioning visuals directly in-graph using Qwen Image 2.1, Flux 2, Z-Image Turbo, or native MiniMax H3 stills.

ComfyUI Edit Template Compliance: Qwen reference and edit setups feature TextEncodeQwenImage21, QwenImage21Cache, and latent switching via ComfySwitchNode.

Multi-Angle Asset Expansion: Automatically generate Front, Left, Right, Back, and Three-Quarter views from an original subject asset to maintain consistency across changing camera angles.

Smart Reference Canvas Packing: Overcomes the standard 9-reference limit by auto-stitching supplementary reference images into unified reference sheets.

⛓️ Advanced Scene Continuity & Drift Control
Three Continuity Modes: Seamless last-frame chaining, dual keyframing (first/last frame constraints), and pure <Picture N> reference-to-video mode (ref2va).

Long-Film Anti-Drift Refresh: Counter cumulative compression blur on extended runs with automated sharpen passes, upscale round-trips, and periodic img2img re-anchoring.

Hard Cut Isolation: Toggle hard scene breaks to sever frame carryovers on dramatic scene transitions while preserving project-wide identity anchors.

⚡ Performance & Scalability
Subgraph Node Packing: Collapses complex per-scene generation logic into single subgraphs, preventing interface lag on multi-dozen scene compositions.

Hardware Preset Recommendations: One-click presets optimized for speed, balanced output, or maximum quality based on detected GPU hardware and VRAM.

Sequence Dependency Locks: Optional hidden link dependencies ensure scenes render sequentially without memory deadlocks.

📦 Long-Form Production Pipeline
Multi-Take Iteration: Render fast, downscaled draft passes, manage scene seeds, output alternative takes (_t2), and selectively re-render specific scene indices (e.g., 3, 5, 10-14).

Lossless Frame Exports: Export individual frames (output/lossless/<scene>/f_#####_.png) for professional post-production.

Audio Stems & Finishing Scripts: Automatically generates a comprehensive assemble_long_video.sh bash script with automated loudness normalization, TTS audio stem replacement, cross-scene stitching, deflickering, and LUT color grading.

Supported Engine Matrix
Model Category	Supported Engines & Frameworks	Primary ComfyUI Nodes
Video Models	MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3, LTX 2.5	MiniMaxH3ReferenceToVideo, MiniMaxH3ImageToVideo
Image / Ref Models	Qwen Image 2.1, Flux 2, Z-Image Turbo, MiniMax H3 Stills	TextEncodeQwenImage21, QwenImage21Cache, LoraLoader
Audio & Music	ACE-Step, External TTS Audio Pipeline	Custom stem mapping, external audio workflows
Post-Processing	Integrated Upscalers, FFmpeg Assembly Pipeline	SamplerCustomAdvanced, SaveVideo, ProRes/H.264/H.265
Continuity Modes Comparison
[Mode 1: Last-Frame Chain]
Scene 1 (End Frame) ──────► Scene 2 (First Frame) ──────► Scene 3 (Accumulates Drift)

[Mode 2: First + Last Keyframes]
Keyframe A ──► [ Scene 1 ] ──► Keyframe B ──► [ Scene 2 ] ──► Keyframe C (Drift-Free)

[Mode 3: Reference-to-Video]
Reference Sheet (<Picture 1...N>) ──┬──► Scene 1 (ref2va)
                                   ├──► Scene 2 (ref2va)
                                   └──► Scene 3 (ref2va)
Last-Frame Chaining: Ideal for continuous, fluid shots. Feeds the final frame of Scene N directly into Scene N+1.

First + Last Keyframes: Eliminates gradual visual degradation across extended sequences by anchoring scene boundaries to predetermined keyframe assets.

Reference-to-Video Mode (ref2va): Injects character identity, locations, and wardrobe sheets across all clips using <Picture N> token associations.

Quick Start
Launch the Interface:
Download and open minimax_h3_comfyui_local_generator_v2.html in any modern web browser (or serve it directly via GitHub Pages by renaming to index.html). No Node.js, Python, or web server required.

Configure Engine & Scenes:

Select your target Video Engine (e.g., MiniMax H3) and Image Model (e.g., Qwen Image 2.1).

Define your scene descriptions, camera perspectives (Auto, Front, Profile, Reverse), and frame durations.

Add character/location definitions to the Reference Images panel.

Export and Queue:

Click Generate Workflow, then Download Workflow JSON.

Drag the exported JSON file directly onto your ComfyUI workspace.

Place any external source assets into your local ComfyUI/input folder and queue the prompt.

AI Prompt Engineering Template Workflow
Accelerate long-form screenwriting by letting Large Language Models structure your scenes:

Click Generate Template to export a formatted JSON schema containing your generation constraints, references, and scene fields.

Provide the downloaded JSON to your preferred LLM alongside your script or premise with instructions to complete the scene entries.

Click Upload Filled Template to load the AI-generated script back into the interface.

Review prompts, refine camera views, adjust continuity parameters, and export your ready-to-run ComfyUI workflow.

Complete Quality Production Pipeline
  1. Draft Pass      2. Scene Select     3. Final Takes     4. Audio & Stems     5. Final Assembly
┌──────────────┐   ┌────────────────┐   ┌──────────────┐   ┌────────────────┐   ┌────────────────┐
│  50% Scale   │──►│ Pick Approved  │──►│ Render Multi-│──►│ ACE-Step Music │──►│ FFmpeg Finish  │
│  Fast Steps  │   │  Indices Only  │   │  Takes (_t2) │   │ External Voice │   │  LUTs & Master │
└──────────────┘   └────────────────┘   └──────────────┘   └────────────────┘   └────────────────┘
Drafting: Run rapid, low-resolution passes without upscale passes to validate pacing and motion blocking.

Filtering: Use the scene index selector (e.g., 1,2,5-8) to render only missing or approved segments.

Takes & Seeds: Maintain base identity via the fixed Project Seed while varying clip seeds using Take Numbers (_t2).

Dialogue & Audio: Export dialogue breakdown sheets (.csv) for TTS generation, generate film scores using the ACE-Step audio workflow, and place spoken lines into dialogue/scene_NN.wav.

Finishing Script: Execute assemble_long_video.sh to assemble video segments, normalize audio mixes, and compile master outputs in ProRes, FFV1, or H.264/H.265.

System Requirements
ComfyUI Environment: A current build of ComfyUI with Subgraph support enabled.

Custom Nodes: Respective model wrapper nodes (MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, or equivalent Wan/LTX/Qwen nodes).

Model Checkpoints: Corresponding Diffusion weights, UNETs, Text Encoders, Video VAEs, and Audio VAEs installed in your ComfyUI model directories.

Assembly Tools (Optional): ffmpeg installed within your terminal path (Linux, macOS, or WSL/Git Bash on Windows).

Troubleshooting Guide
Issue Observed	Root Cause	Resolution
Workflow generation fails	Incomplete required fields or invalid frame values	Check on-screen alerts; confirm all scenes contain prompts and frame rates are non-zero.
Nodes highlight red in ComfyUI	Node name mismatch or missing custom node packs	Update your ComfyUI custom node packages and confirm model filenames match your local folder structure.
ComfyUI UI freezes on load	Extremely high flat-node volume across dozens of scenes	Re-export with Subgraph Packing enabled to group scene logic.
Drift on side/rear camera angles	Single-angle reference conditioning	Enable Multi-Angle References and set the scene's camera angle explicitly.
Missing inputs on execution	ComfyUI cannot find specified image filenames	Ensure all referenced reference files reside inside ComfyUI/input/.
Contributing
Contributions, bug reports, and node template updates are welcome. When opening an issue, please provide your ComfyUI version, active generation options, and any associated error logs.

License
Distributed under the MIT License. Ensure compliance with individual model weights and third-party node licenses.
