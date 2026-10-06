ComfyUI Long Movie Local Generator
A single-file, offline HTML tool that builds ComfyUI workflow JSON for long, multi-scene videos with the MiniMax H3 model. Describe your scenes, set your options, and download a workflow you can drag straight into ComfyUI.
There is nothing to install and no server to run. Open the HTML file in a browser, fill it in, and export.
> **Status:** community project, not affiliated with MiniMax, Comfy, or Qwen. The generated workflows are built from example workflows for the H3 and Qwen Image 2.1 nodes. Test with a short 2-3 scene run before committing to a long render, and check the model filenames against your own install.
---
Features
Video engine selector (top of the page). Choose MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3 or LTX 2.5. Controls that do not apply to the chosen engine are hidden, and defaults (resolution, FPS, sampler, model files) switch with it. MiniMax output is unchanged. Wan and LTX nodes follow ComfyUI's standard templates and are unverified against your install.
Image model selector (top of the page). Next to the video model, choose Qwen Image 2.1, Flux 2, Z-Image Turbo, or "Same as video model" (MiniMax H3 still, MiniMax only). It sets the model files, LoRAs and nodes used for generated reference images and scene first frames. Fields that do not apply to the chosen image model are hidden.
MiniMax resolution presets. A dropdown (MiniMax only) lists your official 16:9 sizes plus 9:16, 1:1, 4:3, 3:4 and 21:9 at the same megapixel steps, all multiples of 32, with a Custom option that reveals width and height. Reference images are padded or center-cropped to the video aspect before reaching MiniMax, so they are never stretched.
Long-film quality. Chaining scenes from the previous scene's last frame loses detail over time, because each frame is already compressed and soft. Three optional steps clean the frame before it starts the next scene: a sharpen pass, an upscale-model round trip, and an image-model img2img refresh that keeps your references. An editable continuation prompt asks for sharp, artifact-free output. Combine with a low "Re-anchor every N scenes" (2 to 4), 0.5 MP or higher resolution, and no Turbo for final renders.
Upscale. A tick box at the bottom adds an upscale model node to every scene, with an optional final resize.
Clear JSON. Generate Workflow scrolls to the top and a Clear button empties the JSON box.
Per-scene reference lists are optional. By default every scene links every reference, so a reference added later is never skipped. The preview lists which references each scene links.
Any number of references. MiniMax accepts at most 9 reference images per node. Extras are stitched side by side into sheets so every reference stays linked.
Multi-scene workflows. Add as many scenes as you like, each with its own prompt, duration, optional first/last frame, and seed.
Several ways to carry continuity between scenes. Last-frame chaining, first and last keyframes, or reference images.
Reference-to-video mode. Uses the `MiniMaxH3ReferenceToVideo` node and the ref2va model. Characters, locations and style sheets are fed in as `<Picture N>` references.
Reference image generation. Optionally generate the reference images inside the same workflow with Qwen Image 2.1, Flux 2, Z-Image Turbo or MiniMax H3, with optional LoRAs.
Consistency bible. A style line, character descriptions and a sound-design line are injected into every scene prompt.
Dialogue and sound. Per-scene dialogue and sound notes. Optionally route dialogue to external TTS and keep H3 audio for ambience.
Long-film tools. Scene ranges, skipping the in-graph join, an ffmpeg assemble script, and a production plan export.
Strict scene ordering. An optional dependency lock so scenes generate in sequence.
Subgraph packing. Each scene becomes one subgraph node, which keeps ComfyUI responsive on workflows with dozens of scenes.
AI template workflow. Generate a template, have an AI fill in the whole film, upload it, tweak, and export.
---
Requirements
A recent ComfyUI (with subgraph support) that includes the MiniMax H3 nodes:
`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, plus the usual `SamplerCustomAdvanced`, `CreateVideo`, `SaveVideo` and related nodes.
The H3 model files you intend to use (first/last-frame model, and/or the ref2va model), the text encoder, video VAE, and audio VAE.
For image generation: the nodes and model files for your chosen engine (Qwen Image 2.1: `TextEncodeQwenImage21`, `QwenImage21Cache`; Flux 2 and Z-Image Turbo: their standard ComfyUI template nodes, which are unverified).
For the ffmpeg assemble script: `ffmpeg` and a bash shell (Linux, macOS, or WSL/Git Bash on Windows).
Default model filenames are filled in on the page and are editable. Make sure they match what is in your ComfyUI `models` folders.
---
Quick start
Open `minimax_h3_comfyui_local_generator_v2.html` in a browser (renaming it to `index.html` also works for GitHub Pages).
Set resolution, FPS and default scene duration.
Add scenes and write a prompt for each, or load them from a template (see below).
Choose your continuity method (see Modes).
Click Generate Workflow, then Download Workflow JSON.
Drag the JSON into ComfyUI. Put any image files you reference into `ComfyUI/input`.
Queue the workflow. Scene videos are saved under `ComfyUI/output/video/`.
If something blocks generation, a pop-up lists the errors. Warnings (for example about memory or missing images) appear in the preview panel and do not block the build.
---
Continuity modes
Mode	How scenes stay consistent	Notes
Last-frame chain	The last frame of scene N is the first frame of scene N+1.	Simple and smooth, but small errors accumulate over many scenes.
First + last keyframes	Each scene runs from one keyframe image to the next.	You create N+1 keyframes with an image model. Keyframes are re-anchored to your references, so no drift builds up.
Reference-to-video mode	Reference images are fed to every scene as `<Picture N>` inputs.	Uses the ref2va model. There is no first or last frame input, so frame-based options are ignored in this mode.
---
Options
Option	What it does
First + last keyframes	Wires first and last frames into H3 using a filename pattern such as `keyframe_{n}.png`. Per-scene "hard cut" starts a fresh keyframe pair.
Consistency bible	Adds a style line, matching character descriptions and a global sound line to each scene prompt.
External TTS	Tells H3 to produce ambience and effects only. Dialogue is left for your TTS.
Film-wide music	Generates music tracks with ACE-Step from one music prompt, tells H3 not to make its own per-scene music, and mixes the tracks under the film in the ffmpeg script.
Skip in-graph join	Saves each scene as its own MP4 and leaves joining to ffmpeg. Strongly recommended for long films.
Plan JSON + ffmpeg export	Enables the Plan JSON and ffmpeg script download buttons.
Force scene order	Adds hidden dependency links so scenes generate in sequence.
Subgraph packing	Packs each scene (and the join) into a subgraph node to reduce node count.
Reference-to-video mode	The only mode that links your reference images to MiniMax; the first-frame model has no reference inputs. Every reference is linked to every scene (any number, 1 to 20 or more), unless you filter by name or use the per-scene list. By default the previous scene's last frame is also passed as `<Picture 1>`. Uses `MiniMaxH3ReferenceToVideo` with the ref2va model and LoRA.
Image engine	Choose how generated images are made: Qwen Image 2.1, Flux 2, Z-Image Turbo, or MiniMax H3 (a 5-frame text-to-video still). Optional LoRAs, one per line as `filename | strength`, are added with `LoraLoader` nodes (not for the H3 engine). Qwen and Flux 2 can use your reference images as conditioning; Z-Image Turbo and H3 use the text prompt and reference descriptions only.
Generate reference images	Creates your reference images in the workflow with the chosen image engine.
Generate scene first frames	Hybrid anchoring for long films. Every Nth scene (and every hard cut) starts from a first frame generated by the image engine from your reference images. The scenes in between continue from the previous scene's last frame. Frames are saved to `ComfyUI/output/first_frames/`.
Scene range	Render only scenes from to to while keeping global scene numbering and seeds.
---
Reference images
The Reference images section holds one card per character, location or style sheet:
Name: a short label, for example `Maya` or `Harbor town`.
Filename: an image in `ComfyUI/input`, or the saved name when generating.
Description: look, outfit and voice. With "Generate reference images" on, this is also the generation prompt.
In reference mode each scene gets a line such as `Reference images: <Picture 1> = Maya; <Picture 2> = Harbor town.` at the start of its prompt. Tags are numbered in connection order. There is no cap on the number of references.
By default, every scene gets all references. Tick Only link references whose name appears in the scene prompt to filter by name, or fill the per-scene "Reference images for this scene" field (numbers or names) to choose manually.
---
AI template workflow
Click Generate Template. A JSON file downloads containing instructions for an AI, plus your current settings, references and scenes.
Give the file to an AI along with your story, premise or long prompt, and ask it to follow the instructions in the file and return only the completed JSON.
Save the reply as `.json` (or `.txt`) and click Upload Filled Template. Scenes, references, consistency fields and settings are filled in. A status line reports what was loaded.
Make any last changes on the page, then Generate Workflow and download the final JSON.
Notes:
The uploader tolerates markdown code fences or extra text around the JSON.
Keys starting with `_` (the instructions) are ignored.
A ComfyUI workflow file is not a template and is rejected with a clear message.
Fields missing from the template are left unchanged.
---
Template coverage
Generate Template exports every setting and option on the page, so anything you change shows up in the JSON: named `settings` and `options`, the `consistency` fields, `reference_images`, `scenes`, and an `advanced` map containing every page control by id. Upload Filled Template applies all of them.
Making very long films
A one-hour film is hundreds of scenes, so plan for it:
Tick Skip in-graph join and join with the generated ffmpeg script instead of an in-graph join. A long in-graph join holds all decoded frames in RAM, and the page warns you when it would be large.
Keep Subgraph packing on and render in ranges of roughly 15-20 scenes per workflow.
For last-frame chain runs split into chunks, the final scene of each run saves `<name>_lastframe_NN`. Copy that image into `ComfyUI/input` and use its filename as the Global first frame filename of the next chunk.
Use keyframes or reference mode for long films, since they do not accumulate drift like chaining.
Benchmark one scene first and multiply to estimate total render time.
ffmpeg assemble script
With Plan JSON + ffmpeg export on, Download ffmpeg script produces `assemble_long_video.sh`. Run it inside your ComfyUI `output` folder. It:
finds each scene file (ComfyUI adds a `_00001_` suffix to saved names),
trims the duplicated frame at shared joins when keyframes are used,
normalizes loudness per segment,
mixes optional TTS dialogue from `dialogue/scene_NN.wav`,
concatenates everything into `<name>_ffmpeg_joined.mp4`.
It skips segments that already exist, so you can re-run it after rendering more scenes. There are no crossfades yet.
---
Troubleshooting
Generate Workflow does nothing. A blocking error appears in a pop-up. Common causes: no scene prompts, or an invalid frame setting.
A node shows red in ComfyUI. The node name or model file does not match your install. Check the filenames on the page, and update ComfyUI and the H3 / Qwen custom nodes.
ComfyUI is slow or stalls on a big workflow. Keep Subgraph packing on, use scene ranges, and consider Skip in-graph join.
The file fails to load with subgraphs on. Untick Subgraph packing and regenerate to get a flat workflow.
Scenes run out of order. Keep Force scene order on. Save nodes can still appear in any order, but generation is sequential.
Reference image missing in a scene. Check you have not enabled name-only matching, and that the reference has a filename (or a generation prompt with "Generate reference images" on).
Reference mode ignores my first or last frames. Expected. The ref2va node has no frame inputs.
Image files not found. Files used by `LoadImage` must be in `ComfyUI/input`. Generated references save to `ComfyUI/output/ref/` and must be copied into `input` to reuse them as files.
---
Known limitations
H3 voice consistency across clips is not guaranteed. For stable character voices, use External TTS and an external voice-cloning TTS.
Reference images condition the video through the ref2va node only. They are not used by the first/last-frame model.
The prompt enhancer from the Qwen Image 2.1 template is not included.
The tool does not generate per-scene keyframes. Create them separately, for example with an image-edit model and your reference sheets.
Node names and widget layouts follow example workflows and may change as ComfyUI and the model nodes are updated.
---
Contributing
Issues and pull requests are welcome. When reporting a problem, please include your ComfyUI version, the option checkboxes you used, and the console or red-node error message.
License
Add your license here (for example MIT) before publishing.
