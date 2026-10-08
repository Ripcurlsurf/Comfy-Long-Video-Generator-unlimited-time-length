<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MiniMax H3 ComfyUI Local Generator - README</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --text-color: #c9d1d9;
            --heading-color: #f0f6fc;
            --accent-color: #58a6ff;
            --border-color: #30363d;
            --code-bg: #161b22;
            --blockquote-bg: #1f6feb15;
            --table-alt: #161b22;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 2rem;
        }

        .markdown-body {
            max-width: 900px;
            margin: 0 auto;
            background-color: var(--bg-color);
            padding: 2rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }

        h1, h2, h3 {
            color: var(--heading-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3em;
            margin-top: 24px;
            margin-bottom: 16px;
        }

        h1 { font-size: 2em; }
        h2 { font-size: 1.5em; }
        h3 { font-size: 1.25em; border-bottom: none; }

        p {
            margin-top: 0;
            margin-bottom: 16px;
        }

        a {
            color: var(--accent-color);
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        ul {
            padding-left: 2rem;
            margin-top: 0;
            margin-bottom: 16px;
        }

        li {
            margin-bottom: 0.25em;
        }

        strong {
            color: var(--heading-color);
        }

        code {
            background-color: var(--code-bg);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 85%;
        }

        blockquote {
            margin: 0 0 16px 0;
            padding: 0.5rem 1rem;
            color: #8b949e;
            border-left: 0.25em solid var(--accent-color);
            background-color: var(--blockquote-bg);
            border-radius: 0 6px 6px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16px;
            overflow: hidden;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        th, td {
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            text-align: left;
        }

        th {
            background-color: var(--code-bg);
            color: var(--heading-color);
        }

        tr:nth-child(even) {
            background-color: var(--table-alt);
        }

        .badges {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-block;
            padding: 4px 8px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1;
            color: #fff;
            background-color: #2ea043;
            border-radius: 6px;
        }

        .badge-blue { background-color: #1f6feb; }
        .badge-orange { background-color: #bc4c00; }

        hr {
            height: 0.25em;
            padding: 0;
            margin: 24px 0;
            background-color: var(--border-color);
            border: 0;
        }
    </style>
</head>
<body>

<div class="markdown-body">

    <h1>MiniMax H3 ComfyUI Local Generator</h1>

    <div class="badges">
        <span class="badge badge-blue">Status: Community Project</span>
        <span class="badge">100% Offline</span>
        <span class="badge badge-orange">ComfyUI Compatible</span>
    </div>

    <blockquote>
        <strong>Developer's Note:</strong> I am developing my own script to create very long movies. I explored ComfyUI and Pinokio—both are great platforms, but they have limitations. ComfyUI requires extra nodes to achieve difficult tasks, which heavily consumes resources, especially RAM. These scripts work well within ComfyUI, but I am branching out externally while keeping this project active in case ComfyUI improves its backend.
    </blockquote>

    <p>A powerful, single-file, offline HTML tool that effortlessly builds advanced ComfyUI workflow JSONs for long, multi-scene AI videos using the <strong>MiniMax H3</strong>, <strong>Wan 2.1</strong>, <strong>Wan 2.2</strong>, <strong>LTX 2.3</strong>, and <strong>LTX 2.5</strong> models.</p>
    
    <p>Describe your scenes, configure your production settings, and instantly download a complete workflow ready to drag straight into ComfyUI. No installation required, no server to run. Just open the HTML file in any browser, fill out your project, and export.</p>

    <blockquote>
        <strong>Status:</strong> Community project. Not affiliated with MiniMax, Comfy, or Qwen. Generated workflows are built from standard example templates for H3, Wan, LTX, and Qwen Image nodes. <br>
        <em>Tip:</em> Test with a short 2–3 scene preview run before committing to a heavy, multi-scene render, and verify model filenames against your local ComfyUI installation.
    </blockquote>

    <hr>

    <h2>🚀 Key Features</h2>
    <ul>
        <li><strong>Multi-Engine Video Support:</strong> Choose between MiniMax H3, Wan 2.1, Wan 2.2, LTX 2.3, or LTX 2.5 right from the top of the page. Unused controls automatically hide, and resolutions, FPS, samplers, and defaults update dynamically.</li>
        <li><strong>Flexible Image Model Selector:</strong> Pair your video engine with Qwen Image 2.1, Flux 2, Z-Image Turbo, or mirror the video model (MiniMax). Automatically configures correct LoRAs, samplers, and nodes for reference images and scene first frames.</li>
        <li><strong>Long-Film Quality Pipeline:</strong> Combat detail degradation over long chains with optional cleaning steps between scenes: sharpen passes, upscale-model round-trips, and image-model img2img refreshes.</li>
        <li><strong>Multi-Angle Reference Consistency:</strong> Automatically generate or map multiple character/location angles (front, back, left, right, three-quarter) tied to a master reference image, preventing structural drift across camera cuts.</li>
        <li><strong>AI Template Integration:</strong> Generate a structural template JSON, feed it along with your script or story to an AI assistant to flesh out a full film production, and upload the completed JSON right back into the tool.</li>
        <li><strong>Subgraph Packing:</strong> Automatically packs scenes and join nodes into clean subgraphs to maintain high ComfyUI responsiveness even on massive workflows containing dozens of scenes.</li>
        <li><strong>FFmpeg Assemble & Audio Scripts:</strong> Automatically exports production plans, dialogue sheets, and shell scripts to stitch together rendered scenes, handle audio ducking, loudness normalization, and final master encoding.</li>
    </ul>

    <hr>

    <h2>🛠️ Quick Start Guide</h2>
    <ol>
        <li>Open <code>minimax_h3_comfyui_local_generator_v2.html</code> in any web browser (rename it to <code>index.html</code> if hosting on GitHub Pages).</li>
        <li>Set your target resolution, FPS, and default scene duration.</li>
        <li>Add your scenes and write individual scene prompts, or load them automatically via an AI Template Workflow.</li>
        <li>Choose your preferred Continuity Mode (Last-frame chain, Keyframes, or Reference-to-video).</li>
        <li>Click <strong>Generate Workflow</strong>, then <strong>Download Workflow JSON</strong>.</li>
        <li>Drag and drop the downloaded JSON file into ComfyUI. Place any referenced image files into your <code>ComfyUI/input/</code> directory.</li>
        <li>Queue the workflow! Scene videos will output to <code>ComfyUI/output/video/</code>.</li>
    </ol>

    <hr>

    <h2>🔗 Continuity Modes</h2>
    <table>
        <thead>
            <tr>
                <th>Mode</th>
                <th>How Scenes Stay Consistent</th>
                <th>Best Suited For</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Last-frame chain</strong></td>
                <td>The last frame of scene $N$ becomes the first frame of scene $N+1$.</td>
                <td>Simple, quick sequences; minor error accumulation over long runs.</td>
            </tr>
            <tr>
                <td><strong>First + last keyframes</strong></td>
                <td>Each scene runs between pre-generated keyframe images re-anchored to references.</td>
                <td>Clean productions with zero long-term visual drift.</td>
            </tr>
            <tr>
                <td><strong>Reference-to-video mode</strong></td>
                <td>Reference images are fed directly into every scene as &lt;Picture N&gt; inputs using the <code>ref2va</code> model.</td>
                <td>MiniMax H3 native multi-reference character consistency.</td>
            </tr>
        </tbody>
    </table>

    <hr>

    <h2>📋 System Requirements</h2>
    <ul>
        <li><strong>ComfyUI:</strong> A recent installation supporting subgraphs and required custom nodes:
            <ul>
                <li>MiniMax H3 nodes: <code>MiniMaxH3ImageToVideo</code>, <code>MiniMaxH3ReferenceToVideo</code>, <code>SamplerCustomAdvanced</code>, <code>CreateVideo</code>, <code>SaveVideo</code>.</li>
                <li>Image generation nodes if utilizing Qwen Image 2.1 (<code>TextEncodeQwenImage21</code>, <code>QwenImage21Cache</code>) or Flux/Z-Image templates.</li>
            </ul>
        </li>
        <li><strong>Model Files:</strong> Appropriate H3 model files (first/last-frame model, <code>ref2va</code>), text encoders, video VAEs, and audio VAEs placed in your ComfyUI model directories.</li>
        <li><strong>FFmpeg & Bash Shell:</strong> Required if running the automated <code>assemble_long_video.sh</code> script (compatible with Linux, macOS, or WSL/Git Bash on Windows).</li>
    </ul>

    <hr>

    <h2>📦 Project Saving & Updates</h2>
    <ul>
        <li><strong>Save/Load Project:</strong> Export your complete production setup (settings, options, scenes, reference sheets, camera views, and dialogues) into a single JSON project file. Browser local backup is also active automatically.</li>
        <li><strong>Built-in Update Checker:</strong> Automatically checks the GitHub repository for updates upon loading so you're always running the latest workflow template generator.</li>
    </ul>

    <hr>

    <h2>🤝 Contributing & License</h2>
    <p>Contributions, bug reports, and pull requests are warmly welcomed! When opening an issue, please include your ComfyUI version, active option checkboxes, and any relevant console error logs.</p>
    <p>Distributed under the MIT License. See <code>LICENSE</code> for more information.</p>

</div>

</body>
</html>
