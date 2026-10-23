"""
troubleshoot.py - ComfyUI custom node "Troubleshooting Log" (package: ripcurlsurf_comfy).

Only used when 'Troubleshooting logs' is ticked in the long-video generator. For every clip it writes:
  - an entry in ComfyUI/output/movie_log.txt (the one log file of the ripcurlsurf nodes): settings and prompt, frame
    counts, where the clip moves and where it stands still, when it reaches its last picture, picture quality over
    time against the gold frame, brightness over time (flicker), sound level (silent gaps), and at the end the same
    numbers on one line for checking by a program
  - ComfyUI/output/troubleshooting/<film>_<clip>.png, a picture sheet: the story first and last picture, frames across
    the raw clip (as the video model made it) and across the finished clip, a movement graph with the still parts
    marked, and the key numbers
The first clip of every ComfyUI session also records the computer: Python, PyTorch, GPU and memory.
It only reads the pictures; nothing it does changes the video. It never stops a render: any problem is written to the
log instead. Plain numpy and Pillow (both part of ComfyUI).
"""
import json
import math
import os
import platform
import re
import time

import numpy as np

from .qualitylock import _out_dir, _resize, movie_log, profile, quality_score, relative
from .sceneguard import _features, _frozen_mask, _moving_end, _moving_start
from .storykeys import frame_change

VERSION = "2026.10.16.5"
_SUB = "troubleshooting"
_ENV_DONE = False


def _np(t):
    if t is None:
        return None
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[0] == 0:
        return None
    if a.shape[-1] > 3:
        a = a[..., :3]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    return np.nan_to_num(np.clip(a, 0, 1))


def _env():
    """The computer, once per ComfyUI session."""
    info = {"python": platform.python_version(), "system": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["vram_gb"] = round(total / 2 ** 30, 1)
            info["vram_free_gb"] = round(free / 2 ** 30, 1)
    except Exception as e:
        info["torch"] = "unknown (%s)" % e
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 2 ** 30, 1)
        info["ram_free_gb"] = round(vm.available / 2 ** 30, 1)
    except Exception:
        pass
    try:
        from . import VERSION as PKG
        info["nodes"] = PKG
    except Exception:
        info["nodes"] = VERSION
    return info


def _motion(arr, fps):
    """Per-frame change (mean, 32x32 grid), the still mask and where the clip stops moving for good."""
    small = np.concatenate([_features(arr[a:a + 32])[0] for a in range(0, arr.shape[0], 32)], 0)
    n = small.shape[0]
    d = np.zeros(n, dtype=np.float64)
    if n > 1:
        d[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2, 3))
        d[0] = d[1]
    still = _frozen_mask(small, fps)[0] if n >= 12 else np.zeros(n, dtype=bool)
    return d, still, (_moving_end(small, fps) if n >= 12 else n), (_moving_start(small, fps) if n >= 12 else 0)


def _runs(mask, fps, min_s=0.25):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) / fps >= min_s:
                out.append((round(i / fps, 2), round(j / fps, 2)))
            i = j
        else:
            i += 1
    return out


def _bins(v, fps, step=0.5):
    k = max(1, int(round(fps * step)))
    return [round(float(np.mean(v[a:a + k])), 4) for a in range(0, len(v), k)]


def _pick(arr, k):
    n = arr.shape[0]
    idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(k, n))))
    return idx


def _to_close(arr, pic, step):
    """How much of the picture differs from `pic`, sampled every `step` frames (0 = the same picture)."""
    idx = list(range(0, arr.shape[0], step))
    if idx[-1] != arr.shape[0] - 1:
        idx.append(arr.shape[0] - 1)
    return idx, [round(frame_change(arr[i], pic), 4) for i in idx]


def _audio_levels(audio, fps):
    try:
        wf = audio["waveform"]
        wf = wf.detach().cpu().numpy() if hasattr(wf, "detach") else np.asarray(wf)
        sr = int(audio["sample_rate"])
        x = np.asarray(wf, dtype=np.float64).reshape(-1, wf.shape[-1]).mean(axis=0)
        k = max(1, sr // 2)
        rms = [float(np.sqrt(np.mean(x[a:a + k] ** 2))) for a in range(0, len(x), k)]
        db = [round(20 * math.log10(max(r, 1e-6)), 1) for r in rms]
        return {"seconds": round(len(x) / sr, 2), "level_db_per_half_second": db,
                "silent_half_seconds": [round(i * 0.5, 1) for i, v in enumerate(db) if v < -55]}
    except Exception as e:
        return {"error": str(e)}


def _sheet(path, title, lines, raw, final, first_pic, last_pic, d_raw, still_raw, d_fin, fps, me):
    from PIL import Image, ImageDraw, ImageFont
    tw = 192
    font = ImageFont.load_default()

    def thumb(img):
        h, w = img.shape[:2]
        th = max(8, int(round(tw * h / float(w))))
        return Image.fromarray((np.clip(_resize(img, th, tw), 0, 1) * 255 + 0.5).astype(np.uint8)), th

    rows = []
    ref = [(p, t) for p, t in ((first_pic, "story first picture"), (last_pic, "story last picture")) if p is not None]
    if ref:
        rows.append(("Story pictures", [(thumb(p)[0], t) for p, t in ref]))
    for arr, name in ((raw, "Raw clip (as the video model made it)"), (final, "Finished clip (after Scene Guard and the locks)")):
        if arr is not None:
            rows.append((name, [(thumb(arr[i])[0], "%.2f s (frame %d)" % (i / fps, i)) for i in _pick(arr, 10)]))
    th = max([im.size[1] for _, r in rows for im, _ in r] + [108])
    W = 16 + 10 * (tw + 8)
    gh = 150
    H = 40 + len(rows) * (th + 44) + gh + 30 + 14 * (len(lines) + 1)
    sh = Image.new("RGB", (W, H), (24, 26, 30))
    dr = ImageDraw.Draw(sh)
    dr.text((10, 10), title, fill=(235, 235, 235), font=font)
    y = 34
    for name, r in rows:
        dr.text((10, y), name, fill=(170, 200, 255), font=font)
        y += 16
        for k, (im, cap) in enumerate(r):
            x = 10 + k * (tw + 8)
            sh.paste(im, (x, y))
            dr.text((x, y + im.size[1] + 2), cap, fill=(200, 200, 200), font=font)
        y += th + 28
    # movement graph: raw (orange), finished (green), still parts of the raw clip shaded, the cut point marked
    dr.text((10, y), "Movement per frame (orange: raw clip, green: finished clip; grey: raw clip standing still; "
                     "red line: where the raw clip stops moving for good)", fill=(170, 200, 255), font=font)
    y += 16
    gx0, gw = 10, W - 20
    dr.rectangle([gx0, y, gx0 + gw, y + gh], outline=(80, 80, 80))
    top = max([1e-3] + ([float(np.percentile(d_raw, 99))] if d_raw is not None and len(d_raw) else []) +
              ([float(np.percentile(d_fin, 99))] if d_fin is not None and len(d_fin) else []))

    def plot(dv, col, n_total):
        if dv is None or len(dv) < 2:
            return
        pts = [(gx0 + gw * i / float(max(1, n_total - 1)), y + gh - gh * min(1.0, dv[i] / top)) for i in range(len(dv))]
        dr.line(pts, fill=col, width=2)

    if d_raw is not None and len(d_raw):
        n = len(d_raw)
        for i in range(n):
            if still_raw[i]:
                x = gx0 + gw * i / float(max(1, n - 1))
                dr.line([(x, y + 1), (x, y + gh - 1)], fill=(60, 60, 64))
        plot(d_raw, (240, 150, 40), n)
        if me < n:
            x = gx0 + gw * me / float(max(1, n - 1))
            dr.line([(x, y), (x, y + gh)], fill=(230, 60, 60), width=2)
        if d_fin is not None and len(d_fin) > 1:
            # the finished clip is drawn over the part of the raw clip it was made from
            span = me if me < n else n
            pts = [(gx0 + gw * (i * (span - 1) / float(max(1, len(d_fin) - 1))) / float(max(1, n - 1)),
                    y + gh - gh * min(1.0, d_fin[i] / top)) for i in range(len(d_fin))]
            dr.line(pts, fill=(80, 210, 110), width=1)
    elif d_fin is not None:
        plot(d_fin, (80, 210, 110), len(d_fin))
    y += gh + 10
    for ln in lines:
        dr.text((10, y), ln[:230], fill=(225, 225, 225), font=font)
        y += 14
    sh.save(path)


class TroubleshootClip:
    """Writes a detailed log and a picture sheet for one clip (Troubleshooting logs in the generator)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "label": ("STRING", {"default": "film_clip_01"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "info": ("STRING", {"default": "", "multiline": True,
                                    "tooltip": "Settings and prompt of this clip, written by the generator."}),
                "picture_sheet": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "final": ("IMAGE", {"tooltip": "The finished clip (after Scene Guard and the quality locks)."}),
                "raw": ("IMAGE", {"tooltip": "The clip as the video model made it (first try)."}),
                "first_picture": ("IMAGE", {"tooltip": "The story first frame the clip should start on."}),
                "last_picture": ("IMAGE", {"tooltip": "The story last frame the clip should end on."}),
                "anchor": ("IMAGE", {"tooltip": "The film's gold frame: picture quality over time is measured against it."}),
                "audio": ("AUDIO", {"tooltip": "The clip's finished sound: level over time, silent gaps."}),
                "guard_report": ("STRING", {"forceInput": True}),
                "lock_report": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "video/ripcurlsurf"

    def run(self, label, fps, info, picture_sheet=True, final=None, raw=None, first_picture=None, last_picture=None,
            anchor=None, audio=None, guard_report=None, lock_report=None):
        global _ENV_DONE
        fps = float(fps) if fps else 24.0
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label or "clip")).strip("._") or "clip"
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "clip": safe, "fps": fps, "node_version": VERSION}
        txt = []
        try:
            d = os.path.join(_out_dir(), _SUB)
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            print("[Troubleshooting] cannot create the troubleshooting folder: %s" % e)
            return {"ui": {"text": [str(e)]}}
        if not _ENV_DONE:
            _ENV_DONE = True
            try:
                rec["computer"] = _env()
            except Exception as e:
                rec["computer"] = {"error": str(e)}
        try:
            rec["settings"] = {}
            for ln in str(info or "").splitlines():
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    rec["settings"][k.strip()] = v.strip()
            raw_a, fin_a = _np(raw), _np(final)
            fp, lp, an = _np(first_picture), _np(last_picture), _np(anchor)
            fp = fp[0] if fp is not None else None
            lp = lp[0] if lp is not None else None
            an = an[0] if an is not None else None
            d_raw = still_raw = d_fin = None
            me = 0
            if raw_a is not None:
                d_raw, still_raw, me, ms = _motion(raw_a, fps)
                n = raw_a.shape[0]
                rec["raw"] = {"frames": n, "seconds": round(n / fps, 2), "size": [int(raw_a.shape[2]), int(raw_a.shape[1])],
                              "moves_from_s": round(ms / fps, 2), "moves_until_s": round(me / fps, 2), "still_share": round(float(np.mean(still_raw)), 3),
                              "still_parts_s": _runs(still_raw, fps), "movement_per_half_second": _bins(d_raw, fps),
                              "brightness_per_half_second": _bins(raw_a.mean(axis=(1, 2, 3)), fps)}
                step = max(1, int(round(fps / 4)))
                if fp is not None:
                    rec["raw"]["start_differs_from_first_picture"] = round(frame_change(raw_a[0], fp), 4)
                if lp is not None:
                    idx, cl = _to_close(raw_a, lp, step)
                    rec["raw"]["differs_from_last_picture"] = {"every_s": round(step / fps, 3), "values": cl}
                    reach = next((i for i, v in zip(idx, cl) if v <= min(cl) + 0.01), None)
                    rec["raw"]["reaches_last_picture_at_s"] = None if reach is None else round(reach / fps, 2)
                    rec["raw"]["end_differs_from_last_picture"] = cl[-1]
            if fin_a is not None:
                d_fin, still_fin, _, _ = _motion(fin_a, fps)
                n = fin_a.shape[0]
                rec["final"] = {"frames": n, "seconds": round(n / fps, 2), "size": [int(fin_a.shape[2]), int(fin_a.shape[1])],
                                "still_share": round(float(np.mean(still_fin)), 3), "still_parts_s": _runs(still_fin, fps),
                                "movement_per_half_second": _bins(d_fin, fps),
                                "brightness_per_half_second": _bins(fin_a.mean(axis=(1, 2, 3)), fps)}
                if fp is not None:
                    rec["final"]["start_differs_from_first_picture"] = round(frame_change(fin_a[0], fp), 4)
                if lp is not None:
                    rec["final"]["end_differs_from_last_picture"] = round(frame_change(fin_a[-1], lp), 4)
                if an is not None:
                    pa = profile(np.clip(_resize(an, fin_a.shape[1], fin_a.shape[2]), 0, 1))
                    qs = []
                    for i in range(0, n, max(1, int(round(fps / 2)))):
                        try:
                            qs.append(round(quality_score(relative(profile(fin_a[i]), pa)), 1))
                        except Exception:
                            qs.append(None)
                    rec["final"]["quality_per_half_second"] = qs
            if audio is not None:
                rec["audio"] = _audio_levels(audio, fps)
            if guard_report:
                rec["scene_guard"] = str(guard_report)
            if lock_report:
                rec["quality_lock"] = str(lock_report)
            # readable summary
            txt.append("=" * 100)
            txt.append("%s | %s | troubleshooting node %s" % (rec["time"], safe, VERSION))
            if "computer" in rec:
                txt.append("computer: " + ", ".join("%s %s" % kv for kv in rec["computer"].items()))
            for k, v in rec["settings"].items():
                txt.append("  %s: %s" % (k, v))
            for key, name in (("raw", "raw clip"), ("final", "finished clip")):
                r = rec.get(key)
                if not r:
                    continue
                txt.append("%s: %d frames (%.2f s), %dx%d" % (name, r["frames"], r["seconds"], r["size"][0], r["size"][1]))
                if key == "raw":
                    txt.append("  moves from %.2f s until %.2f s; standing still %d%% of the time; still parts (s): %s" % (
                        r["moves_from_s"], r["moves_until_s"], round(100 * r["still_share"]), r["still_parts_s"] or "none"))
                    if "start_differs_from_first_picture" in r:
                        txt.append("  first frame vs the story first picture: %.1f%% of the picture differs" % (
                            100 * r["start_differs_from_first_picture"]))
                    if "reaches_last_picture_at_s" in r:
                        txt.append("  reaches the story last picture at %s s; the last frame differs %.1f%% from it" % (
                            r["reaches_last_picture_at_s"], 100 * r["end_differs_from_last_picture"]))
                        dl = r["differs_from_last_picture"]
                        txt.append("  difference from the story last picture every %.2f s: %s" % (dl["every_s"], dl["values"]))
                else:
                    txt.append("  standing still %d%% of the time; still parts (s): %s" % (
                        round(100 * r["still_share"]), r["still_parts_s"] or "none"))
                    for k2, nm in (("start_differs_from_first_picture", "first frame vs the story first picture"),
                                   ("end_differs_from_last_picture", "last frame vs the story last picture")):
                        if k2 in r:
                            txt.append("  %s: %.1f%% of the picture differs" % (nm, 100 * r[k2]))
                    if "quality_per_half_second" in r:
                        txt.append("  quality vs the gold frame every 0.5 s: %s" % r["quality_per_half_second"])
                txt.append("  movement every 0.5 s: %s" % r["movement_per_half_second"])
                txt.append("  brightness every 0.5 s: %s" % r["brightness_per_half_second"])
            if "audio" in rec:
                a = rec["audio"]
                txt.append("sound: %s" % (("%.2f s, silent at (s): %s; level dB every 0.5 s: %s" % (
                    a["seconds"], a["silent_half_seconds"] or "none", a["level_db_per_half_second"])) if "seconds" in a
                    else a.get("error")))
            if guard_report:
                txt.append("scene guard: " + str(guard_report))
            if lock_report:
                txt.append("quality lock: " + str(lock_report))
            if picture_sheet and (raw_a is not None or fin_a is not None):
                sp = os.path.join(d, safe + ".png")
                # measurements first, then the settings; long lines are wrapped
                meas = [ln for ln in txt[2:] if not ln.startswith("  movement") and not ln.startswith("  brightness")
                        and not ln.startswith("  difference from the story last picture every")
                        and not (ln.startswith("  ") and ln[2:].split(":", 1)[0] in rec["settings"])]
                sets = ["settings: " + "; ".join("%s %s" % kv for kv in rec["settings"].items()
                                                 if not kv[0].startswith("prompt"))]
                prm = ["prompt: " + v for k, v in rec["settings"].items() if k.startswith("prompt")]
                lines = []
                for ln in meas + sets + prm:
                    while len(ln) > 225:
                        cut = ln.rfind(" ", 0, 225)
                        cut = cut if cut > 100 else 225
                        lines.append(ln[:cut])
                        ln = "    " + ln[cut:].lstrip()
                    lines.append(ln)
                _sheet(sp, "%s   (%s)" % (safe, rec["time"]), lines[:46], raw_a, fin_a, fp, lp, d_raw,
                       still_raw if still_raw is not None else None, d_fin, fps, me if raw_a is not None else 0)
                rec["picture_sheet"] = "output/%s/%s.png" % (_SUB, safe)
                txt.append("picture sheet: " + rec["picture_sheet"])
        except Exception as e:
            import traceback
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            txt.append("ERROR while measuring this clip (the video itself is not affected): %s" % rec["error"])
            txt.append(traceback.format_exc())
        # one entry in output/movie_log.txt: the first line names the clip, the report follows indented, and the
        # last line has every number for checking by a program
        body = [ln for ln in (txt[2:] if txt[:1] == ["=" * 100] else txt) if not ln.startswith("picture sheet: ")]
        try:
            nums = json.dumps(rec)
        except Exception as e:
            nums = json.dumps({"clip": safe, "error": "numbers not serialisable: %s" % e})
        head = "troubleshooting node %s" % VERSION
        if "picture_sheet" in rec:
            head += " | picture sheet: " + rec["picture_sheet"]
        if "error" in rec:
            head += " | " + rec["error"]
        if movie_log("TroubleshootClip", safe, "ERROR" if "error" in rec else "LOGGED", head,
                     more=body + ["all numbers (for a program): " + nums]) is None:
            print("[Troubleshooting] could not write output/movie_log.txt")
        short = "%s: logged%s" % (safe, (" (" + rec["error"] + ")") if "error" in rec else "")
        print("[Troubleshooting] " + short)
        return {"ui": {"text": [short]}}


NODE_CLASS_MAPPINGS = {"TroubleshootClip": TroubleshootClip}
NODE_DISPLAY_NAME_MAPPINGS = {"TroubleshootClip": "Troubleshooting Log (one clip, for finding problems)"}
