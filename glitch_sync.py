import cv2
import numpy as np
import librosa
import os
import random
import subprocess
import argparse
import sys
import threading
import shutil
import tempfile
import zipfile
import json
import pickle
import hashlib
import bisect
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
import requests
from requests import Response
import psutil

EXPORT_FINAL_VIDEO = "final_video"
EXPORT_CUT_AWARE_MLT = "cut_aware_mlt"
EXPORT_CLIP_MLT = "clip_mlt"

EXPORT_MODE_LABELS = {
    "Final video (MP4)": EXPORT_FINAL_VIDEO,
    "Shotcut: Full Timeline (ZIP)": EXPORT_CUT_AWARE_MLT,
    "Shotcut: Source Clips (ZIP)": EXPORT_CLIP_MLT,
}

OUTPUT_RESOLUTION_LABELS = {
    "Auto (first input)": None,
    "480p (854x480)": (854, 480),
    "720p HD (1280x720)": (1280, 720),
    "1080p Full HD (1920x1080)": (1920, 1080),
    "4K UHD (3840x2160)": (3840, 2160),
    "8K UHD (7680x4320)": (7680, 4320),
}

EXPORT_QUALITY_LABELS = {
    "Master quality (largest)": ("ultrafast", "0"),
    "High quality (slower)": ("medium", "21"),
    "Balanced": ("fast", "22"),
    "Fast preview": ("veryfast", "24"),
}

EFFECT_NAMES = (
    "pixelate", "flash", "rewind", "rgb_shift", "shake", "ghosting",
    "monochrome", "hue_shift", "vignette", "static_pan_zoom",
)
DEFAULT_EFFECT_AMOUNTS = {
    "pixelate": 0.7,
    "flash": 0.6,
    "rewind": 0.25,
    "rgb_shift": 0.6,
    "shake": 0.5,
    "ghosting": 0.55,
    "monochrome": 0.35,
    "hue_shift": 0.35,
    "vignette": 0.45,
    "static_pan_zoom": 0.4,
}
EFFECT_TIMING_LABELS = ("Frame", "Clip", "Random")
DEFAULT_EFFECT_TIMING = {
    "pixelate": "Random",
    "flash": "Frame",
    "rewind": "Clip",
    "rgb_shift": "Random",
    "shake": "Frame",
    "ghosting": "Random",
    "monochrome": "Clip",
    "hue_shift": "Random",
    "vignette": "Clip",
    "static_pan_zoom": "Clip",
}
DEFAULT_STYLE_NAME = "Default"
SCENE_TRANSITION_LABELS = ("Auto", "Hard cut", "Fast fade", "Slow fade")
SCENE_TRANSITION_AUTO_HARD_ACTIVITY = 0.68
SCENE_TRANSITION_AUTO_FAST_ACTIVITY = 0.40
SCENE_TRANSITION_AUTO_HARD_DELTA = 0.30
SCENE_TRANSITION_AUTO_FAST_DELTA = 0.14
SCENE_TRANSITION_FAST_FADE_SECONDS = 0.12
SCENE_TRANSITION_SLOW_FADE_SECONDS = 0.28
STYLE_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".glitchsync_styles.json")
RECENT_PROJECTS_PATH = os.path.join(os.path.expanduser("~"), ".glitchsync_recent_projects.json")
RECENT_PROJECTS_LIMIT = 8
ANALYSIS_CACHE_VERSION = "7"
ANALYSIS_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "glitchsync", "analysis")
STATIC_MOTION_THRESHOLD = 0.018
LAB_EPSILON = 1e-6
SOURCE_LOCK_TRIGGER_STREAK = 4
SOURCE_LOCK_PENALTY_START = 4.0
SOURCE_LOCK_PENALTY_DECAY = 0.82
SOURCE_LOCK_PENALTY_MAX = 10.0
AI_DEFAULT_BACKEND_URL = "http://127.0.0.1:9000"
AI_DEFAULT_PROMPT = "dreamlike transformation of the input image, preserve the original subject and composition, and reimagine it as a surreal cinematic dream scene with soft painterly detail"
AI_DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, watermark, text"
AI_DEFAULT_EVERY_N_FRAMES = 12
AI_DEFAULT_DENOISE = 0.32
AI_DEFAULT_CFG_SCALE = 6.5
AI_DEFAULT_STEPS = 10
AI_DEFAULT_MAX_DIM = 512
AI_DEFAULT_BLEND = 0.35
AI_DEFAULT_SESSION = "glitchsync"
AI_DEFAULT_MODEL_FALLBACK = "sd-v1-4"
AI_OBSOLETE_NEGATIVE_PROMPTS = {
    "blurry, low quality, watermark, text, deformed, extra fingers",
    "blurry, low quality, watermark, text, deformed",
}
AI_OBSOLETE_PROMPTS = {
    "hand drawn illustration, expressive linework",
}
AI_DREAM_TIMING_LABELS = ("Clip", "Random")


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def _schedule_show(self, _event=None):
        if self._after_id is None:
            self._after_id = self.widget.after(450, self.show)

    def show(self, _event=None):
        self.hide()
        if not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            relief=tk.SOLID,
            borderwidth=1,
            padding=(8, 5),
            wraplength=320,
        )
        label.pack()

    def hide(self, _event=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


def file_cache_key(path, kind):
    stat = os.stat(path)
    identity = {
        "version": ANALYSIS_CACHE_VERSION,
        "kind": kind,
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    raw = json.dumps(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def analysis_cache_path(path, kind, extension):
    return os.path.join(ANALYSIS_CACHE_DIR, kind, f"{file_cache_key(path, kind)}.{extension}")


def load_cube_lut(path):
    size = None
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE") or upper.startswith("DOMAIN_"):
                continue
            if upper.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) >= 2:
                    size = int(parts[1])
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    values.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if not size or len(values) < size ** 3:
        raise ValueError("Unsupported or incomplete .cube LUT")
    return np.array(values[:size ** 3], dtype=np.float32).reshape((size, size, size, 3))


def apply_cube_lut(frame, lut):
    if lut is None:
        return frame
    size = lut.shape[0]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    idx = np.clip(np.rint(rgb * (size - 1)).astype(np.int32), 0, size - 1)
    graded = lut[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    graded = np.clip(graded * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(graded, cv2.COLOR_RGB2BGR)


def frame_to_base64_png(frame):
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Could not encode frame for AI stylization")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def base64_png_to_frame(image_b64, target_size):
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    target_width, target_height = target_size
    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return frame


def fit_frame_to_output(frame, width, height):
    frame = crop_dark_edges(frame)
    src_h, src_w = frame.shape[:2]
    if src_w == width and src_h == height:
        return frame
    cover_scale = max(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * cover_scale)))
    new_h = max(1, int(round(src_h * cover_scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA if cover_scale < 1 else cv2.INTER_LINEAR)
    x = max(0, (new_w - width) // 2)
    y = max(0, (new_h - height) // 2)
    fitted = resized[y:y + height, x:x + width]
    if fitted.shape[:2] != (height, width):
        fitted = cv2.resize(fitted, (width, height), interpolation=cv2.INTER_LINEAR)
    return fitted


def crop_dark_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = max(8, int(np.percentile(gray, 15) * 0.6))
    mask = gray > threshold
    rows = np.where(np.mean(mask, axis=1) > 0.12)[0]
    cols = np.where(np.mean(mask, axis=0) > 0.12)[0]
    if rows.size < 2 or cols.size < 2:
        return frame
    y1, y2 = rows[0], rows[-1] + 1
    x1, x2 = cols[0], cols[-1] + 1
    h, w = frame.shape[:2]
    if (y2 - y1) < h * 0.55 or (x2 - x1) < w * 0.55:
        return frame
    if y1 == 0 and y2 == h and x1 == 0 and x2 == w:
        return frame
    return frame[y1:y2, x1:x2]


def apply_center_zoom(frame, zoom):
    if zoom <= 1.001:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    return cv2.resize(frame[y:y + crop_h, x:x + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)


def match_lab_color(frame, source_stats, reference_stats, strength):
    if not source_stats or not reference_stats or strength <= 0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_mean = np.array(source_stats["lab_mean"], dtype=np.float32)
    src_std = np.array(source_stats["lab_std"], dtype=np.float32)
    ref_mean = np.array(reference_stats["lab_mean"], dtype=np.float32)
    ref_std = np.array(reference_stats["lab_std"], dtype=np.float32)
    matched = ((lab - src_mean) * (ref_std / np.maximum(src_std, LAB_EPSILON))) + ref_mean
    blended = lab + ((matched - lab) * np.clip(strength, 0, 1))
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def force_black_and_white(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_style(duration=0.1, fps=30, coherence=0.2, sensitivity=1.0, beat_sync=True,
               beat_step=4, beat_variation=0.0, effects_enabled=None,
               effect_amounts=None, effect_timing=None,
               primary_enabled=False, primary_focus=0.75,
               export_mode_label="Final video (MP4)", source_variety=0.0, music_match=0.35,
               color_match_enabled=False, color_match_strength=0.5,
               output_resolution_label="Auto (first input)",
               export_quality_label="High quality (slower)", render_mode="Full",
               snippet_duration=30.0, lut_path="", ai_dream_chance=0.25, ai_dream_timing="Clip",
               scene_transition_mode="Auto", force_bw=False):
    return {
        "export_mode_label": export_mode_label,
        "output_resolution_label": output_resolution_label,
        "export_quality_label": export_quality_label,
        "duration": duration,
        "fps": fps,
        "render_mode": render_mode,
        "snippet_duration": snippet_duration,
        "coherence": coherence,
        "sensitivity": sensitivity,
        "source_variety": source_variety,
        "music_match": music_match,
        "color_match_enabled": color_match_enabled,
        "color_match_strength": color_match_strength,
        "force_bw": force_bw,
        "lut_path": lut_path,
        "beat_sync": beat_sync,
        "beat_step": beat_step,
        "beat_variation": beat_variation,
        "effects_enabled": {
            "pixelate": True,
            "flash": True,
            "rewind": True,
            "rgb_shift": True,
            "shake": True,
            "ghosting": True,
            "monochrome": False,
            "hue_shift": False,
            "vignette": False,
            "static_pan_zoom": False,
            **(effects_enabled or {}),
        },
        "effect_amounts": {
            **DEFAULT_EFFECT_AMOUNTS,
            **(effect_amounts or {}),
        },
        "effect_timing": {
            **DEFAULT_EFFECT_TIMING,
            **(effect_timing or {}),
        },
        "primary_enabled": primary_enabled,
        "primary_focus": primary_focus,
        "ai_dream_chance": ai_dream_chance,
        "ai_dream_timing": ai_dream_timing,
        "scene_transition_mode": scene_transition_mode if scene_transition_mode in SCENE_TRANSITION_LABELS else "Auto",
    }


def resolve_scene_transition_mode(mode, current_activity, next_activity):
    if mode != "Auto":
        return mode
    current_activity = float(np.clip(current_activity, 0, 1))
    next_activity = float(np.clip(next_activity, 0, 1))
    drive = max(current_activity, next_activity)
    delta = abs(next_activity - current_activity)
    if drive >= SCENE_TRANSITION_AUTO_HARD_ACTIVITY or delta >= SCENE_TRANSITION_AUTO_HARD_DELTA:
        return "Hard cut"
    if drive <= SCENE_TRANSITION_AUTO_FAST_ACTIVITY and delta <= SCENE_TRANSITION_AUTO_FAST_DELTA:
        return "Slow fade"
    return "Fast fade"


def scene_transition_fade_frames(mode, fps, current_frames, next_frames=None):
    if mode == "Hard cut":
        return 0
    fade_seconds = SCENE_TRANSITION_FAST_FADE_SECONDS if mode == "Fast fade" else SCENE_TRANSITION_SLOW_FADE_SECONDS
    max_frames = current_frames if next_frames is None else min(current_frames, next_frames)
    return max(1, min(max_frames, int(round(max(1, fps) * fade_seconds))))


def scene_transition_ffmpeg_name(mode):
    if mode == "Fast fade":
        return "fadeblack"
    if mode == "Slow fade":
        return "fade"
    return None


def ffmpeg_concat_list_line(path):
    abs_path = os.path.abspath(path)
    return "file '" + abs_path.replace("'", "'\\''") + "'"


def preferred_temp_root():
    for path in ("/dev/shm", "/run/shm"):
        if os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK):
            return path
    return None


def human_bytes(num):
    num = float(num or 0)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if num < 1024.0 or unit == "PB":
            return f"{num:.1f} {unit}"
        num /= 1024.0


class LosslessVideoWriter:
    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        self.proc = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(float(fps)),
            '-i', 'pipe:0',
            '-an',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '0',
            '-pix_fmt', 'yuv420p',
            output_path,
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.closed = False

    def write(self, frame):
        if self.closed or self.proc.stdin is None:
            raise RuntimeError("Lossless video writer is already closed")
        try:
            self.proc.stdin.write(frame.astype(np.uint8, copy=False).tobytes())
        except BrokenPipeError:
            stderr = ""
            try:
                if self.proc.stderr is not None:
                    stderr = self.proc.stderr.read().decode(errors="ignore").strip()
            except Exception:
                pass
            raise RuntimeError(stderr or f"ffmpeg stopped while writing {self.output_path}")

    def release(self):
        if self.closed:
            return
        self.closed = True
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        stderr = ""
        if self.proc.stderr is not None:
            try:
                stderr = self.proc.stderr.read().decode(errors="ignore").strip()
            except Exception:
                stderr = ""
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(stderr or f"ffmpeg failed while writing {self.output_path}")


BUILTIN_STYLES = {
    DEFAULT_STYLE_NAME: make_style(ai_dream_chance=0.25, ai_dream_timing="Clip", scene_transition_mode="Auto"),
    "Mellow Story": make_style(
        duration=0.8,
        coherence=0.85,
        sensitivity=0.55,
        source_variety=0.35,
        music_match=0.25,
        color_match_strength=0.55,
        beat_step=8,
        beat_variation=0.12,
        effects_enabled={"monochrome": True, "vignette": True, "static_pan_zoom": True},
        effect_amounts={"pixelate": 0.25, "flash": 0.18, "rewind": 0.1, "rgb_shift": 0.2, "shake": 0.15, "ghosting": 0.7, "monochrome": 0.28, "hue_shift": 0.0, "vignette": 0.35, "static_pan_zoom": 0.45},
        effect_timing={"pixelate": "Clip", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Clip", "shake": "Clip", "ghosting": "Clip", "monochrome": "Clip", "hue_shift": "Clip", "vignette": "Clip", "static_pan_zoom": "Clip"},
        primary_focus=0.9,
        ai_dream_chance=0.20,
        ai_dream_timing="Clip",
        scene_transition_mode="Auto",
    ),
    "Pop Performance": make_style(
        coherence=0.65,
        sensitivity=0.9,
        source_variety=0.55,
        music_match=0.45,
        color_match_strength=0.5,
        beat_step=4,
        beat_variation=0.25,
        effects_enabled={"hue_shift": True, "vignette": True},
        effect_amounts={"pixelate": 0.5, "flash": 0.55, "rewind": 0.2, "rgb_shift": 0.65, "shake": 0.45, "ghosting": 0.5, "monochrome": 0.0, "hue_shift": 0.22, "vignette": 0.25, "static_pan_zoom": 0.35},
        effect_timing={"pixelate": "Random", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Random", "shake": "Frame", "ghosting": "Random", "monochrome": "Clip", "hue_shift": "Random", "vignette": "Random", "static_pan_zoom": "Clip"},
        ai_dream_chance=0.22,
        ai_dream_timing="Clip",
        scene_transition_mode="Fast fade",
    ),
    "EDM Pulse": make_style(
        coherence=0.35,
        sensitivity=1.35,
        source_variety=0.7,
        music_match=0.7,
        color_match_strength=0.45,
        beat_step=2,
        beat_variation=0.45,
        effects_enabled={"hue_shift": True, "vignette": True},
        effect_amounts={"pixelate": 1.2, "flash": 1.25, "rewind": 0.45, "rgb_shift": 1.25, "shake": 1.1, "ghosting": 0.65, "monochrome": 0.0, "hue_shift": 0.75, "vignette": 0.45, "static_pan_zoom": 0.25},
        effect_timing={"pixelate": "Random", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Frame", "shake": "Frame", "ghosting": "Random", "monochrome": "Clip", "hue_shift": "Frame", "vignette": "Random", "static_pan_zoom": "Clip"},
        ai_dream_chance=0.30,
        ai_dream_timing="Clip",
        scene_transition_mode="Hard cut",
    ),
    "Rock Punch": make_style(
        coherence=0.5,
        sensitivity=1.15,
        source_variety=0.6,
        music_match=0.65,
        color_match_strength=0.45,
        beat_step=2,
        beat_variation=0.25,
        effects_enabled={"monochrome": True, "vignette": True},
        effect_amounts={"pixelate": 0.65, "flash": 0.6, "rewind": 0.25, "rgb_shift": 0.65, "shake": 1.25, "ghosting": 0.35, "monochrome": 0.25, "hue_shift": 0.0, "vignette": 0.55, "static_pan_zoom": 0.25},
        effect_timing={"pixelate": "Random", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Random", "shake": "Frame", "ghosting": "Frame", "monochrome": "Random", "hue_shift": "Clip", "vignette": "Clip", "static_pan_zoom": "Clip"},
        ai_dream_chance=0.18,
        ai_dream_timing="Clip",
        scene_transition_mode="Fast fade",
    ),
    "Ambient Drift": make_style(
        duration=1.0,
        coherence=0.95,
        sensitivity=0.4,
        source_variety=0.25,
        music_match=0.2,
        color_match_strength=0.6,
        beat_step=16,
        beat_variation=0.05,
        effects_enabled={"pixelate": False, "rewind": False, "shake": False, "monochrome": True, "vignette": True, "static_pan_zoom": True},
        effect_amounts={"pixelate": 0.0, "flash": 0.1, "rewind": 0.0, "rgb_shift": 0.12, "shake": 0.0, "ghosting": 1.2, "monochrome": 0.35, "hue_shift": 0.0, "vignette": 0.3, "static_pan_zoom": 0.75},
        effect_timing={"pixelate": "Clip", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Clip", "shake": "Clip", "ghosting": "Clip", "monochrome": "Clip", "hue_shift": "Clip", "vignette": "Clip", "static_pan_zoom": "Clip"},
        primary_focus=0.95,
        ai_dream_chance=0.65,
        ai_dream_timing="Clip",
        scene_transition_mode="Slow fade",
    ),
    "Glitch Heavy": make_style(
        coherence=0.15,
        sensitivity=1.75,
        source_variety=0.8,
        music_match=0.75,
        color_match_strength=0.35,
        beat_step=1,
        beat_variation=0.0,
        effects_enabled={"hue_shift": True, "vignette": True},
        effect_amounts={"pixelate": 1.7, "flash": 1.15, "rewind": 0.8, "rgb_shift": 1.8, "shake": 1.4, "ghosting": 1.0, "monochrome": 0.0, "hue_shift": 0.9, "vignette": 0.65, "static_pan_zoom": 0.2},
        effect_timing={"pixelate": "Random", "flash": "Frame", "rewind": "Clip", "rgb_shift": "Random", "shake": "Frame", "ghosting": "Random", "monochrome": "Clip", "hue_shift": "Random", "vignette": "Random", "static_pan_zoom": "Clip"},
        ai_dream_chance=0.24,
        ai_dream_timing="Clip",
        scene_transition_mode="Hard cut",
    ),
}


AUTO_STYLE_PROFILES = {
    "Ambient Drift": {"tempo": 75, "energy": 0.30, "bass": 0.30, "highs": 0.18, "dynamic": 0.35},
    "Mellow Story": {"tempo": 90, "energy": 0.36, "bass": 0.35, "highs": 0.24, "dynamic": 0.42},
    "Pop Performance": {"tempo": 112, "energy": 0.48, "bass": 0.42, "highs": 0.34, "dynamic": 0.50},
    "Rock Punch": {"tempo": 126, "energy": 0.58, "bass": 0.52, "highs": 0.32, "dynamic": 0.58},
    "EDM Pulse": {"tempo": 132, "energy": 0.66, "bass": 0.56, "highs": 0.48, "dynamic": 0.62},
    "Glitch Heavy": {"tempo": 145, "energy": 0.76, "bass": 0.50, "highs": 0.58, "dynamic": 0.72},
}


def frame_count_to_out(frame_count):
    return max(0, frame_count - 1)


def frames_to_timecode(frame, fps):
    frame = max(0, int(frame))
    fps = max(1, int(fps))
    total_seconds, frames = divmod(frame, fps)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = round((frames / fps) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def output_path_for_mode(output, export_mode):
    root, ext = os.path.splitext(output)
    if export_mode == EXPORT_FINAL_VIDEO:
        return output
    if ext.lower() == ".zip":
        return output
    return f"{root or output}_shotcut.zip"


def increment_path_if_exists(path):
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{root}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate
        index += 1


def prettify_xml(element):
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def add_property(parent, name, value):
    prop = ET.SubElement(parent, "property", {"name": name})
    prop.text = str(value)
    return prop


def write_shotcut_mlt(project_path, video_resources, audio_resource, segments, fps, width, height, rendered_segments=False):
    if rendered_segments:
        total_frames = 0
        for index, segment in enumerate(segments):
            segment["shotcut_in"] = total_frames
            segment["shotcut_out"] = total_frames + segment["frame_count"] - 1
            transition = segment.get("transition_out") or {}
            transition_frames = 0 if transition.get("mode") == "Hard cut" else int(transition.get("frames", 0) or 0)
            total_frames += segment["frame_count"] - transition_frames
    else:
        total_frames = sum(segment["frame_count"] for segment in segments)
    out_frame = frame_count_to_out(total_frames)
    out_time = frames_to_timecode(out_frame, fps)
    mlt = ET.Element("mlt", {
        "LC_NUMERIC": "C",
        "version": "7.0.0",
        "title": "GlitchSync Shotcut export",
        "producer": "main_bin",
        "parent": "tractor0",
        "in": "00:00:00.000",
        "out": out_time,
    })

    profile = ET.SubElement(mlt, "profile", {
        "description": "GlitchSync export",
        "width": str(width),
        "height": str(height),
        "progressive": "1",
        "sample_aspect_num": "1",
        "sample_aspect_den": "1",
        "display_aspect_num": str(width),
        "display_aspect_den": str(height),
        "frame_rate_num": str(fps),
        "frame_rate_den": "1",
        "colorspace": "709",
    })
    profile.tail = "\n"

    ET.SubElement(mlt, "producer", {"id": "black", "in": "0", "out": str(out_frame)})
    add_property(mlt[-1], "mlt_service", "color")
    add_property(mlt[-1], "resource", "black")
    add_property(mlt[-1], "aspect_ratio", "1")

    producer_outs = {producer_id: 0 for producer_id in video_resources}
    for segment in segments:
        if rendered_segments:
            producer_outs[segment["producer"]] = max(producer_outs[segment["producer"]], segment["frame_count"] - 1)
        else:
            producer_outs[segment["producer"]] = max(producer_outs[segment["producer"]], segment["out"])

    for producer_id, resource in video_resources.items():
        producer_out = producer_outs[producer_id]
        producer = ET.SubElement(mlt, "producer", {"id": producer_id, "in": "0", "out": str(producer_out)})
        add_property(producer, "length", producer_out + 1)
        add_property(producer, "eof", "pause")
        add_property(producer, "mlt_service", "avformat")
        add_property(producer, "resource", resource)
        add_property(producer, "audio_index", "-1")
        add_property(producer, "video_index", "0")
        add_property(producer, "seekable", "1")
        add_property(producer, "shotcut:caption", os.path.basename(resource))
        add_property(producer, "shotcut:resource", resource)
        add_property(producer, "shotcut:skipConvert", "1")
        if rendered_segments:
            for segment_num, segment in enumerate(segments):
                if segment["producer"] != producer_id:
                    continue
                comment = f"Segment {segment_num + 1}"
                source_name = os.path.basename(segment.get("source_path", ""))
                if source_name:
                    comment += f" from {source_name}"
                transition = segment.get("transition_out") or {}
                transition_mode = transition.get("mode")
                transition_frames = int(transition.get("frames", 0) or 0)
                if transition_mode and transition_frames > 0:
                    comment += f" -> {transition_mode.lower()} {transition_frames}f"
                add_property(producer, "shotcut:comment", comment)
                break

    audio = ET.SubElement(mlt, "producer", {"id": "audio0", "in": "0", "out": str(out_frame)})
    add_property(audio, "length", total_frames)
    add_property(audio, "eof", "pause")
    add_property(audio, "mlt_service", "avformat")
    add_property(audio, "resource", audio_resource)
    add_property(audio, "audio_index", "0")
    add_property(audio, "video_index", "-1")
    add_property(audio, "seekable", "1")
    add_property(audio, "shotcut:caption", os.path.basename(audio_resource))
    add_property(audio, "shotcut:resource", audio_resource)
    add_property(audio, "shotcut:skipConvert", "1")

    main_bin = ET.SubElement(mlt, "playlist", {"id": "main_bin"})
    add_property(main_bin, "xml_retain", "1")
    for producer_id in video_resources:
        ET.SubElement(main_bin, "entry", {
            "producer": producer_id,
            "in": "0",
            "out": str(producer_outs[producer_id]),
        })
    ET.SubElement(main_bin, "entry", {"producer": "audio0", "in": "0", "out": str(out_frame)})

    background = ET.SubElement(mlt, "playlist", {"id": "background"})
    ET.SubElement(background, "entry", {"producer": "black", "in": "0", "out": str(out_frame)})

    if rendered_segments:
        video_track_a = ET.SubElement(mlt, "playlist", {"id": "video_track_a"})
        add_property(video_track_a, "shotcut:name", "V1")
        add_property(video_track_a, "shotcut:video", "1")
        video_track_b = ET.SubElement(mlt, "playlist", {"id": "video_track_b"})
        add_property(video_track_b, "shotcut:name", "V2")
        add_property(video_track_b, "shotcut:video", "1")
        track_offsets = [0, 0]
        for index, segment in enumerate(segments):
            track_index = index % 2
            track = video_track_a if track_index == 0 else video_track_b
            start = segment["shotcut_in"]
            if start > track_offsets[track_index]:
                ET.SubElement(track, "blank", {"length": str(start - track_offsets[track_index])})
            ET.SubElement(track, "entry", {
                "producer": segment["producer"],
                "in": "0",
                "out": str(segment["frame_count"] - 1),
            })
            track_offsets[track_index] = start + segment["frame_count"]
    else:
        video_playlist = ET.SubElement(mlt, "playlist", {"id": "video_track"})
        add_property(video_playlist, "shotcut:name", "V1")
        add_property(video_playlist, "shotcut:video", "1")
        for segment in segments:
            ET.SubElement(video_playlist, "entry", {
                "producer": segment["producer"],
                "in": str(segment["in"]),
                "out": str(segment["out"]),
            })

    audio_playlist = ET.SubElement(mlt, "playlist", {"id": "audio_track"})
    add_property(audio_playlist, "shotcut:name", "A1")
    add_property(audio_playlist, "shotcut:audio", "1")
    ET.SubElement(audio_playlist, "entry", {"producer": "audio0", "in": "0", "out": str(out_frame)})

    tractor = ET.SubElement(mlt, "tractor", {
        "id": "tractor0",
        "title": "GlitchSync Shotcut export",
        "global_feed": "1",
        "in": "0",
        "out": str(out_frame),
    })
    add_property(tractor, "shotcut", "1")
    add_property(tractor, "shotcut:projectFolder", "1")
    add_property(tractor, "shotcut:projectAudioChannels", "2")
    add_property(tractor, "shotcut:scaleFactor", "0")
    multitrack = ET.SubElement(tractor, "multitrack")
    ET.SubElement(multitrack, "track", {"producer": "background"})
    if rendered_segments:
        ET.SubElement(multitrack, "track", {"producer": "video_track_a"})
        ET.SubElement(multitrack, "track", {"producer": "video_track_b"})
        ET.SubElement(multitrack, "track", {"producer": "audio_track"})
        for index in range(max(0, len(segments) - 1)):
            transition = segments[index].get("transition_out") or {}
            transition_frames = 0 if transition.get("mode") == "Hard cut" else int(transition.get("frames", 0) or 0)
            if transition_frames <= 0:
                continue
            transition_start = segments[index + 1]["shotcut_in"]
            transition_end = transition_start + transition_frames - 1
            transition_element = ET.SubElement(tractor, "transition", {
                "id": f"transition{index}",
                "in": str(transition_start),
                "out": str(transition_end),
            })
            add_property(transition_element, "mlt_service", "luma")
            add_property(transition_element, "a_track", "1")
            add_property(transition_element, "b_track", "2")
            add_property(transition_element, "always_active", "1")
    else:
        ET.SubElement(multitrack, "track", {"producer": "video_track"})
        ET.SubElement(multitrack, "track", {"producer": "audio_track"})
        transition_element = ET.SubElement(tractor, "transition", {"id": "transition0", "in": "0", "out": str(out_frame)})
        add_property(transition_element, "mlt_service", "mix")
        add_property(transition_element, "a_track", "0")
        add_property(transition_element, "b_track", "1")
        add_property(transition_element, "always_active", "1")

    with open(project_path, "wb") as f:
        f.write(prettify_xml(mlt))


def zip_directory(source_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                path = os.path.join(root, filename)
                z.write(path, os.path.relpath(path, source_dir))

# --- Effects Functions ---

def apply_pixelate(frame, intensity, high_intensity, sensitivity=1.0, probability_scale=1.0):
    """
    Improved: Only triggers on high-frequency transients.
    Designed to be an occasional accent rather than a theme.
    """
    # Only trigger if there is a significant high-frequency spike AND a random roll
    # This makes it feel much more like an intentional 'glitch'
    trigger_probability = np.clip(0.25 * probability_scale, 0, 1)
    if high_intensity < 0.45 or random.random() > trigger_probability:
        return frame 
    
    h, w = frame.shape[:2]
    # Aggressive scaling for a 'punchy' look
    boosted = ((high_intensity - 0.45) / 0.55) * sensitivity
    block_size = 4 + int(boosted * 80) # Minimum 4px blocks for visibility
    
    small_w, small_h = max(1, w // block_size), max(1, h // block_size)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_flash_boost(frame, intensity, sensitivity=1.0):
    if intensity <= 0: return frame
    val = int(np.clip(intensity, 0, 1) * sensitivity * 90)
    return cv2.add(frame, np.full(frame.shape, max(0, val), dtype='uint8'))

def apply_rgb_shift(frame, intensity, sensitivity=1.0):
    if intensity < 0.2: return frame
    h, w = frame.shape[:2]
    shift = int((intensity - 0.2) * sensitivity * 45)
    if shift < 1: return frame
    shift = min(shift, w - 1)
    res = frame.copy()
    res[:, shift:, 0] = frame[:, :-shift, 0] 
    res[:, :-shift, 2] = frame[:, shift:, 2] 
    return res

def apply_shake(frame, intensity, sensitivity=1.0):
    if intensity < 0.2: return frame
    h, w = frame.shape[:2]
    max_offset = int((intensity - 0.2) * sensitivity * 60)
    if max_offset < 1: return frame
    dx, dy = random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_ghosting(frame, prev_frame, intensity, sensitivity=1.0):
    if prev_frame is None or intensity < 0.15: return frame
    alpha = np.clip(0.1 + (intensity * sensitivity * 0.7), 0, 0.97)
    return cv2.addWeighted(frame, 1 - alpha, prev_frame, alpha, 0)

def apply_monochrome(frame, intensity, sensitivity=1.0, amount=1.0):
    if intensity < 0.35:
        return frame
    blend = np.clip(((intensity - 0.35) / 0.65) * sensitivity * amount, 0, 1)
    if blend <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 1 - blend, gray_bgr, blend, 0)

def apply_hue_shift(frame, intensity, sensitivity=1.0, amount=1.0):
    if intensity < 0.15:
        return frame
    shift = int(np.clip((intensity - 0.15) * sensitivity * amount * 140, -180, 180))
    if abs(shift) < 1:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 180).astype(np.uint8)
    saturation_boost = np.clip(1.0 + (0.25 * np.clip(amount, 0, 2) * np.clip(sensitivity, 0, 3)), 1.0, 1.7)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_boost, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_vignette(frame, intensity, sensitivity=1.0, amount=1.0):
    if intensity < 0.2:
        return frame
    strength = np.clip((intensity - 0.2) * sensitivity * amount, 0, 1)
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    distance = np.clip(np.sqrt((x * x) + (y * y)), 0, 1)
    mask = 1.0 - (distance ** 1.7 * 0.65 * strength)
    return np.clip(frame.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)

def apply_static_pan_zoom(frame, progress, amount, pan_x, pan_y):
    h, w = frame.shape[:2]
    max_zoom = 1.0 + (0.10 * np.clip(amount, 0, 1))
    zoom = 1.0 + ((max_zoom - 1.0) * np.clip(progress, 0, 1))
    crop_w, crop_h = max(1, int(w / zoom)), max(1, int(h / zoom))
    max_x, max_y = max(0, w - crop_w), max(0, h - crop_h)
    x = int((max_x / 2) + (pan_x * max_x * 0.35 * progress))
    y = int((max_y / 2) + (pan_y * max_y * 0.35 * progress))
    x, y = int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))
    cropped = frame[y:y + crop_h, x:x + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def parse_render_seed(seed_text):
    if seed_text is None:
        return None
    text = str(seed_text).strip()
    if not text:
        return None
    return int(text, 0)

# --- Core Logic ---

class GlitchProcessor:
    def __init__(self, inputs, audio, output, duration=0.1, fps=30, 
                 pixelate=False, flash=False, rewind=False,
                 rgb_shift=False, shake=False, ghosting=False,
                 static_pan_zoom=False, monochrome=False, hue_shift=False, vignette=False,
                 beat_sync=False, coherence=0.7, sensitivity=1.0,
                 export_mode=EXPORT_FINAL_VIDEO,
                 progress_callback=None, log_callback=None, frame_callback=None,
                 effect_amounts=None, primary_video_idx=None, primary_focus=0.0,
                 beat_step=4, beat_variation=0.0, render_limit=None,
                 source_variety=0.0, music_match=0.35,
                 color_reference_idx=None, color_match_strength=0.0,
                 lut_path="", output_resolution=None, export_quality_label="High quality (slower)",
                 effect_timing=None,
                 experimental_mode=False,
                 ai_dream_chance=0.25,
                 ai_dream_timing="Clip",
                 scene_transition_mode="Auto",
                 force_bw=False,
                 ai_enabled=False, ai_segment_anchor_only=True, ai_backend_url=AI_DEFAULT_BACKEND_URL,
                 ai_prompt=AI_DEFAULT_PROMPT, ai_negative_prompt=AI_DEFAULT_NEGATIVE_PROMPT,
                 ai_every_n_frames=AI_DEFAULT_EVERY_N_FRAMES, ai_denoise=AI_DEFAULT_DENOISE,
                 ai_cfg_scale=AI_DEFAULT_CFG_SCALE, ai_steps=AI_DEFAULT_STEPS,
                 ai_max_dim=AI_DEFAULT_MAX_DIM, ai_blend=AI_DEFAULT_BLEND,
                 verbose_match_logging=False,
                 debug_match_logging=False,
                 temp_root=None,
                 render_seed=None):
        self.inputs, self.audio, self.output = inputs, audio, output
        self.duration, self.fps = duration, fps
        self.pixelate, self.flash, self.rewind = pixelate, flash, rewind
        self.rgb_shift, self.shake, self.ghosting = rgb_shift, shake, ghosting
        self.static_pan_zoom = static_pan_zoom
        self.monochrome, self.hue_shift, self.vignette = monochrome, hue_shift, vignette
        self.beat_sync, self.coherence, self.sensitivity = beat_sync, coherence, sensitivity
        self.export_mode = export_mode
        self.beat_step = max(1, int(beat_step))
        self.beat_variation = np.clip(float(beat_variation), 0, 1)
        self.render_limit = float(render_limit) if render_limit else None
        self.source_variety = np.clip(float(source_variety), 0, 1)
        self.music_match = np.clip(float(music_match), 0, 1)
        self.color_reference_idx = color_reference_idx if color_reference_idx is not None else None
        self.color_match_strength = np.clip(float(color_match_strength), 0, 1)
        self.lut_path = lut_path
        self.lut = None
        self.output_resolution = output_resolution
        self.export_quality_label = export_quality_label
        self.experimental_mode = bool(experimental_mode)
        self.scene_transition_mode = scene_transition_mode if scene_transition_mode in SCENE_TRANSITION_LABELS else "Auto"
        self.force_bw = bool(force_bw)
        self.ai_dream_chance = float(np.clip(ai_dream_chance, 0.0, 1.0))
        self.ai_dream_timing = ai_dream_timing if ai_dream_timing in AI_DREAM_TIMING_LABELS else "Clip"
        self.ai_enabled = bool(ai_enabled)
        self.ai_segment_anchor_only = bool(ai_segment_anchor_only)
        self.ai_backend_url = ai_backend_url.strip()
        self.ai_prompt = ai_prompt.strip()
        self.ai_negative_prompt = ai_negative_prompt.strip()
        self.ai_every_n_frames = max(1, int(ai_every_n_frames))
        self.ai_denoise = float(np.clip(ai_denoise, 0.05, 0.95))
        self.ai_cfg_scale = float(np.clip(ai_cfg_scale, 1.0, 30.0))
        self.ai_steps = max(1, int(ai_steps))
        self.ai_max_dim = max(64, int(ai_max_dim))
        self.ai_blend = float(np.clip(ai_blend, 0.0, 1.0))
        self.verbose_match_logging = bool(verbose_match_logging)
        self.debug_match_logging = bool(debug_match_logging)
        self.temp_root = temp_root
        self.render_seed = render_seed
        self.ai_session = requests.Session()
        self.ai_backend_warning_shown = False
        self.ai_backend_model_warning_shown = False
        self.effect_amounts = DEFAULT_EFFECT_AMOUNTS.copy()
        if effect_amounts:
            for name in EFFECT_NAMES:
                self.effect_amounts[name] = max(0.0, float(effect_amounts.get(name, 1.0)))
        self.effect_timing = DEFAULT_EFFECT_TIMING.copy()
        if effect_timing:
            for name in EFFECT_NAMES:
                if effect_timing.get(name) in EFFECT_TIMING_LABELS:
                    self.effect_timing[name] = effect_timing[name]
        self.primary_video_idx = primary_video_idx if primary_video_idx is not None else None
        self.primary_focus = np.clip(float(primary_focus), 0, 1)
        self.progress_callback, self.log_callback, self.frame_callback = progress_callback, log_callback, frame_callback
        self.stop_requested = False
        self.current_vid_idx = 0
        self.recent_matches = {idx: deque(maxlen=8) for idx in range(len(self.inputs))}
        self.source_lock_penalties = [0.0 for _ in self.inputs]

    def log(self, msg):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def effect_amount(self, name):
        return self.effect_amounts.get(name, 1.0)

    def effect_uses_frame_timing(self, name):
        return self.effect_timing.get(name, DEFAULT_EFFECT_TIMING.get(name, "Frame")) == "Frame"

    def resolve_effect_frame_timing(self, name):
        timing = self.effect_timing.get(name, DEFAULT_EFFECT_TIMING.get(name, "Frame"))
        if timing == "Random":
            return random.choice((True, False))
        return timing == "Frame"

    def export_quality_args(self):
        return EXPORT_QUALITY_LABELS.get(
            self.export_quality_label,
            EXPORT_QUALITY_LABELS["High quality (slower)"],
        )

    def initialize_rng(self):
        seed = self.render_seed
        if seed is None:
            seed = random.SystemRandom().randint(1, 2**63 - 1)
        seed = int(seed)
        self.render_seed = seed
        random.seed(seed)
        np.random.seed(seed % (2**32))
        return seed

    def ai_stylization_active(self):
        return self.experimental_mode and self.ai_enabled and bool(self.ai_backend_url) and bool(self.ai_prompt)

    def ai_segment_probability(self, clip_rms):
        base = float(np.clip(self.ai_dream_chance, 0.0, 1.0))
        if self.ai_dream_timing == "Random":
            return base
        # Only boost noticeably on genuinely quiet sections; keep the default
        # chance closer to the user's slider value on most clips.
        quietness = float(np.clip((0.75 - clip_rms) / 0.75, 0.0, 1.0))
        boosted = base + ((1.0 - base) * (quietness ** 2.2) * 0.2)
        return float(np.clip(boosted, 0.0, 1.0))

    def ai_backend_base_url(self):
        return self.ai_backend_url.rstrip("/")

    def ai_backend_app_config(self):
        response = self.ai_session.get(f"{self.ai_backend_base_url()}/get/app_config", timeout=(10, 30))
        response.raise_for_status()
        return response.json()

    def ai_backend_model_selection(self):
        config = self.ai_backend_app_config()
        model_config = config.get("model") or {}
        stable_diffusion_model = model_config.get("stable-diffusion")
        vae_model = model_config.get("vae")

        if stable_diffusion_model:
            return stable_diffusion_model, vae_model

        response = self.ai_session.get(f"{self.ai_backend_base_url()}/get/models", timeout=(10, 30))
        response.raise_for_status()
        models_data = response.json().get("models") or []
        stable_diffusion_models = [
            model_info.get("model")
            for model_info in models_data
            if "stable-diffusion" in (model_info.get("tags") or []) and model_info.get("model")
        ]
        if stable_diffusion_models:
            return stable_diffusion_models[0], vae_model

        if not self.ai_backend_model_warning_shown:
            self.log("  Easy Diffusion did not report a configured stable-diffusion model; using a bundled default.")
            self.ai_backend_model_warning_shown = True
        return AI_DEFAULT_MODEL_FALLBACK, vae_model

    @staticmethod
    def ai_parse_stream_response(stream_text):
        decoder = json.JSONDecoder()
        documents = []
        index = 0
        text_length = len(stream_text)
        while index < text_length:
            while index < text_length and stream_text[index].isspace():
                index += 1
            if index >= text_length:
                break
            document, end_index = decoder.raw_decode(stream_text, index)
            documents.append(document)
            index = end_index
        return documents

    def ai_target_size(self, frame):
        height, width = frame.shape[:2]
        if max(width, height) <= self.ai_max_dim:
            return width, height
        if width >= height:
            target_width = self.ai_max_dim
            target_height = max(64, int(round(height * (self.ai_max_dim / max(width, 1)))))
        else:
            target_height = self.ai_max_dim
            target_width = max(64, int(round(width * (self.ai_max_dim / max(height, 1)))))
        return target_width, target_height

    def ai_render_frame(self, frame):
        if not self.ai_stylization_active():
            return frame, False
        try:
            target_width, target_height = self.ai_target_size(frame)
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            stable_diffusion_model, vae_model = self.ai_backend_model_selection()
            payload = {
                "prompt": self.ai_prompt,
                "negative_prompt": self.ai_negative_prompt,
                "seed": random.randint(0, 2**31 - 1),
                "width": target_width,
                "height": target_height,
                "num_outputs": 1,
                "num_inference_steps": self.ai_steps,
                "guidance_scale": self.ai_cfg_scale,
                "prompt_strength": self.ai_denoise,
                "init_image": frame_to_base64_png(resized),
                "mask": "",
                "preserve_init_image_color_profile": False,
                "sampler_name": "ddim",
                "use_stable_diffusion_model": stable_diffusion_model,
                "use_vae_model": vae_model,
                "request_id": f"glitchsync_{random.randint(100000, 999999)}",
                "session_id": AI_DEFAULT_SESSION,
                "vram_usage_level": "balanced",
                "output_format": "png",
                "output_quality": 100,
                "output_lossless": True,
            }
            self.log(f"  Easy Diffusion model: {stable_diffusion_model}")
            response = self.ai_session.post(
                f"{self.ai_backend_base_url()}/render",
                json=payload,
                timeout=(10, 30),
            )
            response.raise_for_status()
            render_data = response.json()
            task_id = render_data.get("task")
            if not task_id:
                raise RuntimeError("Easy Diffusion did not return a task id")
            ping_url = f"{self.ai_backend_base_url()}/ping?session_id={AI_DEFAULT_SESSION}"
            stream_url = f"{self.ai_backend_base_url()}/image/stream/{task_id}"
            deadline = time.time() + 180
            last_status = None
            while time.time() < deadline:
                ping_response: Response = self.ai_session.get(ping_url, timeout=(10, 30))
                ping_response.raise_for_status()
                ping_data = ping_response.json()
                tasks = ping_data.get("tasks") or {}
                task_status = tasks.get(str(task_id))
                if task_status != last_status:
                    self.log(f"  Easy Diffusion task {task_id} status: {task_status or ping_data.get('status')}")
                    last_status = task_status
                if task_status in {"buffer", "completed"}:
                    stream_response: Response = self.ai_session.get(stream_url, timeout=(10, 600))
                    stream_response.raise_for_status()
                    documents = self.ai_parse_stream_response(stream_response.text)
                    final_document = next(
                        (document for document in reversed(documents) if isinstance(document, dict) and document.get("output")),
                        documents[-1] if documents else {},
                    )
                    if final_document.get("status") not in {"succeeded", "success", "completed"}:
                        raise RuntimeError(
                            f"Easy Diffusion stream ended without a successful result: {final_document.get('status')}"
                        )
                    images = final_document.get("output") or []
                    if not images:
                        raise RuntimeError("Easy Diffusion completed without returning an image")
                    output_image = images[0]
                    image_data = output_image.get("data")
                    if not image_data:
                        raise RuntimeError("Easy Diffusion response did not include image data")
                    stylized = base64_png_to_frame(image_data, (target_width, target_height))
                    if stylized.shape[1] != frame.shape[1] or stylized.shape[0] != frame.shape[0]:
                        stylized = cv2.resize(stylized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    return stylized, True
                if task_status in {"failed", "stopped"}:
                    raise RuntimeError(f"Easy Diffusion task ended with status: {task_status}")
                time.sleep(0.5)
            raise TimeoutError("Timed out waiting for Easy Diffusion to complete the stylization task")
        except Exception as exc:
            if not self.ai_backend_warning_shown:
                self.log(f"  AI stylization disabled for this render: {exc}")
                self.log("  Check that Easy Diffusion is running and reachable at the configured URL.")
                self.ai_backend_warning_shown = True
            return frame, False

    @staticmethod
    def describe_audio_style(audio_features):
        bass = audio_features["bass_energy"]
        highs = audio_features["highs_energy"]
        mids = audio_features["mids_energy"]
        rms = audio_features["rms_energy"]
        tempo = float(audio_features["tempo"])
        duration = max(float(audio_features["duration"]), 1e-6)
        p95 = max(float(np.percentile(rms, 95)), 1e-9)
        p50 = float(np.percentile(rms, 50))
        bass_mean = float(np.mean(bass))
        highs_mean = float(np.mean(highs))
        mids_mean = float(np.mean(mids))
        spectral_total = max(bass_mean + highs_mean + mids_mean, 1e-9)
        return {
            "tempo": tempo,
            "energy": float(np.clip(np.mean(rms) / p95, 0, 1)),
            "bass": float(np.clip(bass_mean / spectral_total, 0, 1)),
            "highs": float(np.clip(highs_mean / spectral_total, 0, 1)),
            "dynamic": float(np.clip((p95 - p50) / p95, 0, 1)),
            "beat_density": float(len(audio_features["beats"]) / duration),
        }

    @staticmethod
    def choose_auto_style(audio_features):
        desc = GlitchProcessor.describe_audio_style(audio_features)
        if desc["tempo"] < 85 and desc["energy"] < 0.42:
            return "Ambient Drift", desc
        if desc["tempo"] < 105 and desc["energy"] < 0.50:
            return "Mellow Story", desc
        if desc["tempo"] > 138 and desc["highs"] > 0.42 and desc["dynamic"] > 0.58:
            return "Glitch Heavy", desc
        best_name = "Pop Performance"
        best_score = float("inf")
        for name, profile in AUTO_STYLE_PROFILES.items():
            score = (
                abs(desc["tempo"] - profile["tempo"]) / 80
                + abs(desc["energy"] - profile["energy"]) * 1.2
                + abs(desc["bass"] - profile["bass"]) * 0.8
                + abs(desc["highs"] - profile["highs"]) * 0.8
                + abs(desc["dynamic"] - profile["dynamic"]) * 0.9
            )
            if score < best_score:
                best_name, best_score = name, score
        return best_name, desc

    def choose_candidate(self, candidates, source_use_counts=None, activity_db=None,
                         target_activity=None, frame_count=1, activity_scale=1.0):
        if not candidates:
            return None
        diversity_strength = self.source_variety if self.source_variety > 0 else (0.35 if len(self.inputs) == 1 else 0.0)
        use_variety = bool(source_use_counts) and diversity_strength > 0
        use_activity = (
            activity_db is not None
            and target_activity is not None
            and self.music_match > 0
            and activity_scale > 0
        )
        if not use_variety and not use_activity:
            choice = random.choice(candidates)
            if self.debug_match_logging:
                self.log(
                    f"  Debug pick: {os.path.basename(self.inputs[choice[0]])} frame {choice[1]} "
                    f"(uniform among {len(candidates)} candidates)"
                )
            return choice
        max_count = max(source_use_counts) if source_use_counts else 0
        weights = []
        for v_idx, f_idx in candidates:
            weight = 1.0
            lock_penalty = self.source_lock_penalties[v_idx] if v_idx < len(self.source_lock_penalties) else 0.0
            if lock_penalty > 0:
                weight *= 1.0 / (1.0 + lock_penalty)
            history = self.recent_matches.get(v_idx)
            if history:
                nearest = min(abs(f_idx - prev) for prev in history)
                repeat_scale = max(1.0, frame_count * 12.0)
                repeat_separation = np.clip(nearest / repeat_scale, 0, 1)
                weight *= 0.2 + (0.8 * (repeat_separation ** 1.4))
            if use_variety:
                weight *= ((max_count + 1) / (source_use_counts[v_idx] + 1)) ** (1 + (self.source_variety * 4))
                if history:
                    nearest = min(abs(f_idx - prev) for prev in history)
                    spread = max(1.0, frame_count * (8.0 + (diversity_strength * 24.0)))
                    separation = np.clip(nearest / spread, 0, 1)
                    weight *= 0.25 + (0.75 * (separation ** (0.7 + (diversity_strength * 1.5))))
            if use_activity:
                activity = self.segment_motion_score(activity_db, v_idx, f_idx, frame_count)
                activity_norm = np.clip(activity / activity_scale, 0, 1)
                closeness = 1.0 - abs(activity_norm - target_activity)
                weight *= max(0.05, 1.0 + (self.music_match * 5.0 * closeness))
            weights.append(weight)
        choice = random.choices(candidates, weights=weights, k=1)[0]
        if self.debug_match_logging:
            ranked = sorted(
                zip(candidates, weights),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            summary = ", ".join(
                f"{os.path.basename(self.inputs[v_idx])}@{f_idx}:{weight:.2f}"
                for (v_idx, f_idx), weight in ranked
            )
            self.log(f"  Debug candidates ({len(candidates)}): {summary}")
            self.log(f"  Debug pick: {os.path.basename(self.inputs[choice[0]])} frame {choice[1]}")
        return choice

    def decay_source_lock_penalties(self):
        if not self.source_lock_penalties:
            return
        for idx, penalty in enumerate(self.source_lock_penalties):
            if penalty > 0:
                self.source_lock_penalties[idx] = max(0.0, penalty * SOURCE_LOCK_PENALTY_DECAY)

    def penalize_source_lock(self, source_idx, source_streak, alternate_count=0, current_count=0):
        if source_idx is None or not (0 <= source_idx < len(self.source_lock_penalties)):
            return
        alt_count = max(0, int(alternate_count))
        total_candidates = max(1, alt_count + max(0, int(current_count)))
        alt_volume = np.clip(alt_count / 16.0, 0, 1)
        alt_fraction = np.clip(alt_count / total_candidates, 0, 1)
        penalty_scale = 0.5 + (1.0 * alt_volume) + (0.5 * alt_fraction)
        penalty = min(
            SOURCE_LOCK_PENALTY_MAX,
            (SOURCE_LOCK_PENALTY_START + max(0, source_streak - SOURCE_LOCK_TRIGGER_STREAK) * 0.75) * penalty_scale,
        )
        self.source_lock_penalties[source_idx] = max(self.source_lock_penalties[source_idx], penalty)

    def motion_scale(self, motion_db):
        scores = []
        for _, source_scores in motion_db.values():
            scores.extend(source_scores)
        if not scores:
            return 1.0
        return max(float(np.percentile(scores, 95)), STATIC_MOTION_THRESHOLD, 1e-6)

    def use_primary_video(self):
        return (
            self.primary_video_idx is not None
            and 0 <= self.primary_video_idx < len(self.inputs)
            and self.primary_focus > 0
        )

    def select_beat_cut_times(self, beats, sr):
        beat_times = librosa.frames_to_time(beats, sr=sr)
        if self.beat_step <= 1 and self.beat_variation <= 0:
            return beat_times

        selected = []
        for idx, beat_time in enumerate(beat_times):
            is_main_cut = idx % self.beat_step == 0
            is_variation_cut = random.random() < self.beat_variation
            if is_main_cut or is_variation_cut:
                selected.append(beat_time)

        if len(selected) < 2 and len(beat_times) >= 2:
            selected = [beat_times[0], beat_times[-1]]
        return np.array(selected)

    def analyze_video_file(self, path, emit_log=True, emit_frame_callback=True):
        messages = []
        def log(msg):
            if emit_log:
                self.log(msg)
            else:
                messages.append(msg)
        cache_path = analysis_cache_path(path, "video", "pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if (
                    isinstance(cached, dict)
                    and isinstance(cached.get("buckets"), list)
                    and len(cached["buckets"]) == 256
                    and "brightness_scores" in cached
                    and "color_stats" in cached
                ):
                    log(f"  Using cached index for {os.path.basename(path)}")
                    return cached, messages
            except Exception as e:
                log(f"  Ignoring video cache for {os.path.basename(path)}: {e}")

        log(f"  Scanning {os.path.basename(path)}...")
        buckets = [[] for _ in range(256)]
        brightness_scores = []
        motion_scores = []
        activity_scores = []
        lab_sum = np.zeros(3, dtype=np.float64)
        lab_sq_sum = np.zeros(3, dtype=np.float64)
        lab_pixels = 0
        cap = cv2.VideoCapture(path)
        frame_count = 0
        prev_sample = None
        while not self.stop_requested:
            ret, frame = cap.read()
            if not ret: break
            if emit_frame_callback and self.frame_callback and frame_count % 120 == 0:
                self.frame_callback(frame)
            if frame_count % 5 == 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                b = int(np.mean(hsv[:, :, 2]))
                buckets[b].append(frame_count)
                brightness_scores.append((frame_count, b / 255.0))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sample = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                motion = 0.0 if prev_sample is None else float(np.mean(cv2.absdiff(sample, prev_sample)) / 255.0)
                motion_scores.append((frame_count, motion))
                if prev_sample is None:
                    flow = 0.0
                else:
                    flow_field = cv2.calcOpticalFlowFarneback(
                        prev_sample,
                        sample,
                        None,
                        pyr_scale=0.5,
                        levels=3,
                        winsize=15,
                        iterations=3,
                        poly_n=5,
                        poly_sigma=1.2,
                        flags=0,
                    )
                    flow_mag, _ = cv2.cartToPolar(flow_field[..., 0], flow_field[..., 1])
                    flow = float(np.mean(flow_mag))
                activity_scores.append((frame_count, motion, flow))
                prev_sample = sample
                lab_sample = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
                lab_sum += np.sum(lab_sample, axis=(0, 1))
                lab_sq_sum += np.sum(lab_sample ** 2, axis=(0, 1))
                lab_pixels += lab_sample.shape[0] * lab_sample.shape[1]
            frame_count += 1
        cap.release()
        if lab_pixels:
            lab_mean = lab_sum / lab_pixels
            lab_std = np.sqrt(np.maximum((lab_sq_sum / lab_pixels) - (lab_mean ** 2), LAB_EPSILON))
        else:
            lab_mean = np.array([128.0, 128.0, 128.0])
            lab_std = np.array([1.0, 1.0, 1.0])
        if activity_scores:
            motion_vals = np.array([motion for _, motion, _ in activity_scores], dtype=np.float32)
            flow_vals = np.array([flow for _, _, flow in activity_scores], dtype=np.float32)
            motion_scale = max(float(np.percentile(motion_vals, 95)), LAB_EPSILON)
            flow_scale = max(float(np.percentile(flow_vals, 95)), LAB_EPSILON)
            activity_scores = [
                (
                    frame_idx,
                    float(np.clip(motion / motion_scale, 0, 1)),
                    float(np.clip(flow / flow_scale, 0, 1)),
                    float(np.clip(((motion / motion_scale) * 0.55) + ((flow / flow_scale) * 0.45), 0, 1)),
                )
                for frame_idx, motion, flow in activity_scores
            ]
        video_analysis = {
            "buckets": buckets,
            "brightness_scores": brightness_scores,
            "motion_scores": motion_scores,
            "activity_scores": activity_scores,
            "color_stats": {
                "lab_mean": lab_mean.tolist(),
                "lab_std": lab_std.tolist(),
            },
        }

        if not self.stop_requested:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(video_analysis, f, protocol=pickle.HIGHEST_PROTOCOL)
                log(f"  Cached index for {os.path.basename(path)}")
            except Exception as e:
                log(f"  Could not save video cache for {os.path.basename(path)}: {e}")
        return video_analysis, messages

    def analyze_source_videos(self):
        self.log("Indexing video frames...")
        if self.progress_callback: self.progress_callback(-1, -1)
        frame_db = {i: [] for i in range(256)}
        brightness_db = {}
        motion_db = {}
        activity_db = {}
        color_stats = {}
        worker_count = min(max(1, os.cpu_count() or 1), max(1, min(len(self.inputs), 4)))
        if worker_count > 1:
            self.log(f"  Parallel indexing with {worker_count} workers...")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self.analyze_video_file, path, False, False): video_idx
                for video_idx, path in enumerate(self.inputs)
            }
            completed = 0
            for future in as_completed(futures):
                video_idx = futures[future]
                video_analysis, messages = future.result()
                for msg in messages:
                    self.log(msg)
                buckets = video_analysis["buckets"]
                brightness_scores = video_analysis.get("brightness_scores", [])
                motion_scores = video_analysis.get("motion_scores", [])
                activity_scores = video_analysis.get("activity_scores", [])
                brightness_db[video_idx] = (
                    [frame for frame, _ in brightness_scores],
                    [score for _, score in brightness_scores],
                )
                motion_db[video_idx] = (
                    [frame for frame, _ in motion_scores],
                    [score for _, score in motion_scores],
                )
                activity_db[video_idx] = (
                    [frame for frame, _, _, _ in activity_scores],
                    [score for _, _, _, score in activity_scores],
                )
                color_stats[video_idx] = video_analysis.get("color_stats")
                for brightness, frames in enumerate(buckets):
                    frame_db[brightness].extend((video_idx, frame_idx) for frame_idx in frames)
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, len(self.inputs))
        return frame_db, brightness_db, motion_db, activity_db, color_stats

    def analyze_audio_file(self):
        cache_path = analysis_cache_path(self.audio, "audio", "npz")
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=False)
                self.log(f"Using cached audio analysis for {os.path.basename(self.audio)}")
                return {
                    "sr": int(cached["sr"]),
                    "duration": float(cached["duration"]),
                    "bass_energy": cached["bass_energy"],
                    "highs_energy": cached["highs_energy"],
                    "mids_energy": cached["mids_energy"],
                    "rms_energy": cached["rms_energy"],
                    "beats": cached["beats"],
                    "tempo": float(cached["tempo"]),
                }
            except Exception as e:
                self.log(f"Ignoring audio cache for {os.path.basename(self.audio)}: {e}")

        self.log("Analyzing audio features...")
        y, sr = librosa.load(self.audio, sr=None)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass_energy = np.mean(D[freqs <= 150, :], axis=0)
        highs_energy = np.mean(D[freqs >= 5000, :], axis=0)
        mids_energy = np.mean(D[(freqs > 150) & (freqs < 5000), :], axis=0)
        rms_energy = librosa.feature.rms(y=y)[0]
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path,
                sr=np.array(sr, dtype=np.int64),
                duration=np.array(audio_duration, dtype=np.float64),
                bass_energy=bass_energy,
                highs_energy=highs_energy,
                mids_energy=mids_energy,
                rms_energy=rms_energy,
                beats=beats,
                tempo=np.array(tempo_val, dtype=np.float64),
            )
            self.log(f"Cached audio analysis for {os.path.basename(self.audio)}")
        except Exception as e:
            self.log(f"Could not save audio cache for {os.path.basename(self.audio)}: {e}")

        return {
            "sr": sr,
            "duration": audio_duration,
            "bass_energy": bass_energy,
            "highs_energy": highs_energy,
            "mids_energy": mids_energy,
            "rms_energy": rms_energy,
            "beats": beats,
            "tempo": tempo_val,
        }

    def find_best_match(self, target_b, frame_db, current_vid_idx, source_use_counts=None,
                        activity_db=None, target_activity=None, frame_count=1, activity_scale=1.0,
                        source_streak=0):
        search_range = 8
        candidates_current, candidates_other, candidates_primary = [], [], []
        for b in range(max(0, target_b - search_range), min(255, target_b + search_range) + 1):
            for v_idx, f_idx in frame_db[b]:
                if self.use_primary_video() and v_idx == self.primary_video_idx:
                    candidates_primary.append((v_idx, f_idx))
                if v_idx == current_vid_idx: candidates_current.append((v_idx, f_idx))
                else: candidates_other.append((v_idx, f_idx))
        if self.debug_match_logging:
            current_lock_penalty = (
                self.source_lock_penalties[current_vid_idx]
                if 0 <= current_vid_idx < len(self.source_lock_penalties)
                else 0.0
            )
            alternate_lock_penalty = max(
                [self.source_lock_penalties[v_idx] for v_idx, _ in candidates_other]
                or [0.0]
            )
            self.log(
                f"  Debug pool: target_b={target_b} current={len(candidates_current)} other={len(candidates_other)} "
                f"primary={len(candidates_primary)} streak={source_streak} "
                f"lock={current_lock_penalty:.2f} alt_lock={alternate_lock_penalty:.2f}"
            )
        if candidates_primary and random.random() < self.primary_focus:
            return self.choose_candidate(candidates_primary, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
        if source_streak >= SOURCE_LOCK_TRIGGER_STREAK and candidates_other:
            self.penalize_source_lock(current_vid_idx, source_streak, len(candidates_other), len(candidates_current))
            self.log(
                f"  Source lock detected after {source_streak} same-source segments; "
                f"forcing an alternate source near brightness {target_b} "
                f"({len(candidates_current)} current / {len(candidates_other)} alternate candidates)."
            )
            if self.debug_match_logging:
                self.log(
                    f"  Debug lock: current={len(candidates_current)} alternate={len(candidates_other)} "
                    f"primary={len(candidates_primary)} streak={source_streak}"
                )
            return self.choose_candidate(candidates_other, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
        if source_streak >= SOURCE_LOCK_TRIGGER_STREAK and candidates_current and not candidates_other:
            self.penalize_source_lock(current_vid_idx, source_streak, 0, len(candidates_current))
            self.log(
                f"  Source lock detected after {source_streak} same-source segments; "
                f"no alternate candidates exist near brightness {target_b} "
                f"({len(candidates_current)} current / 0 alternate candidates)."
            )
            if self.debug_match_logging:
                self.log(
                    f"  Debug lock: current={len(candidates_current)} alternate=0 "
                    f"primary={len(candidates_primary)} streak={source_streak}"
                )
        if candidates_current and (random.random() < self.coherence or not candidates_other):
            return self.choose_candidate(candidates_current, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
        if candidates_other:
            return self.choose_candidate(candidates_other, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
        offset = 1
        first_fallback = None
        skipped_single_source_bands = 0
        while offset < 256:
            low, high = target_b - offset, target_b + offset
            fallback = []
            primary_fallback = []
            if low >= 0: fallback.extend(frame_db[low])
            if high <= 255: fallback.extend(frame_db[high])
            if fallback and first_fallback is None:
                first_fallback = list(fallback)
            if self.use_primary_video():
                primary_fallback = [item for item in fallback if item[0] == self.primary_video_idx]
                if primary_fallback and random.random() < self.primary_focus:
                    return self.choose_candidate(primary_fallback, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
            if fallback:
                fallback_sources = {v_idx for v_idx, _ in fallback}
                if source_streak >= SOURCE_LOCK_TRIGGER_STREAK and len(fallback_sources) < 2:
                    skipped_single_source_bands += 1
                    if self.debug_match_logging:
                        self.log(
                            f"  Debug fallback skip: offset={offset} candidates={len(fallback)} "
                            f"sources={len(fallback_sources)} skipped={skipped_single_source_bands}"
                        )
                    offset += 1
                    continue
                if self.debug_match_logging:
                    self.log(
                        f"  Debug fallback: offset={offset} candidates={len(fallback)} "
                        f"primary={len(primary_fallback)} sources={len(fallback_sources)} "
                        f"skipped={skipped_single_source_bands}"
                    )
                return self.choose_candidate(fallback, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
            offset += 1
        if first_fallback:
            if self.debug_match_logging:
                fallback_sources = {v_idx for v_idx, _ in first_fallback}
                self.log(
                    f"  Debug fallback last-resort: candidates={len(first_fallback)} "
                    f"sources={len(fallback_sources)} skipped={skipped_single_source_bands}"
                )
            return self.choose_candidate(first_fallback, source_use_counts, activity_db, target_activity, frame_count, activity_scale)
        return None

    def segment_motion_score(self, motion_db, video_idx, start_frame, frame_count):
        frames, scores = motion_db.get(video_idx, ([], []))
        if not frames:
            return 1.0
        end_frame = start_frame + max(1, frame_count)
        start_pos = bisect.bisect_left(frames, start_frame)
        end_pos = bisect.bisect_right(frames, end_frame)
        window = scores[start_pos:end_pos]
        if not window:
            nearest_pos = min(range(len(frames)), key=lambda idx: abs(frames[idx] - start_frame))
            return scores[nearest_pos]
        return float(np.mean(window))

    def record_match(self, video_idx, start_frame):
        history = self.recent_matches.get(video_idx)
        if history is not None:
            history.append(start_frame)

    def load_lut(self):
        if not self.lut_path:
            return
        self.lut = load_cube_lut(self.lut_path)
        self.log(f"Using LUT: {os.path.basename(self.lut_path)}")

    def process(self):
        package_dir = None
        clip_dir = None
        temp_root = self.temp_root or preferred_temp_root() or tempfile.gettempdir()
        seed = self.initialize_rng()
        self.log(f"  Render seed: {seed}")
        if temp_root in ("/dev/shm", "/run/shm"):
            self.log(f"  Temporary storage: RAM-backed ({temp_root})")
        else:
            self.log(f"  Temporary storage: disk-backed ({temp_root})")
        self.log("  Starting source indexing and audio analysis in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            source_future = executor.submit(self.analyze_source_videos)
            audio_future = executor.submit(self.analyze_audio_file)
            frame_db, brightness_db, motion_db, activity_db, color_stats = source_future.result()
            if self.stop_requested: return
            audio_features = audio_future.result()
        if self.stop_requested: return
        if self.lut_path:
            self.load_lut()
        sr = audio_features["sr"]
        audio_duration = audio_features["duration"]
        bass_energy = audio_features["bass_energy"]
        highs_energy = audio_features["highs_energy"]
        mids_energy = audio_features["mids_energy"]
        rms_energy = audio_features["rms_energy"]
        
        if self.beat_sync:
            beats = audio_features["beats"]
            tempo_val = audio_features["tempo"]
            cut_times = self.select_beat_cut_times(beats, sr)
            self.log(f"  Tempo: {tempo_val:.1f} BPM")
            self.log(f"  Beat step: {self.beat_step}; variation: {self.beat_variation:.2f}")
            if len(cut_times) < 2:
                self.log("  Not enough beats detected; falling back to duration cuts.")
                cut_times = np.arange(0, audio_duration, max(0.01, self.duration))
        else:
            cut_times = np.arange(0, audio_duration, max(0.01, self.duration))

        render_duration = audio_duration
        if self.render_limit:
            render_duration = min(audio_duration, max(self.duration, self.render_limit))
            cut_times = cut_times[cut_times < render_duration]
            if len(cut_times) == 0 or cut_times[0] > 0:
                cut_times = np.insert(cut_times, 0, 0)
            self.log(f"  Render limit: {render_duration:.1f}s")

        times = librosa.times_like(rms_energy, sr=sr)
        def get_val(arr, t): return arr[min(np.searchsorted(times, t), len(arr)-1)]
        def norm_val(arr, max_val, t):
            return np.clip(float(get_val(arr, t)) / max(max_val, 1e-9), 0, 1)
        def norm_range(arr, max_val, start_t, end_t):
            start_idx = min(np.searchsorted(times, start_t), len(arr) - 1)
            end_idx = min(np.searchsorted(times, end_t), len(arr) - 1)
            if end_idx <= start_idx:
                return norm_val(arr, max_val, start_t)
            return np.clip(float(np.mean(arr[start_idx:end_idx])) / max(max_val, 1e-9), 0, 1)
        
        # Aggressive normalization for punchier effects
        max_rms = float(np.percentile(rms_energy, 99.5))
        max_bass = float(np.percentile(bass_energy, 99.5))
        max_highs = float(np.percentile(highs_energy, 99.5))
        max_mids = float(np.percentile(mids_energy, 99.5))
        segment_profiles = []
        for i in range(len(cut_times)):
            t_start = cut_times[i]
            dur = (cut_times[i + 1] if i + 1 < len(cut_times) else render_duration) - t_start
            clip_end = t_start + dur
            profile = {
                "index": i,
                "t_start": t_start,
                "dur": dur,
                "num_frames": max(1, int(dur * self.fps)),
                "curr_rms": norm_val(rms_energy, max_rms, t_start),
                "clip_rms": norm_range(rms_energy, max_rms, t_start, clip_end),
                "clip_bass": norm_range(bass_energy, max_bass, t_start, clip_end),
                "clip_highs": norm_range(highs_energy, max_highs, t_start, clip_end),
                "clip_mids": norm_range(mids_energy, max_mids, t_start, clip_end),
            }
            profile["music_energy"] = np.clip(
                (profile["clip_rms"] * 0.55) + (profile["clip_bass"] * 0.30) + (profile["clip_highs"] * 0.15),
                0,
                1,
            )
            profile["target_brightness"] = int(np.clip(
                ((1 - self.music_match) * profile["curr_rms"]) + (self.music_match * profile["music_energy"]),
                0,
                1,
            ) * 255)
            profile["target_brightness_norm"] = profile["target_brightness"] / 255.0
            profile["target_activity"] = np.clip(
                (profile["clip_rms"] * 0.45) + (profile["clip_bass"] * 0.40) + (profile["clip_highs"] * 0.15),
                0,
                1,
            )
            segment_profiles.append(profile)
        self.log(f"  Planning {len(segment_profiles)} segment(s)...")
        if self.progress_callback:
            self.progress_callback(-1, -1)
        activity_scale = self.motion_scale(activity_db)
        saved_log = self.log
        saved_random_state = random.getstate()
        saved_np_state = np.random.get_state()
        saved_current_vid_idx = self.current_vid_idx
        saved_recent_matches = {
            idx: deque(history, maxlen=history.maxlen)
            for idx, history in self.recent_matches.items()
        }
        saved_source_lock_penalties = list(self.source_lock_penalties)
        planned_segments = []
        match_quality_total = 0.0
        match_brightness_total = 0.0
        match_activity_total = 0.0
        match_segments = 0
        try:
            self.log = lambda *args, **kwargs: None
            planning_vid_idx = self.current_vid_idx
            planning_source_use_counts = [0 for _ in self.inputs]
            planning_source_streak = 0
            for i, profile in enumerate(segment_profiles):
                t_start = profile["t_start"]
                num_frames = profile["num_frames"]
                clip_rms = profile["clip_rms"]
                target_brightness = profile["target_brightness"]
                target_brightness_norm = profile["target_brightness_norm"]
                target_activity = profile["target_activity"]
                self.decay_source_lock_penalties()
                match = self.find_best_match(
                    target_brightness,
                    frame_db,
                    planning_vid_idx,
                    planning_source_use_counts,
                    activity_db,
                    target_activity,
                    num_frames,
                    activity_scale,
                    planning_source_streak,
                )
                if not match:
                    continue
                previous_vid_idx = planning_vid_idx
                planning_vid_idx, start_frame = match
                planning_source_streak = planning_source_streak + 1 if planning_vid_idx == previous_vid_idx else 1
                self.record_match(planning_vid_idx, start_frame)
                planning_source_use_counts[planning_vid_idx] += 1
                selected_brightness = self.segment_motion_score(brightness_db, planning_vid_idx, start_frame, num_frames)
                selected_activity = self.segment_motion_score(activity_db, planning_vid_idx, start_frame, num_frames)
                brightness_error = abs(selected_brightness - target_brightness_norm)
                activity_error = abs(selected_activity - target_activity)
                segment_quality = float(np.clip(1.0 - ((brightness_error + activity_error) * 0.5), 0, 1))
                match_quality_total += segment_quality
                match_brightness_total += float(np.clip(1.0 - brightness_error, 0, 1))
                match_activity_total += float(np.clip(1.0 - activity_error, 0, 1))
                match_segments += 1
                planned_segments.append({
                    "index": i,
                    "profile": profile,
                    "video_idx": planning_vid_idx,
                    "start_frame": start_frame,
                    "num_frames": num_frames,
                    "target_brightness_norm": target_brightness_norm,
                    "target_activity": target_activity,
                    "selected_brightness": selected_brightness,
                    "selected_activity": selected_activity,
                    "segment_quality": segment_quality,
                    "clip_rms": clip_rms,
                    "source_path": self.inputs[planning_vid_idx],
                })
        finally:
            self.log = saved_log
            random.setstate(saved_random_state)
            np.random.set_state(saved_np_state)
            self.current_vid_idx = saved_current_vid_idx
            self.recent_matches = saved_recent_matches
            self.source_lock_penalties = saved_source_lock_penalties

        self.log(f"  Planned {len(planned_segments)} segment(s); preparing output assembly...")
        transition_boundaries = []
        for i in range(max(0, len(planned_segments) - 1)):
            current = planned_segments[i]
            next_segment = planned_segments[i + 1]
            same_source = current["source_path"] == next_segment["source_path"]
            transition_mode = "Hard cut" if same_source else resolve_scene_transition_mode(
                self.scene_transition_mode,
                current["target_activity"],
                next_segment["target_activity"],
            )
            transition_frames = 0 if transition_mode == "Hard cut" else scene_transition_fade_frames(
                transition_mode,
                self.fps,
                current["num_frames"],
                next_segment["num_frames"],
            )
            transition_boundaries.append({
                "mode": transition_mode,
                "frames": transition_frames,
                "current_activity": current["target_activity"],
                "next_activity": next_segment["target_activity"],
                "same_source": same_source,
            })

        cap = cv2.VideoCapture(self.inputs[0])
        _, first_frame = cap.read()
        cap.release()
        source_height, source_width = first_frame.shape[:2]
        if self.output_resolution:
            width, height = self.output_resolution
        else:
            width, height = source_width, source_height
        self.log(f"  Output resolution: {width}x{height}")
        temp_video = None
        out = None
        segment_dir = None
        if self.export_mode == EXPORT_FINAL_VIDEO:
            temp_fd, temp_video = tempfile.mkstemp(suffix=".mp4", prefix="glitchsync_render_", dir=temp_root)
            os.close(temp_fd)
            out = LosslessVideoWriter(temp_video, width, height, self.fps)
        caps, prev_f = [cv2.VideoCapture(f) for f in self.inputs], None
        reference_stats = color_stats.get(self.color_reference_idx) if self.color_reference_idx is not None else None
        segments = []
        rendered_frames = 0
        if self.export_mode in (EXPORT_FINAL_VIDEO, EXPORT_CUT_AWARE_MLT, EXPORT_CLIP_MLT):
            package_dir = tempfile.mkdtemp(prefix="glitchsync_shotcut_", dir=temp_root)
            if self.export_mode == EXPORT_CLIP_MLT:
                clip_dir = os.path.join(package_dir, "media", "clips")
                os.makedirs(clip_dir, exist_ok=True)
            else:
                segment_dir = os.path.join(package_dir, "media", "segments")
                os.makedirs(segment_dir, exist_ok=True)

        self.log("Rendering...")
        if self.use_primary_video():
            self.log(f"  Primary focus: {os.path.basename(self.inputs[self.primary_video_idx])} ({self.primary_focus:.2f})")
        if self.source_variety > 0:
            self.log(f"  Source variety: {self.source_variety:.2f}")
        if self.music_match > 0:
            self.log(f"  Music match: {self.music_match:.2f}")
        if self.scene_transition_mode == "Auto":
            self.log("  Scene transitions: Auto (activity-driven cuts/fades)")
        else:
            self.log(f"  Scene transitions: {self.scene_transition_mode}")
        if reference_stats and self.color_match_strength > 0:
            self.log(f"  Color reference: {os.path.basename(self.inputs[self.color_reference_idx])} ({self.color_match_strength:.2f})")
        if not planned_segments:
            self.log("  No segment matches were selected.")
        for i, plan in enumerate(planned_segments):
            if self.stop_requested: break
            profile = plan["profile"]
            t_start = profile["t_start"]
            dur = profile["dur"]
            num_frames = plan["num_frames"]
            curr_rms = profile["curr_rms"]
            clip_rms = plan["clip_rms"]
            clip_bass = profile["clip_bass"]
            clip_highs = profile["clip_highs"]
            clip_mids = profile["clip_mids"]
            target_brightness = profile["target_brightness"]
            target_brightness_norm = plan["target_brightness_norm"]
            target_activity = plan["target_activity"]
            current_vid_idx = plan["video_idx"]
            start_frame = plan["start_frame"]
            self.current_vid_idx = current_vid_idx
            ai_segment_probability = self.ai_segment_probability(clip_rms)
            if self.debug_match_logging:
                self.log(
                    f"  Debug AI: segment {i + 1}/{len(planned_segments)} clip_rms={clip_rms:.2f} "
                    f"prob={ai_segment_probability:.2f}"
                )
            caps[current_vid_idx].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            chunk = []
            for _ in range(num_frames):
                r, f = caps[current_vid_idx].read()
                if r:
                    chunk.append(f)
            rewind_amount = self.effect_amount("rewind")
            if self.rewind and clip_rms > 0.75 and random.random() < min(1.0, rewind_amount):
                chunk = chunk[::-1]

            if self.verbose_match_logging:
                self.log(
                    f"  Match {i + 1}/{len(planned_segments)}: quality {plan['segment_quality']:.2f} "
                    f"(brightness {plan['selected_brightness']:.2f}/{target_brightness_norm:.2f}, "
                    f"activity {plan['selected_activity']:.2f}/{target_activity:.2f})"
                )
            transition_in = transition_boundaries[i - 1] if i > 0 and i - 1 < len(transition_boundaries) else None
            transition_out = transition_boundaries[i] if i < len(transition_boundaries) else None
            if self.debug_match_logging and transition_out:
                self.log(
                    f"  Debug transition {i + 1}->{i + 2}: {transition_out['mode'].lower()} "
                    f"({transition_out['current_activity']:.2f} -> {transition_out['next_activity']:.2f}, "
                    f"{transition_out['frames']} frames)"
                )
            pixelate_amount = self.effect_amount("pixelate")
            flash_amount = self.effect_amount("flash")
            rgb_shift_amount = self.effect_amount("rgb_shift")
            shake_amount = self.effect_amount("shake")
            ghosting_amount = self.effect_amount("ghosting")
            monochrome_amount = self.effect_amount("monochrome")
            hue_shift_amount = self.effect_amount("hue_shift")
            vignette_amount = self.effect_amount("vignette")
            static_pan_zoom_amount = self.effect_amount("static_pan_zoom")
            pixelate_frame_timing = self.resolve_effect_frame_timing("pixelate")
            flash_frame_timing = self.resolve_effect_frame_timing("flash")
            rgb_shift_frame_timing = self.resolve_effect_frame_timing("rgb_shift")
            shake_frame_timing = self.resolve_effect_frame_timing("shake")
            ghosting_frame_timing = self.resolve_effect_frame_timing("ghosting")
            monochrome_frame_timing = self.resolve_effect_frame_timing("monochrome")
            hue_shift_frame_timing = self.resolve_effect_frame_timing("hue_shift")
            vignette_frame_timing = self.resolve_effect_frame_timing("vignette")
            motion_score = self.segment_motion_score(motion_db, self.current_vid_idx, start_frame, len(chunk))
            use_static_pan_zoom = (
                self.static_pan_zoom
                and static_pan_zoom_amount > 0
                and motion_score <= STATIC_MOTION_THRESHOLD
                and len(chunk) > 1
            )
            transform_overscan = 1.0
            if use_static_pan_zoom:
                transform_overscan += 0.10 * np.clip(static_pan_zoom_amount, 0, 1)
            if self.shake and shake_amount > 0:
                max_shake_offset = max(0, (1.0 - 0.2) * self.sensitivity * shake_amount * 60)
                transform_overscan += min(0.12, (max_shake_offset * 2) / max(width, height))
            pan_x, pan_y = random.uniform(-1, 1), random.uniform(-1, 1)
            flash_decay_frames = max(3, int(self.fps * 0.12))
            flash_decay_until = -1
            flash_peak = 0.0
            ai_segment_active = (
                self.ai_stylization_active()
                and random.random() < ai_segment_probability
            )
            ai_anchor = None
            ai_anchor_frame_idx = -1
            segment_writer = None
            segment_path = None
            if self.export_mode in (EXPORT_FINAL_VIDEO, EXPORT_CUT_AWARE_MLT):
                segment_path = os.path.join(segment_dir, f"segment_{i + 1:04d}.mp4")
                segment_writer = LosslessVideoWriter(segment_path, width, height, self.fps)
            elif self.export_mode == EXPORT_CLIP_MLT:
                clip_name = f"clip_{len(segments) + 1:04d}.mp4"
                segment_path = os.path.join(clip_dir, clip_name)
                segment_writer = LosslessVideoWriter(segment_path, width, height, self.fps)
            for frame_idx, f in enumerate(chunk):
                if self.stop_requested:
                    break
                frame_time = t_start + (frame_idx / self.fps)
                frame_rms = norm_val(rms_energy, max_rms, frame_time)
                frame_bass = norm_val(bass_energy, max_bass, frame_time)
                frame_highs = norm_val(highs_energy, max_highs, frame_time)
                frame_mids = norm_val(mids_energy, max_mids, frame_time)
                pixelate_rms = frame_rms if pixelate_frame_timing else clip_rms
                pixelate_highs = frame_highs if pixelate_frame_timing else clip_highs
                flash_rms = frame_rms if flash_frame_timing else clip_rms
                flash_highs = frame_highs if flash_frame_timing else clip_highs
                rgb_shift_highs = frame_highs if rgb_shift_frame_timing else clip_highs
                shake_bass = frame_bass if shake_frame_timing else clip_bass
                ghosting_mids = frame_mids if ghosting_frame_timing else clip_mids
                monochrome_mids = frame_mids if monochrome_frame_timing else clip_mids
                hue_shift_highs = frame_highs if hue_shift_frame_timing else clip_highs
                vignette_bass = frame_bass if vignette_frame_timing else clip_bass
                f = fit_frame_to_output(f, width, height)
                f = apply_center_zoom(f, transform_overscan)
                if use_static_pan_zoom:
                    progress = frame_idx / max(1, len(chunk) - 1)
                    f = apply_static_pan_zoom(f, progress, static_pan_zoom_amount, pan_x, pan_y)
                if reference_stats and self.color_match_strength > 0:
                    f = match_lab_color(f, color_stats.get(self.current_vid_idx), reference_stats, self.color_match_strength)
                if self.pixelate and pixelate_amount > 0:
                    f = apply_pixelate(f, pixelate_rms, pixelate_highs, self.sensitivity * pixelate_amount, pixelate_amount)
                flash_can_trigger = (
                    self.flash
                    and flash_amount > 0
                    and flash_highs > 0.55
                    and flash_rms > 0.25
                    and (flash_frame_timing or frame_idx == 0)
                )
                if flash_can_trigger:
                    new_flash_peak = ((flash_highs - 0.55) / 0.45) * flash_rms
                    flash_peak = max(flash_peak, new_flash_peak) if frame_idx < flash_decay_until else new_flash_peak
                    flash_decay_until = frame_idx + flash_decay_frames
                if self.flash and flash_amount > 0 and frame_idx < flash_decay_until:
                    decay = (flash_decay_until - frame_idx) / flash_decay_frames
                    f = apply_flash_boost(f, flash_peak * decay, self.sensitivity * flash_amount)
                if self.rgb_shift and rgb_shift_amount > 0:
                    f = apply_rgb_shift(f, rgb_shift_highs, self.sensitivity * rgb_shift_amount)
                if self.shake and shake_amount > 0:
                    f = apply_shake(f, shake_bass, self.sensitivity * shake_amount)
                if self.ghosting and ghosting_amount > 0:
                    f = apply_ghosting(f, prev_f, ghosting_mids, self.sensitivity * ghosting_amount)
                if self.monochrome and monochrome_amount > 0:
                    f = apply_monochrome(f, monochrome_mids, self.sensitivity, monochrome_amount)
                if self.hue_shift and hue_shift_amount > 0:
                    f = apply_hue_shift(f, hue_shift_highs, self.sensitivity, hue_shift_amount)
                if self.vignette and vignette_amount > 0:
                    f = apply_vignette(f, vignette_bass, self.sensitivity, vignette_amount)
                if ai_segment_active:
                    if self.ai_segment_anchor_only:
                        if ai_anchor is None:
                            ai_anchor, ai_ok = self.ai_render_frame(f)
                            if ai_ok:
                                self.log(f"  AI anchor generated for segment {i + 1}/{len(planned_segments)}")
                            else:
                                ai_anchor = None
                    else:
                        should_refresh_ai = (
                            ai_anchor is None
                            or frame_idx == 0
                            or (frame_idx - ai_anchor_frame_idx) >= self.ai_every_n_frames
                        )
                        if should_refresh_ai:
                            ai_anchor, ai_ok = self.ai_render_frame(f)
                            if ai_ok:
                                ai_anchor_frame_idx = frame_idx
                                self.log(f"  AI frame refreshed at segment {i + 1}/{len(planned_segments)} frame {frame_idx + 1}/{len(chunk)}")
                            else:
                                ai_anchor = None
                    if ai_anchor is not None and self.ai_blend > 0:
                        if self.ai_segment_anchor_only:
                            f = cv2.addWeighted(f, 1 - self.ai_blend, ai_anchor, self.ai_blend, 0)
                        else:
                            since_anchor = max(0, frame_idx - ai_anchor_frame_idx)
                            interval_progress = np.clip(since_anchor / max(1, self.ai_every_n_frames), 0, 1)
                            blend = self.ai_blend * (1.0 - (interval_progress * 0.5))
                            if blend > 0:
                                f = cv2.addWeighted(f, 1 - blend, ai_anchor, blend, 0)
                if self.lut is not None:
                    f = apply_cube_lut(f, self.lut)
                if self.force_bw:
                    f = force_black_and_white(f)
                segment_frame = f
                if out is not None:
                    transition_alpha = 1.0
                    if transition_in and transition_in["frames"] > 0 and frame_idx < transition_in["frames"]:
                        if transition_in["frames"] > 1:
                            transition_alpha = min(transition_alpha, frame_idx / (transition_in["frames"] - 1))
                    if transition_out and transition_out["frames"] > 0 and frame_idx >= (len(chunk) - transition_out["frames"]):
                        if transition_out["frames"] > 0:
                            transition_alpha = min(transition_alpha, (len(chunk) - frame_idx - 1) / transition_out["frames"])
                    if transition_alpha < 1.0:
                        f = cv2.convertScaleAbs(f, alpha=max(0.0, transition_alpha), beta=0)
                    out.write(f)
                if segment_writer is not None:
                    segment_writer.write(segment_frame)
                prev_f = f.copy()

            if segment_writer is not None:
                segment_writer.release()

            if chunk and not self.stop_requested:
                frame_count = len(chunk)
                segment = {
                    "timeline_in": rendered_frames,
                    "timeline_out": rendered_frames + frame_count - 1,
                    "frame_count": frame_count,
                    "transition_in": transition_in,
                    "transition_out": transition_out,
                }
                if self.export_mode == EXPORT_CUT_AWARE_MLT:
                    segment["producer"] = f"video{len(segments)}"
                    segment["resource"] = f"media/segments/{os.path.basename(segment_path)}"
                    segment["in"] = 0
                    segment["out"] = frame_count - 1
                elif self.export_mode == EXPORT_CLIP_MLT:
                    segment["producer"] = f"video{len(segments)}"
                    segment["resource"] = f"media/clips/{os.path.basename(segment_path)}"
                    segment["in"] = 0
                    segment["out"] = frame_count - 1
                elif self.export_mode == EXPORT_FINAL_VIDEO:
                    segment["resource"] = segment_path
                segments.append(segment)
                rendered_frames += frame_count

            if self.frame_callback and chunk:
                self.frame_callback(chunk[0])

            if self.progress_callback:
                self.progress_callback(i, len(planned_segments))
            
        for c in caps: c.release()
        if out is not None:
            out.release()
        if not self.stop_requested:
            self.log("Exporting final stage...")
            if self.progress_callback: self.progress_callback(-1, -1)
            if self.export_mode == EXPORT_FINAL_VIDEO:
                self.export_final_video_from_segments(segments, width, height, package_dir)
            else:
                self.export_shotcut_archive(temp_video, segments, width, height, package_dir)
            if match_segments:
                self.log(
                    f"Source match summary: quality {match_quality_total / match_segments:.2f}, "
                    f"brightness {match_brightness_total / match_segments:.2f}, "
                    f"activity {match_activity_total / match_segments:.2f} over {match_segments} segments"
                )
        if temp_video and os.path.exists(temp_video): os.remove(temp_video)
        if package_dir and os.path.exists(package_dir): shutil.rmtree(package_dir)
        if self.stop_requested:
            self.log("Stopped.")
            return False
        self.log("Done!")
        return True

    def export_final_video(self, temp_video):
        root, ext = os.path.splitext(self.output)
        final_temp = f"{root or self.output}.tmp_{random.randint(1000, 9999)}{ext or '.mp4'}"
        preset, crf = self.export_quality_args()
        self.log(f"  Export quality: {self.export_quality_label} (preset {preset}, CRF {crf})")
        result = subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-i', self.audio, '-c:v', 'libx264', '-preset', preset, '-crf', crf, '-c:a', 'aac', '-b:a', '192k', '-shortest', final_temp], capture_output=True)
        if result.returncode != 0:
            if os.path.exists(final_temp):
                os.remove(final_temp)
            raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while creating final video")
        os.replace(final_temp, self.output)

    def _concat_video_segments(self, input_paths, output_path):
        if not input_paths:
            raise RuntimeError("No segment clips available for concatenation")
        if len(input_paths) == 1:
            shutil.copy2(input_paths[0], output_path)
            return
        list_file = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8")
        try:
            for path in input_paths:
                list_file.write(ffmpeg_concat_list_line(path) + "\n")
            list_file.flush()
            list_file.close()
            result = subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file.name,
                '-c', 'copy',
                '-an',
                output_path
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while concatenating a segment block")
        finally:
            try:
                os.remove(list_file.name)
            except Exception:
                pass

    def export_final_video_from_segments(self, segments, width, height, package_dir):
        if not segments:
            raise RuntimeError("No rendered segments to export")

        preset, crf = self.export_quality_args()
        self.log(f"  Export quality: {self.export_quality_label} (preset {preset}, CRF {crf})")
        work_dir = os.path.join(package_dir, "media")
        block_dir = os.path.join(work_dir, "blocks")
        os.makedirs(block_dir, exist_ok=True)

        blocks = []
        current_block = [segments[0]]
        for idx in range(len(segments) - 1):
            transition = segments[idx].get("transition_out") or {}
            if transition.get("mode") == "Hard cut":
                current_block.append(segments[idx + 1])
                continue
            blocks.append({
                "segments": current_block,
                "transition": transition,
            })
            current_block = [segments[idx + 1]]
        blocks.append({"segments": current_block, "transition": None})

        block_paths = []
        block_durations = []
        self.log(f"  Assembling {len(blocks)} transition block(s)...")
        block_worker_count = min(max(1, os.cpu_count() or 1), max(1, min(len(blocks), 4)))
        if block_worker_count > 1 and len(blocks) > 1:
            self.log(f"  Parallel block assembly with {block_worker_count} workers...")
        block_results = [None] * len(blocks)
        with ThreadPoolExecutor(max_workers=block_worker_count) as executor:
            future_to_idx = {}
            for block_idx, block in enumerate(blocks):
                block_segments = block["segments"]
                block_paths_input = [segment["resource"] for segment in block_segments]
                block_path = os.path.join(block_dir, f"block_{block_idx + 1:04d}.mp4")
                self.log(f"    Building block {block_idx + 1}/{len(blocks)} from {len(block_segments)} segment(s)...")
                future_to_idx[executor.submit(self._concat_video_segments, block_paths_input, block_path)] = (
                    block_idx,
                    block_path,
                    sum(segment["frame_count"] / max(self.fps, 1) for segment in block_segments),
                )
            for future in as_completed(future_to_idx):
                block_idx, block_path, block_duration = future_to_idx[future]
                future.result()
                block_results[block_idx] = (block_path, block_duration)
        for block_path, block_duration in block_results:
            block_paths.append(block_path)
            block_durations.append(block_duration)

        if len(block_paths) == 1:
            temp_video = block_paths[0]
        else:
            self.log("  Applying dissolve transitions between blocks...")
            xfade_inputs = []
            filter_parts = []
            output_label = None
            output_cursor = block_durations[0]
            for block_idx, block_path in enumerate(block_paths):
                xfade_inputs.extend(['-i', block_path])
            for idx in range(len(block_paths) - 1):
                transition = blocks[idx].get("transition") or {}
                mode = transition.get("mode", "Fast fade")
                xfade_name = scene_transition_ffmpeg_name(mode)
                if not xfade_name:
                    xfade_name = "fade"
                transition_frames = max(1, int(transition.get("frames", 1)))
                duration_seconds = transition_frames / max(self.fps, 1)
                offset = max(0.0, output_cursor - duration_seconds)
                output_label = f"v{idx + 1}"
                if idx == 0:
                    filter_parts.append(
                        f"[0:v][1:v]xfade=transition={xfade_name}:duration={duration_seconds:.6f}:offset={offset:.6f}[{output_label}]"
                    )
                else:
                    prev_label = f"v{idx}"
                    filter_parts.append(
                        f"[{prev_label}][{idx + 1}:v]xfade=transition={xfade_name}:duration={duration_seconds:.6f}:offset={offset:.6f}[{output_label}]"
                    )
                output_cursor = output_cursor + block_durations[idx + 1] - duration_seconds
            temp_video = os.path.join(work_dir, f"glitchsync_dissolve_{random.randint(1000, 9999)}.mp4")
            result = subprocess.run([
                'ffmpeg', '-y',
                *xfade_inputs,
                '-filter_complex', ';'.join(filter_parts),
                '-map', f"[{output_label}]",
                '-an',
                '-c:v', 'libx264',
                '-preset', preset,
                '-crf', crf,
                '-pix_fmt', 'yuv420p',
                temp_video,
            ], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while assembling dissolve transitions")

        root, ext = os.path.splitext(self.output)
        final_temp = f"{root or self.output}.tmp_{random.randint(1000, 9999)}{ext or '.mp4'}"
        self.log("  Muxing audio...")
        result = subprocess.run([
            'ffmpeg', '-y', '-i', temp_video,
            '-i', self.audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            final_temp,
        ], capture_output=True)
        if result.returncode != 0:
            if os.path.exists(final_temp):
                os.remove(final_temp)
            raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while muxing final audio")
        os.replace(final_temp, self.output)

    def export_shotcut_archive(self, temp_video, segments, width, height, package_dir=None):
        if not segments:
            raise RuntimeError("No rendered segments to export")

        output_zip = output_path_for_mode(self.output, self.export_mode)
        created_package_dir = package_dir is None
        if package_dir is None:
            package_dir = tempfile.mkdtemp(prefix="glitchsync_shotcut_")
        try:
            media_dir = os.path.join(package_dir, "media")
            os.makedirs(media_dir, exist_ok=True)

            audio_name = os.path.basename(self.audio)
            audio_resource = os.path.join("media", audio_name).replace(os.sep, "/")
            shutil.copy2(self.audio, os.path.join(media_dir, audio_name))

            if self.export_mode == EXPORT_CUT_AWARE_MLT:
                video_resources = {segment["producer"]: segment["resource"] for segment in segments}
            elif self.export_mode == EXPORT_CLIP_MLT:
                video_resources = {segment["producer"]: segment["resource"] for segment in segments}
            else:
                raise RuntimeError(f"Unsupported Shotcut export mode: {self.export_mode}")

            project_path = os.path.join(package_dir, "glitchsync_project.mlt")
            if self.export_mode == EXPORT_CUT_AWARE_MLT:
                self.log("  Shotcut export: using rendered segment chunks with transition metadata")
            write_shotcut_mlt(
                project_path,
                video_resources,
                audio_resource,
                segments,
                self.fps,
                width,
                height,
                rendered_segments=self.export_mode == EXPORT_CUT_AWARE_MLT,
            )
            zip_directory(package_dir, output_zip)
            self.log(f"Shotcut archive written: {output_zip}")
        finally:
            if created_package_dir and os.path.exists(package_dir):
                shutil.rmtree(package_dir)

# --- GUI ---

class GlitchGUI:
    def __init__(self, root):
        self.root = root
        self.current_project_path = None
        self.root.title("GlitchSync Pro v3.8")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        initial_w = min(1180, max(1100, screen_w - 60))
        initial_h = min(1500, max(1100, screen_h - 40))
        self.root.geometry(f"{initial_w}x{initial_h}")
        self.root.minsize(1100, 1040)
        self.inputs, self.audio = [], tk.StringVar()
        self.output = tk.StringVar(value=f"glitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        self.output_auto_managed = True
        self.suppress_output_trace = False
        self.output.trace_add("write", self.on_output_changed)
        self.export_mode_label = tk.StringVar(value="Final video (MP4)")
        self.output_resolution_label = tk.StringVar(value="Auto (first input)")
        self.export_quality_label = tk.StringVar(value="High quality (slower)")
        self.increment_output_if_exists = tk.BooleanVar(value=False)
        self.verbose_match_logging = tk.BooleanVar(value=False)
        self.debug_match_logging = tk.BooleanVar(value=False)
        self.render_seed = tk.StringVar(value="")
        self.experimental_mode = tk.BooleanVar(value=False)
        self.ai_panel_visible = self.experimental_mode
        self.ai_dream_chance = tk.DoubleVar(value=0.25)
        self.ai_dream_timing = tk.StringVar(value="Clip")
        self.ai_enabled = tk.BooleanVar(value=False)
        self.ai_segment_anchor_only = tk.BooleanVar(value=True)
        self.ai_backend_url = tk.StringVar(value=AI_DEFAULT_BACKEND_URL)
        self.ai_prompt = tk.StringVar(value=AI_DEFAULT_PROMPT)
        self.ai_negative_prompt = tk.StringVar(value=AI_DEFAULT_NEGATIVE_PROMPT)
        self.ai_every_n_frames = tk.IntVar(value=AI_DEFAULT_EVERY_N_FRAMES)
        self.ai_denoise = tk.DoubleVar(value=AI_DEFAULT_DENOISE)
        self.ai_denoise_label = None
        self.ai_cfg_scale = tk.DoubleVar(value=AI_DEFAULT_CFG_SCALE)
        self.ai_cfg_scale_label = None
        self.ai_steps = tk.IntVar(value=AI_DEFAULT_STEPS)
        self.ai_max_dim = tk.IntVar(value=AI_DEFAULT_MAX_DIM)
        self.ai_blend = tk.DoubleVar(value=AI_DEFAULT_BLEND)
        self.ai_blend_label = None
        self.duration, self.fps, self.coherence, self.sensitivity = tk.DoubleVar(value=0.10), tk.IntVar(value=30), tk.DoubleVar(value=0.20), tk.DoubleVar(value=1.0)
        self.source_variety = tk.DoubleVar(value=0.0)
        self.source_variety_label = None
        self.music_match = tk.DoubleVar(value=0.35)
        self.music_match_label = None
        self.scene_transition_mode = tk.StringVar(value="Auto")
        self.scene_transition_mode_label = None
        self.color_match_enabled = tk.BooleanVar(value=False)
        self.color_match_strength = tk.DoubleVar(value=0.5)
        self.color_match_strength_label = None
        self.force_bw = tk.BooleanVar(value=False)
        self.color_reference_idx = 0
        self.color_reference_label = tk.StringVar(value="Reference: first input")
        self.lut_path = tk.StringVar()
        self.render_mode = tk.StringVar(value="Full")
        self.snippet_duration = tk.DoubleVar(value=30.0)
        self.beat_step = tk.IntVar(value=4)
        self.beat_variation = tk.DoubleVar(value=0.0)
        self.beat_variation_label = None
        self.effect_amounts = {name: tk.DoubleVar(value=DEFAULT_EFFECT_AMOUNTS[name]) for name in EFFECT_NAMES}
        self.effect_timing = {name: tk.StringVar(value=DEFAULT_EFFECT_TIMING[name]) for name in EFFECT_NAMES}
        self.effect_amount_labels = {}
        self.primary_enabled = tk.BooleanVar(value=False)
        self.primary_focus = tk.DoubleVar(value=0.75)
        self.primary_focus_label = None
        self.primary_video_idx = 0
        self.primary_video_label = tk.StringVar(value="Primary: first input")
        self.beat_sync = tk.BooleanVar(value=True)
        self.pixelate, self.flash, self.rewind, self.rgb_shift, self.shake, self.ghosting = [tk.BooleanVar(value=True) for _ in range(6)]
        self.monochrome = tk.BooleanVar(value=False)
        self.hue_shift = tk.BooleanVar(value=False)
        self.vignette = tk.BooleanVar(value=False)
        self.static_pan_zoom = tk.BooleanVar(value=False)
        self.style_name = tk.StringVar(value=DEFAULT_STYLE_NAME)
        self.style_choice = tk.StringVar()
        self.style_combo = None
        self.styles = self.load_styles_file()
        self.tooltips = []
        self.recent_projects = self.load_recent_projects()
        self.recent_projects_menu = None
        self.render_seed_status_label = None
        self.last_render_seed = None
        self.last_render_output_path = None
        self.last_progress_val = 0
        self.active_processor = None
        self.render_was_stopped = False
        self.build_menu()
        self.build_ui()
        self.refresh_style_choices()

    def on_output_changed(self, *args):
        if not self.suppress_output_trace:
            self.output_auto_managed = False

    def set_output(self, value, auto_managed=None):
        self.suppress_output_trace = True
        try:
            self.output.set(value)
        finally:
            self.suppress_output_trace = False
        if auto_managed is not None:
            self.output_auto_managed = auto_managed

    def default_output_for_project(self, project_path):
        folder = os.path.dirname(project_path)
        name = os.path.basename(project_path)
        if name.endswith(".glitchsync.json"):
            stem = name[:-len(".glitchsync.json")]
        else:
            stem = os.path.splitext(name)[0]
        return os.path.join(folder, f"{stem}.mp4")

    def sync_auto_output_to_project(self, project_path):
        if self.output_auto_managed:
            self.set_output(self.default_output_for_project(project_path), True)

    def build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_project)
        file_menu.add_command(label="Load Project...", command=self.load_project)
        self.recent_projects_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self.recent_projects_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_project)
        file_menu.add_command(label="Save As...", command=self.save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Analysis Cache...", command=self.clear_analysis_cache)
        menubar.add_cascade(label="File", menu=file_menu)
        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_checkbutton(
            label="Experimental mode",
            variable=self.experimental_mode,
            command=self.toggle_experimental_mode,
        )
        menubar.add_cascade(label="Options", menu=options_menu)
        self.root.config(menu=menubar)
        self.refresh_recent_projects_menu()

    def clear_analysis_cache(self):
        if not os.path.exists(ANALYSIS_CACHE_DIR):
            messagebox.showinfo("Analysis Cache", "No analysis cache exists yet.")
            return
        if not messagebox.askyesno("Clear Analysis Cache", "Delete cached video and audio analysis files?"):
            return
        try:
            shutil.rmtree(ANALYSIS_CACHE_DIR)
            messagebox.showinfo("Analysis Cache", "Analysis cache cleared.")
        except Exception as e:
            messagebox.showerror("Analysis Cache", f"Could not clear analysis cache:\n{e}")

    def load_recent_projects(self):
        try:
            with open(RECENT_PROJECTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            recent = []
            seen = set()
            for path in data:
                if not isinstance(path, str):
                    continue
                abs_path = os.path.abspath(path)
                if abs_path in seen or not os.path.exists(abs_path):
                    continue
                seen.add(abs_path)
                recent.append(abs_path)
            return recent[:RECENT_PROJECTS_LIMIT]
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def save_recent_projects(self):
        try:
            with open(RECENT_PROJECTS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.recent_projects[:RECENT_PROJECTS_LIMIT], f, indent=2)
        except Exception as e:
            self.log_msg(f"Could not save recent projects: {e}")

    def refresh_recent_projects_menu(self):
        if not self.recent_projects_menu:
            return
        self.recent_projects_menu.delete(0, tk.END)
        if not self.recent_projects:
            self.recent_projects_menu.add_command(label="(No recent projects)", state=tk.DISABLED)
            return
        for path in self.recent_projects[:RECENT_PROJECTS_LIMIT]:
            label = os.path.basename(path)
            self.recent_projects_menu.add_command(
                label=label,
                command=lambda p=path: self.load_project_path(p),
            )

    def add_recent_project(self, path):
        if not path:
            return
        abs_path = os.path.abspath(path)
        recent = [p for p in self.recent_projects if os.path.abspath(p) != abs_path]
        recent.insert(0, abs_path)
        self.recent_projects = recent[:RECENT_PROJECTS_LIMIT]
        self.save_recent_projects()
        self.refresh_recent_projects_menu()

    def load_project_path(self, path):
        if not path:
            return
        path = os.path.abspath(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                project = json.load(f)
            self.apply_project(project)
            self.current_project_path = path
            self.update_project_title()
            self.add_recent_project(path)
        except Exception as e:
            return messagebox.showerror("Error", f"Could not load project: {e}")
        messagebox.showinfo("Loaded", f"Loaded project: {os.path.basename(path)}")

    def build_ui(self):
        m = ttk.Frame(self.root, padding="15"); m.pack(fill=tk.BOTH, expand=True)
        m.columnconfigure(0, weight=1)
        m.columnconfigure(1, weight=1)
        m.rowconfigure(0, weight=0)
        m.rowconfigure(1, weight=0)
        m.rowconfigure(2, weight=0)
        m.rowconfigure(3, weight=1)
        m.rowconfigure(4, weight=0)
        m.rowconfigure(5, weight=0)
        
        io = ttk.LabelFrame(m, text="Files", padding="10"); io.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        io.columnconfigure(1, weight=1)
        ttk.Label(io, text="Seed:").grid(row=0, column=0, sticky="w")
        seed_controls = ttk.Frame(io); seed_controls.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 5))
        seed_controls.columnconfigure(0, weight=1)
        render_seed_entry = ttk.Entry(seed_controls, textvariable=self.render_seed)
        render_seed_entry.grid(row=0, column=0, sticky="ew")
        seed_copy_button = ttk.Button(seed_controls, text="Copy", command=self.copy_render_seed)
        seed_copy_button.grid(row=0, column=1, padx=(5, 0))
        seed_randomize_button = ttk.Button(seed_controls, text="Randomize", command=self.randomize_render_seed)
        seed_randomize_button.grid(row=0, column=2, padx=(5, 0))
        use_last_seed_button = ttk.Button(seed_controls, text="Use Last", command=self.use_last_render_seed)
        use_last_seed_button.grid(row=0, column=3, padx=(5, 0))
        self.render_seed_status_label = ttk.Label(seed_controls, text="Auto", width=5)
        self.render_seed_status_label.grid(row=0, column=4, padx=(8, 0))

        ttk.Label(io, text="Inputs:").grid(row=1, column=0, sticky="nw")
        self.lb = tk.Listbox(io, height=5, selectmode=tk.EXTENDED); self.lb.grid(row=1, column=1, sticky="ew", padx=5)
        input_buttons = ttk.Frame(io); input_buttons.grid(row=1, column=2, sticky="n")
        ttk.Button(input_buttons, text="+ Add", command=self.add_v).pack(fill=tk.X)
        ttk.Button(input_buttons, text="Remove", command=self.remove_selected_inputs).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(input_buttons, text="Clean Missing", command=self.clean_missing_inputs).pack(fill=tk.X, pady=(5, 0))
        ttk.Label(io, text="Audio:").grid(row=2, column=0, pady=5)
        ttk.Entry(io, textvariable=self.audio).grid(row=2, column=1, sticky="ew")
        ttk.Button(io, text="...", command=self.add_a).grid(row=2, column=2)
        ttk.Label(io, text="Export:").grid(row=3, column=0)
        export_mode_combo = ttk.Combobox(io, textvariable=self.export_mode_label, values=list(EXPORT_MODE_LABELS.keys()), state="readonly")
        export_mode_combo.grid(row=3, column=1, sticky="ew")
        ttk.Label(io, text="Resolution:").grid(row=4, column=0)
        ttk.Combobox(io, textvariable=self.output_resolution_label, values=list(OUTPUT_RESOLUTION_LABELS.keys()), state="readonly").grid(row=4, column=1, sticky="ew")
        ttk.Label(io, text="Quality:").grid(row=5, column=0)
        ttk.Combobox(io, textvariable=self.export_quality_label, values=list(EXPORT_QUALITY_LABELS.keys()), state="readonly").grid(row=5, column=1, sticky="ew")
        ttk.Label(io, text="Render length:").grid(row=6, column=0, sticky="w")
        render_length_controls = ttk.Frame(io); render_length_controls.grid(row=6, column=1, sticky="w", pady=5)
        ttk.Combobox(render_length_controls, textvariable=self.render_mode, values=["Full", "Snippet"], state="readonly", width=10).pack(side=tk.LEFT)
        ttk.Spinbox(render_length_controls, from_=1, to=3600, textvariable=self.snippet_duration, width=7).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(render_length_controls, text="sec").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(io, text="Out:").grid(row=7, column=0)
        ttk.Entry(io, textvariable=self.output).grid(row=7, column=1, sticky="ew")
        output_name_mode = ttk.Checkbutton(io, text="Increment if exists", variable=self.increment_output_if_exists)
        output_name_mode.grid(row=8, column=1, sticky="w")
        ttk.Checkbutton(io, text="Primary focus", variable=self.primary_enabled).grid(row=9, column=0, sticky="w")
        primary_controls = ttk.Frame(io); primary_controls.grid(row=9, column=1, sticky="w", pady=5)
        ttk.Button(primary_controls, text="Set Selected", command=self.set_primary_video).pack(side=tk.LEFT)
        ttk.Label(primary_controls, textvariable=self.primary_video_label).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(io, text="Focus:").grid(row=10, column=0, sticky="w")
        primary_focus_controls = ttk.Frame(io); primary_focus_controls.grid(row=10, column=1, sticky="ew")
        primary_focus_controls.columnconfigure(0, weight=1)
        ttk.Scale(primary_focus_controls, from_=0.0, to=1.0, variable=self.primary_focus, command=lambda e: self.update_primary_focus_label()).grid(row=0, column=0, sticky="ew")
        self.primary_focus_label = ttk.Label(primary_focus_controls, text="0.75", width=5); self.primary_focus_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(io, text="Color match", variable=self.color_match_enabled).grid(row=11, column=0, sticky="w")
        color_ref_controls = ttk.Frame(io); color_ref_controls.grid(row=11, column=1, sticky="w", pady=5)
        ttk.Button(color_ref_controls, text="Set Selected", command=self.set_color_reference_video).pack(side=tk.LEFT)
        ttk.Label(color_ref_controls, textvariable=self.color_reference_label).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(io, text="Ref match:").grid(row=12, column=0, sticky="w")
        color_match_controls = ttk.Frame(io); color_match_controls.grid(row=12, column=1, sticky="ew")
        color_match_controls.columnconfigure(0, weight=1)
        ttk.Scale(color_match_controls, from_=0.0, to=1.0, variable=self.color_match_strength, command=lambda e: self.update_color_match_strength_label()).grid(row=0, column=0, sticky="ew")
        self.color_match_strength_label = ttk.Label(color_match_controls, text="0.50", width=5); self.color_match_strength_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(io, text="LUT:").grid(row=13, column=0, sticky="w")
        lut_controls = ttk.Frame(io); lut_controls.grid(row=13, column=1, sticky="ew", pady=5)
        lut_controls.columnconfigure(0, weight=1)
        ttk.Entry(lut_controls, textvariable=self.lut_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(lut_controls, text="...", command=self.pick_lut, width=3).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(lut_controls, text="Clear", command=self.clear_lut).grid(row=0, column=2, padx=(5, 0))
        force_bw_cb = ttk.Checkbutton(io, text="Force black & white", variable=self.force_bw)
        force_bw_cb.grid(row=14, column=0, columnspan=3, sticky="w")
        self.add_tooltip(render_seed_entry, "Leave blank for a fresh random seed. Enter a number to reproduce the same render later.")
        self.add_tooltip(seed_copy_button, "Copy the current seed to the clipboard.")
        self.add_tooltip(seed_randomize_button, "Fill the seed field with a new random value.")
        self.add_tooltip(use_last_seed_button, "Restore the seed from the most recent render in this session.")
        self.add_tooltip(export_mode_combo, "Full Timeline bakes GlitchSync effects and transitions into rendered segment clips for a multi-track Shotcut project. Source Clips keeps the project editable from original clips, but does not include the full rendered transition look.")
        self.add_tooltip(force_bw_cb, "Convert the final output to black and white after all other effects and color matching have been applied.")
        self.render_seed.trace_add("write", lambda *_: self.update_render_seed_status())
        self.update_render_seed_status()

        right_top = ttk.Frame(m)
        right_top.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        right_top.columnconfigure(0, weight=1)
        right_top.rowconfigure(0, weight=0)
        right_top.rowconfigure(1, weight=0)
        right_top.rowconfigure(2, weight=0)
        pv = ttk.LabelFrame(right_top, text="Live Preview & Review", padding="8"); pv.grid(row=0, column=0, sticky="ew")
        self.cv = tk.Canvas(pv, width=480, height=220, bg="black"); self.cv.pack(pady=(2, 4), fill=tk.X)
        self.rv_btn = ttk.Button(pv, text="REVIEW WITH AUDIO", command=self.review_render, state=tk.DISABLED); self.rv_btn.pack(fill=tk.X)
        ttk.Label(pv, text="Click REVIEW to watch with full audio sync.", wraplength=430, justify=tk.CENTER).pack(pady=(4, 2))
        resource_f = ttk.LabelFrame(right_top, text="System Resources", padding="6")
        resource_f.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        resource_f.columnconfigure(1, weight=1)
        self.resource_cpu_label = ttk.Label(resource_f, text="CPU:")
        self.resource_cpu_label.grid(row=0, column=0, sticky="w")
        self.resource_cpu_bar = ttk.Progressbar(resource_f, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.resource_cpu_bar.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.resource_cpu_value = ttk.Label(resource_f, text="0.0%", width=18)
        self.resource_cpu_value.grid(row=0, column=2, sticky="e")
        self.resource_ram_label = ttk.Label(resource_f, text="RAM:")
        self.resource_ram_label.grid(row=1, column=0, sticky="w")
        self.resource_ram_bar = ttk.Progressbar(resource_f, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.resource_ram_bar.grid(row=1, column=1, sticky="ew", padx=(8, 8))
        self.resource_ram_value = ttk.Label(resource_f, text="0.0%", width=18)
        self.resource_ram_value.grid(row=1, column=2, sticky="e")
        self.resource_gpu_label = ttk.Label(resource_f, text="GPU:")
        self.resource_gpu_label.grid(row=2, column=0, sticky="w")
        self.resource_gpu_bar = ttk.Progressbar(resource_f, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.resource_gpu_bar.grid(row=2, column=1, sticky="ew", padx=(8, 8))
        self.resource_gpu_value = ttk.Label(resource_f, text="0.0%", width=18)
        self.resource_gpu_value.grid(row=2, column=2, sticky="e")
        self.resource_temp_label = ttk.Label(resource_f, text="Disk:")
        self.resource_temp_label.grid(row=3, column=0, sticky="w")
        self.resource_temp_bar = ttk.Progressbar(resource_f, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.resource_temp_bar.grid(row=3, column=1, sticky="ew", padx=(8, 8))
        self.resource_temp_value = ttk.Label(resource_f, text="0.0%", width=18)
        self.resource_temp_value.grid(row=3, column=2, sticky="e")
        self.resource_temp_type = ttk.Label(resource_f, text="")
        self.resource_temp_type.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.temp_storage_root = preferred_temp_root() or tempfile.gettempdir()
        self.init_resource_monitor()

        set_f = ttk.LabelFrame(m, text="Parameters", padding="10"); set_f.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        set_f.columnconfigure(1, weight=1)
        beat_sync_cb = ttk.Checkbutton(set_f, text="Beat Sync", variable=self.beat_sync)
        beat_sync_cb.grid(row=0, column=0, sticky="w")
        beat_interval_label = ttk.Label(set_f, text="Beat interval:")
        beat_interval_label.grid(row=1, column=0)
        beat_interval_spin = ttk.Spinbox(set_f, from_=1, to=64, textvariable=self.beat_step, width=5)
        beat_interval_spin.grid(row=1, column=1, sticky="w")
        ttk.Label(set_f, text="beats").grid(row=1, column=2, sticky="w")
        beat_variation_label = ttk.Label(set_f, text="Beat variation:")
        beat_variation_label.grid(row=2, column=0)
        beat_variation_scale = ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.beat_variation, command=lambda e: self.update_beat_variation_label())
        beat_variation_scale.grid(row=2, column=1, sticky="ew")
        self.beat_variation_label = ttk.Label(set_f, text="0.00"); self.beat_variation_label.grid(row=2, column=2)
        duration_label = ttk.Label(set_f, text="Duration (no Beat Sync):")
        duration_label.grid(row=3, column=0)
        duration_scale = ttk.Scale(set_f, from_=0.0, to=8.0, variable=self.duration, command=lambda e: self.l_dur.config(text=f"{self.duration.get():.2f}s"))
        duration_scale.grid(row=3, column=1, sticky="ew")
        self.l_dur = ttk.Label(set_f, text="0.10s"); self.l_dur.grid(row=3, column=2)
        coherence_label = ttk.Label(set_f, text="Coherence:")
        coherence_label.grid(row=4, column=0)
        coherence_scale = ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.coherence, command=lambda e: self.l_coh.config(text=f"{self.coherence.get():.2f}"))
        coherence_scale.grid(row=4, column=1, sticky="ew")
        self.l_coh = ttk.Label(set_f, text="0.20"); self.l_coh.grid(row=4, column=2)
        source_variety_label = ttk.Label(set_f, text="Source variety:")
        source_variety_label.grid(row=5, column=0)
        source_variety_scale = ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.source_variety, command=lambda e: self.update_source_variety_label())
        source_variety_scale.grid(row=5, column=1, sticky="ew")
        self.source_variety_label = ttk.Label(set_f, text="0.00"); self.source_variety_label.grid(row=5, column=2)
        music_match_label = ttk.Label(set_f, text="Music match:")
        music_match_label.grid(row=6, column=0)
        music_match_scale = ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.music_match, command=lambda e: self.update_music_match_label())
        music_match_scale.grid(row=6, column=1, sticky="ew")
        self.music_match_label = ttk.Label(set_f, text="0.35"); self.music_match_label.grid(row=6, column=2)
        sensitivity_label = ttk.Label(set_f, text="Sensitivity:")
        sensitivity_label.grid(row=8, column=0)
        sensitivity_scale = ttk.Scale(set_f, from_=0.1, to=3.0, variable=self.sensitivity, command=lambda e: self.l_sen.config(text=f"{self.sensitivity.get():.2f}"))
        sensitivity_scale.grid(row=8, column=1, sticky="ew")
        self.l_sen = ttk.Label(set_f, text="1.00"); self.l_sen.grid(row=8, column=2)
        scene_transition_label = ttk.Label(set_f, text="Scene transitions:")
        scene_transition_label.grid(row=9, column=0)
        scene_transition_combo = ttk.Combobox(set_f, textvariable=self.scene_transition_mode, values=SCENE_TRANSITION_LABELS, state="readonly")
        scene_transition_combo.grid(row=9, column=1, sticky="ew")
        fps_label = ttk.Label(set_f, text="FPS:")
        fps_label.grid(row=10, column=0)
        fps_spin = ttk.Spinbox(set_f, from_=1, to=120, textvariable=self.fps, width=5)
        fps_spin.grid(row=10, column=1, sticky="w")
        style_name_label = ttk.Label(set_f, text="Style name:")
        style_name_label.grid(row=11, column=0)
        style_name_entry = ttk.Entry(set_f, textvariable=self.style_name)
        style_name_entry.grid(row=11, column=1, sticky="ew")
        ttk.Button(set_f, text="Save Style", command=self.save_named_style).grid(row=11, column=2, sticky="ew")
        load_style_label = ttk.Label(set_f, text="Load style:")
        load_style_label.grid(row=12, column=0)
        self.style_combo = ttk.Combobox(set_f, textvariable=self.style_choice, state="readonly")
        self.style_combo.grid(row=12, column=1, sticky="ew")
        ttk.Button(set_f, text="Load Style", command=self.load_named_style).grid(row=12, column=2, sticky="ew")
        ttk.Button(set_f, text="Auto Style", command=self.auto_style).grid(row=13, column=1, sticky="ew")
        ttk.Button(set_f, text="Delete Style", command=self.delete_named_style).grid(row=13, column=2, sticky="ew")
        self.ai_frame = ttk.LabelFrame(set_f, text="Experimental AI Stylization", padding="10")
        self.ai_frame.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.ai_frame.columnconfigure(1, weight=1)
        ttk.Label(self.ai_frame, text="AI stylization is enabled from the Effects panel.").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self.ai_frame, text="Easy Diffusion URL:").grid(row=1, column=0, sticky="w")
        ai_backend_entry = ttk.Entry(self.ai_frame, textvariable=self.ai_backend_url)
        ai_backend_entry.grid(row=1, column=1, sticky="ew")
        ttk.Label(self.ai_frame, text="Prompt:").grid(row=2, column=0, sticky="w")
        ai_prompt_entry = ttk.Entry(self.ai_frame, textvariable=self.ai_prompt)
        ai_prompt_entry.grid(row=2, column=1, sticky="ew")
        ttk.Label(self.ai_frame, text="Negative:").grid(row=3, column=0, sticky="w")
        ai_negative_entry = ttk.Entry(self.ai_frame, textvariable=self.ai_negative_prompt)
        ai_negative_entry.grid(row=3, column=1, sticky="ew")
        ttk.Label(self.ai_frame, text="Every N frames:").grid(row=4, column=0, sticky="w")
        ai_every_spin = ttk.Spinbox(self.ai_frame, from_=1, to=120, textvariable=self.ai_every_n_frames, width=6)
        ai_every_spin.grid(row=4, column=1, sticky="w")
        ttk.Label(self.ai_frame, text="Denoise:").grid(row=5, column=0, sticky="w")
        ai_denoise_controls = ttk.Frame(self.ai_frame)
        ai_denoise_controls.grid(row=5, column=1, sticky="ew")
        ai_denoise_controls.columnconfigure(0, weight=1)
        ai_denoise_scale = ttk.Scale(ai_denoise_controls, from_=0.05, to=0.95, variable=self.ai_denoise, command=lambda e: self.update_ai_denoise_label())
        ai_denoise_scale.grid(row=0, column=0, sticky="ew")
        self.ai_denoise_label = ttk.Label(ai_denoise_controls, text=f"{self.ai_denoise.get():.2f}", width=5)
        self.ai_denoise_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(self.ai_frame, text="CFG scale:").grid(row=6, column=0, sticky="w")
        ai_cfg_controls = ttk.Frame(self.ai_frame)
        ai_cfg_controls.grid(row=6, column=1, sticky="ew")
        ai_cfg_controls.columnconfigure(0, weight=1)
        ai_cfg_scale = ttk.Scale(ai_cfg_controls, from_=1.0, to=20.0, variable=self.ai_cfg_scale, command=lambda e: self.update_ai_cfg_scale_label())
        ai_cfg_scale.grid(row=0, column=0, sticky="ew")
        self.ai_cfg_scale_label = ttk.Label(ai_cfg_controls, text=f"{self.ai_cfg_scale.get():.2f}", width=5)
        self.ai_cfg_scale_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(self.ai_frame, text="Steps:").grid(row=7, column=0, sticky="w")
        ai_steps_spin = ttk.Spinbox(self.ai_frame, from_=1, to=40, textvariable=self.ai_steps, width=6)
        ai_steps_spin.grid(row=7, column=1, sticky="w")
        ttk.Label(self.ai_frame, text="Max dimension:").grid(row=8, column=0, sticky="w")
        ai_max_dim_spin = ttk.Spinbox(self.ai_frame, from_=128, to=1024, increment=64, textvariable=self.ai_max_dim, width=6)
        ai_max_dim_spin.grid(row=8, column=1, sticky="w")
        ttk.Label(self.ai_frame, text="Blend strength:").grid(row=9, column=0, sticky="w")
        ai_blend_controls = ttk.Frame(self.ai_frame)
        ai_blend_controls.grid(row=9, column=1, sticky="ew")
        ai_blend_controls.columnconfigure(0, weight=1)
        ai_blend_scale = ttk.Scale(ai_blend_controls, from_=0.0, to=1.0, variable=self.ai_blend, command=lambda e: self.update_ai_blend_label())
        ai_blend_scale.grid(row=0, column=0, sticky="ew")
        self.ai_blend_label = ttk.Label(ai_blend_controls, text=f"{self.ai_blend.get():.2f}", width=5)
        self.ai_blend_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ai_segment_anchor = ttk.Checkbutton(self.ai_frame, text="Segment anchor only", variable=self.ai_segment_anchor_only)
        ai_segment_anchor.grid(row=10, column=0, sticky="w")

        self.add_tooltip(ai_backend_entry, "Base URL for your local Easy Diffusion instance, usually http://127.0.0.1:9000.")
        self.add_tooltip(ai_prompt_entry, "Prompt sent to the image model for stylization.")
        self.add_tooltip(ai_negative_entry, "Negative prompt to suppress unwanted artifacts.")
        self.add_tooltip(ai_every_spin, "How often to refresh the AI anchor frame; higher values are faster.")
        self.add_tooltip(ai_denoise_scale, "How strongly the model reinterprets the frame.")
        self.add_tooltip(ai_cfg_scale, "Prompt adherence. Lower values usually preserve motion better.")
        self.add_tooltip(ai_steps_spin, "Inference steps per stylized anchor frame.")
        self.add_tooltip(ai_max_dim_spin, "Downscale the AI input to cap VRAM usage and latency.")
        self.add_tooltip(ai_blend_scale, "How strongly the stylized anchor influences the final video frame.")
        self.add_tooltip(ai_segment_anchor, "When enabled, only one stylized anchor is generated per source segment.")
        self.update_ai_denoise_label()
        self.update_ai_cfg_scale_label()
        self.update_ai_blend_label()
        self.toggle_experimental_mode()
        self.adjust_window_for_mode()

        self.add_tooltip(beat_sync_cb, "Enable beat detection so clip boundaries follow the music.")
        self.add_tooltip(output_name_mode, "When enabled, existing output files are preserved and a numeric suffix is added.")
        self.add_tooltip(beat_interval_label, "Cut on every Nth detected beat. Higher values make cuts less frequent.")
        self.add_tooltip(beat_interval_spin, "Number of beats between cuts when Beat Sync is enabled.")
        self.add_tooltip(beat_variation_label, "Randomly add extra beat cuts for less predictable pacing.")
        self.add_tooltip(beat_variation_scale, "How often to insert extra cuts on skipped beats.")
        self.add_tooltip(duration_label, "Fallback segment duration when Beat Sync is off.")
        self.add_tooltip(duration_scale, "How long each segment lasts when Beat Sync is disabled.")
        self.add_tooltip(coherence_label, "Prefer staying on the current source video instead of switching often.")
        self.add_tooltip(coherence_scale, "Higher values keep the same source video more often.")
        self.add_tooltip(source_variety_label, "Prefer sources that have been used less often.")
        self.add_tooltip(source_variety_scale, "Higher values spread selections across sources more aggressively.")
        self.add_tooltip(music_match_label, "Bias frame selection toward source motion that matches the audio energy.")
        self.add_tooltip(music_match_scale, "Higher values make audio energy influence source choice more strongly.")
        self.add_tooltip(scene_transition_label, "Automatically choose hard cuts, quick fades, or longer dissolves from audio activity, or force one transition style.")
        self.add_tooltip(scene_transition_combo, "Auto favors hard cuts in active sections, quick fades in moderate sections, and longer dissolves in calmer sections.")
        self.add_tooltip(sensitivity_label, "Amplify or soften all enabled video effects.")
        self.add_tooltip(sensitivity_scale, "Higher values make enabled effects stronger and easier to trigger.")
        self.add_tooltip(fps_label, "Frames per second for the exported video.")
        self.add_tooltip(fps_spin, "Lower values render faster; higher values give smoother motion.")
        self.add_tooltip(style_name_label, "Name for saving the current settings as a reusable style.")
        self.add_tooltip(style_name_entry, "Enter a name before saving a custom style.")
        self.add_tooltip(load_style_label, "Choose a saved style to restore a preset configuration.")
        self.add_tooltip(self.style_combo, "Select a built-in or saved style.")

        fx = ttk.LabelFrame(m, text="Effects", padding="10"); fx.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        fx.columnconfigure(1, weight=1)
        ttk.Label(fx, text="Amount").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(fx, text="Timing").grid(row=0, column=3, sticky="w", padx=(8, 0))
        self.add_effect_control(fx, 1, "Pixelate", self.pixelate, "pixelate")
        self.add_effect_control(fx, 2, "Flash", self.flash, "flash")
        self.add_effect_control(fx, 3, "Rewind", self.rewind, "rewind", 1.0)
        self.add_effect_control(fx, 4, "RGB Shift", self.rgb_shift, "rgb_shift")
        self.add_effect_control(fx, 5, "Shake", self.shake, "shake")
        self.add_effect_control(fx, 6, "Ghosting", self.ghosting, "ghosting")
        self.add_effect_control(fx, 7, "Monochrome", self.monochrome, "monochrome")
        self.add_effect_control(fx, 8, "Hue Shift", self.hue_shift, "hue_shift")
        self.add_effect_control(fx, 9, "Vignette", self.vignette, "vignette")
        self.add_effect_control(fx, 10, "Static Pan/Zoom", self.static_pan_zoom, "static_pan_zoom", 1.0)
        ai_dream_label, ai_dream_scale, ai_dream_timing = self.add_effect_control(
            fx,
            11,
            "AI Dream Chance",
            self.ai_enabled,
            "ai_dream_chance",
            1.0,
            amount_var=self.ai_dream_chance,
            timing_var=self.ai_dream_timing,
            timing_values=AI_DREAM_TIMING_LABELS,
        )
        self.ai_dream_row_widgets = (ai_dream_label, ai_dream_scale, self.effect_amount_labels["ai_dream_chance"], ai_dream_timing)
        self.add_tooltip(ai_dream_label, "Probability that a clip gets AI stylization, with quieter sections favored automatically.")
        self.add_tooltip(ai_dream_scale, "Higher values make AI more likely to appear on a clip; quieter clips get a stronger boost.")
        self.add_tooltip(ai_dream_timing, "Clip biases the chance upward in quieter clips. Random uses the base chance unchanged, regardless of audio energy.")

        log_f = ttk.LabelFrame(m, text="Engine Log", padding="5"); log_f.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)
        log_toggle_row = ttk.Frame(log_f)
        log_toggle_row.pack(anchor="w", fill=tk.X, pady=(0, 4))
        verbose_match_cb = ttk.Checkbutton(log_toggle_row, text="Verbose match logging", variable=self.verbose_match_logging)
        verbose_match_cb.pack(side=tk.LEFT)
        debug_match_cb = ttk.Checkbutton(log_toggle_row, text="Debug match logging", variable=self.debug_match_logging)
        debug_match_cb.pack(side=tk.LEFT, padx=(12, 0))
        self.log_t = tk.Text(log_f, height=8, font=('Consolas', 9)); self.log_t.pack(fill=tk.BOTH, expand=True)
        self.add_tooltip(verbose_match_cb, "Show every segment match in the log instead of only the final summary.")
        self.add_tooltip(debug_match_cb, "Show candidate pools, weights, and selection reasons to diagnose sticky source selection.")
        self.pg = ttk.Progressbar(m, orient=tk.HORIZONTAL, mode='determinate'); self.pg.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        render_buttons = ttk.Frame(m); render_buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        render_buttons.columnconfigure(0, weight=1)
        render_buttons.columnconfigure(1, weight=0)
        self.btn = ttk.Button(render_buttons, text="RENDER", command=self.start_p); self.btn.grid(row=0, column=0, sticky="ew")
        self.stop_btn = ttk.Button(render_buttons, text="STOP", command=self.stop_render, state=tk.DISABLED); self.stop_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))

    def log_msg(self, msg):
        self.root.after(0, self._log_msg_ui, msg)
    def _log_msg_ui(self, msg):
        self.log_t.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_t.see(tk.END)
    def add_tooltip(self, widget, text):
        self.tooltips.append(ToolTip(widget, text))
    def toggle_experimental_mode(self):
        if not hasattr(self, "ai_frame"):
            return
        if self.experimental_mode.get():
            for widget in getattr(self, "ai_dream_row_widgets", ()):
                widget.grid()
            self.ai_frame.grid()
        else:
            self.ai_frame.grid_remove()
            for widget in getattr(self, "ai_dream_row_widgets", ()):
                widget.grid_remove()
        self.adjust_window_for_mode()
    def adjust_window_for_mode(self):
        if not hasattr(self, "root"):
            return
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        base_width = 1180
        requested_height = self.root.winfo_reqheight() + 20
        base_height = 1340 if not self.experimental_mode.get() else max(1500, requested_height)
        width = min(base_width, max(1100, screen_w - 60))
        height = min(base_height, max(1100, screen_h - 10))
        self.root.geometry(f"{width}x{height}")
    def add_effect_control(self, parent, row, label, enabled_var, amount_name, max_value=2.0, amount_var=None, timing_var=None, timing_values=None, show_check=True):
        amount_var = amount_var or self.effect_amounts[amount_name]
        timing_var = timing_var or self.effect_timing[amount_name]
        timing_values = timing_values or EFFECT_TIMING_LABELS
        if show_check:
            check = ttk.Checkbutton(parent, text=label, variable=enabled_var)
            check.grid(row=row, column=0, sticky="w")
        else:
            check = ttk.Label(parent, text=label)
            check.grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(parent, from_=0.0, to=max_value, variable=amount_var, command=lambda e, name=amount_name: self.update_effect_amount_label(name))
        scale.grid(row=row, column=1, sticky="ew", padx=5)
        self.effect_amount_labels[amount_name] = ttk.Label(parent, text=f"{amount_var.get():.2f}", width=5)
        self.effect_amount_labels[amount_name].grid(row=row, column=2, sticky="e")
        timing = ttk.Combobox(parent, textvariable=timing_var, values=timing_values, state="readonly", width=6)
        timing.grid(row=row, column=3, sticky="w", padx=(8, 0))
        return check, scale, timing
    def update_effect_amount_label(self, amount_name):
        if amount_name not in self.effect_amount_labels:
            return
        if amount_name == "ai_dream_chance":
            value = self.ai_dream_chance.get()
        else:
            value = self.effect_amounts[amount_name].get()
        self.effect_amount_labels[amount_name].config(text=f"{value:.2f}")
    def update_primary_focus_label(self):
        if self.primary_focus_label:
            self.primary_focus_label.config(text=f"{self.primary_focus.get():.2f}")
    def update_beat_variation_label(self):
        if self.beat_variation_label:
            self.beat_variation_label.config(text=f"{self.beat_variation.get():.2f}")
    def update_source_variety_label(self):
        if self.source_variety_label:
            self.source_variety_label.config(text=f"{self.source_variety.get():.2f}")
    def update_music_match_label(self):
        if self.music_match_label:
            self.music_match_label.config(text=f"{self.music_match.get():.2f}")
    def update_color_match_strength_label(self):
        if self.color_match_strength_label:
            self.color_match_strength_label.config(text=f"{self.color_match_strength.get():.2f}")
    def update_ai_denoise_label(self):
        if self.ai_denoise_label:
            self.ai_denoise_label.config(text=f"{self.ai_denoise.get():.2f}")
    def update_ai_cfg_scale_label(self):
        if self.ai_cfg_scale_label:
            self.ai_cfg_scale_label.config(text=f"{self.ai_cfg_scale.get():.2f}")
    def update_ai_blend_label(self):
        if self.ai_blend_label:
            self.ai_blend_label.config(text=f"{self.ai_blend.get():.2f}")
    def update_render_seed_status(self):
        if self.render_seed_status_label:
            self.render_seed_status_label.config(text="Auto" if not self.render_seed.get().strip() else "Fixed")
    def probe_gpu_usage(self):
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=1.5)
            if result.returncode != 0:
                return None
            values = []
            for line in result.stdout.splitlines():
                line = line.strip().rstrip("%")
                if not line:
                    continue
                try:
                    values.append(float(line))
                except ValueError:
                    continue
            if not values:
                return None
            return sum(values) / len(values)
        except Exception:
            return None
    def init_resource_monitor(self):
        self.temp_storage_root = preferred_temp_root() or tempfile.gettempdir()
        if hasattr(psutil, "cpu_percent"):
            psutil.cpu_percent(None)
        self.root.after(2000, self.refresh_resource_monitor)
    def refresh_resource_monitor(self):
        if not self.root.winfo_exists():
            return
        if not getattr(self, "active_processor", None):
            if self.resource_cpu_bar:
                self.resource_cpu_bar["value"] = 0
            if self.resource_cpu_value:
                self.resource_cpu_value.config(text="0.0%")
            if self.resource_ram_bar:
                self.resource_ram_bar["value"] = 0
            if self.resource_ram_value:
                self.resource_ram_value.config(text="0.0%")
            if self.resource_gpu_bar:
                self.resource_gpu_bar["value"] = 0
            if self.resource_gpu_value:
                self.resource_gpu_value.config(text="0.0%")
            if self.resource_temp_bar:
                self.resource_temp_bar["value"] = 0
            if self.resource_temp_value:
                self.resource_temp_value.config(text="0.0%")
            if self.resource_temp_type:
                self.resource_temp_type.config(text="Idle")
            self.root.after(4000, self.refresh_resource_monitor)
            return
        try:
            cpu_pct = float(psutil.cpu_percent(None))
            mem = psutil.virtual_memory()
            gpu_pct = self.probe_gpu_usage()
            if gpu_pct is None:
                gpu_pct = 0.0
            temp_root = getattr(self, "temp_storage_root", tempfile.gettempdir())
            temp_usage = shutil.disk_usage(temp_root)
            temp_pct = (temp_usage.used / temp_usage.total * 100.0) if temp_usage.total else 0.0
            temp_is_ram = temp_root in ("/dev/shm", "/run/shm")
        except Exception:
            self.root.after(1000, self.refresh_resource_monitor)
            return
        if self.resource_cpu_bar:
            self.resource_cpu_bar["value"] = max(0.0, min(100.0, cpu_pct))
        if self.resource_cpu_value:
            self.resource_cpu_value.config(text=f"{cpu_pct:.1f}%")
        if self.resource_ram_bar:
            self.resource_ram_bar["value"] = max(0.0, min(100.0, float(mem.percent)))
        if self.resource_ram_value:
            self.resource_ram_value.config(text=f"{mem.percent:.1f}% ({human_bytes(mem.used)} / {human_bytes(mem.total)})")
        if self.resource_gpu_bar:
            self.resource_gpu_bar["value"] = max(0.0, min(100.0, float(gpu_pct)))
        if self.resource_gpu_value:
            self.resource_gpu_value.config(text=f"{gpu_pct:.1f}%")
        if self.resource_temp_bar:
            self.resource_temp_bar["value"] = max(0.0, min(100.0, temp_pct))
        if self.resource_temp_value:
            self.resource_temp_value.config(
                text=f"{temp_pct:.1f}% ({human_bytes(temp_usage.used)} / {human_bytes(temp_usage.total)})"
            )
        if self.resource_temp_type:
            self.resource_temp_type.config(text=f"Using {'RAM-backed' if temp_is_ram else 'disk-backed'} storage")
        self.root.after(2500, self.refresh_resource_monitor)
    def randomize_render_seed(self):
        self.render_seed.set(str(random.SystemRandom().randint(1, 2**63 - 1)))
    def use_last_render_seed(self):
        if self.last_render_seed is None:
            return messagebox.showinfo("Seed", "There is no previous render seed in this session yet.")
        self.render_seed.set(str(self.last_render_seed))
    def copy_render_seed(self):
        seed = self.render_seed.get().strip()
        if not seed:
            return messagebox.showinfo("Seed", "The seed is currently auto-generated. Run a render first, or enter a seed to copy.")
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(seed)
            self.root.update()
        except Exception as e:
            return messagebox.showerror("Seed", f"Could not copy seed: {e}")
    def load_styles_file(self):
        try:
            with open(STYLE_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            styles = data if isinstance(data, dict) else {}
        except FileNotFoundError:
            styles = {}
        except Exception as e:
            print(f"Could not load styles: {e}")
            styles = {}
        return self.with_builtin_styles(styles)
    def normalize_ai_settings(self, settings):
        if not isinstance(settings, dict):
            return settings
        ai_negative_prompt = settings.get("ai_negative_prompt")
        ai_prompt = settings.get("ai_prompt")
        if (
            isinstance(ai_negative_prompt, str)
            and ai_negative_prompt.strip().rstrip(",") in AI_OBSOLETE_NEGATIVE_PROMPTS
        ) or (
            isinstance(ai_prompt, str)
            and ai_prompt.strip().rstrip(",") in AI_OBSOLETE_PROMPTS
        ):
            settings = dict(settings)
            if isinstance(ai_negative_prompt, str) and ai_negative_prompt.strip().rstrip(",") in AI_OBSOLETE_NEGATIVE_PROMPTS:
                settings["ai_negative_prompt"] = AI_DEFAULT_NEGATIVE_PROMPT
            if isinstance(ai_prompt, str) and ai_prompt.strip().rstrip(",") in AI_OBSOLETE_PROMPTS:
                settings["ai_prompt"] = AI_DEFAULT_PROMPT
        return settings
    def with_builtin_styles(self, styles):
        deleted = set(styles.get("_deleted_builtin_styles", []))
        merged = json.loads(json.dumps({
            name: style for name, style in BUILTIN_STYLES.items()
            if name == DEFAULT_STYLE_NAME or name not in deleted
        }))
        for name, style in styles.items():
            merged[name] = self.normalize_ai_settings(style) if isinstance(style, dict) else style
        merged[DEFAULT_STYLE_NAME] = json.loads(json.dumps(BUILTIN_STYLES[DEFAULT_STYLE_NAME]))
        return merged
    def save_styles_file(self):
        with open(STYLE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.styles, f, indent=2, sort_keys=True)
    def style_names(self):
        names = sorted(name for name in self.styles if not name.startswith("_") and name != DEFAULT_STYLE_NAME)
        if DEFAULT_STYLE_NAME in self.styles:
            return [DEFAULT_STYLE_NAME, *names]
        return names
    def refresh_style_choices(self):
        names = self.style_names()
        if self.style_combo:
            self.style_combo["values"] = names
        if DEFAULT_STYLE_NAME in names and not self.style_choice.get():
            self.style_choice.set(DEFAULT_STYLE_NAME)
        elif names and self.style_choice.get() not in names:
            self.style_choice.set(names[0])
    def collect_settings(self, include_media_refs=False, include_seed=False):
        settings = {
            "export_mode_label": self.export_mode_label.get(),
            "output_resolution_label": self.output_resolution_label.get(),
            "export_quality_label": self.export_quality_label.get(),
            "increment_output_if_exists": self.increment_output_if_exists.get(),
            "verbose_match_logging": self.verbose_match_logging.get(),
            "debug_match_logging": self.debug_match_logging.get(),
            "experimental_mode": self.experimental_mode.get(),
            "ai_dream_chance": self.ai_dream_chance.get(),
            "ai_dream_timing": self.ai_dream_timing.get(),
            "ai_panel_visible": self.ai_panel_visible.get(),
            "ai_enabled": self.ai_enabled.get(),
            "ai_segment_anchor_only": self.ai_segment_anchor_only.get(),
            "ai_backend_url": self.ai_backend_url.get(),
            "ai_prompt": self.ai_prompt.get(),
            "ai_negative_prompt": self.ai_negative_prompt.get(),
            "ai_every_n_frames": self.ai_every_n_frames.get(),
            "ai_denoise": self.ai_denoise.get(),
            "ai_cfg_scale": self.ai_cfg_scale.get(),
            "ai_steps": self.ai_steps.get(),
            "ai_max_dim": self.ai_max_dim.get(),
            "ai_blend": self.ai_blend.get(),
            "duration": self.duration.get(),
            "fps": self.fps.get(),
            "render_mode": self.render_mode.get(),
            "snippet_duration": self.snippet_duration.get(),
            "coherence": self.coherence.get(),
            "sensitivity": self.sensitivity.get(),
            "source_variety": self.source_variety.get(),
            "music_match": self.music_match.get(),
            "scene_transition_mode": self.scene_transition_mode.get(),
            "color_match_enabled": self.color_match_enabled.get(),
            "color_match_strength": self.color_match_strength.get(),
            "force_bw": self.force_bw.get(),
            "lut_path": self.lut_path.get(),
            "beat_sync": self.beat_sync.get(),
            "beat_step": self.beat_step.get(),
            "beat_variation": self.beat_variation.get(),
            "effects_enabled": {
                "pixelate": self.pixelate.get(),
                "flash": self.flash.get(),
                "rewind": self.rewind.get(),
                "rgb_shift": self.rgb_shift.get(),
                "shake": self.shake.get(),
                "ghosting": self.ghosting.get(),
                "monochrome": self.monochrome.get(),
                "hue_shift": self.hue_shift.get(),
                "vignette": self.vignette.get(),
                "static_pan_zoom": self.static_pan_zoom.get(),
            },
            "effect_amounts": {name: var.get() for name, var in self.effect_amounts.items()},
            "effect_timing": {name: var.get() for name, var in self.effect_timing.items()},
            "primary_enabled": self.primary_enabled.get(),
            "primary_focus": self.primary_focus.get(),
        }
        if include_seed:
            settings["render_seed"] = self.render_seed.get()
        if include_media_refs:
            settings["color_reference_idx"] = self.color_reference_idx
            settings["color_reference_label"] = self.color_reference_label.get()
        return settings
    def collect_project(self):
        return {
            "version": 1,
            "inputs": self.inputs,
            "audio": self.audio.get(),
            "output": self.output.get(),
            "output_auto_managed": self.output_auto_managed,
            "output_resolution_label": self.output_resolution_label.get(),
            "export_quality_label": self.export_quality_label.get(),
            "primary_video_idx": self.primary_video_idx,
            "primary_video_label": self.primary_video_label.get(),
            "color_reference_idx": self.color_reference_idx,
            "color_reference_label": self.color_reference_label.get(),
            "lut_path": self.lut_path.get(),
            "style_name": self.style_name.get(),
            "style_choice": self.style_choice.get(),
            "settings": self.collect_settings(include_media_refs=True, include_seed=True),
        }
    def update_project_title(self):
        if self.current_project_path:
            name = os.path.basename(self.current_project_path)
            self.root.title(f"GlitchSync Pro v3.8 - {name}")
        else:
            self.root.title("GlitchSync Pro v3.8")
    def reset_project_state(self):
        self.current_project_path = None
        self.inputs = []
        self.lb.delete(0, tk.END)
        self.audio.set("")
        self.set_output(f"glitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4", True)
        self.output_resolution_label.set("Auto (first input)")
        self.export_quality_label.set("High quality (slower)")
        self.render_seed.set("")
        self.primary_video_idx = 0
        self.primary_video_label.set("Primary: first input")
        self.color_reference_idx = 0
        self.color_reference_label.set("Reference: first input")
        self.lut_path.set("")
        self.style_name.set(DEFAULT_STYLE_NAME)
        if DEFAULT_STYLE_NAME in self.styles:
            self.style_choice.set(DEFAULT_STYLE_NAME)
        elif self.style_names():
            self.style_choice.set(self.style_names()[0])
        default_settings = make_style()
        default_settings["render_mode"] = "Full"
        default_settings["snippet_duration"] = 30.0
        self.apply_settings(default_settings)
        self.btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.rv_btn.config(state=tk.DISABLED)
        self.pg.stop()
        self.pg.config(mode='determinate')
        self.pg['value'] = 0
        self.last_progress_val = 0
        self.cv.delete("all")
        self.log_t.delete("1.0", tk.END)
        self.update_project_title()
    def new_project(self):
        has_project_data = bool(self.current_project_path or self.inputs or self.audio.get())
        if has_project_data and not messagebox.askyesno("New Project", "Start a new project and clear the current setup?"):
            return
        self.reset_project_state()
    def apply_project(self, project):
        if not isinstance(project, dict):
            raise ValueError("Invalid project file")
        self.set_inputs(list(project.get("inputs", [])), int(project.get("primary_video_idx", 0) or 0))
        self.audio.set(project.get("audio", ""))
        self.set_output(project.get("output", self.output.get()), bool(project.get("output_auto_managed", False)))
        if project.get("output_resolution_label") in OUTPUT_RESOLUTION_LABELS:
            self.output_resolution_label.set(project["output_resolution_label"])
        if project.get("export_quality_label") in EXPORT_QUALITY_LABELS:
            self.export_quality_label.set(project["export_quality_label"])
        self.color_reference_idx = int(project.get("color_reference_idx", 0) or 0)
        self.update_color_reference_label(project.get("color_reference_label", "Reference: first input"))
        self.lut_path.set(project.get("lut_path", self.lut_path.get()))
        if not self.inputs:
            self.primary_video_label.set(project.get("primary_video_label", "Primary: first input"))
        self.style_name.set(project.get("style_name", self.style_name.get()))
        if project.get("style_choice"):
            self.style_choice.set(project["style_choice"])
        settings = project.get("settings", {})
        self.apply_settings(settings)
        if not isinstance(settings, dict) or "render_seed" not in settings:
            self.render_seed.set("")
    def write_project(self, path):
        path = os.path.abspath(path)
        self.sync_auto_output_to_project(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.collect_project(), f, indent=2, sort_keys=True)
        self.current_project_path = path
        self.update_project_title()
        self.add_recent_project(path)
    def save_project(self):
        if not self.current_project_path:
            return self.save_project_as()
        try:
            self.write_project(self.current_project_path)
        except Exception as e:
            return messagebox.showerror("Error", f"Could not save project: {e}")
        messagebox.showinfo("Saved", f"Saved project: {os.path.basename(self.current_project_path)}")
    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".glitchsync.json",
            filetypes=[("GlitchSync Project", "*.glitchsync.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            self.write_project(path)
        except Exception as e:
            return messagebox.showerror("Error", f"Could not save project: {e}")
        messagebox.showinfo("Saved", f"Saved project: {os.path.basename(path)}")
    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("GlitchSync Project", "*.glitchsync.json"), ("JSON", "*.json")])
        if not path:
            return
        self.load_project_path(path)
    def apply_settings(self, settings):
        if not isinstance(settings, dict):
            return
        if settings.get("export_mode_label") in EXPORT_MODE_LABELS:
            self.export_mode_label.set(settings["export_mode_label"])
        if settings.get("output_resolution_label") in OUTPUT_RESOLUTION_LABELS:
            self.output_resolution_label.set(settings["output_resolution_label"])
        self.export_quality_label.set(
            settings["export_quality_label"]
            if settings.get("export_quality_label") in EXPORT_QUALITY_LABELS
            else "High quality (slower)"
        )
        self.increment_output_if_exists.set(settings.get("increment_output_if_exists", self.increment_output_if_exists.get()))
        self.verbose_match_logging.set(settings.get("verbose_match_logging", self.verbose_match_logging.get()))
        self.debug_match_logging.set(settings.get("debug_match_logging", self.debug_match_logging.get()))
        render_seed = settings.get("render_seed", self.render_seed.get())
        self.render_seed.set("" if render_seed is None else str(render_seed))
        experimental_mode = settings.get("experimental_mode")
        if experimental_mode is None:
            experimental_mode = settings.get("ai_panel_visible", self.experimental_mode.get())
        self.experimental_mode.set(bool(experimental_mode))
        self.ai_panel_visible.set(self.experimental_mode.get())
        self.ai_dream_chance.set(settings.get("ai_dream_chance", self.ai_dream_chance.get()))
        ai_dream_timing = settings.get("ai_dream_timing", self.ai_dream_timing.get())
        self.ai_dream_timing.set(ai_dream_timing if ai_dream_timing in AI_DREAM_TIMING_LABELS else "Clip")
        self.ai_enabled.set(settings.get("ai_enabled", self.ai_enabled.get()))
        self.ai_segment_anchor_only.set(settings.get("ai_segment_anchor_only", self.ai_segment_anchor_only.get()))
        self.ai_backend_url.set(settings.get("ai_backend_url", self.ai_backend_url.get()))
        self.ai_prompt.set(settings.get("ai_prompt", self.ai_prompt.get()))
        self.ai_negative_prompt.set(self.normalize_ai_settings(settings).get("ai_negative_prompt", self.ai_negative_prompt.get()))
        self.ai_every_n_frames.set(settings.get("ai_every_n_frames", self.ai_every_n_frames.get()))
        self.ai_denoise.set(settings.get("ai_denoise", self.ai_denoise.get()))
        self.ai_cfg_scale.set(settings.get("ai_cfg_scale", self.ai_cfg_scale.get()))
        self.ai_steps.set(settings.get("ai_steps", self.ai_steps.get()))
        self.ai_max_dim.set(settings.get("ai_max_dim", self.ai_max_dim.get()))
        self.ai_blend.set(settings.get("ai_blend", self.ai_blend.get()))
        self.duration.set(settings.get("duration", self.duration.get()))
        self.fps.set(settings.get("fps", self.fps.get()))
        if settings.get("render_mode") in ("Full", "Snippet"):
            self.render_mode.set(settings["render_mode"])
        self.snippet_duration.set(settings.get("snippet_duration", self.snippet_duration.get()))
        self.coherence.set(settings.get("coherence", self.coherence.get()))
        self.sensitivity.set(settings.get("sensitivity", self.sensitivity.get()))
        self.source_variety.set(settings.get("source_variety", self.source_variety.get()))
        self.music_match.set(settings.get("music_match", self.music_match.get()))
        scene_transition_mode = settings.get("scene_transition_mode", self.scene_transition_mode.get())
        self.scene_transition_mode.set(scene_transition_mode if scene_transition_mode in SCENE_TRANSITION_LABELS else "Auto")
        self.color_match_enabled.set(settings.get("color_match_enabled", self.color_match_enabled.get()))
        self.color_match_strength.set(settings.get("color_match_strength", self.color_match_strength.get()))
        self.force_bw.set(settings.get("force_bw", self.force_bw.get()))
        self.color_reference_idx = int(settings.get("color_reference_idx", self.color_reference_idx) or 0)
        self.update_color_reference_label(settings.get("color_reference_label", self.color_reference_label.get()))
        self.lut_path.set(settings.get("lut_path", self.lut_path.get()))
        self.beat_sync.set(settings.get("beat_sync", self.beat_sync.get()))
        self.beat_step.set(settings.get("beat_step", self.beat_step.get()))
        self.beat_variation.set(settings.get("beat_variation", self.beat_variation.get()))
        effects_enabled = settings.get("effects_enabled", {})
        self.pixelate.set(effects_enabled.get("pixelate", self.pixelate.get()))
        self.flash.set(effects_enabled.get("flash", self.flash.get()))
        self.rewind.set(effects_enabled.get("rewind", self.rewind.get()))
        self.rgb_shift.set(effects_enabled.get("rgb_shift", self.rgb_shift.get()))
        self.shake.set(effects_enabled.get("shake", self.shake.get()))
        self.ghosting.set(effects_enabled.get("ghosting", self.ghosting.get()))
        self.monochrome.set(effects_enabled.get("monochrome", False))
        self.hue_shift.set(effects_enabled.get("hue_shift", False))
        self.vignette.set(effects_enabled.get("vignette", False))
        self.static_pan_zoom.set(effects_enabled.get("static_pan_zoom", False))
        for name, value in settings.get("effect_amounts", {}).items():
            if name in self.effect_amounts:
                self.effect_amounts[name].set(value)
                self.update_effect_amount_label(name)
        effect_timing = settings.get("effect_timing", {})
        for name, var in self.effect_timing.items():
            value = effect_timing.get(name, DEFAULT_EFFECT_TIMING[name])
            var.set(value if value in EFFECT_TIMING_LABELS else DEFAULT_EFFECT_TIMING[name])
        self.primary_enabled.set(settings.get("primary_enabled", self.primary_enabled.get()))
        self.primary_focus.set(settings.get("primary_focus", self.primary_focus.get()))
        self.l_dur.config(text=f"{self.duration.get():.2f}s")
        self.l_coh.config(text=f"{self.coherence.get():.2f}")
        self.l_sen.config(text=f"{self.sensitivity.get():.2f}")
        self.update_beat_variation_label()
        self.update_source_variety_label()
        self.update_music_match_label()
        self.update_color_match_strength_label()
        self.update_ai_denoise_label()
        self.update_ai_cfg_scale_label()
        self.update_ai_blend_label()
        self.update_effect_amount_label("ai_dream_chance")
        self.update_primary_focus_label()
        self.toggle_experimental_mode()
    def save_named_style(self):
        name = self.style_name.get().strip()
        if not name:
            return messagebox.showerror("Error", "Style name is required")
        if name == DEFAULT_STYLE_NAME:
            return messagebox.showerror("Error", "The default style cannot be overwritten")
        if name in BUILTIN_STYLES and name in self.styles.get("_deleted_builtin_styles", []):
            self.styles["_deleted_builtin_styles"].remove(name)
        self.styles[name] = self.collect_settings()
        try:
            self.save_styles_file()
        except Exception as e:
            return messagebox.showerror("Error", f"Could not save style: {e}")
        self.refresh_style_choices()
        self.style_choice.set(name)
        messagebox.showinfo("Saved", f"Saved style: {name}")
    def load_named_style(self):
        name = self.style_choice.get() or self.style_name.get().strip()
        if name not in self.styles:
            return messagebox.showerror("Error", "Select a saved style to load")
        self.apply_settings(self.styles[name])
        self.style_name.set(name)
    def auto_style(self):
        audio_path = self.audio.get()
        if not audio_path:
            return messagebox.showerror("Auto Style", "Select an audio file first.")
        if not os.path.exists(audio_path):
            return messagebox.showerror("Auto Style", "The selected audio file does not exist.")
        self.log_msg("Auto Style: analyzing audio...")
        threading.Thread(target=self._auto_style_worker, args=(audio_path,), daemon=True).start()
    def _auto_style_worker(self, audio_path):
        try:
            processor = GlitchProcessor([], audio_path, "", log_callback=self.log_msg)
            audio_features = processor.analyze_audio_file()
            style_name, desc = GlitchProcessor.choose_auto_style(audio_features)
            self.root.after(0, self._apply_auto_style_ui, style_name, desc)
        except Exception as e:
            self.log_msg(f"Auto Style error: {e}")
            self.root.after(0, messagebox.showerror, "Auto Style", f"Could not analyze audio:\n{e}")
    def _apply_auto_style_ui(self, style_name, desc):
        if style_name not in self.styles:
            style_name = DEFAULT_STYLE_NAME
        self.apply_settings(self.styles[style_name])
        self.style_name.set(style_name)
        self.style_choice.set(style_name)
        self.log_msg(
            "Auto Style selected "
            f"{style_name} (tempo {desc['tempo']:.1f} BPM, energy {desc['energy']:.2f}, "
            f"bass {desc['bass']:.2f}, highs {desc['highs']:.2f})"
        )
        messagebox.showinfo("Auto Style", f"Selected style: {style_name}")
    def delete_named_style(self):
        name = self.style_choice.get() or self.style_name.get().strip()
        if not name or name not in self.styles:
            return messagebox.showerror("Error", "Select a saved style to delete")
        if name == "_last":
            return messagebox.showerror("Error", "The automatic last-used style cannot be deleted")
        if name == DEFAULT_STYLE_NAME:
            return messagebox.showerror("Error", "The default style cannot be deleted")
        if not messagebox.askyesno("Delete Style", f"Delete style '{name}'?"):
            return
        if name in BUILTIN_STYLES:
            deleted = self.styles.setdefault("_deleted_builtin_styles", [])
            if name not in deleted:
                deleted.append(name)
        del self.styles[name]
        try:
            self.save_styles_file()
        except Exception as e:
            return messagebox.showerror("Error", f"Could not delete style: {e}")
        self.refresh_style_choices()
        if self.style_choice.get() == name:
            self.style_choice.set(self.style_names()[0] if self.style_names() else "")
        messagebox.showinfo("Deleted", f"Deleted style: {name}")
    def save_last_settings(self):
        self.styles["_last"] = self.collect_settings()
        try:
            self.save_styles_file()
        except Exception as e:
            self.log_msg(f"Could not save last-used settings: {e}")
    def set_inputs(self, inputs, primary_idx=None):
        self.inputs = list(inputs)
        self.lb.delete(0, tk.END)
        for path in self.inputs:
            self.lb.insert(tk.END, os.path.basename(path))
        if not self.inputs:
            self.primary_video_idx = 0
            self.primary_video_label.set("Primary: first input")
            self.color_reference_idx = 0
            self.color_reference_label.set("Reference: first input")
            return
        if primary_idx is None:
            primary_idx = self.primary_video_idx
        self.primary_video_idx = int(np.clip(primary_idx, 0, len(self.inputs) - 1))
        self.primary_video_label.set(f"Primary: {os.path.basename(self.inputs[self.primary_video_idx])}")
        self.color_reference_idx = int(np.clip(self.color_reference_idx, 0, len(self.inputs) - 1))
        self.update_color_reference_label()
    def update_color_reference_label(self, fallback="Reference: first input"):
        if self.inputs and 0 <= self.color_reference_idx < len(self.inputs):
            self.color_reference_label.set(f"Reference: {os.path.basename(self.inputs[self.color_reference_idx])}")
        else:
            self.color_reference_idx = 0
            self.color_reference_label.set(fallback)
    def add_v(self):
        f = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        for x in f:
            if x not in self.inputs:
                self.inputs.append(x); self.lb.insert(tk.END, os.path.basename(x))
                if len(self.inputs) == 1:
                    self.primary_video_idx = 0
                    self.primary_video_label.set(f"Primary: {os.path.basename(x)}")
                    self.color_reference_idx = 0
                    self.update_color_reference_label()
    def remove_selected_inputs(self):
        selected = set(self.lb.curselection())
        if not selected:
            return messagebox.showinfo("Remove Inputs", "Select one or more input videos to remove.")
        new_inputs = [path for idx, path in enumerate(self.inputs) if idx not in selected]
        new_primary = self.primary_video_idx - sum(1 for idx in selected if idx < self.primary_video_idx)
        self.color_reference_idx -= sum(1 for idx in selected if idx < self.color_reference_idx)
        self.set_inputs(new_inputs, new_primary)
    def clean_missing_inputs(self):
        missing = [path for path in self.inputs if not os.path.exists(path)]
        if not missing:
            return messagebox.showinfo("Clean Missing Inputs", "All selected input files still exist.")
        if not messagebox.askyesno("Clean Missing Inputs", f"Remove {len(missing)} missing input file(s) from this project?"):
            return
        existing = [path for path in self.inputs if os.path.exists(path)]
        primary_path = self.inputs[self.primary_video_idx] if self.inputs and 0 <= self.primary_video_idx < len(self.inputs) else None
        color_ref_path = self.inputs[self.color_reference_idx] if self.inputs and 0 <= self.color_reference_idx < len(self.inputs) else None
        new_primary = existing.index(primary_path) if primary_path in existing else 0
        self.color_reference_idx = existing.index(color_ref_path) if color_ref_path in existing else 0
        self.set_inputs(existing, new_primary)
        self.log_msg(f"Removed {len(missing)} missing input file(s).")
    def add_a(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a")])
        if f: self.audio.set(f)
    def set_primary_video(self):
        sel = self.lb.curselection()
        if not sel: return
        self.primary_video_idx = sel[0]
        self.primary_video_label.set(f"Primary: {os.path.basename(self.inputs[self.primary_video_idx])}")
    def set_color_reference_video(self):
        sel = self.lb.curselection()
        if not sel: return
        self.color_reference_idx = sel[0]
        self.update_color_reference_label()
    def pick_lut(self):
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")])
        if path:
            self.lut_path.set(path)
    def clear_lut(self):
        self.lut_path.set("")
    def resolve_render_output(self, export_mode):
        base_output = self.output.get()
        resolved_output = output_path_for_mode(base_output, export_mode)
        if self.increment_output_if_exists.get():
            resolved_output = increment_path_if_exists(resolved_output)
        return resolved_output
    def update_p(self, curr, total):
        self.root.after(0, self._update_p_ui, curr, total)
    def _update_p_ui(self, curr, total):
        if curr == -1:
            if self.pg['mode'] != 'indeterminate': self.pg.config(mode='indeterminate'); self.pg.start(10)
        else:
            if self.pg['mode'] != 'determinate': self.pg.stop(); self.pg.config(mode='determinate')
            val = (curr / total) * 100
            if abs(self.last_progress_val - val) >= 0.5: self.pg['value'] = val; self.last_progress_val = val
    def start_p(self):
        if not self.inputs or not self.audio.get(): return messagebox.showerror("Error", "Missing files")
        self.save_last_settings()
        self.render_was_stopped = False
        self.last_render_output_path = None
        self.btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL); self.rv_btn.config(state=tk.DISABLED); self.last_progress_val = 0
        threading.Thread(target=self.run_e, daemon=True).start()
    def stop_render(self):
        self.render_was_stopped = True
        if self.active_processor:
            self.active_processor.stop_requested = True
        self.stop_btn.config(state=tk.DISABLED)
        self.log_msg("Stop requested; finishing current operation...")
    def run_e(self):
        try:
            export_mode = EXPORT_MODE_LABELS[self.export_mode_label.get()]
            resolved_output = self.resolve_render_output(export_mode)
            self.last_render_output_path = resolved_output
            if resolved_output != self.output.get():
                self.log_msg(f"Output resolved to: {resolved_output}")
            effect_amounts = {name: var.get() for name, var in self.effect_amounts.items()}
            effect_timing = {name: var.get() for name, var in self.effect_timing.items()}
            primary_idx = self.primary_video_idx if self.primary_enabled.get() and self.inputs else None
            primary_focus = self.primary_focus.get() if self.primary_enabled.get() else 0.0
            color_reference_idx = self.color_reference_idx if self.color_match_enabled.get() and self.inputs else None
            color_match_strength = self.color_match_strength.get() if self.color_match_enabled.get() else 0.0
            render_limit = self.snippet_duration.get() if self.render_mode.get() == "Snippet" else None
            output_resolution = OUTPUT_RESOLUTION_LABELS.get(self.output_resolution_label.get())
            seed = parse_render_seed(self.render_seed.get())
            if seed is None:
                seed = random.SystemRandom().randint(1, 2**63 - 1)
                self.render_seed.set(str(seed))
            self.last_render_seed = seed
            p = GlitchProcessor(
                self.inputs,
                self.audio.get(),
                resolved_output,
                self.duration.get(),
                self.fps.get(),
                self.pixelate.get(),
                self.flash.get(),
                self.rewind.get(),
                self.rgb_shift.get(),
                self.shake.get(),
                self.ghosting.get(),
                self.static_pan_zoom.get(),
                self.monochrome.get(),
                self.hue_shift.get(),
                self.vignette.get(),
                self.beat_sync.get(),
                self.coherence.get(),
                self.sensitivity.get(),
                export_mode,
                self.update_p,
                self.log_msg,
                self.display_frame,
                effect_amounts,
                primary_idx,
                primary_focus,
                self.beat_step.get(),
                self.beat_variation.get(),
                render_limit,
                self.source_variety.get(),
                self.music_match.get(),
                color_reference_idx,
                color_match_strength,
                self.lut_path.get().strip(),
                output_resolution,
                self.export_quality_label.get(),
                effect_timing,
                self.experimental_mode.get(),
                self.ai_dream_chance.get(),
                self.ai_dream_timing.get(),
                self.scene_transition_mode.get(),
                self.force_bw.get(),
                self.ai_enabled.get(),
                self.ai_segment_anchor_only.get(),
                self.ai_backend_url.get(),
                self.ai_prompt.get(),
                self.ai_negative_prompt.get(),
                self.ai_every_n_frames.get(),
                self.ai_denoise.get(),
                self.ai_cfg_scale.get(),
                self.ai_steps.get(),
                self.ai_max_dim.get(),
                self.ai_blend.get(),
                self.verbose_match_logging.get(),
                self.debug_match_logging.get(),
                temp_root=self.temp_storage_root,
                render_seed=seed,
            )
            self.active_processor = p
            if self.render_was_stopped:
                p.stop_requested = True
            completed = p.process()
            if completed:
                self.root.after(0, self._render_complete_ui, export_mode)
            else:
                self.root.after(0, self._render_stopped_ui)
        except Exception as e:
            self.log_msg(f"ERROR: {e}")
            self.root.after(0, self._render_error_ui, str(e))
        finally:
            self.active_processor = None
            self.root.after(0, self._render_finished_ui)
    def _render_complete_ui(self, export_mode):
        if export_mode == EXPORT_FINAL_VIDEO:
            self.rv_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Complete!")
    def _render_error_ui(self, error):
        messagebox.showerror("Error", error)
    def _render_stopped_ui(self):
        messagebox.showinfo("Stopped", "Render stopped.")
    def _render_finished_ui(self):
        self.pg.stop(); self.pg.config(mode='determinate'); self.btn.config(state=tk.NORMAL); self.stop_btn.config(state=tk.DISABLED); self.pg['value'] = 0
    def display_frame(self, f):
        if getattr(self, "force_bw", None) is not None and self.force_bw.get():
            f = force_black_and_white(f)
        h, w = f.shape[:2]
        canvas_w = max(1, int(self.cv.winfo_width() or self.cv.winfo_reqwidth() or 480))
        canvas_h = max(1, int(self.cv.winfo_height() or self.cv.winfo_reqheight() or 220))
        s = min(canvas_w / max(w, 1), canvas_h / max(h, 1))
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((nw, nh), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        self.root.after(0, self._update_cv, img_tk)
    def _update_cv(self, img_tk):
        self.cv.delete("all")
        self.cv.create_image(self.cv.winfo_width() // 2, self.cv.winfo_height() // 2, anchor=tk.CENTER, image=img_tk); self.cv._img_ref = img_tk
    def review_render(self):
        path = self.last_render_output_path or self.resolve_render_output(EXPORT_MODE_LABELS[self.export_mode_label.get()])
        self.log_msg(f"Launching ffplay: {path}")
        subprocess.Popen(['ffplay', '-i', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs='+'); p.add_argument("--audio"); p.add_argument("--output", default="output.mp4"); p.add_argument("--beat_sync", action="store_true"); p.add_argument("--gui", action="store_true")
    p.add_argument("--export-mode", choices=[EXPORT_FINAL_VIDEO, EXPORT_CUT_AWARE_MLT, EXPORT_CLIP_MLT], default=EXPORT_FINAL_VIDEO)
    p.add_argument("--export-quality", choices=list(EXPORT_QUALITY_LABELS.keys()), default="High quality (slower)")
    p.add_argument("--beat-step", type=int, default=4)
    p.add_argument("--beat-variation", type=float, default=0.0)
    p.add_argument("--render-limit", type=float)
    p.add_argument("--music-match", type=float, default=0.35)
    p.add_argument("--force-bw", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.gui or not (args.inputs and args.audio):
        r = tk.Tk(); g = GlitchGUI(r); r.mainloop()
    else:
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(int(args.seed) % (2**32))
        proc = GlitchProcessor(args.inputs, args.audio, args.output, beat_sync=args.beat_sync, export_mode=args.export_mode, progress_callback=lambda c, t: print(f"Progress: {c}/{t}", end='\r'), beat_step=args.beat_step, beat_variation=args.beat_variation, render_limit=args.render_limit, music_match=args.music_match, export_quality_label=args.export_quality, force_bw=args.force_bw, render_seed=args.seed)
        proc.process()
