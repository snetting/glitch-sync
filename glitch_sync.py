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
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk

EXPORT_FINAL_VIDEO = "final_video"
EXPORT_CUT_AWARE_MLT = "cut_aware_mlt"
EXPORT_CLIP_MLT = "clip_mlt"

EXPORT_MODE_LABELS = {
    "Final video (MP4)": EXPORT_FINAL_VIDEO,
    "Cut-aware Shotcut MLT (ZIP)": EXPORT_CUT_AWARE_MLT,
    "Clip Shotcut MLT (ZIP)": EXPORT_CLIP_MLT,
}

EFFECT_NAMES = ("pixelate", "flash", "rewind", "rgb_shift", "shake", "ghosting")
DEFAULT_EFFECT_AMOUNTS = {name: 1.0 for name in EFFECT_NAMES}
STYLE_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".glitchsync_styles.json")


def make_style(duration=0.1, fps=30, coherence=0.2, sensitivity=1.0, beat_sync=True,
               beat_step=1, beat_variation=0.0, effects_enabled=None,
               effect_amounts=None, primary_enabled=False, primary_focus=0.75,
               export_mode_label="Final video (MP4)"):
    return {
        "export_mode_label": export_mode_label,
        "duration": duration,
        "fps": fps,
        "coherence": coherence,
        "sensitivity": sensitivity,
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
            **(effects_enabled or {}),
        },
        "effect_amounts": {
            **DEFAULT_EFFECT_AMOUNTS,
            **(effect_amounts or {}),
        },
        "primary_enabled": primary_enabled,
        "primary_focus": primary_focus,
    }


BUILTIN_STYLES = {
    "Mellow Story": make_style(
        duration=0.8,
        coherence=0.85,
        sensitivity=0.55,
        beat_step=8,
        beat_variation=0.12,
        effect_amounts={"pixelate": 0.25, "flash": 0.25, "rewind": 0.1, "rgb_shift": 0.3, "shake": 0.2, "ghosting": 0.7},
        primary_focus=0.9,
    ),
    "Pop Performance": make_style(
        coherence=0.65,
        sensitivity=0.9,
        beat_step=4,
        beat_variation=0.25,
        effect_amounts={"pixelate": 0.5, "flash": 0.55, "rewind": 0.2, "rgb_shift": 0.65, "shake": 0.45, "ghosting": 0.5},
    ),
    "EDM Pulse": make_style(
        coherence=0.35,
        sensitivity=1.35,
        beat_step=2,
        beat_variation=0.45,
        effect_amounts={"pixelate": 1.2, "flash": 1.35, "rewind": 0.45, "rgb_shift": 1.25, "shake": 1.1, "ghosting": 0.65},
    ),
    "Rock Punch": make_style(
        coherence=0.5,
        sensitivity=1.15,
        beat_step=2,
        beat_variation=0.25,
        effect_amounts={"pixelate": 0.65, "flash": 0.7, "rewind": 0.25, "rgb_shift": 0.75, "shake": 1.25, "ghosting": 0.35},
    ),
    "Ambient Drift": make_style(
        duration=1.0,
        coherence=0.95,
        sensitivity=0.4,
        beat_step=16,
        beat_variation=0.05,
        effects_enabled={"pixelate": False, "rewind": False, "shake": False},
        effect_amounts={"pixelate": 0.0, "flash": 0.15, "rewind": 0.0, "rgb_shift": 0.2, "shake": 0.0, "ghosting": 1.2},
        primary_focus=0.95,
    ),
    "Glitch Heavy": make_style(
        coherence=0.15,
        sensitivity=1.75,
        beat_step=1,
        beat_variation=0.0,
        effect_amounts={"pixelate": 1.7, "flash": 1.2, "rewind": 0.8, "rgb_shift": 1.8, "shake": 1.4, "ghosting": 1.0},
    ),
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


def prettify_xml(element):
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")


def add_property(parent, name, value):
    prop = ET.SubElement(parent, "property", {"name": name})
    prop.text = str(value)
    return prop


def write_shotcut_mlt(project_path, video_resources, audio_resource, segments, fps, width, height):
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
    ET.SubElement(multitrack, "track", {"producer": "video_track"})
    ET.SubElement(multitrack, "track", {"producer": "audio_track"})
    ET.SubElement(tractor, "transition", {"id": "transition0", "in": "0", "out": str(out_frame)})
    add_property(tractor[-1], "mlt_service", "mix")
    add_property(tractor[-1], "a_track", "0")
    add_property(tractor[-1], "b_track", "1")
    add_property(tractor[-1], "always_active", "1")

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

def apply_brightness_boost(frame, intensity, sensitivity=1.0):
    if intensity < 0.15: return frame
    val = int((intensity - 0.15) * sensitivity * 160)
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

# --- Core Logic ---

class GlitchProcessor:
    def __init__(self, inputs, audio, output, duration=0.1, fps=30, 
                 pixelate=False, flash=False, rewind=False, 
                 rgb_shift=False, shake=False, ghosting=False,
                 beat_sync=False, coherence=0.7, sensitivity=1.0,
                 export_mode=EXPORT_FINAL_VIDEO,
                 progress_callback=None, log_callback=None, frame_callback=None,
                 effect_amounts=None, primary_video_idx=None, primary_focus=0.0,
                 beat_step=1, beat_variation=0.0, render_limit=None):
        self.inputs, self.audio, self.output = inputs, audio, output
        self.duration, self.fps = duration, fps
        self.pixelate, self.flash, self.rewind = pixelate, flash, rewind
        self.rgb_shift, self.shake, self.ghosting = rgb_shift, shake, ghosting
        self.beat_sync, self.coherence, self.sensitivity = beat_sync, coherence, sensitivity
        self.export_mode = export_mode
        self.beat_step = max(1, int(beat_step))
        self.beat_variation = np.clip(float(beat_variation), 0, 1)
        self.render_limit = float(render_limit) if render_limit else None
        self.effect_amounts = DEFAULT_EFFECT_AMOUNTS.copy()
        if effect_amounts:
            for name in EFFECT_NAMES:
                self.effect_amounts[name] = max(0.0, float(effect_amounts.get(name, 1.0)))
        self.primary_video_idx = primary_video_idx if primary_video_idx is not None else None
        self.primary_focus = np.clip(float(primary_focus), 0, 1)
        self.progress_callback, self.log_callback, self.frame_callback = progress_callback, log_callback, frame_callback
        self.stop_requested = False
        self.current_vid_idx = 0

    def log(self, msg):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def effect_amount(self, name):
        return self.effect_amounts.get(name, 1.0)

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

    def analyze_source_videos(self):
        self.log("Indexing video frames...")
        if self.progress_callback: self.progress_callback(-1, -1)
        frame_db = {i: [] for i in range(256)}
        for video_idx, path in enumerate(self.inputs):
            self.log(f"  Scanning {os.path.basename(path)}...")
            cap = cv2.VideoCapture(path)
            frame_count = 0
            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret: break
                if frame_count % 5 == 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    b = int(np.mean(hsv[:, :, 2]))
                    frame_db[b].append((video_idx, frame_count))
                frame_count += 1
            cap.release()
        return frame_db

    def find_best_match(self, target_b, frame_db, current_vid_idx):
        search_range = 8
        candidates_current, candidates_other, candidates_primary = [], [], []
        for b in range(max(0, target_b - search_range), min(255, target_b + search_range) + 1):
            for v_idx, f_idx in frame_db[b]:
                if self.use_primary_video() and v_idx == self.primary_video_idx:
                    candidates_primary.append((v_idx, f_idx))
                if v_idx == current_vid_idx: candidates_current.append((v_idx, f_idx))
                else: candidates_other.append((v_idx, f_idx))
        if candidates_primary and random.random() < self.primary_focus:
            return random.choice(candidates_primary)
        if candidates_current and (random.random() < self.coherence or not candidates_other):
            return random.choice(candidates_current)
        if candidates_other: return random.choice(candidates_other)
        offset = 1
        while offset < 256:
            low, high = target_b - offset, target_b + offset
            fallback = []
            primary_fallback = []
            if low >= 0: fallback.extend(frame_db[low])
            if high <= 255: fallback.extend(frame_db[high])
            if self.use_primary_video():
                primary_fallback = [item for item in fallback if item[0] == self.primary_video_idx]
                if primary_fallback and random.random() < self.primary_focus:
                    return random.choice(primary_fallback)
            if fallback: return random.choice(fallback)
            offset += 1
        return None

    def process(self):
        package_dir = None
        clip_dir = None
        frame_db = self.analyze_source_videos()
        if self.stop_requested: return
        self.log("Analyzing audio features...")
        y, sr = librosa.load(self.audio, sr=None)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass_energy = np.mean(D[freqs <= 150, :], axis=0)
        highs_energy = np.mean(D[freqs >= 5000, :], axis=0)
        mids_energy = np.mean(D[(freqs > 150) & (freqs < 5000), :], axis=0)
        rms_energy = librosa.feature.rms(y=y)[0]
        
        if self.beat_sync:
            self.log("Detecting beats...")
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            cut_times = self.select_beat_cut_times(beats, sr)
            self.log(f"  Tempo: {tempo_val:.1f} BPM")
            self.log(f"  Beat step: {self.beat_step}; variation: {self.beat_variation:.2f}")
            if len(cut_times) < 2:
                self.log("  Not enough beats detected; falling back to duration cuts.")
                cut_times = np.arange(0, audio_duration, self.duration)
        else:
            cut_times = np.arange(0, audio_duration, self.duration)

        render_duration = audio_duration
        if self.render_limit:
            render_duration = min(audio_duration, max(self.duration, self.render_limit))
            cut_times = cut_times[cut_times < render_duration]
            if len(cut_times) == 0 or cut_times[0] > 0:
                cut_times = np.insert(cut_times, 0, 0)
            self.log(f"  Render limit: {render_duration:.1f}s")

        times = librosa.times_like(rms_energy, sr=sr)
        def get_val(arr, t): return arr[min(np.searchsorted(times, t), len(arr)-1)]
        
        # Aggressive normalization for punchier effects
        max_rms = float(np.percentile(rms_energy, 99.5))
        max_bass = float(np.percentile(bass_energy, 99.5))
        max_highs = float(np.percentile(highs_energy, 99.5))
        max_mids = float(np.percentile(mids_energy, 99.5))

        cap = cv2.VideoCapture(self.inputs[0])
        _, first_frame = cap.read()
        cap.release()
        height, width = first_frame.shape[:2]
        temp_video = f"temp_{random.randint(1000, 9999)}.avi"
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))
        caps, prev_f = [cv2.VideoCapture(f) for f in self.inputs], None
        segments = []
        rendered_frames = 0
        if self.export_mode == EXPORT_CLIP_MLT:
            package_dir = tempfile.mkdtemp(prefix="glitchsync_shotcut_")
            clip_dir = os.path.join(package_dir, "media", "clips")
            os.makedirs(clip_dir, exist_ok=True)

        self.log("Rendering...")
        if self.use_primary_video():
            self.log(f"  Primary focus: {os.path.basename(self.inputs[self.primary_video_idx])} ({self.primary_focus:.2f})")
        for i in range(len(cut_times)):
            if self.stop_requested: break
            t_start = cut_times[i]
            dur = (cut_times[i+1] if i+1 < len(cut_times) else render_duration) - t_start
            num_frames = max(1, int(dur * self.fps))
            
            curr_rms = np.clip(float(get_val(rms_energy, t_start)) / max_rms, 0, 1)
            curr_bass = np.clip(float(get_val(bass_energy, t_start)) / max_bass, 0, 1)
            curr_highs = np.clip(float(get_val(highs_energy, t_start)) / max_highs, 0, 1)
            curr_mids = np.clip(float(get_val(mids_energy, t_start)) / max_mids, 0, 1)
            
            match = self.find_best_match(int(curr_rms * 255), frame_db, self.current_vid_idx)
            if match:
                self.current_vid_idx, start_frame = match
                caps[self.current_vid_idx].set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                chunk = []
                for _ in range(num_frames):
                    r, f = caps[self.current_vid_idx].read()
                    if r: chunk.append(f)
                rewind_amount = self.effect_amount("rewind")
                if self.rewind and curr_rms > 0.75 and random.random() < min(1.0, rewind_amount):
                    chunk = chunk[::-1]

                pixelate_amount = self.effect_amount("pixelate")
                flash_amount = self.effect_amount("flash")
                rgb_shift_amount = self.effect_amount("rgb_shift")
                shake_amount = self.effect_amount("shake")
                ghosting_amount = self.effect_amount("ghosting")
                segment_frames = []
                for f in chunk:
                    if f.shape[:2] != (height, width): f = cv2.resize(f, (width, height))
                    if self.pixelate and pixelate_amount > 0:
                        f = apply_pixelate(f, curr_rms, curr_highs, self.sensitivity * pixelate_amount, pixelate_amount)
                    if self.flash and flash_amount > 0:
                        f = apply_brightness_boost(f, curr_rms, self.sensitivity * flash_amount)
                    if self.rgb_shift and rgb_shift_amount > 0:
                        f = apply_rgb_shift(f, curr_highs, self.sensitivity * rgb_shift_amount)
                    if self.shake and shake_amount > 0:
                        f = apply_shake(f, curr_bass, self.sensitivity * shake_amount)
                    if self.ghosting and ghosting_amount > 0:
                        f = apply_ghosting(f, prev_f, curr_mids, self.sensitivity * ghosting_amount)
                    out.write(f)
                    if self.export_mode == EXPORT_CLIP_MLT:
                        segment_frames.append(f)
                    prev_f = f.copy()

                if chunk:
                    frame_count = len(chunk)
                    segment = {
                        "timeline_in": rendered_frames,
                        "timeline_out": rendered_frames + frame_count - 1,
                        "frame_count": frame_count,
                    }
                    if self.export_mode == EXPORT_CUT_AWARE_MLT:
                        segment["producer"] = "video0"
                        segment["in"] = rendered_frames
                        segment["out"] = rendered_frames + frame_count - 1
                    elif self.export_mode == EXPORT_CLIP_MLT:
                        clip_name = f"clip_{len(segments) + 1:04d}.avi"
                        clip_path = os.path.join(clip_dir, clip_name)
                        clip_out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))
                        for frame in segment_frames:
                            clip_out.write(frame)
                        clip_out.release()
                        segment["producer"] = f"video{len(segments)}"
                        segment["resource"] = f"media/clips/{clip_name}"
                        segment["in"] = 0
                        segment["out"] = frame_count - 1
                    segments.append(segment)
                    rendered_frames += frame_count

                if self.frame_callback and chunk: self.frame_callback(chunk[0])

            if self.progress_callback: self.progress_callback(i, len(cut_times))
            
        for c in caps: c.release()
        out.release()
        if not self.stop_requested:
            self.log("Exporting final stage...")
            if self.progress_callback: self.progress_callback(-1, -1)
            if self.export_mode == EXPORT_FINAL_VIDEO:
                self.export_final_video(temp_video)
            else:
                self.export_shotcut_archive(temp_video, segments, width, height, package_dir)
        if os.path.exists(temp_video): os.remove(temp_video)
        if package_dir and os.path.exists(package_dir): shutil.rmtree(package_dir)
        self.log("Done!")

    def export_final_video(self, temp_video):
        root, ext = os.path.splitext(self.output)
        final_temp = f"{root or self.output}.tmp_{random.randint(1000, 9999)}{ext or '.mp4'}"
        result = subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-i', self.audio, '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-c:a', 'aac', '-b:a', '192k', '-shortest', final_temp], capture_output=True)
        if result.returncode != 0:
            if os.path.exists(final_temp):
                os.remove(final_temp)
            raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while creating final video")
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
                video_name = "glitchsync_render.mp4"
                video_path = os.path.join(media_dir, video_name)
                result = subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', video_path], capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode(errors="ignore") or "ffmpeg failed while creating Shotcut video")
                video_resources = {"video0": f"media/{video_name}"}
            elif self.export_mode == EXPORT_CLIP_MLT:
                video_resources = {segment["producer"]: segment["resource"] for segment in segments}
            else:
                raise RuntimeError(f"Unsupported Shotcut export mode: {self.export_mode}")

            project_path = os.path.join(package_dir, "glitchsync_project.mlt")
            write_shotcut_mlt(project_path, video_resources, audio_resource, segments, self.fps, width, height)
            zip_directory(package_dir, output_zip)
            self.log(f"Shotcut archive written: {output_zip}")
        finally:
            if created_package_dir and os.path.exists(package_dir):
                shutil.rmtree(package_dir)

# --- GUI ---

class GlitchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GlitchSync Pro v3.8")
        self.root.geometry("1180x1120")
        self.root.minsize(1100, 1050)
        self.inputs, self.audio = [], tk.StringVar()
        self.output = tk.StringVar(value=f"glitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        self.export_mode_label = tk.StringVar(value="Final video (MP4)")
        self.duration, self.fps, self.coherence, self.sensitivity = tk.DoubleVar(value=0.10), tk.IntVar(value=30), tk.DoubleVar(value=0.20), tk.DoubleVar(value=1.0)
        self.render_mode = tk.StringVar(value="Full")
        self.snippet_duration = tk.DoubleVar(value=30.0)
        self.beat_step = tk.IntVar(value=1)
        self.beat_variation = tk.DoubleVar(value=0.0)
        self.beat_variation_label = None
        self.effect_amounts = {name: tk.DoubleVar(value=1.0) for name in EFFECT_NAMES}
        self.effect_amount_labels = {}
        self.primary_enabled = tk.BooleanVar(value=False)
        self.primary_focus = tk.DoubleVar(value=0.75)
        self.primary_focus_label = None
        self.primary_video_idx = 0
        self.primary_video_label = tk.StringVar(value="Primary: first input")
        self.beat_sync = tk.BooleanVar(value=True)
        self.pixelate, self.flash, self.rewind, self.rgb_shift, self.shake, self.ghosting = [tk.BooleanVar(value=True) for _ in range(6)]
        self.style_name = tk.StringVar(value="Default")
        self.style_choice = tk.StringVar()
        self.style_combo = None
        self.styles = self.load_styles_file()
        self.last_progress_val = 0
        self.build_menu()
        self.build_ui()
        self.refresh_style_choices()
        if "_last" in self.styles:
            self.apply_settings(self.styles["_last"])

    def build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Project...", command=self.save_project)
        file_menu.add_command(label="Load Project...", command=self.load_project)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def build_ui(self):
        m = ttk.Frame(self.root, padding="15"); m.pack(fill=tk.BOTH, expand=True)
        m.columnconfigure(0, weight=1); m.columnconfigure(1, weight=1)
        
        io = ttk.LabelFrame(m, text="Files", padding="10"); io.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Label(io, text="Inputs:").grid(row=0, column=0, sticky="nw")
        self.lb = tk.Listbox(io, height=5); self.lb.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(io, text="+ Add", command=self.add_v).grid(row=0, column=2, sticky="n")
        ttk.Label(io, text="Audio:").grid(row=1, column=0, pady=5)
        ttk.Entry(io, textvariable=self.audio).grid(row=1, column=1, sticky="ew")
        ttk.Button(io, text="...", command=self.add_a).grid(row=1, column=2)
        ttk.Label(io, text="Export:").grid(row=2, column=0)
        ttk.Combobox(io, textvariable=self.export_mode_label, values=list(EXPORT_MODE_LABELS.keys()), state="readonly").grid(row=2, column=1, sticky="ew")
        ttk.Label(io, text="Out:").grid(row=3, column=0)
        ttk.Entry(io, textvariable=self.output).grid(row=3, column=1, sticky="ew")
        ttk.Label(io, text="Render length:").grid(row=4, column=0, sticky="w")
        ttk.Combobox(io, textvariable=self.render_mode, values=["Full", "Snippet"], state="readonly", width=10).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Spinbox(io, from_=1, to=3600, textvariable=self.snippet_duration, width=7).grid(row=4, column=2, sticky="w")
        ttk.Checkbutton(io, text="Primary focus", variable=self.primary_enabled).grid(row=5, column=0, sticky="w")
        ttk.Button(io, text="Set Selected", command=self.set_primary_video).grid(row=5, column=1, sticky="w", pady=5)
        ttk.Label(io, textvariable=self.primary_video_label).grid(row=5, column=2, sticky="w")
        ttk.Label(io, text="Focus:").grid(row=6, column=0, sticky="w")
        ttk.Scale(io, from_=0.0, to=1.0, variable=self.primary_focus, command=lambda e: self.update_primary_focus_label()).grid(row=6, column=1, sticky="ew")
        self.primary_focus_label = ttk.Label(io, text="0.75"); self.primary_focus_label.grid(row=6, column=2, sticky="w")

        pv = ttk.LabelFrame(m, text="Live Preview & Review", padding="10"); pv.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.cv = tk.Canvas(pv, width=480, height=270, bg="black"); self.cv.pack(pady=5)
        self.rv_btn = ttk.Button(pv, text="REVIEW WITH AUDIO", command=self.review_render, state=tk.DISABLED); self.rv_btn.pack(fill=tk.X)
        ttk.Label(pv, text="Click REVIEW to watch with full audio sync.", wraplength=450, justify=tk.CENTER).pack(pady=5)

        set_f = ttk.LabelFrame(m, text="Parameters", padding="10"); set_f.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        set_f.columnconfigure(1, weight=1)
        ttk.Checkbutton(set_f, text="Beat Sync", variable=self.beat_sync).grid(row=0, column=0, sticky="w")
        ttk.Label(set_f, text="Beat interval:").grid(row=1, column=0)
        ttk.Spinbox(set_f, from_=1, to=64, textvariable=self.beat_step, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(set_f, text="beats").grid(row=1, column=2, sticky="w")
        ttk.Label(set_f, text="Beat variation:").grid(row=2, column=0)
        ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.beat_variation, command=lambda e: self.update_beat_variation_label()).grid(row=2, column=1, sticky="ew")
        self.beat_variation_label = ttk.Label(set_f, text="0.00"); self.beat_variation_label.grid(row=2, column=2)
        ttk.Label(set_f, text="Duration (no Beat Sync):").grid(row=3, column=0)
        ttk.Scale(set_f, from_=0.01, to=1.0, variable=self.duration, command=lambda e: self.l_dur.config(text=f"{self.duration.get():.2f}")).grid(row=3, column=1, sticky="ew")
        self.l_dur = ttk.Label(set_f, text="0.10"); self.l_dur.grid(row=3, column=2)
        ttk.Label(set_f, text="Coherence:").grid(row=4, column=0)
        ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.coherence, command=lambda e: self.l_coh.config(text=f"{self.coherence.get():.2f}")).grid(row=4, column=1, sticky="ew")
        self.l_coh = ttk.Label(set_f, text="0.20"); self.l_coh.grid(row=4, column=2)
        ttk.Label(set_f, text="Sensitivity:").grid(row=5, column=0)
        ttk.Scale(set_f, from_=0.1, to=3.0, variable=self.sensitivity, command=lambda e: self.l_sen.config(text=f"{self.sensitivity.get():.2f}")).grid(row=5, column=1, sticky="ew")
        self.l_sen = ttk.Label(set_f, text="1.00"); self.l_sen.grid(row=5, column=2)
        ttk.Label(set_f, text="FPS:").grid(row=6, column=0)
        ttk.Spinbox(set_f, from_=1, to=120, textvariable=self.fps, width=5).grid(row=6, column=1, sticky="w")
        ttk.Label(set_f, text="Style name:").grid(row=7, column=0)
        ttk.Entry(set_f, textvariable=self.style_name).grid(row=7, column=1, sticky="ew")
        ttk.Button(set_f, text="Save Style", command=self.save_named_style).grid(row=7, column=2, sticky="ew")
        ttk.Label(set_f, text="Load style:").grid(row=8, column=0)
        self.style_combo = ttk.Combobox(set_f, textvariable=self.style_choice, state="readonly")
        self.style_combo.grid(row=8, column=1, sticky="ew")
        ttk.Button(set_f, text="Load Style", command=self.load_named_style).grid(row=8, column=2, sticky="ew")
        ttk.Button(set_f, text="Delete Style", command=self.delete_named_style).grid(row=9, column=2, sticky="ew")

        fx = ttk.LabelFrame(m, text="Effects", padding="10"); fx.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        fx.columnconfigure(1, weight=1)
        self.add_effect_control(fx, 0, "Pixelate", self.pixelate, "pixelate")
        self.add_effect_control(fx, 1, "Flash", self.flash, "flash")
        self.add_effect_control(fx, 2, "Rewind", self.rewind, "rewind", 1.0)
        self.add_effect_control(fx, 3, "RGB Shift", self.rgb_shift, "rgb_shift")
        self.add_effect_control(fx, 4, "Shake", self.shake, "shake")
        self.add_effect_control(fx, 5, "Ghosting", self.ghosting, "ghosting")

        log_f = ttk.LabelFrame(m, text="Engine Log", padding="5"); log_f.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        self.log_t = tk.Text(log_f, height=8, font=('Consolas', 9)); self.log_t.pack(fill=tk.BOTH, expand=True)
        self.pg = ttk.Progressbar(m, orient=tk.HORIZONTAL, mode='determinate'); self.pg.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.btn = ttk.Button(m, text="RENDER", command=self.start_p); self.btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

    def log_msg(self, msg):
        self.root.after(0, self._log_msg_ui, msg)
    def _log_msg_ui(self, msg):
        self.log_t.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_t.see(tk.END)
    def add_effect_control(self, parent, row, label, enabled_var, amount_name, max_value=2.0):
        ttk.Checkbutton(parent, text=label, variable=enabled_var).grid(row=row, column=0, sticky="w")
        ttk.Scale(parent, from_=0.0, to=max_value, variable=self.effect_amounts[amount_name], command=lambda e, name=amount_name: self.update_effect_amount_label(name)).grid(row=row, column=1, sticky="ew", padx=5)
        self.effect_amount_labels[amount_name] = ttk.Label(parent, text=f"{self.effect_amounts[amount_name].get():.2f}", width=5)
        self.effect_amount_labels[amount_name].grid(row=row, column=2, sticky="e")
    def update_effect_amount_label(self, amount_name):
        self.effect_amount_labels[amount_name].config(text=f"{self.effect_amounts[amount_name].get():.2f}")
    def update_primary_focus_label(self):
        if self.primary_focus_label:
            self.primary_focus_label.config(text=f"{self.primary_focus.get():.2f}")
    def update_beat_variation_label(self):
        if self.beat_variation_label:
            self.beat_variation_label.config(text=f"{self.beat_variation.get():.2f}")
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
    def with_builtin_styles(self, styles):
        deleted = set(styles.get("_deleted_builtin_styles", []))
        merged = json.loads(json.dumps({name: style for name, style in BUILTIN_STYLES.items() if name not in deleted}))
        merged.update(styles)
        return merged
    def save_styles_file(self):
        with open(STYLE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.styles, f, indent=2, sort_keys=True)
    def style_names(self):
        return sorted(name for name in self.styles if not name.startswith("_"))
    def refresh_style_choices(self):
        names = self.style_names()
        if self.style_combo:
            self.style_combo["values"] = names
        if names and self.style_choice.get() not in names:
            self.style_choice.set(names[0])
    def collect_settings(self):
        return {
            "export_mode_label": self.export_mode_label.get(),
            "duration": self.duration.get(),
            "fps": self.fps.get(),
            "render_mode": self.render_mode.get(),
            "snippet_duration": self.snippet_duration.get(),
            "coherence": self.coherence.get(),
            "sensitivity": self.sensitivity.get(),
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
            },
            "effect_amounts": {name: var.get() for name, var in self.effect_amounts.items()},
            "primary_enabled": self.primary_enabled.get(),
            "primary_focus": self.primary_focus.get(),
        }
    def collect_project(self):
        return {
            "version": 1,
            "inputs": self.inputs,
            "audio": self.audio.get(),
            "output": self.output.get(),
            "primary_video_idx": self.primary_video_idx,
            "primary_video_label": self.primary_video_label.get(),
            "style_name": self.style_name.get(),
            "style_choice": self.style_choice.get(),
            "settings": self.collect_settings(),
        }
    def apply_project(self, project):
        if not isinstance(project, dict):
            raise ValueError("Invalid project file")
        self.inputs = list(project.get("inputs", []))
        self.lb.delete(0, tk.END)
        for path in self.inputs:
            self.lb.insert(tk.END, os.path.basename(path))
        self.audio.set(project.get("audio", ""))
        self.output.set(project.get("output", self.output.get()))
        self.primary_video_idx = int(project.get("primary_video_idx", 0) or 0)
        if self.inputs and not (0 <= self.primary_video_idx < len(self.inputs)):
            self.primary_video_idx = 0
        if self.inputs:
            self.primary_video_label.set(f"Primary: {os.path.basename(self.inputs[self.primary_video_idx])}")
        else:
            self.primary_video_label.set(project.get("primary_video_label", "Primary: first input"))
        self.style_name.set(project.get("style_name", self.style_name.get()))
        if project.get("style_choice"):
            self.style_choice.set(project["style_choice"])
        self.apply_settings(project.get("settings", {}))
    def save_project(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".glitchsync.json",
            filetypes=[("GlitchSync Project", "*.glitchsync.json"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.collect_project(), f, indent=2, sort_keys=True)
        except Exception as e:
            return messagebox.showerror("Error", f"Could not save project: {e}")
        messagebox.showinfo("Saved", f"Saved project: {os.path.basename(path)}")
    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("GlitchSync Project", "*.glitchsync.json"), ("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                project = json.load(f)
            self.apply_project(project)
        except Exception as e:
            return messagebox.showerror("Error", f"Could not load project: {e}")
        messagebox.showinfo("Loaded", f"Loaded project: {os.path.basename(path)}")
    def apply_settings(self, settings):
        if not isinstance(settings, dict):
            return
        if settings.get("export_mode_label") in EXPORT_MODE_LABELS:
            self.export_mode_label.set(settings["export_mode_label"])
        self.duration.set(settings.get("duration", self.duration.get()))
        self.fps.set(settings.get("fps", self.fps.get()))
        if settings.get("render_mode") in ("Full", "Snippet"):
            self.render_mode.set(settings["render_mode"])
        self.snippet_duration.set(settings.get("snippet_duration", self.snippet_duration.get()))
        self.coherence.set(settings.get("coherence", self.coherence.get()))
        self.sensitivity.set(settings.get("sensitivity", self.sensitivity.get()))
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
        for name, value in settings.get("effect_amounts", {}).items():
            if name in self.effect_amounts:
                self.effect_amounts[name].set(value)
                self.update_effect_amount_label(name)
        self.primary_enabled.set(settings.get("primary_enabled", self.primary_enabled.get()))
        self.primary_focus.set(settings.get("primary_focus", self.primary_focus.get()))
        self.l_dur.config(text=f"{self.duration.get():.2f}")
        self.l_coh.config(text=f"{self.coherence.get():.2f}")
        self.l_sen.config(text=f"{self.sensitivity.get():.2f}")
        self.update_beat_variation_label()
        self.update_primary_focus_label()
    def save_named_style(self):
        name = self.style_name.get().strip()
        if not name:
            return messagebox.showerror("Error", "Style name is required")
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
    def delete_named_style(self):
        name = self.style_choice.get() or self.style_name.get().strip()
        if not name or name not in self.styles:
            return messagebox.showerror("Error", "Select a saved style to delete")
        if name == "_last":
            return messagebox.showerror("Error", "The automatic last-used style cannot be deleted")
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
    def add_v(self):
        f = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        for x in f:
            if x not in self.inputs:
                self.inputs.append(x); self.lb.insert(tk.END, os.path.basename(x))
                if len(self.inputs) == 1:
                    self.primary_video_idx = 0
                    self.primary_video_label.set(f"Primary: {os.path.basename(x)}")
    def add_a(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a")])
        if f: self.audio.set(f)
    def set_primary_video(self):
        sel = self.lb.curselection()
        if not sel: return
        self.primary_video_idx = sel[0]
        self.primary_video_label.set(f"Primary: {os.path.basename(self.inputs[self.primary_video_idx])}")
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
        self.btn.config(state=tk.DISABLED); self.rv_btn.config(state=tk.DISABLED); self.last_progress_val = 0
        threading.Thread(target=self.run_e, daemon=True).start()
    def run_e(self):
        try:
            export_mode = EXPORT_MODE_LABELS[self.export_mode_label.get()]
            effect_amounts = {name: var.get() for name, var in self.effect_amounts.items()}
            primary_idx = self.primary_video_idx if self.primary_enabled.get() and self.inputs else None
            primary_focus = self.primary_focus.get() if self.primary_enabled.get() else 0.0
            render_limit = self.snippet_duration.get() if self.render_mode.get() == "Snippet" else None
            p = GlitchProcessor(self.inputs, self.audio.get(), self.output.get(), self.duration.get(), self.fps.get(), self.pixelate.get(), self.flash.get(), self.rewind.get(), self.rgb_shift.get(), self.shake.get(), self.ghosting.get(), self.beat_sync.get(), self.coherence.get(), self.sensitivity.get(), export_mode, self.update_p, self.log_msg, self.display_frame, effect_amounts, primary_idx, primary_focus, self.beat_step.get(), self.beat_variation.get(), render_limit)
            p.process()
            self.root.after(0, self._render_complete_ui, export_mode)
        except Exception as e:
            self.log_msg(f"ERROR: {e}")
            self.root.after(0, self._render_error_ui, str(e))
        finally:
            self.root.after(0, self._render_finished_ui)
    def _render_complete_ui(self, export_mode):
        if export_mode == EXPORT_FINAL_VIDEO:
            self.rv_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Complete!")
    def _render_error_ui(self, error):
        messagebox.showerror("Error", error)
    def _render_finished_ui(self):
        self.pg.stop(); self.pg.config(mode='determinate'); self.btn.config(state=tk.NORMAL); self.pg['value'] = 0
    def display_frame(self, f):
        h, w = f.shape[:2]; s = min(480/w, 270/h); nw, nh = int(w*s), int(h*s)
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((nw, nh), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        self.root.after(0, self._update_cv, img_tk)
    def _update_cv(self, img_tk):
        self.cv.delete("all")
        self.cv.create_image(240, 135, anchor=tk.CENTER, image=img_tk); self.cv._img_ref = img_tk
    def review_render(self):
        self.log_msg(f"Launching ffplay: {self.output.get()}")
        subprocess.Popen(['ffplay', '-i', self.output.get()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs='+'); p.add_argument("--audio"); p.add_argument("--output", default="output.mp4"); p.add_argument("--beat_sync", action="store_true"); p.add_argument("--gui", action="store_true")
    p.add_argument("--export-mode", choices=[EXPORT_FINAL_VIDEO, EXPORT_CUT_AWARE_MLT, EXPORT_CLIP_MLT], default=EXPORT_FINAL_VIDEO)
    p.add_argument("--beat-step", type=int, default=1)
    p.add_argument("--beat-variation", type=float, default=0.0)
    p.add_argument("--render-limit", type=float)
    args = p.parse_args()
    if args.gui or not (args.inputs and args.audio):
        r = tk.Tk(); g = GlitchGUI(r); r.mainloop()
    else:
        proc = GlitchProcessor(args.inputs, args.audio, args.output, beat_sync=args.beat_sync, export_mode=args.export_mode, progress_callback=lambda c, t: print(f"Progress: {c}/{t}", end='\r'), beat_step=args.beat_step, beat_variation=args.beat_variation, render_limit=args.render_limit)
        proc.process()
