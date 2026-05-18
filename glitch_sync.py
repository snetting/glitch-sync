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

def apply_pixelate(frame, intensity, high_intensity, sensitivity=1.0):
    """
    Improved: Only triggers on high-frequency transients.
    Designed to be an occasional accent rather than a theme.
    """
    # Only trigger if there is a significant high-frequency spike AND a random roll
    # This makes it feel much more like an intentional 'glitch'
    if high_intensity < 0.45 or random.random() > 0.25: 
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
                 progress_callback=None, log_callback=None, frame_callback=None):
        self.inputs, self.audio, self.output = inputs, audio, output
        self.duration, self.fps = duration, fps
        self.pixelate, self.flash, self.rewind = pixelate, flash, rewind
        self.rgb_shift, self.shake, self.ghosting = rgb_shift, shake, ghosting
        self.beat_sync, self.coherence, self.sensitivity = beat_sync, coherence, sensitivity
        self.export_mode = export_mode
        self.progress_callback, self.log_callback, self.frame_callback = progress_callback, log_callback, frame_callback
        self.stop_requested = False
        self.current_vid_idx = 0

    def log(self, msg):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

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
        candidates_current, candidates_other = [], []
        for b in range(max(0, target_b - search_range), min(255, target_b + search_range) + 1):
            for v_idx, f_idx in frame_db[b]:
                if v_idx == current_vid_idx: candidates_current.append((v_idx, f_idx))
                else: candidates_other.append((v_idx, f_idx))
        if candidates_current and (random.random() < self.coherence or not candidates_other):
            return random.choice(candidates_current)
        if candidates_other: return random.choice(candidates_other)
        offset = 1
        while offset < 256:
            low, high = target_b - offset, target_b + offset
            fallback = []
            if low >= 0: fallback.extend(frame_db[low])
            if high <= 255: fallback.extend(frame_db[high])
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
            cut_times = librosa.frames_to_time(beats, sr=sr)
            self.log(f"  Tempo: {tempo_val:.1f} BPM")
        else:
            cut_times = np.arange(0, audio_duration, self.duration)

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
        for i in range(len(cut_times)):
            if self.stop_requested: break
            t_start = cut_times[i]
            dur = (cut_times[i+1] if i+1 < len(cut_times) else audio_duration) - t_start
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
                if self.rewind and curr_rms > 0.75: chunk = chunk[::-1]

                segment_frames = []
                for f in chunk:
                    if f.shape[:2] != (height, width): f = cv2.resize(f, (width, height))
                    if self.pixelate: f = apply_pixelate(f, curr_rms, curr_highs, self.sensitivity)
                    if self.flash: f = apply_brightness_boost(f, curr_rms, self.sensitivity)
                    if self.rgb_shift: f = apply_rgb_shift(f, curr_highs, self.sensitivity)
                    if self.shake: f = apply_shake(f, curr_bass, self.sensitivity)
                    if self.ghosting: f = apply_ghosting(f, prev_f, curr_mids, self.sensitivity)
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
                subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-i', self.audio, '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-c:a', 'aac', '-b:a', '192k', '-shortest', self.output], capture_output=True)
            else:
                self.export_shotcut_archive(temp_video, segments, width, height, package_dir)
        if os.path.exists(temp_video): os.remove(temp_video)
        if package_dir and os.path.exists(package_dir): shutil.rmtree(package_dir)
        self.log("Done!")

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
        self.root.geometry("1100x950")
        self.inputs, self.audio = [], tk.StringVar()
        self.output = tk.StringVar(value=f"glitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        self.export_mode_label = tk.StringVar(value="Final video (MP4)")
        self.duration, self.fps, self.coherence, self.sensitivity = tk.DoubleVar(value=0.10), tk.IntVar(value=30), tk.DoubleVar(value=0.20), tk.DoubleVar(value=1.0)
        self.beat_sync = tk.BooleanVar(value=True)
        self.pixelate, self.flash, self.rewind, self.rgb_shift, self.shake, self.ghosting = [tk.BooleanVar(value=True) for _ in range(6)]
        self.last_progress_val = 0
        self.build_ui()

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

        pv = ttk.LabelFrame(m, text="Live Preview & Review", padding="10"); pv.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.cv = tk.Canvas(pv, width=480, height=270, bg="black"); self.cv.pack(pady=5)
        self.rv_btn = ttk.Button(pv, text="REVIEW WITH AUDIO", command=self.review_render, state=tk.DISABLED); self.rv_btn.pack(fill=tk.X)
        ttk.Label(pv, text="Click REVIEW to watch with full audio sync.", wraplength=450, justify=tk.CENTER).pack(pady=5)

        set_f = ttk.LabelFrame(m, text="Parameters", padding="10"); set_f.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Checkbutton(set_f, text="Beat Sync", variable=self.beat_sync).grid(row=0, column=0, sticky="w")
        ttk.Label(set_f, text="Dur:").grid(row=1, column=0)
        ttk.Scale(set_f, from_=0.01, to=1.0, variable=self.duration, command=lambda e: self.l_dur.config(text=f"{self.duration.get():.2f}")).grid(row=1, column=1, sticky="ew")
        self.l_dur = ttk.Label(set_f, text="0.10"); self.l_dur.grid(row=1, column=2)
        ttk.Label(set_f, text="Coh:").grid(row=2, column=0)
        ttk.Scale(set_f, from_=0.0, to=1.0, variable=self.coherence, command=lambda e: self.l_coh.config(text=f"{self.coherence.get():.2f}")).grid(row=2, column=1, sticky="ew")
        self.l_coh = ttk.Label(set_f, text="0.20"); self.l_coh.grid(row=2, column=2)
        ttk.Label(set_f, text="Sens:").grid(row=3, column=0)
        ttk.Scale(set_f, from_=0.1, to=3.0, variable=self.sensitivity, command=lambda e: self.l_sen.config(text=f"{self.sensitivity.get():.2f}")).grid(row=3, column=1, sticky="ew")
        self.l_sen = ttk.Label(set_f, text="1.00"); self.l_sen.grid(row=3, column=2)
        ttk.Label(set_f, text="FPS:").grid(row=4, column=0)
        ttk.Spinbox(set_f, from_=1, to=120, textvariable=self.fps, width=5).grid(row=4, column=1, sticky="w")

        fx = ttk.LabelFrame(m, text="Effects", padding="10"); fx.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        ttk.Checkbutton(fx, text="Pixelate", variable=self.pixelate).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(fx, text="Flash", variable=self.flash).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(fx, text="Rewind", variable=self.rewind).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(fx, text="RGB Shift", variable=self.rgb_shift).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(fx, text="Shake", variable=self.shake).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(fx, text="Ghosting", variable=self.ghosting).grid(row=2, column=1, sticky="w")

        log_f = ttk.LabelFrame(m, text="Engine Log", padding="5"); log_f.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        self.log_t = tk.Text(log_f, height=8, font=('Consolas', 9)); self.log_t.pack(fill=tk.BOTH, expand=True)
        self.pg = ttk.Progressbar(m, orient=tk.HORIZONTAL, mode='determinate'); self.pg.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.btn = ttk.Button(m, text="RENDER", command=self.start_p); self.btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

    def log_msg(self, msg):
        self.log_t.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_t.see(tk.END); self.root.update_idletasks()
    def add_v(self):
        f = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        for x in f:
            if x not in self.inputs: self.inputs.append(x); self.lb.insert(tk.END, os.path.basename(x))
    def add_a(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a")])
        if f: self.audio.set(f)
    def update_p(self, curr, total):
        if curr == -1:
            if self.pg['mode'] != 'indeterminate': self.pg.config(mode='indeterminate'); self.pg.start(10)
        else:
            if self.pg['mode'] != 'determinate': self.pg.stop(); self.pg.config(mode='determinate')
            val = (curr / total) * 100
            if abs(self.last_progress_val - val) >= 0.5: self.pg['value'] = val; self.last_progress_val = val
        self.root.update_idletasks()
    def start_p(self):
        if not self.inputs or not self.audio.get(): return messagebox.showerror("Error", "Missing files")
        self.btn.config(state=tk.DISABLED); self.rv_btn.config(state=tk.DISABLED); self.last_progress_val = 0
        threading.Thread(target=self.run_e, daemon=True).start()
    def run_e(self):
        try:
            export_mode = EXPORT_MODE_LABELS[self.export_mode_label.get()]
            p = GlitchProcessor(self.inputs, self.audio.get(), self.output.get(), self.duration.get(), self.fps.get(), self.pixelate.get(), self.flash.get(), self.rewind.get(), self.rgb_shift.get(), self.shake.get(), self.ghosting.get(), self.beat_sync.get(), self.coherence.get(), self.sensitivity.get(), export_mode, self.update_p, self.log_msg, self.display_frame)
            p.process()
            if export_mode == EXPORT_FINAL_VIDEO:
                self.root.after(0, lambda: self.rv_btn.config(state=tk.NORMAL))
            messagebox.showinfo("Success", "Complete!")
        except Exception as e: self.log_msg(f"ERROR: {e}"); messagebox.showerror("Error", str(e))
        finally: self.pg.stop(); self.pg.config(mode='determinate'); self.btn.config(state=tk.NORMAL); self.pg['value'] = 0
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
    args = p.parse_args()
    if args.gui or not (args.inputs and args.audio):
        r = tk.Tk(); g = GlitchGUI(r); r.mainloop()
    else:
        proc = GlitchProcessor(args.inputs, args.audio, args.output, beat_sync=args.beat_sync, export_mode=args.export_mode, progress_callback=lambda c, t: print(f"Progress: {c}/{t}", end='\r'))
        proc.process()
