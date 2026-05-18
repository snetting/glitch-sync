# GlitchSync Pro

Audio-reactive video glitch tool that synchronizes video cuts and visual effects to the beats and frequency transients of a music track.

## Features
- **Beat Synchronization**: Automatically detects BPM and aligns cuts to the rhythm.
- **Audio-Reactive Effects**:
  - **Pixelate**: Rhythmic block-glitch triggered by high-frequency transients.
  - **RGB Shift**: Chromatic aberration synced to audio peaks.
  - **Shake**: Bass-driven camera movement.
  - **Ghosting**: Temporal motion blur synced to mids.
  - **Flash**: Brightness bursts on transients.
  - **Rewind**: Stutter/rewind effects on intense peaks.
- **Live Render Preview**: Watch the video being built in real-time.
- **Professional Review**: One-click high-performance preview with full audio sync using `ffplay`.
- **Source Coherence**: Control how often the engine switches between different source videos.
- **Shotcut Export Modes**:
  - **Final video (MP4)**: Existing flattened MP4 export with audio muxed into the video.
  - **Cut-aware Shotcut MLT (ZIP)**: A Shotcut project archive with the rendered video split into editable timeline cuts and the original audio on its own track.
  - **Clip Shotcut MLT (ZIP)**: A Shotcut project archive with each rendered segment as its own video clip and the original audio on its own track.

## CLI Export Modes

Use `--export-mode final_video`, `--export-mode cut_aware_mlt`, or `--export-mode clip_mlt`.

The Shotcut modes create a ZIP archive containing `glitchsync_project.mlt` and relative `media/` files.
