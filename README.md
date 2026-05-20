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
- **Per-Effect Amount Controls**: Fine tune each enabled effect independently instead of relying only on on/off toggles.
- **Beat Interval and Variation**: Cut every N beats during Beat Sync, with optional extra cuts on skipped beats for variety.
- **Saved Styles**: Load starter styles, save/delete named settings presets, and automatically restore the last-used controls on startup.
- **Snippet Rendering**: Render a short preview section, such as 30 seconds, before committing to a full export.
- **Project Files**: Save and load full project state, including media paths, output path, primary focus selection, style, and unsaved setting tweaks.
- **Optional Primary Focus**: Select one input video as the primary visual source and bias the cut selection toward it while still allowing alternate videos for variation.
- **Live Render Preview**: Watch the video being built in real-time.
- **Professional Review**: One-click high-performance preview with full audio sync using `ffplay`.
- **Source Coherence**: Control how often the engine switches between different source videos.
- **Shotcut Export Modes**:
  - **Final video (MP4)**: Existing flattened MP4 export with audio muxed into the video.
  - **Cut-aware Shotcut MLT (ZIP)**: A Shotcut project archive with the rendered video split into editable timeline cuts and the original audio on its own track.
  - **Clip Shotcut MLT (ZIP)**: A Shotcut project archive with each rendered segment as its own video clip and the original audio on its own track.

## CLI Export Modes

Use `--export-mode final_video`, `--export-mode cut_aware_mlt`, or `--export-mode clip_mlt`.
Use `--beat-step 8 --beat-variation 0.2` with `--beat_sync` to make main cuts every 8 beats while allowing occasional extra cuts.

The Shotcut modes create a ZIP archive containing `glitchsync_project.mlt` and relative `media/` files.
