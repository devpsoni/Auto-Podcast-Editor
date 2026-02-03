# AutoPodcastEditor

An automated video editing tool that intelligently switches between multiple camera angles in podcast recordings based on audio levels and facial emotion detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

AutoPodcastEditor analyzes multiple synchronized video files and automatically creates a single edited output by:

1. **Audio Analysis** — Detects which speaker is talking based on audio levels
2. **Emotion Detection** — Uses facial emotion recognition to identify expressive moments worth highlighting
3. **Smart Switching** — Combines both signals to switch camera angles at natural breakpoints

Perfect for podcasters, interviewers, and content creators who record with multiple cameras but want to automate the tedious editing process.

## Features

- Multi-camera support (2+ video inputs)
- Audio-based speaker detection with configurable thresholds
- Real-time facial emotion detection using FER (Facial Expression Recognition)
- Audio overlap option to maintain all microphone feeds
- Progress tracking with time estimation
- Configurable output settings
- GUI interface for easy operation

## Requirements

### System Requirements
- Python 3.8+
- 8GB RAM (recommended)
- FFmpeg installed and accessible in PATH

### Python Dependencies

```bash
pip install tensorflow
pip install moviepy
pip install opencv-python
pip install numpy
pip install scipy
pip install fer
pip install psutil
```

Or install all at once:

```bash
pip install tensorflow moviepy opencv-python numpy scipy fer psutil
```

### FFmpeg Installation

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoPodcastEditor.git
cd AutoPodcastEditor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

### Quick Start

1. Launch the application
2. Click **"Add File"** to add your synchronized video files
3. Adjust settings if needed (or use defaults)
4. Click **"Process"** to start editing
5. Find your output in the `output/` folder

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Sample Rate** | 24 | Audio samples per second for analysis |
| **Threshold** | 5 | Consecutive samples needed before switching |
| **Exceeds By** | 4 | How much louder a speaker must be to trigger switch |
| **Face Threshold** | 0.5 | Confidence threshold for emotion detection (0-1) |
| **Overlap Audio** | Enabled | Mix audio from all sources vs. only active speaker |

### Tips for Best Results

- **Sync your clips** — All input videos must be synchronized at the start
- **Consistent audio** — Each speaker should have their own microphone/audio track
- **Good lighting** — Helps with facial emotion detection accuracy
- **Start with defaults** — The default settings work well for most podcasts

## Project Structure

```
AutoPodcastEditor/
├── main.py          # Main application
├── tmp/             # Temporary files (auto-created)
├── output/          # Output videos (auto-created)
└── README.md
```

## How It Works

### Audio Analysis Pipeline

1. Extract audio from each video using FFmpeg
2. Downsample to configured sample rate
3. Normalize audio levels across all tracks
4. Compare levels frame-by-frame
5. Apply threshold smoothing to prevent rapid switching

### Emotion Detection Pipeline

1. Sample frames at configured intervals
2. Detect faces using MTCNN
3. Classify emotions using FER
4. Override audio-based switching for high-confidence emotional moments

### Video Assembly

1. Identify all switch points from combined audio/emotion analysis
2. Extract subclips from appropriate source videos
3. Optionally composite audio from all sources
4. Concatenate clips into final output

## Troubleshooting

### Common Issues

**"Failed to initialize face detector"**
- Ensure TensorFlow is properly installed
- Try: `pip install --upgrade tensorflow fer`

**"Could not open video file"**
- Check file path contains no special characters
- Ensure video codec is supported (H.264 recommended)

**Low memory warning**
- Close other applications
- Process shorter clips
- Reduce video resolution before processing

**FFmpeg not found**
- Ensure FFmpeg is installed and in your system PATH
- Try running `ffmpeg -version` in terminal to verify

### Performance Tips

- Use H.264/MP4 format for fastest processing
- Lower resolution videos process faster
- SSD storage improves read/write speeds
- Increase `CHECK_EMOTION_INTERVAL` to reduce emotion detection overhead

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FER](https://github.com/justinshenk/fer) for facial emotion recognition
- [MoviePy](https://zulko.github.io/moviepy/) for video editing capabilities
- [OpenCV](https://opencv.org/) for computer vision processing

---

Made with ☕ for podcast creators everywhere
