# QR Code Position Tracker

[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-blue)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)

Real-time QR code tracking system with position plotting and data export capabilities.

![Demo Screenshot](docs/screenshot.png)

## Features
- 📷 Real-time webcam QR code detection
- 📈 Live x-position vs time plotting
- 🎨 Unique colors per QR code
- 📥 CSV data export
- 🔄 Persistent tracking between frames
- 📏 Pixel position tracking

## Requirements
- Python 3.8+
- Webcam
- Standard QR codes

## Installation
```bash
git clone https://github.com/yourusername/qr-position-tracker.git
cd qr-position-tracker
pip install opencv-python matplotlib numpy
