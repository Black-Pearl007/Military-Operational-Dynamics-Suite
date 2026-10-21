# MODLS - Military Operational Dynamics and Logistics Suite

A Streamlit app for object tracking, cadet face registration and recognition,
armory data management and visualization, intrusion detection, and a software
night-vision module. It runs on a normal webcam or on uploaded videos.

## Features

| Page | What it does |
|------|--------------|
| Object Tracking | Lock onto an object and follow it. Recovers the target after it is hidden (see below). |
| Cadet Face Registration | Register a cadet's face from a webcam snapshot or an uploaded photo. |
| Cadet Face Recognition | Identify registered cadets from a snapshot or live webcam. |
| Armory Data Registration | Add weapons records to a CSV file. |
| Armory Data Visualization | Histograms, box plots and distribution charts for any column. |
| Intrusion Detection | Motion detection on a webcam or video, with optional snapshots. |
| Night Vision | Low-light enhancement with noise removal and a green tint. |

### Object tracking and target recovery

The tracker can use ViT, CSRT, KCF or MIL. On top of that, three optional aids
help it find the target again after it is lost:

- **YOLO detector (YOLOX):** after losing the target, it looks for objects of the
  same class (for example "cell phone"). Covers the 80 COCO classes; other objects
  fall back to template search.
- **Kalman filter:** predicts where the hidden target is going and shows a yellow
  `PREDICTED` box. The prediction guides the search.
- **Recognition model (MobileNetV2 features):** compares appearance so a similar
  looking object is not mistaken for the target. If two candidates match about
  equally well, the tracker refuses to guess.

Set these in the "Recovery aids" box on the Object Tracking page **before** locking
a target. The "Re-identification strictness" slider trades fewer wrong locks
(higher) against faster recovery (lower).

### Camera options (sidebar)

- **Camera index:** choose which camera to use.
- **Camera resolution:** 640x480, 1280x720 or 1920x1080. The camera may pick the
  nearest size it supports.
- **Processing width:** frames are scaled to this width before processing. Larger
  means more detail but slower.
- **Infrared / grayscale camera:** tick this when the selected camera is an IR camera.
  Colour matching is replaced by brightness matching.

## Installation

Python 3.9 or newer is recommended.

```bash
pip install -r requirements.txt
```

Use `opencv-contrib-python` (not the headless build). The face and tracking features
need the contrib modules and the pop-up target selector needs GUI support.

## Run

```bash
streamlit run modls_app.py
```

## Models

Models download automatically into `./models` the first time a feature needs them,
so an internet connection is required once:

| Model | Used for |
|-------|----------|
| YuNet + SFace | Face detection and recognition |
| ViT tracker | Object tracking |
| YOLOX-s | Re-finding a lost target |
| MobileNetV2 | Telling look-alike objects apart |

All come from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo). If the YOLOX or
MobileNetV2 download fails, the tracker warns you and falls back to its colour and
template search.

## Data created by the app

- `registered_faces/` holds face embeddings and cropped face images. This is personal
  data, so do not commit it to a public repository.
- `armory_data.csv` holds the armory records you add.
- `intrusions/` holds snapshots saved by Intrusion Detection.

## Limitations

- Software cannot create light. Night Vision only amplifies what the webcam captures.
  For true night vision use an IR-capable camera.
- The tracker cannot tell apart two visually identical objects. A lone identical
  object appearing far from the predicted position may be accepted as the target.
- The default re-identification threshold is a starting point. Tune it on your own
  objects and lighting.
- Live webcam pages are meant for local use. In a browser-only deployment, use the
  snapshot and upload modes.

## Project layout

```
modls_app.py      main application
nightvision.py    standalone night-vision demo (OpenCV window)
requirements.txt  dependencies
.gitignore        keeps models, face data and snapshots out of git
```
<img width="1910" height="870" alt="Screenshot 2026-10-21 203813" src="https://github.com/user-attachments/assets/0a97888b-9d26-4d8e-ae76-cfdc580cd895" />
