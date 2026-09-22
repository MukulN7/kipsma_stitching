# MicroStitch — Project Context & Specification

## 1. Executive Summary
**MicroStitch** is a lightweight, real-time microscope image-stitching application built in Python. It captures frames from a fixed USB microscope camera while a specimen slide is moved underneath it, registers overlapping frames in real-time, incrementally builds a global high-resolution mosaic, blends overlapping regions, and displays the expanding mosaic via an intuitive Streamlit interface.

This document serves as the **single source of truth** for all architectural decisions, technology choices, reference evaluations, and development plans.

---

## 2. Physical System & Motion Model
* **Physical Configuration**: A USB microscope camera fixed in a stationary position overhead, pointing down at a specimen slide.
* **User Workflow**: The operator slowly slides the specimen manually or mechanically under the field of view.
* **Primary Motion Model**: 2D Rigid Translation ($\Delta x, \Delta y$). Because the camera optics and height relative to the slide remain constant, scale and rotation are negligible.
* **Registration Focus**: Phase correlation (FFT-based shift estimation) and 2D translation registration.

---

## 3. Technology Stack

| Layer / Component | Chosen Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Ecosystem support for computer vision & science |
| **Computer Vision / Algorithmic Core** | OpenCV (`opencv-python`), NumPy | High-performance C++ underlying operations (`cv2.phaseCorrelate`, warpAffine, blending) |
| **User Interface** | Streamlit | Rapid, clean web UI running locally on the laptop connected to the microscope |
| **Image Processing & Export** | Pillow (`PIL`), `tifffile` | Support for standard image formats and multi-page/large scientific TIFF export |
| **Reuse Components** | OpenCV stitching modules, Python phase correlation | Standardized, permissive open-source routines |

---

## 4. Key Engineering Directives & Constraints

1. **Practical & Lightweight**: Avoid unnecessary framework abstractions or over-engineering. Keep code readable, modular, and maintainable.
2. **Reuse Proven Code**: Adapt standard open-source algorithms (such as OpenCV's phase correlation and blending primitives). Do **not** reimplement Fourier transforms or feature detectors from scratch.
3. **Single Stitching Engine**: Maintain ONE unified, robust translation-based stitching pipeline. Do not build parallel or competing stitching backends.
4. **No Unnecessary Machine Learning**: Rely on proven deterministic computer vision algorithms (Phase Correlation / Feature-based translation RANSAC). ML is out of scope unless traditional methods demonstrably fail on actual slide samples.
5. **Local Workstation Deployment**: Designed to run locally on the workstation/laptop directly connected to the USB camera.

---

## 5. System Architecture & Pipeline Flow

```
[ Camera / Frame Source ] 
           │
           ▼
[ Movement Detector ] ──(Static Frame)──► [ Skip Processing ]
           │
     (Motion Detected)
           ▼
[ Registration Engine ] ──► Estimate Relative Shift (Δx, Δy)
           │
           ▼
[ Global Mosaic Manager ] ──► Update Global Coordinates (X, Y) & Canvas Boundaries
           │
           ▼
[ Image Blender ] ──► Apply Feathering / Linear Blending on Overlaps
           │
           ▼
[ Streamlit UI Display ] ──► Render Live Feed, Global Mosaic & Status Controls
```

### Module Breakdown

1. **Frame Source (`frame_source.py`)**:
   - Abstract interface supporting both `USBCameraSource` (OpenCV `VideoCapture`) and `SampleImageSequenceSource` / `VideoFileSource`.
   - Allows seamless offline development today and live hardware integration tomorrow.

2. **Movement Detector (`movement_detector.py`)**:
   - Computes frame-to-frame displacement or structural difference (e.g., Frame Difference / SSIM threshold / Phase Correlation shift magnitude).
   - Prevents stitching identical frames when the slide is stationary, reducing computation and avoiding position drift accumulation.

3. **Registration Engine (`stitching_engine.py`)**:
   - **Primary**: Phase Correlation (`cv2.phaseCorrelate` with Hanning window) for fast sub-pixel 2D shift $(\Delta x, \Delta y)$ computation.
   - **Fallback**: ORB feature extraction + translation RANSAC when phase correlation confidence falls below threshold (e.g. low-texture regions).

4. **Global Mosaic & Canvas Tracker (`mosaic_canvas.py`)**:
   - Tracks cumulative global coordinate offsets $(X_{global}, Y_{global})$.
   - Manages an expanding image canvas or dynamic bounding grid to accommodate multi-directional slide movement (left/right, up/down).

5. **Image Blender (`blender.py`)**:
   - Applies feathering or linear weighted alpha blending across overlapping tile margins to eliminate visible seam lines in real time.

6. **Streamlit User Interface (`app.py`)**:
   - **Live Feed Panel**: Shows real-time video feed from the USB camera/sample source.
   - **Stitched Mosaic Panel**: Displays the current composite global mosaic view.
   - **Control Buttons**: `Start Scan`, `Pause Scan`, `Reset Mosaic`, `Save Mosaic`.
   - **Status & Info Metrics**: Camera Connection Status, Processed Frame Count, FPS / Latency, Motion Status, Mosaic Canvas Dimensions ($\text{Width} \times \text{Height}$).

---

## 6. Reference Repositories & Legal Provenance

The following four reference repositories located in `references/` were inspected:

| Repository | Focus & Key Concepts | License | Planned Usage / Adaptation |
| :--- | :--- | :--- | :--- |
| `references/MIST` | NIST Microscopy Image Stitching Tool (Phase correlation & grid optimization) | Public Domain (US Govt Work) | Conceptual reference for FFT phase correlation shift calculation & grid mapping |
| `references/ashlar` | ASHLAR Microscopy Image Registration (Pairwise phase correlation + global optimization) | MIT License | Reference for sub-pixel phase cross-correlation alignment and translation-based microscopy registration |
| `references/multifocal-stitching` | Multi-focal microscope image translation & registration | MIT License | Reference for microscopy translation registration techniques |
| `references/stitching` | OpenStitching OpenCV Python wrapper | Apache License 2.0 | Reference for OpenCV blending, seam management, and feature estimation utilities |

*Note*: All third-party routines or adapted snippets integrated into MicroStitch will be documented with exact version/commit and license details in `THIRD_PARTY_LICENSES.md`.

---

## 7. Immediate Development Roadmap

* **Stage 1 (Today — Offline Development)**:
  - Establish project structure and core python dependencies (`opencv-python`, `streamlit`, `numpy`, `pillow`, `tifffile`).
  - Implement Frame Source abstraction with simulated/sample image sequence.
  - Build Phase Correlation Registration Engine and Global Mosaic Canvas accumulator.
  - Develop Streamlit UI layout with controls (`Start`, `Pause`, `Reset`, `Save`).

* **Stage 2 (Tomorrow — Live Hardware Testing)**:
  - Connect USB microscope camera to laptop.
  - Switch Frame Source to `USBCameraSource`.
  - Calibrate movement detection thresholds, registration parameters, and blending performance under actual physical slide movement.
