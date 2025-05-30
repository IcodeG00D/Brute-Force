

# Object Matching with CLIP & OpenCV

## Overview

This project implements a simple object matching system using OpenCV for webcam capture and OpenAI's CLIP model for image similarity detection. The goal is to open the camera feed, detect when an object from a small dataset is presented, and display the most similar image side-by-side in real time.

---

## Features

* Real-time webcam preview
* Efficient object matching using CLIP’s pretrained image embeddings
* Displays the closest matching dataset image next to the live camera feed when a match is found
* Lightweight solution tailored for small datasets (e.g., 5 images)
* Threshold tuning for balancing accuracy and detection sensitivity

---

## Technologies Used

| Technology       | Purpose                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------- |
| **OpenCV**       | Capture video from the webcam, process frames, and display results.                                            |
| **OpenAI CLIP**  | Compute semantic embeddings for images and measure similarity between the live camera feed and dataset images. |
| **PyTorch**      | Deep learning framework to run the CLIP model efficiently on CPU or GPU.                                       |
| **PIL (Pillow)** | Image preprocessing before feeding images to CLIP.                                                             |
| **NumPy**        | Handle array operations for image processing and similarity computations.                                      |

---

## Why These Technologies?

* **OpenCV** is the de facto standard for real-time video and image processing in Python, providing easy access to camera streams and visualization tools.
* **CLIP** (Contrastive Language–Image Pretraining) is a state-of-the-art model that produces rich, semantic image embeddings without requiring additional training. This makes it ideal for small projects where retraining large models isn’t feasible.
* Using **CLIP** allows comparing visual similarity at a semantic level rather than just pixel matching or local features, improving robustness to lighting, angle, and minor variations.
* **PyTorch** provides a flexible backend for running CLIP efficiently, taking advantage of GPUs if available, but also running on CPU for smaller-scale projects.
* **PIL** is used to preprocess images to the format CLIP expects.
* **NumPy** enables efficient numerical operations needed for similarity scoring.

---

## Setup and Usage

### Requirements

* Python 3.7+
* pip packages: `opencv-python`, `torch`, `openai-clip`, `numpy`, `Pillow`

### Installation

```bash
pip install opencv-python torch openai-clip numpy Pillow
```

### Running the Program

1. Place your dataset images in a folder (e.g., `testimg/`).
2. Run the main script:

   ```bash
   python BFSearch.py
   ```
3. The camera window will open.
4. Place an object from your dataset in front of the camera.
5. When a matching object is detected, its image will be displayed side-by-side with the camera feed.

---

## Limitations & Future Work

* The matching threshold might require tuning depending on lighting and dataset quality.
* Works best with small datasets; larger datasets may require optimization.
* Future improvements may include more advanced object detection or real-time bounding boxes.
* Integration with GUI frameworks for better user interaction.

