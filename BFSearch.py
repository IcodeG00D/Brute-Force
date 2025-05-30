import cv2
import torch
import numpy as np
from PIL import Image
import os
import clip
  # Using openai-clip package

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder with your 5 images
dataset_folder = r"D:\Projects\Brute-Force-main\testimg"
dataset_embeddings = []
image_files = []

print("Loading dataset images...")
for img_name in os.listdir(dataset_folder):
    path = os.path.join(dataset_folder, img_name)
    try:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            dataset_embeddings.append(embedding)
            image_files.append(path)
    except Exception as e:
        print(f"Failed loading {path}: {e}")

print(f"Loaded {len(image_files)} images.")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    image = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model.encode_image(image)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    similarities = [torch.cosine_similarity(query_embedding, emb).item() for emb in dataset_embeddings]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > 0.60:
        matched_img = cv2.imread(image_files[best_idx])
        matched_img = cv2.resize(matched_img, (small_frame.shape[1], small_frame.shape[0]))
        combined = np.hstack((small_frame, matched_img))
        label = f"Match: {os.path.basename(image_files[best_idx])} ({best_score:.2f})"
        cv2.putText(combined, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Object Match", combined)
    else:
        cv2.imshow("Object Match", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
