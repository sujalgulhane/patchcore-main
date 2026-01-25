import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# --------------------------------
# CONFIG
# --------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_OK_DIR = r"E:\Downloads\patchcore-main\patchcore-main\src\data\images\wood\test\OK"
TEST_NG_DIR = r"E:\Downloads\patchcore-main\patchcore-main\src\data\images\wood\test\NG"

WEIGHTS_PATH = r"E:\Downloads\patchcore-main\patchcore-main\data\weights\wide_resnet50_size224_param_0.1_9_wood.pth"

IMAGE_SIZE = 224

# --------------------------------
# SAFE IMAGE LOADER
# --------------------------------
def load_images(folder):
    images = []
    filenames = []

    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)

        # Skip subfolders
        if not os.path.isfile(path):
            continue

        img = cv2.imread(path)

        # Skip unreadable files
        if img is None:
            print(f"[WARNING] Skipping unreadable file: {path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        images.append(img)
        filenames.append(fname)

    if len(images) == 0:
        raise RuntimeError(f"No valid images found in {folder}")

    images_np = np.stack(images)  # shape: (N, C, H, W)
    return torch.from_numpy(images_np).float(), filenames

# --------------------------------
# LOAD TEST DATA
# --------------------------------
print("Loading test images...")
test_ok, ok_names = load_images(TEST_OK_DIR)
test_ng, ng_names = load_images(TEST_NG_DIR)

test_ok = test_ok.to(DEVICE)
test_ng = test_ng.to(DEVICE)

print(f"OK images loaded: {len(test_ok)}")
print(f"NG images loaded: {len(test_ng)}")

# --------------------------------
# LOAD PATCHCORE MODEL
# --------------------------------
from models.patch_core import PatchCore

print("Loading trained PatchCore model...")
model = PatchCore.load_weights(WEIGHTS_PATH, DEVICE)
# Put backbone into eval mode if available
try:
    model.backborn.eval()
except Exception:
    pass

# --------------------------------
# INFERENCE
# --------------------------------
def get_scores(images):
    scores = []
    with torch.no_grad():
        for img in tqdm(images):
            anomaly_score, _, _ = model.predict(img.unsqueeze(0))
            scores.append(anomaly_score.item())
    return np.array(scores)

print("Running inference...")
scores_ok = get_scores(test_ok)
scores_ng = get_scores(test_ng)

# --------------------------------
# GROUND TRUTH
# --------------------------------
y_true = np.array(
    [0] * len(scores_ok) +   # OK
    [1] * len(scores_ng)     # NG
)

scores = np.concatenate([scores_ok, scores_ng])

# --------------------------------
# THRESHOLD SWEEP & PLOT (Threshold vs Accuracy)
# --------------------------------
thresholds = np.linspace(scores.min(), scores.max(), 200)
accuracies = [accuracy_score(y_true, (scores > t).astype(int)) for t in thresholds]

best_idx = int(np.argmax(accuracies))
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

# Try to plot and save the figure
try:
    import matplotlib.pyplot as plt

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "threshold_accuracy.png")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, label="Accuracy")
    plt.axvline(best_threshold, color='r', linestyle='--', label=f"Best: {best_threshold:.4f}")
    plt.scatter([best_threshold], [best_accuracy], color='r')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold vs Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved threshold vs accuracy plot to {save_path}")
except Exception as e:
    print("Could not plot threshold vs accuracy:", e)

# --------------------------------
# THRESHOLD (OK-ONLY) - original heuristic
# --------------------------------
threshold = np.mean(scores_ok) + 3 * np.std(scores_ok)
y_pred = (scores > threshold).astype(int)

# --------------------------------
# METRICS
# --------------------------------
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, scores)

print("\n========== RESULTS ==========")
print(f"Best Threshold (sweep) : {best_threshold:.4f} (accuracy={best_accuracy:.4f})")
print(f"Threshold (heuristic)  : {threshold:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["OK", "NG"]))

# --------------------------------
# OPTIONAL: PRINT MISCLASSIFIED IMAGES
# --------------------------------
all_names = ok_names + ng_names
print("\nMisclassified images:")
for name, true, pred, score in zip(all_names, y_true, y_pred, scores):
    if true != pred:
        print(f"{name} | GT={true} | PRED={pred} | SCORE={score:.4f}")
