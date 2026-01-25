### (1) Preprocessing
import torch
from PIL import Image
from torchvision import transforms

# Preprocessing settings
transform_list = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
transform = transforms.Compose(transform_list)

# Load image
im = Image.open("data/images/wood/train/OK/IMG_3790_0000.png").convert('RGB')

# Preprocess
im = transform(im)

# Convert to [N, C, H, W]
x = torch.unsqueeze(im, 0)

print(x.shape)
# torch.Size([1, 3, 224, 224])

### (2) Feature extraction
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

device = "cpu"

# WideResNet50 pretrained on ImageNet
try:
    model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").to(device)
except Exception as e:
    print("Warning: could not load pretrained weights:", e)
    # Fallback: use model without pretrained weights (for offline)
    model = torchvision.models.wide_resnet50_2(weights=None).to(device)

# Settings for extracting intermediate layer features
layers = ['layer2', 'layer3'] # Layer names to extract features from
extractor = create_feature_extractor(model, layers)

# Set evaluation mode
extractor.eval()
    
# Extract features from specified layers
with torch.no_grad():
    features = extractor(x) # x is a preprocessed image tensor in [N, C, H, W] format

# 'features' is a dict containing tensors for 'layer2' and 'layer3', e.g.:
# {
#   'layer2': layer2 feature tensor (torch.Tensor),
#   'layer3': layer3 feature tensor (torch.Tensor),
# }

print("layer2: ", features['layer2'].shape) # layer2:  torch.Size([1, 512, 28, 28])
print("layer3: ", features['layer3'].shape) # layer3:  torch.Size([1, 1024, 14, 14])

### (3) Average Pooling
import torch

# Apply average pooling per intermediate layer
pooling = torch.nn.AvgPool2d(3, 1, 1)
features = { layer: pooling(feature) for layer, feature in features.items() }

# Average pooing後のシェイプを表示
for k, v in features.items():
    print(f"average pooling - > {k}: {v.shape}")
# average pooling - > layer2: torch.Size([1, 512, 28, 28])
# average pooling - > layer3: torch.Size([1, 1024, 14, 14])

### (4) Combine feature vectors
import torch.nn.functional as F

# Upsample deeper layer features to match spatial size of the shallower layer
upsample_layer3 = F.interpolate(features['layer3'], size=features['layer2'].shape[-2:], mode="nearest")

# Concatenate shallower and upsampled deeper features along channel dimension
features = torch.cat((features['layer2'], upsample_layer3,), 1)

print(features.shape) # torch.Size([1, 1536, 28, 28])

### (5) Reshape feature vectors
features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
print(features.shape) # torch.Size([784, 1536])

### (6) k-center greedy (coreset sampling)
from sklearn.random_projection import SparseRandomProjection
import random

def k_center_greedy(x, sampling_ratio, seed=None):
    """K-Center Greedy algorithm.

    Args:
        x: Matrix of shape [num_samples, feature_dim].
        sampling_ratio: Ratio for coreset sampling.
        seed: Random seed.

    Returns:
        Tuple of (sampled_matrix, number_of_samples).
    """
    device = x.device
    
    # 比率からサンプリングするデータ数を決定 
    n_sample = int(len(x) * sampling_ratio)
    
    # 乱数シード設定
    random.seed(seed)

    # 処理高速化のためにスパースランダム投影を使って特徴ベクトルの次元を削減
    random_projection = SparseRandomProjection(n_components='auto', eps=0.9)
    random_projection.fit(x.detach().cpu())
    x_dash = x.detach().cpu()
    x_dash = random_projection.transform(x_dash)
    x_dash = torch.tensor(x_dash).to(device)

    # 初期中心をランダムで選択
    center_index = random.randint(0, len(x) - 1)
    # サンプリングした要素のインデックスリストを初期化
    selected_indexes = [center_index]

    # 予定サンプリング数だけ実行
    min_distance = None
    for _ in range(n_sample - 1):
        # 中心と各データのユークリッド距離を計算
        distance = F.pairwise_distance(x_dash, x_dash[center_index])

        # 新たに算出した距離とこれまで算出した距離を、各データごとに比較して最小のものを選択
        if min_distance is None:
            min_distance = distance
        else:
            min_distance = torch.minimum(min_distance, distance)

        # 中心から一番遠いデータを次の中心とする
        center_index = int(torch.argmax(min_distance).item())
        selected_indexes.append(center_index)
        # 一度中心としたデータの距離は以後使わない
        min_distance[selected_indexes] = 0

    return x[selected_indexes], n_sample

# Downsample patch feature vectors using k-center greedy
memory_bank, _ = k_center_greedy(features, sampling_ratio=0.1)

print(f"k_center_greedy: {features.shape} -> {memory_bank.shape}")
# k_center_greedy: torch.Size([784, 1536]) -> torch.Size([78, 1536])


test_paths = [
    "data/images/wood/test/OK/IMG_3790_0102.png",
    "data/images/wood/test/NG/IMG_3790_0200.png",
]

# Labels (0: normal, 1: abnormal)
labels = [0, 1]
labels = torch.tensor([labels], dtype=torch.long).T

features = []
for path in test_paths:
    im = Image.open(path).convert('RGB')
    im = transform(im)
    x = torch.unsqueeze(im, 0)

    with torch.no_grad():
        f = extractor(x)

    f = { layer: pooling(feature) for layer, feature in f.items() }

    upsample_layer3 = F.interpolate(f['layer3'], size=f['layer2'].shape[-2:], mode="nearest")
    f = torch.cat((f['layer2'], upsample_layer3,), 1)

    f = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
    features.append(f)


### (8) 標準パラメータの算出
from torchmetrics import PrecisionRecallCurve

# Top-K neighbors
n_neighbors = 9

# Input image size
input_image_size = (512, 512)

# Prepare for computing normalization parameters
min_value = torch.tensor(float("inf"))
max_value = torch.tensor(float("-inf"))
precision_recall_curve = PrecisionRecallCurve(num_classes=1)

for target, label in zip(features, labels):
    # Compute pairwise Euclidean distances between input features and memory bank
    distances = torch.cdist(target, memory_bank, p=2.0)

    # For each patch, take the smallest n_neighbors distances
    patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)

    print("distance top_k: ", patch_scores.shape) # distance top_k:  torch.Size([784, 9])

    # ==== Anomaly map
    # Reshape nearest-neighbor distances per patch to [N, C, H, W]
    anomaly_map = patch_scores[:, 0].reshape((1, 1, 28, 28))

    # Resize to input image size
    anomaly_map = F.interpolate(anomaly_map, size=input_image_size, mode='nearest')

    print("anomaly_map: ", anomaly_map.shape) # anomaly_map:  torch.Size([1, 1, 512, 512])
    
    # ==== Anomaly score
    # 1. Get the element with the largest distance (contains the Top-k distances)
    max_scores_index = torch.argmax(patch_scores[:, 0])
    max_scores = torch.index_select(patch_scores, 0, max_scores_index) # [1, Top-N]

    # 2. Compute anomaly score from the smallest top-k distances considering their variance
    #       weights = 1 - softmax(max_scores)
    weights = 1 - (torch.max(torch.exp(max_scores)) / torch.sum(torch.exp(max_scores)))
    anomaly_score = weights * torch.max(patch_scores[:, 0])

    print("anomaly_score: ", anomaly_score) # anomaly_score:  tensor(1.8167) ※値は入力画像により異なる

    # Update min/max normalization parameters
    min_value = torch.min(min_value, torch.min(anomaly_map))
    max_value = torch.max(max_value, torch.max(anomaly_map))

    # Update precision-recall curve
    precision_recall_curve(anomaly_score.unsqueeze(0).cpu(), label)

# 複数のしきい値でF1-scoreを求める
precision, recall, thresoulds = precision_recall_curve.compute()
f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

# 一番精度の良い（=F1-scoreの高い）しきい値を求める
print("f1_score", f1_score)
print("thresoulds", thresoulds)
best_index = torch.argmax(f1_score)
print(best_index)
if thresoulds.dim() == 0:
    thresould = thresoulds
else:
    thresould = thresoulds[best_index]
print(f"thresould: {thresould} min: {min_value} max: {max_value}")
