from __future__ import annotations

from sklearn import metrics
import omegaconf

from models.patch_core import PatchCore
from common.pytorch_custom_dataset import ImagePaths

def test(
        test_loader: ImagePaths,
        weights_path: str,
        device: str,
    ):
    """Run tests and display results.

    Args:
        test_loader (ImagePaths): Test data loader
        weights_path (str): Path to weights file
        device (str): Device id
    """
    net = PatchCore.load_weights(weights_path, device)

    all_preds = []
    all_labes = []

    for i, (x, label, paths) in enumerate(test_loader):
        # Inference
        anomaly_score, anomaly_map, result = net.predict(x)

        #print(f"{i} [{paths[0]}] preds: {anomaly_map} / label: {label[0]}")

        label = label.tolist()

        all_preds.append(result)
        all_labes.extend(label)

    # Calculate evaluation metrics
    precision = metrics.precision_score(all_labes, all_preds, average=None)
    recall = metrics.recall_score(all_labes, all_preds, average=None)
    f1_score = metrics.f1_score(all_labes, all_preds, average=None)

    # Compute macro-averaged metrics
    macro_precision = metrics.precision_score(all_labes, all_preds, average='macro')
    macro_recall = metrics.recall_score(all_labes, all_preds, average='macro')
    macro_f1_score = metrics.f1_score(all_labes, all_preds, average='macro')

    # Build confusion matrix
    cm = metrics.confusion_matrix(all_labes, all_preds)

    # Display results
    print(f"model: {weights_path}")
    print(f"\tprecision\trecall\t\tf1_score")
    print(f"0:OK\t{precision[0]:.4f}\t\t{recall[0]:.4f}\t\t{f1_score[0]:.4f}")
    print(f"1:NG\t{precision[1]:.4f}\t\t{recall[1]:.4f}\t\t{f1_score[1]:.4f}")
    print(f"macro\t{macro_precision:.4f}\t\t{macro_recall:.4f}\t\t{macro_f1_score:.4f}")
    print()

    print(cm)
