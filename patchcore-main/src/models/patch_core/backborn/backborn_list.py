from .wide_resnet50 import WideResNet50
from .resnet50 import ResNet50
from .resnet18 import ResNet18

backborn_list = {
    "wide_resnet50": WideResNet50,
    "resnet50": ResNet50,
    "resnet18": ResNet18,
}

# Optional ONNX backbones (only add if dependencies available)
try:
    from .resnet18_onnx import ResNet18_ONNX
    backborn_list["resnet18_onnx"] = ResNet18_ONNX
except Exception as e:
    print("Warning: resnet18_onnx backbone unavailable:", e)

try:
    from .resnet18_quantization_onnx import ResNet18_quantization_ONNX
    backborn_list["resnet18_quantization_onnx"] = ResNet18_quantization_ONNX
except Exception as e:
    print("Warning: resnet18_quantization_onnx backbone unavailable:", e)
