from ultralytics import YOLO
import torch

# 加载模型（确保best.pt路径正确）
model = YOLO("best.pt")

# 仅使用YOLO官方支持的导出参数
onnx_path = model.export(
    format="rknn",        # 导出格式为ONNX（必填）
    opset=18,             # 匹配PyTorch自动升级的opset版本
    dynamic=False,        # 关闭动态轴（避免ARM环境兼容性问题）
    simplify=True,        # 简化ONNX模型（移除冗余节点）
    imgsz=640,            # 固定输入尺寸（640x640）
    batch=1,              # 固定批次大小
    device="cpu",         # 指定CPU（RK3588无NVIDIA GPU）
    half=False            # 禁用半精度（ARM CPU不支持FP16）
)

print(f"ONNX模型导出成功，路径：{onnx_path}")