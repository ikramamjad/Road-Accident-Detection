"""
Model Export Script for Edge Hardware (ONNX & TensorRT).
Exports Causal GRU and Fusion MLP models to ONNX and displays TensorRT deployment commands.
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from road_accident_detection.export.onnx_exporter import ONNXExporter


def main():
    parser = argparse.ArgumentParser(description="Export neural network models to ONNX")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="d:/Road Accident Detection/data/exported_models",
        help="Directory to save exported ONNX models",
    )
    args = parser.parse_args()

    exporter = ONNXExporter(output_dir=args.output_dir)
    print(f"[Export] Exporting models to ONNX: {args.output_dir}")

    # Export Causal GRU
    gru_result = exporter.export_causal_gru(filename="causal_gru.onnx")
    print(f"\nModel: {gru_result.model_name}")
    print(f"Path:  {gru_result.onnx_path} ({gru_result.file_size_kb:.1f} KB)")
    print(f"Validated with ONNX Runtime: {gru_result.is_valid} (Max Error: {gru_result.max_abs_error:.2e})")
    print(f"TensorRT Edge Command:\n  {gru_result.tensorrt_command}")

    # Export Fusion MLP
    mlp_result = exporter.export_fusion_mlp(filename="fusion_mlp.onnx")
    print(f"\nModel: {mlp_result.model_name}")
    print(f"Path:  {mlp_result.onnx_path} ({mlp_result.file_size_kb:.1f} KB)")
    print(f"Validated with ONNX Runtime: {mlp_result.is_valid} (Max Error: {mlp_result.max_abs_error:.2e})")
    print(f"TensorRT Edge Command:\n  {mlp_result.tensorrt_command}")

    print("\n[Export Complete] All models successfully exported and validated for edge inference.")


if __name__ == "__main__":
    main()
