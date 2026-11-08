import subprocess
import sys

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

print("\nPython executable:")
print(sys.executable)

# --------------------------------------------------
# 1. Check NVIDIA GPU / Driver
# --------------------------------------------------
print("\n[1] Checking NVIDIA GPU...")

try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=True
    )

    print("NVIDIA GPU detected:")
    print(result.stdout)

except Exception as e:
    print("nvidia-smi could not run.")
    print("Possible reasons:")
    print("- NVIDIA driver is not installed")
    print("- No NVIDIA GPU is available")
    print("- nvidia-smi is not in PATH")
    print("Error:", e)


# --------------------------------------------------
# 2. Check PyTorch
# --------------------------------------------------
print("\n" + "=" * 60)
print("[2] Checking PyTorch...")
print("=" * 60)

try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        device = torch.device("cuda")
        print("\nPyTorch will use:", device)

        # Small GPU test
        x = torch.rand(1000, 1000).to(device)
        y = torch.rand(1000, 1000).to(device)
        z = x @ y

        print("Test tensor device:", z.device)
        print("GPU computation successful.")

    else:
        print("PyTorch is NOT using CUDA.")
        print("Current device: CPU")

except ImportError:
    print("PyTorch is not installed in this Python environment.")


# --------------------------------------------------
# 3. Check TensorFlow
# --------------------------------------------------
print("\n" + "=" * 60)
print("[3] Checking TensorFlow...")
print("=" * 60)

try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")

    print("TensorFlow GPUs:", gpus)

    if gpus:
        print("\nTensorFlow GPU detected successfully.")

        for gpu in gpus:
            print("GPU:", gpu)

    else:
        print("TensorFlow does NOT detect a GPU.")
        print("TensorFlow will use CPU.")

except ImportError:
    print("TensorFlow is not installed in this Python environment.")

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)