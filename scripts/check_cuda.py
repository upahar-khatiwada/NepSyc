"""Reports whether the installed torch build has CUDA support and, if so, whether a GPU is
actually visible to it. Standalone diagnostic -- not part of the pipeline, no project imports.

Usage: python scripts/check_cuda.py
"""

import torch


def main() -> None:
    print(f"torch version:         {torch.__version__}")
    print(f"torch built with CUDA: {torch.version.cuda or 'no (CPU-only build)'}")
    print(f"cuda available:        {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"device count:           {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print("\n-> config.yaml's `local` provider can use device: cuda")
    else:
        print("\n-> config.yaml's `local` provider must use device: cpu")


if __name__ == "__main__":
    main()
