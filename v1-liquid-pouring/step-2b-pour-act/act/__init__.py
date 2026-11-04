"""The learned pour: ACT (Action Chunking with Transformers), and a simpler CNN + MLP policy to compare."""

# MuJoCo before PyTorch: both bring their own OpenMP, and loading PyTorch's first stops the program.
import mujoco  # noqa: F401
