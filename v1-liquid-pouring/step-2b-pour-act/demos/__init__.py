"""Recording the programmed pour as demonstrations for the learned pour (act/)."""

# MuJoCo before PyTorch: both bring their own OpenMP, and loading PyTorch's first stops the program.
import mujoco  # noqa: F401
