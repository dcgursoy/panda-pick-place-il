"""panda_il: imitation learning study on a simulated Franka Panda (cube pick-and-place).

Subpackages, one per stage of the pipeline:
  env       MuJoCo scene, gymnasium environment, differential IK, cameras, task definition
  teleop    keyboard/gamepad teleoperation and the scripted expert
  data      multi-rate logger, Zarr schema, sync, validator, stats, replay, rendering, splits, LeRobot export
  policies  behavior cloning (MLP, ResNet18) and Diffusion Policy
  train     one training entry point per method
  eval      held-out evaluation, aggregation with confidence intervals, plots, GIFs, failure analysis
"""

__version__ = "0.1.0"
