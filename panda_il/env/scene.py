"""Build the MuJoCo scene: our scene.xml with the vendored Menagerie Panda attached.

MjSpec composition is used instead of ``<include>`` because the Panda's mesh
paths resolve relative to the main model file. Attaching with the prefix
``panda/`` gives element names such as ``panda/joint1``, ``panda/actuator8`` and
the keyframe ``panda/home`` (whose qpos is padded with the cube's free joint,
which comes first). Everything is addressed by name, never by position.

Two models are compiled:

- the scene model (table, cube, bin, cameras, Panda), used for simulation;
- an arm-only model (the Panda alone, same base pose, same ``tcp`` site), used
  by the differential IK as its kinematic reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from panda_il.paths import ASSETS_DIR

SCENE_XML = Path(__file__).with_name("scene.xml")
PANDA_XML = ASSETS_DIR / "franka_emika_panda" / "panda.xml"

PREFIX = "panda/"
ARM_JOINTS = tuple(f"{PREFIX}joint{i}" for i in range(1, 8))
FINGER_JOINTS = (f"{PREFIX}finger_joint1", f"{PREFIX}finger_joint2")
ARM_ACTUATORS = tuple(f"{PREFIX}actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = f"{PREFIX}actuator8"
HOME_KEY = f"{PREFIX}home"
HAND_BODY = f"{PREFIX}hand"
FINGER_BODIES = (f"{PREFIX}left_finger", f"{PREFIX}right_finger")
TCP_SITE = "tcp"
TCP_OFFSET_IN_HAND = (0.0, 0.0, 0.1034)  # Franka's hand-to-TCP offset

CUBE_JOINT = "cube"
CUBE_BODY = "cube"
CUBE_GEOM = "cube"
CUBE_HALF = 0.02

BIN_BODY = "bin"
BIN_FLOOR_GEOM = "bin_floor"
BIN_WALL_GEOMS = ("bin_wall_px", "bin_wall_nx", "bin_wall_py", "bin_wall_ny")
BIN_CENTER = np.array([0.5, 0.25])
BIN_INNER_HALF = 0.06
BIN_FLOOR_TOP = 0.005
BIN_WALL_TOP = 0.06

TABLE_GEOM = "table"

GRIPPER_OPEN_CTRL = 255.0  # Menagerie's actuator8 drives the finger tendon on a 0-255 scale
GRIPPER_CLOSED_CTRL = 0.0

# Where the cameras look (positions are in scene.xml).
CAMERA_TARGETS = {"front": (0.50, 0.00, 0.02), "viewer": (0.45, 0.00, 0.10)}
WRIST_CAMERA = "wrist"
WRIST_CAMERA_POS_IN_HAND = (0.065, 0.0, 0.0)
WRIST_CAMERA_TILT = np.deg2rad(
    20.0
)  # tilt from the hand's z axis toward the fingers' approach line
WRIST_CAMERA_FOVY = 70.0


def quat_from_axes(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """MuJoCo (w, x, y, z) quaternion of a frame whose x and y axes are given in the parent frame."""
    x = np.asarray(x_axis, dtype=float)
    x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, dtype=float)
    y = y - x * np.dot(x, y)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    rotation = np.stack([x, y, z], axis=1)  # columns are the frame axes
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
    return quat


def lookat_quat(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Orientation of a MuJoCo camera at ``position`` looking at ``target``.

    MuJoCo cameras look along their -z axis with +y up and +x to the right.
    """
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return quat_from_axes(right, cam_up)


def wrist_camera_quat() -> np.ndarray:
    """Wrist camera frame in the hand frame: looks along the hand's +z (toward the fingertips),
    tilted toward the hand's -x so the camera mounted on the +x side of the hand sees the TCP."""
    tilt = WRIST_CAMERA_TILT
    approach = np.array(
        [-np.sin(tilt), 0.0, np.cos(tilt)]
    )  # the camera's viewing direction (-z of the camera)
    cam_x = np.array([0.0, 1.0, 0.0])
    cam_z = -approach
    cam_y = np.cross(cam_z, cam_x)
    return quat_from_axes(cam_x, cam_y)


def load_panda_spec() -> mujoco.MjSpec:
    if not PANDA_XML.exists():
        raise FileNotFoundError(f"{PANDA_XML} missing; run `uv run python scripts/fetch_assets.py`")
    return mujoco.MjSpec.from_file(str(PANDA_XML))


def build_scene_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for name, target in CAMERA_TARGETS.items():
        camera = spec.camera(name)
        camera.quat = lookat_quat(camera.pos, target)

    panda = load_panda_spec()
    frame = spec.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
    frame.attach_body(panda.body("link0"), PREFIX, "")

    hand = spec.body(HAND_BODY)
    # group 3 keeps the site out of rendered images (the renderer draws site groups 0-2)
    hand.add_site(name=TCP_SITE, pos=list(TCP_OFFSET_IN_HAND), size=[0.004, 0.004, 0.004], group=3)
    hand.add_camera(
        name=WRIST_CAMERA,
        pos=list(WRIST_CAMERA_POS_IN_HAND),
        quat=wrist_camera_quat().tolist(),
        fovy=WRIST_CAMERA_FOVY,
    )
    return spec


def enable_gravity_compensation(spec: mujoco.MjSpec, prefix: str = PREFIX) -> None:
    """Cancel gravity on every Panda body. Menagerie's position actuators have no gravity
    compensation, so without this the TCP sags 2-7 mm under its own weight; the real robot's
    controller compensates gravity, and so does mjctrl's reference IK example. It must be set
    on the spec: the compiler counts compensated bodies (``ngravcomp``), and setting
    ``body_gravcomp`` on a compiled model that had none does nothing."""
    for body in spec.bodies:
        if body.name.startswith(prefix):
            body.gravcomp = 1.0


def compile_scene_model() -> mujoco.MjModel:
    spec = build_scene_spec()
    enable_gravity_compensation(spec)
    model = spec.compile()
    model.light_castshadow[:] = 0
    return model


def compile_arm_model() -> mujoco.MjModel:
    """The Panda alone (no prefix), with the same ``tcp`` site, for the IK reference."""
    spec = load_panda_spec()
    spec.body("hand").add_site(name=TCP_SITE, pos=list(TCP_OFFSET_IN_HAND))
    return spec.compile()


@dataclass(frozen=True)
class SceneIds:
    """Integer handles into the scene model, looked up once by name."""

    arm_qpos: np.ndarray  # 7 qpos addresses
    arm_dof: np.ndarray  # 7 dof addresses
    finger_qpos: np.ndarray  # 2
    cube_qpos: int  # start of the free joint's 7 qpos
    cube_dof: int  # start of its 6 dof
    arm_actuators: np.ndarray  # 7 actuator ids
    gripper_actuator: int
    tcp_site: int
    cube_body: int
    cube_geom: int
    bin_floor_geom: int
    bin_wall_geoms: np.ndarray
    table_geom: int
    finger_geoms: np.ndarray  # every collision geom of both fingers
    home_key: int
    front_camera: int
    wrist_camera: int
    viewer_camera: int


def scene_ids(model: mujoco.MjModel) -> SceneIds:
    finger_geoms = []
    for body in FINGER_BODIES:
        body_id = model.body(body).id
        start = model.body_geomadr[body_id]
        for geom_id in range(start, start + model.body_geomnum[body_id]):
            if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
                finger_geoms.append(geom_id)
    return SceneIds(
        arm_qpos=np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS]),
        arm_dof=np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS]),
        finger_qpos=np.array([model.joint(j).qposadr[0] for j in FINGER_JOINTS]),
        cube_qpos=int(model.joint(CUBE_JOINT).qposadr[0]),
        cube_dof=int(model.joint(CUBE_JOINT).dofadr[0]),
        arm_actuators=np.array([model.actuator(a).id for a in ARM_ACTUATORS]),
        gripper_actuator=int(model.actuator(GRIPPER_ACTUATOR).id),
        tcp_site=int(model.site(TCP_SITE).id),
        cube_body=int(model.body(CUBE_BODY).id),
        cube_geom=int(model.geom(CUBE_GEOM).id),
        bin_floor_geom=int(model.geom(BIN_FLOOR_GEOM).id),
        bin_wall_geoms=np.array([model.geom(g).id for g in BIN_WALL_GEOMS]),
        table_geom=int(model.geom(TABLE_GEOM).id),
        finger_geoms=np.array(finger_geoms),
        home_key=int(model.key(HOME_KEY).id),
        front_camera=int(model.camera("front").id),
        wrist_camera=int(model.camera(WRIST_CAMERA).id),
        viewer_camera=int(model.camera("viewer").id),
    )
