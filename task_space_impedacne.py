import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import genesis as gs
import pinocchio as pin
import matplotlib.pyplot as plt

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=1/60,
        substeps=16,
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
)
franka = scene.add_entity(
    gs.morphs.URDF(file="urdf/panda_bullet/panda.urdf", fixed=True),
    material=gs.materials.Rigid(gravity_compensation=1.0),
)

TARGET_POS  = np.array([0.65, 0.0, 0.25])
TARGET_QUAT = np.array([0, 0, 0, 1])  # wxyz

mouse_plugin = gs.vis.viewer_plugins.MouseInteractionPlugin(
    use_force=True,
    spring_const=100.0,
)
scene.viewer.add_plugin(mouse_plugin)

scene.build()

# ===== 목표 시각화 (scene.build 후) =====
scene.draw_debug_sphere(pos=TARGET_POS, radius=0.02, color=(1.0, 0.1, 0.1, 0.9))
T_goal = np.eye(4)
T_goal[:3, 3] = TARGET_POS
w, x, y, z = TARGET_QUAT
T_goal[:3, :3] = np.array([
    [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
    [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
    [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
])
scene.draw_debug_frame(T=T_goal, axis_length=0.1, origin_size=0.015, axis_radius=0.005)

# ===== Pinocchio 모델 로드 =====
urdf_path = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet/panda.urdf")
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()
hand_frame_id = pin_model.getFrameId("panda_hand")
if hand_frame_id >= pin_model.nframes:
    hand_frame_id = pin_model.getFrameId("panda_link8")
print(f"Pinocchio 로드 완료: nq={pin_model.nq}, hand_frame_id={hand_frame_id}")

motors_dof = np.arange(7)
end_effector = franka.get_link("panda_link7")

franka.set_dofs_kp(np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]))
franka.set_dofs_kv(np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# ===== 임피던스 게인 =====
K_stiffness = np.diag([50, 50, 50, 50, 50, 50])   # 6x6 (회전 게인 올림)
K_damping   = np.diag([5,  5,  5,  5,  5,  5])    # 6x6

# ===== Null space 게인 =====
K_null = np.diag([50, 50, 50, 50, 50, 50, 50])   # 7x7
D_null = np.diag([10, 10, 10, 10,  10,  10,  10])   # 7x7
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

# ===== Task Space Impedance 제어 루프 =====
print("Task Space Impedance 제어 시작...")
print(f"목표: pos={TARGET_POS}, quat={TARGET_QUAT}")
tau_max_arr = np.array([87, 87, 87, 87, 12, 12, 12])

# 에러 기록용
log_pos_err = []
log_rot_err = []
log_pos_xyz = []    # xyz 각 축 위치 오차
log_rot_xyz = []    # xyz 각 축 회전 오차
N_STEPS = 1000

for i in range(N_STEPS):
    # 현재 상태
    q_current  = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()
    current_pos  = end_effector.get_pos().cpu().numpy()
    current_quat = end_effector.get_quat().cpu().numpy()

    # Pinocchio 동역학 계산
    q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_current
    dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_current
    pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)

    M_q     = pin_data.M[:7, :7]
    J_full  = pin.getFrameJacobian(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J       = J_full[:, :7]
    dJ_full = pin.getFrameJacobianTimeVariation(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
    dJ      = dJ_full[:, :7]

    # 중력은 Genesis 처리 → 코리올리만 보상
    g_q      = pin.computeGeneralizedGravity(pin_model, pin_data, q_pin)[:7]
    coriolis = pin_data.nle[:7] - g_q

    # Λ = (J M⁻¹ Jᵀ)⁻¹
    M_q_inv = np.linalg.inv(M_q)
    Lambda  = np.linalg.inv(J @ M_q_inv @ J.T + 1e-6 * np.eye(6))
    J_bar   = M_q_inv @ J.T @ Lambda

    # EE twist
    ee_twist = J @ dq_current

    # 위치/자세 오차 (Genesis end_effector 기준)
    error_pos    = TARGET_POS - current_pos
    error_quat   = gs.transform_quat_by_quat(gs.inv_quat(current_quat), TARGET_QUAT)
    error_rotvec = gs.quat_to_rotvec(error_quat)
    if hasattr(error_rotvec, 'cpu'):
        error_rotvec = error_rotvec.cpu().numpy()
    error_pose = np.concatenate([error_pos, error_rotvec])

    # 임피던스: ddx_cmd = K*e - D*twist (목표 속도/가속도 = 0)
    ddx_cmd = K_stiffness @ error_pose - K_damping @ ee_twist

    # Task space 토크
    tau_task = J.T @ (Lambda @ (ddx_cmd - dJ @ dq_current)) + coriolis

    # Null space 토크
    N_proj   = np.eye(7) - J.T @ J_bar.T
    tau_null = N_proj @ (K_null @ (q_home - q_current) - D_null @ dq_current)

    tau_total = np.clip(tau_task + tau_null, -tau_max_arr, tau_max_arr)
    franka.control_dofs_force(tau_total, motors_dof)
    scene.step()

    # 에러 기록
    log_pos_err.append(np.linalg.norm(error_pos))
    log_rot_err.append(np.linalg.norm(error_rotvec))
    log_pos_xyz.append(error_pos.copy())
    log_rot_xyz.append(error_rotvec.copy())

    if i % 60 == 0:
        print(f"[{i:5d}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
              f"rot_err: {np.linalg.norm(error_rotvec):.4f}rad")

# ===== 에러 플롯 =====
t = np.arange(N_STEPS) / 60.0  # 시간 (초)
log_pos_xyz = np.array(log_pos_xyz)
log_rot_xyz = np.array(log_rot_xyz)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 위치 오차 노름
axes[0, 0].plot(t, log_pos_err, 'b-')
axes[0, 0].set_ylabel('Position Error (m)')
axes[0, 0].set_title('Position Error Norm')
axes[0, 0].grid(True)

# 위치 오차 xyz
axes[0, 1].plot(t, log_pos_xyz[:, 0], 'r-', label='x')
axes[0, 1].plot(t, log_pos_xyz[:, 1], 'g-', label='y')
axes[0, 1].plot(t, log_pos_xyz[:, 2], 'b-', label='z')
axes[0, 1].set_ylabel('Position Error (m)')
axes[0, 1].set_title('Position Error XYZ')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 회전 오차 노름
axes[1, 0].plot(t, log_rot_err, 'r-')
axes[1, 0].set_ylabel('Rotation Error (rad)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Rotation Error Norm')
axes[1, 0].grid(True)

# 회전 오차 xyz
axes[1, 1].plot(t, log_rot_xyz[:, 0], 'r-', label='rx')
axes[1, 1].plot(t, log_rot_xyz[:, 1], 'g-', label='ry')
axes[1, 1].plot(t, log_rot_xyz[:, 2], 'b-', label='rz')
axes[1, 1].set_ylabel('Rotation Error (rad)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Rotation Error XYZ')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('task_space_impedance_error.png', dpi=150)
plt.show()
print("플롯 저장: task_space_impedance_error.png")
