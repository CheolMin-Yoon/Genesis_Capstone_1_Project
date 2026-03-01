# task_space_gripper.py - Task Space Impedance + Gripper Impedance
# arm: Pinocchio 기반 Task Space Impedance (토크 제어)
# gripper: Joint Space Impedance (토크 제어)
# 전체 9 DOF 토크 제어
#
# 시나리오: 목표 지점으로 이동 → 그리퍼 닫기 (물체 잡기) → 들어올리기
# 필요: pip install pin

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
)

mouse_plugin = gs.vis.viewer_plugins.MouseInteractionPlugin(
    use_force=True,
    spring_const=100.0,
)
scene.viewer.add_plugin(mouse_plugin)

scene.build()

# ===== Pinocchio 모델 로드 =====
urdf_path = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet/panda.urdf")
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()
hand_frame_id = pin_model.getFrameId("panda_hand")
if hand_frame_id >= pin_model.nframes:
    hand_frame_id = pin_model.getFrameId("panda_link8")
print(f"Pinocchio 로드 완료: nq={pin_model.nq}, hand_frame_id={hand_frame_id}")

motors_dof  = np.arange(7)
gripper_dof = np.arange(7, 9)
all_dof     = np.arange(9)
end_effector = franka.get_link("panda_link7")

# ===== 전체 9 DOF 토크 제어 (kp=0, kv=0) =====
franka.set_dofs_kp(np.zeros(9))
franka.set_dofs_kv(np.zeros(9))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -20, -20]),
    np.array([87, 87, 87, 87, 12, 12, 12, 20, 20]),
)

# ===== Arm 임피던스 게인 (Task Space 6x6) =====
K_stiffness = np.diag([500, 500, 500, 200, 200, 200])
K_damping   = np.diag([50,  50,  50,  20,  20,  20])

# ===== Arm Null space 게인 (7x7) =====
K_null = np.diag([50, 50, 50, 50, 20, 20, 20])
D_null = np.diag([10, 10, 10, 10,  5,  5,  5])
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

# ===== Gripper 임피던스 게인 =====
kp_gripper = 100.0   # 그리퍼 강성 (N/m)
kv_gripper = 10.0    # 그리퍼 댐핑
max_grip_force = 20.0  # 최대 그립 힘 (N)

# ===== 시나리오 정의 =====
# phase 0: 목표 위치로 이동 (그리퍼 열림)
# phase 1: 그리퍼 닫기 (물체 잡기)
# phase 2: 들어올리기
PHASES = [
    {"target_pos": np.array([0.65, 0.0, 0.06]), "target_quat": np.array([0, 1, 0, 0]),
     "gripper_target": np.array([0.04, 0.04]), "duration": 300, "name": "접근"},
    {"target_pos": np.array([0.65, 0.0, 0.06]), "target_quat": np.array([0, 1, 0, 0]),
     "gripper_target": np.array([0.0, 0.0]), "duration": 200, "name": "그립"},
    {"target_pos": np.array([0.65, 0.0, 0.35]), "target_quat": np.array([0, 1, 0, 0]),
     "gripper_target": np.array([0.0, 0.0]), "duration": 300, "name": "들어올리기"},
]

tau_max_arm = np.array([87, 87, 87, 87, 12, 12, 12])

# ===== 제어 루프 =====
log_pos_err = []
log_rot_err = []
log_grip_err = []
log_grip_force = []

step = 0
for phase_idx, phase in enumerate(PHASES):
    target_pos  = phase["target_pos"]
    target_quat = phase["target_quat"]
    gripper_target = phase["gripper_target"]
    duration = phase["duration"]
    print(f"\n===== Phase {phase_idx}: {phase['name']} ({duration} steps) =====")
    print(f"  arm target: {target_pos}, gripper target: {gripper_target}")

    # 목표 시각화
    scene.draw_debug_sphere(pos=target_pos, radius=0.015, color=(1.0, 0.2, 0.2, 0.8))

    for j in range(duration):
        # ===== 현재 상태 =====
        q_current  = franka.get_dofs_position(motors_dof).cpu().numpy()
        dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()
        current_pos  = end_effector.get_pos().cpu().numpy()
        current_quat = end_effector.get_quat().cpu().numpy()

        q_gripper  = franka.get_dofs_position(gripper_dof).cpu().numpy()
        dq_gripper = franka.get_dofs_velocity(gripper_dof).cpu().numpy()

        # ===== Pinocchio 동역학 (arm) =====
        q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_current
        dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_current
        pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)

        M_q     = pin_data.M[:7, :7]
        J_full  = pin.getFrameJacobian(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
        J       = J_full[:, :7]
        dJ_full = pin.getFrameJacobianTimeVariation(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
        dJ      = dJ_full[:, :7]
        nle     = pin_data.nle[:7]

        # Λ, J_bar
        M_q_inv = np.linalg.inv(M_q)
        Lambda  = np.linalg.inv(J @ M_q_inv @ J.T + 1e-6 * np.eye(6))
        J_bar   = M_q_inv @ J.T @ Lambda

        ee_twist = J @ dq_current

        # ===== Arm Task Space Impedance =====
        error_pos    = target_pos - current_pos
        error_quat   = gs.transform_quat_by_quat(gs.inv_quat(current_quat), target_quat)
        error_rotvec = gs.quat_to_rotvec(error_quat)
        if hasattr(error_rotvec, 'cpu'):
            error_rotvec = error_rotvec.cpu().numpy()
        error_pose = np.concatenate([error_pos, error_rotvec])

        ddx_cmd = K_stiffness @ error_pose - K_damping @ ee_twist
        tau_task = J.T @ (Lambda @ (ddx_cmd - dJ @ dq_current)) + nle

        # Null space
        N_proj   = np.eye(7) - J.T @ J_bar.T
        tau_null = N_proj @ (K_null @ (q_home - q_current) - D_null @ dq_current)

        tau_arm = np.clip(tau_task + tau_null, -tau_max_arm, tau_max_arm)

        # ===== Gripper Impedance =====
        e_grip  = gripper_target - q_gripper
        de_grip = -dq_gripper  # 목표 속도 = 0
        tau_gripper = kp_gripper * e_grip + kv_gripper * de_grip
        tau_gripper = np.clip(tau_gripper, -max_grip_force, max_grip_force)

        # ===== 전체 토크 인가 =====
        tau_all = np.concatenate([tau_arm, tau_gripper])
        franka.control_dofs_force(tau_all, all_dof)
        scene.step()

        # 로깅
        log_pos_err.append(np.linalg.norm(error_pos))
        log_rot_err.append(np.linalg.norm(error_rotvec))
        log_grip_err.append(np.linalg.norm(e_grip))
        log_grip_force.append(np.linalg.norm(tau_gripper))

        if j % 60 == 0:
            print(f"  [{step:5d}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
                  f"grip_err: {np.linalg.norm(e_grip):.4f}, "
                  f"grip_force: {np.linalg.norm(tau_gripper):.2f}N")
        step += 1

print("\n제어 완료!")

# ===== 마지막 위치 유지 =====
print("시뮬레이션 유지 중... (Ctrl+C로 종료)")
q_hold = franka.get_dofs_position(motors_dof).cpu().numpy()
grip_hold = np.array([0.0, 0.0])  # 그리퍼 닫은 상태 유지
try:
    while True:
        q_c  = franka.get_dofs_position(motors_dof).cpu().numpy()
        dq_c = franka.get_dofs_velocity(motors_dof).cpu().numpy()
        q_g  = franka.get_dofs_position(gripper_dof).cpu().numpy()
        dq_g = franka.get_dofs_velocity(gripper_dof).cpu().numpy()

        q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_c
        dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_c
        pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)
        M_q = pin_data.M[:7, :7]
        nle = pin_data.nle[:7]
        M_inv = np.linalg.inv(M_q)
        J_full = pin.getFrameJacobian(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
        J = J_full[:, :7]
        Lambda = np.linalg.inv(J @ M_inv @ J.T + 1e-6 * np.eye(6))
        J_bar = M_inv @ J.T @ Lambda

        ee_twist = J @ dq_c
        e_pos = PHASES[-1]["target_pos"] - end_effector.get_pos().cpu().numpy()
        e_quat = gs.transform_quat_by_quat(gs.inv_quat(end_effector.get_quat().cpu().numpy()), PHASES[-1]["target_quat"])
        e_rv = gs.quat_to_rotvec(e_quat)
        if hasattr(e_rv, 'cpu'):
            e_rv = e_rv.cpu().numpy()
        e_pose = np.concatenate([e_pos, e_rv])
        ddx = K_stiffness @ e_pose - K_damping @ ee_twist
        dJ_full = pin.getFrameJacobianTimeVariation(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
        dJ = dJ_full[:, :7]
        tau_a = J.T @ (Lambda @ (ddx - dJ @ dq_c)) + nle
        N_p = np.eye(7) - J.T @ J_bar.T
        tau_n = N_p @ (K_null @ (q_home - q_c) - D_null @ dq_c)
        tau_arm_h = np.clip(tau_a + tau_n, -tau_max_arm, tau_max_arm)

        tau_g = np.clip(kp_gripper * (grip_hold - q_g) + kv_gripper * (-dq_g), -max_grip_force, max_grip_force)
        franka.control_dofs_force(np.concatenate([tau_arm_h, tau_g]), all_dof)
        scene.step()
except KeyboardInterrupt:
    pass


# ===== 에러 플롯 =====
N = len(log_pos_err)
t = np.arange(N) / 60.0

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(t, log_pos_err, 'b-')
axes[0, 0].set_ylabel('Position Error (m)')
axes[0, 0].set_title('Arm Position Error')
axes[0, 0].grid(True)
# phase 구분선
cum = 0
for p in PHASES:
    cum += p["duration"]
    axes[0, 0].axvline(x=cum/60, color='gray', linestyle='--', alpha=0.5)

axes[0, 1].plot(t, log_rot_err, 'r-')
axes[0, 1].set_ylabel('Rotation Error (rad)')
axes[0, 1].set_title('Arm Rotation Error')
axes[0, 1].grid(True)
cum = 0
for p in PHASES:
    cum += p["duration"]
    axes[0, 1].axvline(x=cum/60, color='gray', linestyle='--', alpha=0.5)

axes[1, 0].plot(t, log_grip_err, 'g-')
axes[1, 0].set_ylabel('Gripper Error')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Gripper Position Error')
axes[1, 0].grid(True)
cum = 0
for p in PHASES:
    cum += p["duration"]
    axes[1, 0].axvline(x=cum/60, color='gray', linestyle='--', alpha=0.5)

axes[1, 1].plot(t, log_grip_force, 'm-')
axes[1, 1].set_ylabel('Gripper Force (N)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Gripper Applied Force')
axes[1, 1].grid(True)
cum = 0
for p in PHASES:
    cum += p["duration"]
    axes[1, 1].axvline(x=cum/60, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('task_space_gripper_error.png', dpi=150)
plt.show()
print("플롯 저장: task_space_gripper_error.png")
