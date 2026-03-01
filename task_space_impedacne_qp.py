# task_space_impedacne_qp.py - QP 기반 Task Space Impedance Control
# X = [ddq(7), tau(7)] ∈ R^14
# min  ||J*ddq + dJ*dq - ddx_cmd||²_W + ||ddq||²_reg + ||tau||²_reg
# s.t. M*ddq - tau = -nle  (동역학)
#      -tau_max <= tau <= tau_max  (토크 한계)
#      ddq_min_safe <= ddq <= ddq_max_safe  (관절 한계 CBF)
#
# 필요: pip install pin osqp

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import genesis as gs
import pinocchio as pin
import osqp
from scipy import sparse
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

TARGET_POS  = np.array([0.65, 0.0, 0.25])
TARGET_QUAT = np.array([0, 1, 0, 0])  # wxyz

mouse_plugin = gs.vis.viewer_plugins.MouseInteractionPlugin(
    use_force=True,
    spring_const=100.0,
)
scene.viewer.add_plugin(mouse_plugin)

scene.build()

# ===== 목표 시각화 =====
scene.draw_debug_sphere(pos=TARGET_POS, radius=0.02, color=(1.0, 0.1, 0.1, 0.9))

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

# arm: torque control, gripper: position control
franka.set_dofs_kp(np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]))
franka.set_dofs_kv(np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# ===== 관절/토크 한계 =====
tau_max_arr = np.array([87, 87, 87, 87, 12, 12, 12])
# Franka Panda 관절 한계 (URDF 기준)
q_lower = pin_model.lowerPositionLimit[:7].copy()
q_upper = pin_model.upperPositionLimit[:7].copy()
# Franka Panda 관절 속도 한계 (스펙)
dq_max = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
print(f"관절 한계: lower={q_lower}, upper={q_upper}")

# ===== 임피던스 게인 =====
K_stiffness = np.diag([500, 500, 500, 200, 200, 200])   # 6x6
K_damping   = np.diag([50,  50,  50,  20,  20,  20])    # 6x6

# ===== QP 가중치 =====
W_task    = np.diag([500, 500, 500, 100, 100, 100])  # 6x6 task tracking
W_reg_ddq = 1e-3 * np.eye(7)                          # 가속도 정규화
W_reg_tau = 1e-5 * np.eye(7)                           # 토크 정규화

# ===== CBF 게인 =====
k_cbf   = 20.0   # 관절 위치 마진
k_cbf_v = 5.0    # 관절 속도 마진

# ===== QP 차원 =====
n_dof = 7
n_x = 2 * n_dof  # 14 (ddq + tau)
# 제약: 동역학(7) + 토크한계(7) + 관절한계CBF(7) = 21행
n_constraints = 3 * n_dof  # 21


# ===== QP 제어 루프 =====
print("QP Task Space Impedance 제어 시작...")
print(f"목표: pos={TARGET_POS}, quat={TARGET_QUAT}")

log_pos_err = []
log_rot_err = []
log_pos_xyz = []
log_rot_xyz = []
N_STEPS = 1000

for i in range(N_STEPS):
    # ===== 1. 상태 읽기 =====
    q_current  = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()
    current_pos  = end_effector.get_pos().cpu().numpy()
    current_quat = end_effector.get_quat().cpu().numpy()

    # ===== 2. Pinocchio 동역학 =====
    q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_current
    dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_current
    pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)

    M_q     = pin_data.M[:7, :7]
    J_full  = pin.getFrameJacobian(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J       = J_full[:, :7]   # 6x7
    dJ_full = pin.getFrameJacobianTimeVariation(pin_model, pin_data, hand_frame_id, pin.LOCAL_WORLD_ALIGNED)
    dJ      = dJ_full[:, :7]  # 6x7
    nle     = pin_data.nle[:7]  # C*dq + g

    # ===== 3. 임피던스 명령 가속도 ddx_cmd =====
    ee_twist = J @ dq_current  # 6x1

    error_pos    = TARGET_POS - current_pos
    error_quat   = gs.transform_quat_by_quat(gs.inv_quat(current_quat), TARGET_QUAT)
    error_rotvec = gs.quat_to_rotvec(error_quat)
    if hasattr(error_rotvec, 'cpu'):
        error_rotvec = error_rotvec.cpu().numpy()
    error_pose = np.concatenate([error_pos, error_rotvec])  # 6x1

    # ddx_cmd = K_stiffness * e - K_damping * twist (목표 속도/가속도 = 0)
    ddx_cmd = K_stiffness @ error_pose - K_damping @ ee_twist  # 6x1

    # ===== 4. QP 행렬 조립 =====
    # --- 목적함수: H, g ---
    # H (14x14)
    H = np.zeros((n_x, n_x))
    H[:7, :7] = J.T @ W_task @ J + W_reg_ddq   # ddq 블록
    H[7:, 7:] = W_reg_tau                        # tau 블록
    # 대칭 보장
    H = 0.5 * (H + H.T)

    # g (14,)
    residual = dJ @ dq_current - ddx_cmd  # 6x1
    g_vec = np.zeros(n_x)
    g_vec[:7] = J.T @ W_task @ residual

    # --- 등식 제약: M*ddq - tau = -nle ---
    A_eq = np.hstack([M_q, -np.eye(7)])  # 7x14
    b_eq = -nle                           # 7x1

    # --- 부등식 제약: 토크 한계 ---
    A_tau = np.hstack([np.zeros((7, 7)), np.eye(7)])  # 7x14
    l_tau = -tau_max_arr
    u_tau =  tau_max_arr

    # --- 부등식 제약: 관절 한계 CBF ---
    A_ddq = np.hstack([np.eye(7), np.zeros((7, 7))])  # 7x14
    ddq_max_safe = k_cbf * (q_upper - q_current) + k_cbf_v * ( dq_max - dq_current)
    ddq_min_safe = k_cbf * (q_lower - q_current) + k_cbf_v * (-dq_max - dq_current)

    # --- 전체 제약 합치기 ---
    A_all = np.vstack([A_eq, A_tau, A_ddq])  # 21x14
    l_all = np.concatenate([b_eq, l_tau, ddq_min_safe])
    u_all = np.concatenate([b_eq, u_tau, ddq_max_safe])

    # ===== 5. OSQP 풀기 =====
    P_sparse = sparse.csc_matrix(H)
    A_sparse = sparse.csc_matrix(A_all)

    qp_solver = osqp.OSQP()
    qp_solver.setup(
        P=P_sparse, q=g_vec, A=A_sparse, l=l_all, u=u_all,
        verbose=False, warm_start=True,
        eps_abs=1e-4, eps_rel=1e-4, max_iter=4000,
        polish=True,
    )

    result = qp_solver.solve()

    if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
        tau_opt = result.x[7:]
    else:
        print(f"[{i}] QP 실패: {result.info.status}, fallback PD")
        tau_opt = J.T @ (K_stiffness @ error_pose - K_damping @ ee_twist) + nle

    # ===== 6. 제어 입력 =====
    tau_opt = np.clip(tau_opt, -tau_max_arr, tau_max_arr)
    franka.control_dofs_force(tau_opt, motors_dof)
    scene.step()

    # 에러 기록
    log_pos_err.append(np.linalg.norm(error_pos))
    log_rot_err.append(np.linalg.norm(error_rotvec))
    log_pos_xyz.append(error_pos.copy())
    log_rot_xyz.append(error_rotvec.copy())

    if i % 60 == 0:
        print(f"[{i:5d}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
              f"rot_err: {np.linalg.norm(error_rotvec):.4f}rad")

print("제어 완료!")


# ===== 에러 플롯 =====
t = np.arange(N_STEPS) / 60.0
log_pos_xyz = np.array(log_pos_xyz)
log_rot_xyz = np.array(log_rot_xyz)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(t, log_pos_err, 'b-')
axes[0, 0].set_ylabel('Position Error (m)')
axes[0, 0].set_title('Position Error Norm')
axes[0, 0].grid(True)

axes[0, 1].plot(t, log_pos_xyz[:, 0], 'r-', label='x')
axes[0, 1].plot(t, log_pos_xyz[:, 1], 'g-', label='y')
axes[0, 1].plot(t, log_pos_xyz[:, 2], 'b-', label='z')
axes[0, 1].set_ylabel('Position Error (m)')
axes[0, 1].set_title('Position Error XYZ')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(t, log_rot_err, 'r-')
axes[1, 0].set_ylabel('Rotation Error (rad)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Rotation Error Norm')
axes[1, 0].grid(True)

axes[1, 1].plot(t, log_rot_xyz[:, 0], 'r-', label='rx')
axes[1, 1].plot(t, log_rot_xyz[:, 1], 'g-', label='ry')
axes[1, 1].plot(t, log_rot_xyz[:, 2], 'b-', label='rz')
axes[1, 1].set_ylabel('Rotation Error (rad)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Rotation Error XYZ')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('task_space_impedance_qp_error.png', dpi=150)
plt.show()
print("플롯 저장: task_space_impedance_qp_error.png")
