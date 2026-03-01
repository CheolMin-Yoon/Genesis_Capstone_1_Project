# my_test5.py - QP-based Task Space Impedance Control
# 방법 3: OMPL → Pinocchio FK → Impedance → QP Solver
#
# 파이프라인:
#   1. OMPL: waypoint 생성
#   2. Minimum Jerk: x_d(t), dx_d(t), ddx_d(t) 생성
#   3. Impedance Law: ddx_cmd 계산
#   4. QP Solver: 토크 제한 + 동역학 + null space 최적화
#
# 필요: pip install pin qpsolvers quadprog

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import genesis as gs

try:
    import pinocchio as pin
except ImportError:
    print("Pinocchio가 필요합니다: pip install pin")
    sys.exit(1)

try:
    from qpsolvers import solve_qp
except ImportError:
    print("qpsolvers가 필요합니다: pip install qpsolvers quadprog")
    sys.exit(1)

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.001,  # 1000Hz
    ),
    show_viewer=True,
)

plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

franka.set_dofs_kp(np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]))
franka.set_dofs_kv(np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")

# ===== Pinocchio 모델 로드 =====
urdf_path = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet/panda.urdf")
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

hand_frame_id = pin_model.getFrameId("panda_hand")
if hand_frame_id >= pin_model.nframes:
    hand_frame_id = pin_model.getFrameId("panda_link8")
    if hand_frame_id >= pin_model.nframes:
        hand_frame_id = pin_model.nframes - 1

print(f"Pinocchio 모델 로드 완료: {pin_model.nq} DOF")


def pinocchio_fk(q_7dof):
    """Pinocchio FK: q → (pos, quat)"""
    q_pin = np.zeros(pin_model.nq)
    q_pin[:7] = q_7dof
    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf = pin_data.oMf[hand_frame_id]
    pos = oMf.translation.copy()
    quat = pin.Quaternion(oMf.rotation)
    return pos, np.array([quat.w, quat.x, quat.y, quat.z])


def minimum_jerk_trajectory(q_waypoints, total_time, dt):
    """Joint Space Minimum Jerk"""
    N = len(q_waypoints)
    segment_time = total_time / (N - 1)
    t_traj = np.arange(0, total_time, dt)
    M = len(t_traj)

    q_traj = np.zeros((M, 7))
    dq_traj = np.zeros((M, 7))
    ddq_traj = np.zeros((M, 7))

    for seg_idx in range(N - 1):
        q_start = q_waypoints[seg_idx]
        q_end = q_waypoints[seg_idx + 1]
        t_seg_start = seg_idx * segment_time
        mask = (t_traj >= t_seg_start) & (t_traj < (seg_idx + 1) * segment_time)
        t_seg = t_traj[mask]
        if len(t_seg) == 0:
            continue

        tau = (t_seg - t_seg_start) / segment_time
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / segment_time
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (segment_time**2)

        for j in range(7):
            q_diff = q_end[j] - q_start[j]
            q_traj[mask, j] = q_start[j] + q_diff * s
            dq_traj[mask, j] = q_diff * ds
            ddq_traj[mask, j] = q_diff * dds

    q_traj[-1] = q_waypoints[-1]
    dq_traj[-1] = 0
    ddq_traj[-1] = 0
    return q_traj, dq_traj, ddq_traj


# ===== 임피던스 게인 =====
K_stiffness = np.diag([1000, 1000, 1000, 500, 500, 500])
K_damping = np.diag([100, 100, 100, 20, 20, 20])

# Null space 목표
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
w_null = 0.1  # null space 가중치

# 토크 제한
tau_min = np.array([-87, -87, -87, -87, -12, -12, -12])
tau_max = np.array([87, 87, 87, 87, 12, 12, 12])


def solve_impedance_qp(J_arm, M, C_dq, g, ddx_cmd, q_current, dq_current, q_desired):
    """
    QP Solver: 임피던스 제어 + 토크 제한 + 동역학 + null space 최적화
    
    최적화 문제:
        min  || J*ddq - (ddx_cmd - dJ*dq) ||^2 + w_null * || ddq - ddq_null ||^2
        s.t. tau_min <= M*ddq + C*dq + g <= tau_max
    
    결정 변수: ddq (7x1)
    
    Args:
        J_arm: (6, 7) Jacobian
        M: (7, 7) Mass matrix
        C_dq: (7,) Coriolis + centrifugal (C*dq)
        g: (7,) Gravity vector
        ddx_cmd: (6,) 임피던스에서 계산된 목표 가속도
        q_current: (7,) 현재 관절 각도
        dq_current: (7,) 현재 관절 속도
        q_desired: (7,) null space 목표 관절 각도
    
    Returns:
        tau: (7,) 최적 토크
    """
    n_dof = 7

    # Null space 목표 가속도 (관절 공간에서 home으로 복귀)
    K_ns = 10.0
    D_ns = 5.0
    ddq_null = K_ns * (q_desired - q_current) - D_ns * dq_current

    # QP: min 0.5 * ddq^T * H * ddq + f^T * ddq
    # Task: || J*ddq - ddx_cmd ||^2 = ddq^T * J^T*J * ddq - 2*ddx_cmd^T*J * ddq + ...
    # Null: w_null * || ddq - ddq_null ||^2

    H_task = J_arm.T @ J_arm  # 7x7
    f_task = -J_arm.T @ ddx_cmd  # 7x1

    H_null = w_null * np.eye(n_dof)  # 7x7
    f_null = -w_null * ddq_null  # 7x1

    H = H_task + H_null  # 7x7
    f = f_task + f_null  # 7x1

    # 대칭 보장
    H = 0.5 * (H + H.T) + 1e-8 * np.eye(n_dof)

    # 부등식 제약: tau_min <= M*ddq + C_dq + g <= tau_max
    # → M*ddq <= tau_max - C_dq - g  (상한)
    # → -M*ddq <= -tau_min + C_dq + g  (하한)
    # → G*ddq <= h

    bias = C_dq + g  # 7x1

    G = np.vstack([M, -M])  # 14x7
    h = np.concatenate([tau_max - bias, -tau_min + bias])  # 14x1

    # QP 풀기
    try:
        ddq_opt = solve_qp(H, f, G, h, solver="quadprog")
    except Exception:
        # QP 실패 시 fallback: 단순 pseudoinverse
        ddq_opt = np.linalg.pinv(J_arm) @ ddx_cmd

    if ddq_opt is None:
        ddq_opt = np.linalg.pinv(J_arm) @ ddx_cmd

    # 토크 계산: tau = M*ddq + C*dq + g
    tau = M @ ddq_opt + bias

    # 안전 클리핑
    tau = np.clip(tau, tau_min, tau_max)

    return tau


# ===== Path Planning =====
qpos_goal = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
qpos_goal[-2:] = 0.04

path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=200)
path_debug = scene.draw_debug_path(path, franka)

# ===== Minimum Jerk 보간 =====
path_np = path[:, :7].cpu().numpy()
total_time = 5.0
dt = 0.001

print("Minimum Jerk 궤적 생성 중...")
q_traj, dq_traj, ddq_traj = minimum_jerk_trajectory(path_np, total_time, dt)
print(f"보간 완료: {len(q_traj)}개 포인트")

# ===== Pinocchio FK =====
print("Pinocchio FK 계산 중...")
cartesian_targets = []
for i in range(len(q_traj)):
    pos, quat = pinocchio_fk(q_traj[i])
    cartesian_targets.append((pos, quat))
    if i % 500 == 0:
        print(f"  {i}/{len(q_traj)} 완료")
print(f"총 {len(cartesian_targets)}개 target 생성 완료")


# ===== QP-based Impedance 제어 루프 =====
print("QP-based Impedance 제어 시작...")

for i in range(len(q_traj)):
    target_pos, target_quat = cartesian_targets[i]
    q_desired = q_traj[i]

    # 현재 상태
    current_pos = end_effector.get_pos().cpu().numpy()
    current_quat = end_effector.get_quat().cpu().numpy()
    q_current = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()

    # Jacobian
    J = franka.get_jacobian(link=end_effector)
    if hasattr(J, 'cpu'):
        J = J.cpu().numpy()
    J_arm = J[:, :7] if J.shape[1] == 9 else J

    # End-effector velocity
    current_vel = J_arm @ dq_current

    # 에러 계산
    error_pos = target_pos - current_pos
    error_quat = gs.transform_quat_by_quat(gs.inv_quat(current_quat), target_quat)
    error_rotvec = gs.quat_to_rotvec(error_quat)
    if hasattr(error_rotvec, 'cpu'):
        error_rotvec = error_rotvec.cpu().numpy()
    error_twist = np.concatenate([error_pos, error_rotvec])

    # ===== Step 3: Impedance Law → ddx_cmd =====
    # ddx_cmd = K_stiffness * error - K_damping * velocity
    ddx_cmd = K_stiffness @ error_twist - K_damping @ current_vel

    # ===== Step 4: 동역학 파라미터 (Genesis에서 가져오기) =====
    M = franka.get_mass_mat(decompose=False)[:7, :7]
    if hasattr(M, 'cpu'):
        M = M.cpu().numpy()

    # Coriolis + Gravity: Genesis에서 직접 가져오기
    # Genesis는 get_dofs_force()로 C*dq + g를 얻을 수 있음
    # 또는 Pinocchio로 계산
    q_pin = np.zeros(pin_model.nq)
    q_pin[:7] = q_current
    dq_pin = np.zeros(pin_model.nv)
    dq_pin[:7] = dq_current

    pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)
    C_dq = pin.nonLinearEffects(pin_model, pin_data, q_pin, dq_pin)[:7]  # C*dq + g
    g = pin.computeGeneralizedGravity(pin_model, pin_data, q_pin)[:7]
    C_dq_only = C_dq - g  # Coriolis만

    # ===== Step 4: QP Solver =====
    tau = solve_impedance_qp(
        J_arm=J_arm,
        M=M,
        C_dq=C_dq_only,
        g=g,
        ddx_cmd=ddx_cmd,
        q_current=q_current,
        dq_current=dq_current,
        q_desired=q_desired,
    )

    # 제어 입력
    franka.control_dofs_force(tau, motors_dof)
    scene.step()

    # 디버깅 출력
    if i % 500 == 0:
        print(f"[Step {i:5d}/{len(q_traj)}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
              f"rot_err: {np.linalg.norm(error_rotvec):.4f}rad, "
              f"tau_max: {np.max(np.abs(tau)):.1f}Nm")

print("QP-based Impedance 제어 완료!")

# 3초 대기
print("3초 대기 중...")
for _ in range(3000):
    scene.step()
