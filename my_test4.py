# my_test4.py - Task Space Impedance Control + Pinocchio FK
# 방법 2: OMPL → Pinocchio FK → Minimum Jerk (Cartesian) → Task Space Impedance
#
# 필요: pip install pin (pinocchio)

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

# ===== Pinocchio 모델 로드 (FK 전용) =====
# Genesis 내장 URDF 경로
urdf_path = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet/panda.urdf")
mesh_dir = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet")

pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()

# Pinocchio에서 hand 링크의 frame ID 찾기
hand_frame_id = pin_model.getFrameId("panda_hand")
if hand_frame_id >= pin_model.nframes:
    # frame이 없으면 마지막 joint 사용
    hand_frame_id = pin_model.getFrameId("panda_link8")
    if hand_frame_id >= pin_model.nframes:
        print("Warning: hand frame을 찾을 수 없음. 마지막 joint 사용")
        hand_frame_id = pin_model.nframes - 1

print(f"Pinocchio 모델 로드 완료: {pin_model.nq} DOF, hand_frame_id={hand_frame_id}")


def pinocchio_fk(q_7dof):
    """
    Pinocchio로 Forward Kinematics 계산 (시뮬레이션 없이!)
    
    Args:
        q_7dof: (7,) 관절 각도
    
    Returns:
        pos: (3,) end-effector 위치
        quat: (4,) end-effector 쿼터니언 (w, x, y, z)
    """
    # Pinocchio는 gripper DOF도 포함할 수 있으므로 패딩
    q_pin = np.zeros(pin_model.nq)
    q_pin[:7] = q_7dof

    pin.forwardKinematics(pin_model, pin_data, q_pin)
    pin.updateFramePlacements(pin_model, pin_data)

    oMf = pin_data.oMf[hand_frame_id]
    pos = oMf.translation.copy()

    # Rotation matrix → quaternion (w, x, y, z)
    quat = pin.Quaternion(oMf.rotation)
    quat_wxyz = np.array([quat.w, quat.x, quat.y, quat.z])

    return pos, quat_wxyz


# ===== Minimum Jerk (Joint Space) =====
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


# ===== 임피던스 게인 (Task Space: 6x6) =====
K_stiffness = np.diag([1000, 1000, 1000, 500, 500, 500])
K_damping = np.diag([100, 100, 100, 20, 20, 20])

# Null Space 게인 (7x7)
K_null = np.diag([10, 10, 10, 10, 5, 5, 5])
D_null = np.diag([5, 5, 5, 5, 2, 2, 2])
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

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

# ===== Pinocchio FK로 Cartesian target 계산 (시뮬레이션 없이!) =====
print("Pinocchio FK로 Cartesian target 계산 중...")
cartesian_targets = []

for i in range(len(q_traj)):
    pos, quat = pinocchio_fk(q_traj[i])
    cartesian_targets.append((pos, quat))
    if i % 500 == 0:
        print(f"  {i}/{len(q_traj)} 계산 완료")

print(f"총 {len(cartesian_targets)}개 Cartesian target 생성 완료")

# ===== Task Space Impedance 제어 루프 =====
print("Task Space Impedance 제어 시작...")

for i in range(len(q_traj)):
    target_pos, target_quat = cartesian_targets[i]
    q_desired = q_traj[i]
    dq_desired = dq_traj[i]

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
    current_vel = J_arm @ dq_current  # 6x1

    # 에러 계산
    error_pos = target_pos - current_pos
    error_quat = gs.transform_quat_by_quat(gs.inv_quat(current_quat), target_quat)
    error_rotvec = gs.quat_to_rotvec(error_quat)
    if hasattr(error_rotvec, 'cpu'):
        error_rotvec = error_rotvec.cpu().numpy()
    error_twist = np.concatenate([error_pos, error_rotvec])

    # Wrench (임피던스)
    F_task = K_stiffness @ error_twist - K_damping @ current_vel

    # Task Space Torque
    tau_task = J_arm.T @ F_task

    # Null Space Control
    N = np.eye(7) - np.linalg.pinv(J_arm) @ J_arm
    tau_null = N @ (K_null @ (q_desired - q_current) - D_null @ (dq_current - dq_desired))

    # 최종 토크
    tau_total = tau_task + tau_null

    # 토크 제한
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_total = np.clip(tau_total, -tau_max, tau_max)

    # 제어 입력
    franka.control_dofs_force(tau_total, motors_dof)
    scene.step()

    # 디버깅 출력
    if i % 500 == 0:
        print(f"[Step {i:5d}/{len(q_traj)}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
              f"rot_err: {np.linalg.norm(error_rotvec):.4f}rad")

print("제어 완료!")

# 3초 대기
print("3초 대기 중...")
for _ in range(3000):
    scene.step()
