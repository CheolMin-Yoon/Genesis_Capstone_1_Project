# custom project genesis - 임피던스 제어 + Null Space

import sys
import os
# Genesis 루트를 sys.path 맨 앞에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.001,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 일단 그리퍼는 position control, manipulator만 torque control
franka.set_dofs_kp(
    np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]),
)
franka.set_dofs_kv(
    np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")

# ===== 임피던스 게인 (Task Space: 6x6 - 위치 3 + 자세 3) =====
K_stiffness = np.diag([1000, 1000, 1000, 500, 500, 500])  # 위치 강성, 자세 강성
K_damping = np.diag([100, 100, 100, 20, 20, 20])  # 위치 댐핑, 자세 댐핑

# ===== Null Space 게인 (Joint Space: 7x7) =====
K_null = np.diag([10, 10, 10, 10, 5, 5, 5])  # 관절 공간 강성
D_null = np.diag([5, 5, 5, 5, 2, 2, 2])  # 관절 공간 댐핑
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])  # Franka home position

# ===== Path Planning =====
# 목표점 정의
qpos_goal = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
# gripper set open
qpos_goal[-2:] = 0.04

# OMPL path planning from current to qpos_goal
path = franka.plan_path(
    qpos_goal=qpos_goal,
    num_waypoints=300,
)

# 시각화
path_debug = scene.draw_debug_path(path, franka)

# ===== Minimum Jerk Trajectory 생성 함수 =====
def minimum_jerk_trajectory(q_waypoints, total_time, dt):
    """
    Minimum Jerk Trajectory 생성 (5차 다항식)
    
    Args:
        q_waypoints: (N, 7) OMPL waypoints
        total_time: 전체 시간 (초)
        dt: 시간 간격 (초)
    
    Returns:
        q_traj: (M, 7) 위치 궤적
        dq_traj: (M, 7) 속도 궤적
        ddq_traj: (M, 7) 가속도 궤적
    """
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
        t_seg_end = (seg_idx + 1) * segment_time
        
        mask = (t_traj >= t_seg_start) & (t_traj < t_seg_end)
        t_seg = t_traj[mask]
        
        if len(t_seg) == 0:
            continue
        
        # 정규화된 시간 [0, 1]
        tau = (t_seg - t_seg_start) / segment_time
        
        # Minimum Jerk 5차 다항식
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / segment_time
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (segment_time**2)
        
        for joint_idx in range(7):
            q_diff = q_end[joint_idx] - q_start[joint_idx]
            q_traj[mask, joint_idx] = q_start[joint_idx] + q_diff * s
            dq_traj[mask, joint_idx] = q_diff * ds
            ddq_traj[mask, joint_idx] = q_diff * dds
    
    # 마지막 waypoint
    q_traj[-1] = q_waypoints[-1]
    dq_traj[-1] = 0
    ddq_traj[-1] = 0
    
    return q_traj, dq_traj, ddq_traj


# ===== Minimum Jerk 보간 =====
path_np = path[:, :7].cpu().numpy()  # (300, 7) arm만
total_time = 3.0  # 3초 동안 이동
dt = 0.001  # 10ms

print("Minimum Jerk 궤적 생성 중...")
q_traj, dq_traj, ddq_traj = minimum_jerk_trajectory(path_np, total_time, dt)
print(f"보간 완료: {len(q_traj)}개 포인트 생성")

# ===== Cartesian target 미리 계산 (FK) =====
print("Cartesian target 계산 중...")
cartesian_targets = []

for i in range(len(q_traj)):
    # FK로 Cartesian 위치/자세 계산
    q_with_gripper = np.concatenate([q_traj[i], [0.04, 0.04]])
    franka.set_dofs_position(q_with_gripper, np.arange(9))
    scene.step()
    
    target_pos = end_effector.get_pos().cpu().numpy().copy()
    target_quat = end_effector.get_quat().cpu().numpy().copy()
    cartesian_targets.append((target_pos, target_quat))
    
    if i % 50 == 0:
        print(f"  {i}/{len(q_traj)} 계산 완료")

print(f"총 {len(cartesian_targets)}개 Cartesian target 생성 완료")

# ===== 로봇을 시작 위치로 리셋 =====
print("로봇 초기 위치로 리셋 중...")
franka.set_dofs_position(path[0], np.arange(9))  # 첫 waypoint로 리셋
for _ in range(100):  # 안정화
    scene.step()
print("리셋 완료!")

# ===== 임피던스 제어 루프 =====
print("임피던스 제어 시작...")

for i in range(len(q_traj)):
    target_pos, target_quat = cartesian_targets[i]
    q_desired = q_traj[i]
    dq_desired = dq_traj[i]
    ddq_desired = ddq_traj[i]
    
    # 현재 상태 가져오기
    current_pos = end_effector.get_pos().cpu().numpy()
    current_quat = end_effector.get_quat().cpu().numpy()
    
    # Jacobian & joint velocity
    J = franka.get_jacobian(link=end_effector)  # 6x9 또는 6x7
    q_current = franka.get_dofs_position(motors_dof).cpu().numpy()  # 7x1
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()  # 7x1
    
    # Jacobian을 numpy로 변환
    if hasattr(J, 'cpu'):
        J = J.cpu().numpy()
    
    # Jacobian 크기 확인 후 슬라이싱
    if J.shape[1] == 9:
        J_arm = J[:, :7]  # 6x7 (arm만)
    else:
        J_arm = J  # 이미 6x7
    
    # End-effector twist (linear + angular velocity)
    ee_twist = J_arm @ dq_current  # 6x1
    current_vel = ee_twist  # 6x1 (이미 numpy)
    
    # 에러 계산 (Lie group 기반)
    error_pos = target_pos - current_pos  # 위치 오차
    error_quat = gs.transform_quat_by_quat(gs.inv_quat(current_quat), target_quat)  # 자세 오차
    error_rotvec = gs.quat_to_rotvec(error_quat)  # 회전 벡터로 변환
    
    # numpy로 변환
    if hasattr(error_rotvec, 'cpu'):
        error_rotvec = error_rotvec.cpu().numpy()
    
    error_twist = np.concatenate([error_pos, error_rotvec])  # 6x1 트위스트 오차
    
    # 디버깅 출력
    if i % 30 == 0:
        print(f"[Step {i:3d}] pos_err: {np.linalg.norm(error_pos):.4f}m, "
              f"rot_err: {np.linalg.norm(error_rotvec):.4f}rad, "
              f"vel: {np.linalg.norm(current_vel):.3f}")
    
    # Wrench 계산 (임피던스 제어)
    F_task = K_stiffness @ error_twist - K_damping @ current_vel  # 6x1
    
    # Task Space Torque
    tau_task = J_arm.T @ F_task  # 7x1
    
    # Mass Matrix
    M = franka.get_mass_mat(decompose=False)[:7, :7]  # 7x7 (arm만)
    if hasattr(M, 'cpu'):
        M = M.cpu().numpy()
    
    # Null Space Control (목표 관절 각도 추적 + 속도 feedforward)
    N = np.eye(7) - np.linalg.pinv(J_arm) @ J_arm  # 7x7 null space projection
    tau_null = N @ (K_null @ (q_desired - q_current) - D_null @ (dq_current - dq_desired))  # 7x1
    
    # Feedforward 토크 (가속도 보상)
    tau_feedforward = M @ ddq_desired  # 7x1
    
    # 최종 토크
    tau_total = tau_task + tau_null + tau_feedforward  # 7x1
    
    # 토크 제한 (안전)
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_total = np.clip(tau_total, -tau_max, tau_max)
    
    # 제어 입력
    franka.control_dofs_force(tau_total, motors_dof)
    scene.step()

print("임피던스 제어 완료!")

# 5초 대기
print("5초 대기 중...")
for _ in range(500):
    scene.step()
