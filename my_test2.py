# custom project genesis - 임피던스 제어 (단일 목표점)

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

# ===== 최종 목표점 정의 =====
target_pos = np.array([0.65, 0.0, 0.25])
target_quat = np.array([0, 1, 0, 0])

print(f"목표: pos={target_pos}, quat={target_quat}")

# ===== 임피던스 제어 루프 =====
print("임피던스 제어 시작...")

max_steps = 10000  # 최대 10초
tolerance_pos = 0.01  # 1cm
tolerance_rot = 0.05  # ~3도

for i in range(max_steps):
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
    
    # 수렴 체크
    pos_err_norm = np.linalg.norm(error_pos)
    rot_err_norm = np.linalg.norm(error_rotvec)
    
    if pos_err_norm < tolerance_pos and rot_err_norm < tolerance_rot:
        print(f"\n목표 도달! (Step {i})")
        print(f"  최종 위치 오차: {pos_err_norm:.4f}m")
        print(f"  최종 자세 오차: {rot_err_norm:.4f}rad")
        break
    
    # 디버깅 출력
    if i % 100 == 0:
        print(f"[Step {i:4d}] pos_err: {pos_err_norm:.4f}m, "
              f"rot_err: {rot_err_norm:.4f}rad, "
              f"vel: {np.linalg.norm(current_vel):.3f}")
    
    # Wrench 계산 (임피던스 제어)
    F_task = K_stiffness @ error_twist - K_damping @ current_vel  # 6x1
    
    # Task Space Torque
    tau_task = J_arm.T @ F_task  # 7x1
    
    # Mass Matrix
    M = franka.get_mass_mat(decompose=False)[:7, :7]  # 7x7 (arm만)
    if hasattr(M, 'cpu'):
        M = M.cpu().numpy()
    
    # Null Space Control (home position 유지)
    N = np.eye(7) - np.linalg.pinv(J_arm) @ J_arm  # 7x7 null space projection
    tau_null = N @ (K_null @ (q_home - q_current) - D_null @ dq_current)  # 7x1
    
    # 최종 토크
    tau_total = tau_task + tau_null  # 7x1
    
    # 토크 제한 (안전)
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_total = np.clip(tau_total, -tau_max, tau_max)
    
    # 제어 입력
    franka.control_dofs_force(tau_total, motors_dof)
    scene.step()

print("임피던스 제어 완료!")

# 5초 대기
print("5초 대기 중...")
for _ in range(5000):
    scene.step()
