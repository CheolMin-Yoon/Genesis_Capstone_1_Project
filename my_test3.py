# my_test3.py - Joint Space Impedance Control + Minimum Jerk
# 방법 1: OMPL → Minimum Jerk (Joint Space) → Joint Space Impedance

import sys
import os
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
        dt=1/60,    # 60Hz 제어 주파수
        substeps=16,  # 내부 물리 적분: ~1000Hz 정밀도
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

# ===== MouseInteractionPlugin 추가 (scene.build() 전에 등록) =====
mouse_plugin = gs.vis.viewer_plugins.MouseInteractionPlugin(
    use_force=True,       # True: 스프링 힘 적용 (MuJoCo 방식)
    spring_const=1000.0,
)
scene.viewer.add_plugin(mouse_plugin)

scene.build()

motors_dof = np.arange(7)       # arm: panda_joint1~7
fingers_dof = np.arange(7, 9)  # gripper: panda_finger_joint1/2

# fixed=True 시 9 DOF (arm 7 + gripper 2)
# arm: torque control (kp=0, kv=0), gripper: position control
franka.set_dofs_kp(np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]))
franka.set_dofs_kv(np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("panda_link7")

# ===== Joint Space 임피던스 게인 (7x7) =====
K_p = np.diag([600, 600, 600, 600, 250, 150, 50])  # 관절 강성
K_d = np.diag([50, 50, 50, 50, 30, 25, 10])  # 관절 댐핑

# ===== Minimum Jerk Trajectory 생성 함수 =====
def minimum_jerk_trajectory(q_waypoints, total_time, dt):
    """
    Joint Space Minimum Jerk Trajectory (5차 다항식)
    q_d(t), dq_d(t), ddq_d(t) 생성
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

        tau = (t_seg - t_seg_start) / segment_time

        # Minimum Jerk 5차 다항식
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


# ===== Path Planning =====
qpos_goal = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
qpos_goal[-2:] = 0.04

path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=200)
path_debug = scene.draw_debug_path(path, franka)

# ===== Minimum Jerk 보간 (Joint Space) =====
path_np = path[:, :7].cpu().numpy()
total_time = 5.0  # 5초
dt = 1/60  # 60Hz 제어 주파수

print("Minimum Jerk 궤적 생성 중...")
q_traj, dq_traj, ddq_traj = minimum_jerk_trajectory(path_np, total_time, dt)
print(f"보간 완료: {len(q_traj)}개 포인트 ({total_time}초)")

# ===== Joint Space Impedance 제어 루프 =====
print("Joint Space Impedance 제어 시작...")

for i in range(len(q_traj)):
    q_desired = q_traj[i]
    dq_desired = dq_traj[i]
    ddq_desired = ddq_traj[i]

    # 현재 상태
    q_current = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()

    # Mass Matrix
    M = franka.get_mass_mat(decompose=False)[:7, :7]
    if hasattr(M, 'cpu'):
        M = M.cpu().numpy()

    # Joint Space Impedance: tau = M*ddq_d + K_p*(q_d - q) + K_d*(dq_d - dq)
    tau_impedance = K_p @ (q_desired - q_current) + K_d @ (dq_desired - dq_current)
    tau_feedforward = M @ ddq_desired
    tau_total = tau_impedance + tau_feedforward

    # 토크 제한
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_total = np.clip(tau_total, -tau_max, tau_max)

    # 제어 입력
    franka.control_dofs_force(tau_total, motors_dof)
    scene.step()

    # 디버깅 출력
    if i % 30 == 0:  # 0.5초마다 출력 (60Hz 기준)
        q_err = np.linalg.norm(q_desired - q_current)
        print(f"[Step {i:5d}/{len(q_traj)}] q_err: {q_err:.4f}rad, "
              f"tau_max: {np.max(np.abs(tau_total)):.1f}Nm")

print("제어 완료!")

# 시뮬레이션 지속 - 마지막 목표 위치 유지
print("시뮬레이션 유지 중... (Ctrl+C로 종료)")
q_hold = q_traj[-1]  # 마지막 목표 관절 각도 유지
while True:
    q_current = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()
    tau_hold = K_p @ (q_hold - q_current) + K_d @ (0 - dq_current)
    tau_max = np.array([87, 87, 87, 87, 12, 12, 12])
    tau_hold = np.clip(tau_hold, -tau_max, tau_max)
    franka.control_dofs_force(tau_hold, motors_dof)
    scene.step()
