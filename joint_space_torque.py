# joint_space_torque.py - Computed Torque Control (Pinocchio 동역학)
# OMPL → Minimum Jerk → Computed Torque: tau = M*(ddq_d + Kp*e + Kv*de) + nle

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
        dt=0.001
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
    spring_const=1000.0,
)
scene.viewer.add_plugin(mouse_plugin)

scene.build()

# ===== Pinocchio 모델 로드 =====
urdf_path = os.path.join(gs._get_src_dir(), "assets/urdf/panda_bullet/panda.urdf")
pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()
print(f"Pinocchio 로드 완료: nq={pin_model.nq}, nv={pin_model.nv}")

motors_dof = np.arange(7)
end_effector = franka.get_link("panda_link7")

# arm: torque control (kp=0, kv=0), gripper: position control
franka.set_dofs_kp(np.array([0, 0, 0, 0, 0, 0, 0, 100, 100]))
franka.set_dofs_kv(np.array([0, 0, 0, 0, 0, 0, 0, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# ===== Computed Torque 게인 (7x7) =====
Kp = np.diag([600, 600, 600, 600, 250, 150, 50])
Kv = np.diag([50,  50,  50,  50,  30,  25,  10])
tau_max_arr = np.array([87, 87, 87, 87, 12, 12, 12])


# ===== Minimum Jerk 보간 =====
def minimum_jerk_trajectory(q_waypoints, total_time, dt):
    """Joint Space Minimum Jerk (5차 다항식) → q, dq, ddq"""
    N = len(q_waypoints)
    segment_time = total_time / (N - 1)
    t_traj = np.arange(0, total_time, dt)
    M = len(t_traj)

    q_traj   = np.zeros((M, 7))
    dq_traj  = np.zeros((M, 7))
    ddq_traj = np.zeros((M, 7))

    for seg in range(N - 1):
        q_s, q_e = q_waypoints[seg], q_waypoints[seg + 1]
        t0 = seg * segment_time
        mask = (t_traj >= t0) & (t_traj < (seg + 1) * segment_time)
        t_seg = t_traj[mask]
        if len(t_seg) == 0:
            continue

        tau = (t_seg - t0) / segment_time
        s   = 10*tau**3 - 15*tau**4 + 6*tau**5
        ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / segment_time
        dds = (60*tau - 180*tau**2 + 120*tau**3) / segment_time**2

        for j in range(7):
            d = q_e[j] - q_s[j]
            q_traj[mask, j]   = q_s[j] + d * s
            dq_traj[mask, j]  = d * ds
            ddq_traj[mask, j] = d * dds

    q_traj[-1]   = q_waypoints[-1]
    dq_traj[-1]  = 0
    ddq_traj[-1] = 0
    return q_traj, dq_traj, ddq_traj


# ===== OMPL 경로 계획 =====
qpos_goal = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
qpos_goal[-2:] = 0.04

path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=200)
scene.draw_debug_path(path, franka)

# ===== Minimum Jerk 보간 =====
path_np = path[:, :7].cpu().numpy()
total_time = 5.0
dt = 0.01

print("Minimum Jerk 궤적 생성 중...")
q_traj, dq_traj, ddq_traj = minimum_jerk_trajectory(path_np, total_time, dt)
print(f"보간 완료: {len(q_traj)}개 포인트 ({total_time}초)")

# ===== Computed Torque 제어 루프 =====
print("Computed Torque 제어 시작...")
log_q_err = []
log_q_err_each = []
log_dq_err = []
log_dq_err_each = []

for i in range(len(q_traj)):
    q_desired   = q_traj[i]
    dq_desired  = dq_traj[i]
    ddq_desired = ddq_traj[i]

    q_current  = franka.get_dofs_position(motors_dof).cpu().numpy()
    dq_current = franka.get_dofs_velocity(motors_dof).cpu().numpy()

    # Pinocchio 동역학
    q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_current
    dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_current
    pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)

    M_q = pin_data.M[:7, :7]
    nle = pin_data.nle[:7]  # C(q,dq)*dq + g(q)

    # Computed Torque: tau = M*(ddq_d + Kp*e + Kv*de) + nle
    e_q  = q_desired  - q_current
    de_q = dq_desired - dq_current
    ddq_cmd = ddq_desired + Kp @ e_q + Kv @ de_q
    tau = M_q @ ddq_cmd + nle

    tau = np.clip(tau, -tau_max_arr, tau_max_arr)
    franka.control_dofs_force(tau, motors_dof)
    scene.step()

    log_q_err.append(np.linalg.norm(e_q))
    log_q_err_each.append(e_q.copy())
    log_dq_err.append(np.linalg.norm(de_q))
    log_dq_err_each.append(de_q.copy())

    if i % 30 == 0:
        print(f"[{i:5d}/{len(q_traj)}] q_err: {np.linalg.norm(e_q):.4f}rad, "
              f"tau_max: {np.max(np.abs(tau)):.1f}Nm")

print("제어 완료!")

# ===== 마지막 위치 유지 =====
print("시뮬레이션 유지 중... (Ctrl+C로 종료)")
q_hold = q_traj[-1]
try:
    while True:
        q_c  = franka.get_dofs_position(motors_dof).cpu().numpy()
        dq_c = franka.get_dofs_velocity(motors_dof).cpu().numpy()

        q_pin  = np.zeros(pin_model.nq);  q_pin[:7]  = q_c
        dq_pin = np.zeros(pin_model.nv);  dq_pin[:7] = dq_c
        pin.computeAllTerms(pin_model, pin_data, q_pin, dq_pin)
        M_q = pin_data.M[:7, :7]
        nle = pin_data.nle[:7]

        ddq_cmd = Kp @ (q_hold - q_c) + Kv @ (0 - dq_c)
        tau = np.clip(M_q @ ddq_cmd + nle, -tau_max_arr, tau_max_arr)
        franka.control_dofs_force(tau, motors_dof)
        scene.step()
except KeyboardInterrupt:
    pass

# ===== 에러 플롯 =====
N = len(log_q_err)
t = np.arange(N) / 60.0
log_q_err_each = np.array(log_q_err_each)
log_dq_err_each = np.array(log_dq_err_each)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 관절 오차 노름
axes[0, 0].plot(t, log_q_err, 'b-')
axes[0, 0].set_ylabel('Joint Error Norm (rad)')
axes[0, 0].set_title('Joint Position Error Norm')
axes[0, 0].grid(True)

# 각 관절별 위치 오차
for j in range(7):
    axes[0, 1].plot(t, log_q_err_each[:, j], label=f'j{j+1}')
axes[0, 1].set_ylabel('Joint Error (rad)')
axes[0, 1].set_title('Joint Position Error (per joint)')
axes[0, 1].legend(fontsize=7)
axes[0, 1].grid(True)

# 관절 속도 오차 노름
axes[1, 0].plot(t, log_dq_err, 'r-')
axes[1, 0].set_ylabel('Velocity Error Norm (rad/s)')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Joint Velocity Error Norm')
axes[1, 0].grid(True)

# 각 관절별 속도 오차
for j in range(7):
    axes[1, 1].plot(t, log_dq_err_each[:, j], label=f'j{j+1}')
axes[1, 1].set_ylabel('Velocity Error (rad/s)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Joint Velocity Error (per joint)')
axes[1, 1].legend(fontsize=7)
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('joint_space_torque_error.png', dpi=150)
plt.show()
print("플롯 저장: joint_space_torque_error.png")
