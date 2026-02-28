# Genesis 제어 공학 레퍼런스

Genesis 예제 전체를 분석하여 제어 공학적으로 활용 가능한 API와 패턴을 정리한 문서.

---

## 1. 제어 모드 (ctrl_mode)

Genesis 내부 제어기는 **한 DOF에 하나의 모드만 활성화**된다. 마지막 호출이 덮어쓴다.

| 모드 | API | 내부 동작 |
|------|-----|-----------|
| POSITION | `control_dofs_position(pos, dofs_idx)` | `tau = kp * (target - pos)` |
| POSITION+VELOCITY | `control_dofs_position_velocity(pos, vel, dofs_idx)` | `tau = kp * (target_pos - pos) + kv * (target_vel - vel)` |
| VELOCITY | `control_dofs_velocity(vel, dofs_idx)` | `tau = kv * (target_vel - vel)` |
| FORCE | `control_dofs_force(force, dofs_idx)` | `tau = force` (직접 토크) |

> 같은 DOF에 position과 force를 동시에 쓸 수 없다. 다른 DOF에는 가능하다.
> 예: arm은 force, gripper는 position → OK

**예제**: `rigid/control_franka.py`

```python
# 처음 250스텝: position 제어
franka.control_dofs_position([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04], motors_dof_idx)
# 1000스텝 이후: force 제어로 전환
franka.control_dofs_force(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), motors_dof_idx)
```

---

## 2. PD 게인 설정

```python
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([ 450,  450,  350,  350,  200,  200,  200,  10,  10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),  # lower
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),  # upper
)
```

> Genesis에 I 항(적분)은 없다. PID가 필요하면 `control_dofs_force`로 직접 구현해야 한다.

---

## 3. Inverse Kinematics

```python
end_effector = franka.get_link("hand")
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)
```

배치 IK (여러 환경 동시):
```python
scene.build(n_envs=64)
qpos = robot.inverse_kinematics(
    link=ee_link,
    pos=target_positions,  # (n_envs, 3)
    quat=target_quats,     # (n_envs, 4)
)
```

**예제**: `rigid/ik_franka.py`, `rigid/ik_franka_batched.py`

---

## 4. Jacobian 기반 제어

### 4-1. Differential IK (resolved-rate)

```python
J = robot.get_jacobian(link=ee_link).cpu().numpy()  # (6, n_dofs)
error = np.concatenate([pos_error, rot_error])       # (6,)
damping = 1e-4
dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), error)
q_new = robot.get_qpos().cpu().numpy() + dq
robot.control_dofs_position(q_new)
```

**예제**: `rigid/diffik_controller.py`

### 4-2. Task-space Impedance Control

```python
J = franka.get_jacobian(end_effector).cpu().numpy()[:, :7]
F = Kp * pos_error + Kd * vel_error  # (6,)
tau = J.T @ F                         # (7,)
franka.control_dofs_force(tau, motors_dof)
```

> `get_jacobian` 사용 시 morph에 `requires_jac_and_IK=True` 필요 (MJCF/URDF는 기본 True)

### 4-3. Orientation Error 계산

```python
ee_quat = ee_link.get_quat().cpu().numpy()
error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), target_quat)
rot_error = gs.quat_to_rotvec(error_quat)  # (3,)
```

---

## 5. 외력/외부 토크 인가

```python
# 링크에 직접 외력 적용
force = [[[fx, fy, fz]]]
scene.sim.rigid_solver.apply_links_external_force(force=force, links_idx=[link_idx])

# 링크에 직접 외부 토크 적용
torque = [[[tx, ty, tz]]]
scene.sim.rigid_solver.apply_links_external_torque(torque=torque, links_idx=[link_idx])
```

**예제**: `rigid/apply_external_force_torque.py`

---

## 6. 궤적 계획 (Motion Planning)

```python
qpos_goal = franka.inverse_kinematics(link=ee, pos=target_pos, quat=target_quat)
qpos_goal[-2:] = 0.04  # 그리퍼 열림

path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=200)

for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
```

> 내부적으로 OMPL 사용. 충돌 회피 경로 생성.

**예제**: `tutorials/08_IK_motion_planning_grasp.py`

---

## 7. 센서

### 7-1. ContactForce (접촉력)

```python
sensor = scene.add_sensor(
    gs.sensors.ContactForce(
        entity_idx=robot.idx,
        link_idx_local=robot.get_link("left_finger").idx_local,
        draw_debug=True,
    )
)
scene.build()
# 매 스텝 후:
force = sensor.get_data().cpu().numpy()  # (3,) xyz
```

> MPM + Rigid 혼합 scene에서는 entity 추가 순서에 주의.
> Rigid entity를 먼저 추가해야 scene idx == rigid solver idx가 일치한다.

센서 없이 접촉력 읽기:
```python
contact_forces = franka.get_links_net_contact_force()  # (n_links, 3)
finger_force = contact_forces[link.idx_local].cpu().numpy()
```

### 7-2. IMU

```python
imu = scene.add_sensor(
    gs.sensors.IMU(
        entity_idx=franka.idx,
        link_idx_local=end_effector.idx_local,
        pos_offset=(0.0, 0.0, 0.15),
        acc_noise=(0.01, 0.01, 0.01),
        gyro_noise=(0.01, 0.01, 0.01),
    )
)
# 매 스텝 후:
data = imu.read()
lin_acc = data.lin_acc  # 선가속도
ang_vel = data.ang_vel  # 각속도
```

### 7-3. Accelerometer (가속도)

```python
links_acc = franka.get_links_acc()  # (n_links, 3)
links_pos = franka.get_links_pos()  # (n_links, 3)
```

**예제**: `rigid/accelerometer_franka.py`, `sensors/imu_franka.py`

---

## 8. 드론 PID 캐스케이드 제어

9개 PID 컨트롤러를 캐스케이드로 연결:

```
위치 PID (x,y,z) → 속도 PID (x,y,z) → 자세 PID (roll,pitch,yaw) → 모터 RPM 믹서
```

```python
drone.set_propellels_rpm(rpms)  # (4,) 각 모터 RPM
```

**예제**: `drone/quadcopter_controller.py`, `drone/fly.py`

---

## 9. RL 기반 제어 (Locomotion / Manipulation)

### Go2 보행 제어
```python
robot.set_dofs_kp([20] * 12, motors_dof_idx)
robot.set_dofs_kv([0.5] * 12, motors_dof_idx)
# RL 정책 출력 → 관절 각도 목표
target = default_pos + action * action_scale
robot.control_dofs_position(target, motors_dof_idx)
```

### Franka 그래스핑 (RL + Vision)
```python
# 스테레오 카메라 → CNN → 행동
left_img = left_cam.render(rgb=True)
right_img = right_cam.render(rgb=True)
# RL 정책 → IK → 관절 제어
```

**예제**: `locomotion/go2_env.py`, `manipulation/grasp_env.py`

---

## 10. 물리 파라미터 런타임 변경 (Domain Randomization)

```python
# 환경별 다른 게인 설정
scene.build(n_envs=2)
franka.set_dofs_kp(kp_array, motors_dof_idx, envs_idx=[0])
franka.set_dofs_kp(kp_array * 0.5, motors_dof_idx, envs_idx=[1])

# 질량, 마찰 등 변경
franka.set_links_inertial_mass(mass, links_idx)
franka.set_friction_ratio(ratio, links_idx)
```

> `batch_dofs_info=True`, `batch_links_info=True` 필요

**예제**: `rigid/set_phys_attr.py`, `rigid/domain_randomization.py`

---

## 11. Soft Body 제어 (Coupling)

### Rigid + MPM (soft cube grasp)
```python
cube = scene.add_entity(
    material=gs.materials.MPM.Elastic(),
    morph=gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.025)),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    material=gs.materials.Rigid(coup_friction=1.0),
)
# 그리퍼 force 제어로 부드럽게 잡기
franka.control_dofs_force(np.array([-1, -1]), fingers_dof)
```

### Rigid + Cloth (PBD)
```python
cloth.fix_particles_to_link(franka, link=ee_link)
```

### Rigid + FEM (IPC Solver)
```python
coupler_options = gs.options.IPCCouplerOptions(
    constraint_strength_translation=10.0,
    constraint_strength_rotation=10.0,
)
```

**예제**: `coupling/grasp_soft_cube.py`, `coupling/cloth_attached_to_rigid.py`, `IPC_Solver/ipc_robot_grasp_cube.py`

---

## 12. 중력 보상

```python
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    material=gs.materials.Rigid(gravity_compensation=1.0),  # 0.0~1.0
)
```

**예제**: `rigid/gravity_compensation.py`

---

## 13. 미분 가능 시뮬레이션 (Differentiable Physics)

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(requires_grad=True),
)
# forward
for _ in range(steps):
    scene.step()
# backward
loss.backward()
scene.backward()
```

> 그래디언트 기반 궤적 최적화, 파라미터 추정에 활용

**예제**: `differentiable_push.py`

---

## 14. 상태 읽기 API 요약

| API | 반환값 | 용도 |
|-----|--------|------|
| `get_qpos()` | 관절 위치 | 상태 피드백 |
| `get_dofs_velocity()` | 관절 속도 | 속도 피드백 |
| `get_dofs_force()` | 관절 힘 | 토크 모니터링 |
| `get_dofs_control_force()` | 제어 출력 토크 | 디버깅 |
| `get_links_pos()` | 링크 위치 | task-space 피드백 |
| `get_links_vel()` | 링크 속도 | task-space 속도 |
| `get_links_acc()` | 링크 가속도 | 가속도 피드백 |
| `get_links_net_contact_force()` | 접촉력 | 힘 피드백 |
| `get_jacobian(link)` | (6, n_dofs) Jacobian | task-space 제어 |
| `link.get_pos()` | 특정 링크 위치 | EE 추적 |
| `link.get_quat()` | 특정 링크 자세 | 자세 제어 |
| `link.get_vel()` | 특정 링크 속도 | 댐핑 |
| `link.get_ang()` | 특정 링크 각속도 | 자세 댐핑 |

---

## 15. 시간 관련 주의사항

```python
scene.t    # 스텝 카운트 (정수)
scene.dt   # 시뮬레이션 dt (초)
# 실제 시간(초) = scene.t * scene.dt
```

---

## 16. 성능 벤치마크

| 로봇 | 환경 수 | FPS |
|------|---------|-----|
| Franka | 30,000 | 43M |
| ANYmal C | 30,000 | 14.4M |

**예제**: `speed_benchmark/franka.py`, `speed_benchmark/anymal_c.py`


---

# 고급 제어 공학 레퍼런스 (Advanced Control Engineering)

Genesis 소스 코드 심층 분석을 통해 정리한 고급 제어 API. Task-space impedance 제어, MPC, computed torque 등에 활용 가능.

---

## 17. 질량 행렬 (Mass Matrix)

```python
# 전체 질량 행렬 M(q) — (n_dofs, n_dofs) 또는 (n_envs, n_dofs, n_dofs)
M = scene.sim.rigid_solver.get_mass_mat()

# LDL 분해 — L (하삼각), D_inv (대각 역행렬)
L, D_inv = scene.sim.rigid_solver.get_mass_mat(decompose=True)
# M = L * diag(1/D_inv) * L^T

# 특정 DOF만 추출
M_arm = scene.sim.rigid_solver.get_mass_mat(dofs_idx=np.arange(7))  # (7, 7)
```

> 질량 행렬은 매 스텝 자동 갱신된다. Computed torque 제어에 직접 사용 가능.

### Computed Torque 제어 패턴

```python
# tau = M(q) * (q_ddot_desired + Kp*e + Kd*e_dot) + h(q, q_dot)
# Genesis에는 h(q, q_dot) (코리올리+중력) 직접 API가 없으므로:
# 방법 1: gravity_compensation=1.0 설정 후 M*q_ddot만 사용
# 방법 2: 제로 가속도에서의 토크를 측정하여 h 추정

M = scene.sim.rigid_solver.get_mass_mat(dofs_idx=motors_dof).cpu().numpy()
q_ddot_des = Kp * pos_err + Kd * vel_err
tau = M @ q_ddot_des  # gravity_compensation=1.0 사용 시
robot.control_dofs_force(tau, motors_dof)
```

---

## 18. Jacobian 상세

```python
# 기본: 링크 원점 기준 Jacobian
J = robot.get_jacobian(link=ee_link)  # (6, n_dofs) 또는 (n_envs, 6, n_dofs)

# 특정 로컬 포인트 기준 Jacobian (예: 툴 끝점)
J = robot.get_jacobian(link=ee_link, local_point=torch.tensor([0.0, 0.0, 0.1]))
```

> `morph.requires_jac_and_IK=True` 필요 (MJCF/URDF는 기본 True)

### Jacobian 활용 패턴

```python
# 1. Task-space impedance: tau = J^T * F
J = robot.get_jacobian(ee_link).cpu().numpy()[:, :n_arm_dofs]
F = Kp * error_6d + Kd * vel_error_6d  # (6,)
tau = J.T @ F

# 2. Operational space inertia: Lambda = (J * M^-1 * J^T)^-1
M = scene.sim.rigid_solver.get_mass_mat(dofs_idx=motors_dof).cpu().numpy()
M_inv = np.linalg.inv(M)
Lambda = np.linalg.inv(J @ M_inv @ J.T)  # (6, 6)

# 3. Dynamically consistent pseudo-inverse: J_bar = M^-1 * J^T * Lambda
J_bar = M_inv @ J.T @ Lambda

# 4. Null-space projection: N = I - J_bar * J
N = np.eye(n_dofs) - J_bar @ J
tau_null = N @ tau_secondary  # 자세 유지 등 보조 목표
```

---

## 19. 외력/외부 토크 API 상세

```python
rigid_solver = scene.sim.rigid_solver

# 외력 적용 — 참조 프레임 선택 가능
rigid_solver.apply_links_external_force(
    force=force,           # (n_links, 3) 또는 (n_envs, n_links, 3)
    links_idx=[link_idx],
    ref="link_origin",     # "link_origin" | "link_com" | "root_com"
    local=False,           # True: 로컬 좌표계, False: 월드 좌표계
)

# 외부 토크 적용
rigid_solver.apply_links_external_torque(
    torque=torque,
    links_idx=[link_idx],
    ref="link_origin",
    local=False,
)
```

| ref | 설명 |
|-----|------|
| `"link_origin"` | 링크 원점 (조인트 위치) |
| `"link_com"` | 링크 질량 중심 |
| `"root_com"` | 전체 kinematic tree의 질량 중심 |

> `local=True`는 `"root_com"`과 함께 사용 불가.
> 외력은 매 스텝 초기화되므로 매 루프마다 다시 적용해야 한다.

### 외력 활용 예시: 외란 주입

```python
# 로봇 EE에 외란 힘 적용 (robustness 테스트)
ee_link_idx = end_effector.idx
disturbance = np.array([[[5.0, 0.0, 0.0]]])  # 5N x방향
rigid_solver.apply_links_external_force(
    force=disturbance,
    links_idx=[ee_link_idx],
    ref="link_origin",
)
scene.step()
```

---

## 20. DOF 물리 속성 API

런타임에 변경 가능한 DOF 속성들:

| Setter | Getter | 설명 |
|--------|--------|------|
| `set_dofs_kp(kp)` | `get_dofs_kp()` | PD 위치 게인 |
| `set_dofs_kv(kv)` | `get_dofs_kv()` | PD 속도 게인 |
| `set_dofs_force_range(lo, hi)` | `get_dofs_force_range()` | 토크 제한 |
| `set_dofs_stiffness(s)` | `get_dofs_stiffness()` | 관절 강성 |
| `set_dofs_armature(a)` | `get_dofs_armature()` | 관절 armature (관성 추가) |
| `set_dofs_damping(d)` | `get_dofs_damping()` | 관절 댐핑 |
| `set_dofs_frictionloss(f)` | `get_dofs_frictionloss()` | 관절 마찰 손실 |
| — | `get_dofs_limit()` | 관절 한계 (lower, upper) |
| — | `get_dofs_invweight()` | DOF 역가중치 |

### 링크 물리 속성

| API | 설명 |
|-----|------|
| `get_links_inertial_mass(links_idx)` | 링크 질량 |
| `set_links_inertial_mass(mass, links_idx)` | 링크 질량 변경 |
| `get_links_invweight(links_idx)` | 링크 역가중치 |
| `get_links_root_COM(links_idx)` | kinematic tree 전체 COM |
| `get_links_mass_shift(links_idx)` | 질량 변화량 |
| `get_links_COM_shift(links_idx)` | COM 변화량 |

> `batch_dofs_info=True`, `batch_links_info=True` 설정 시 환경별 다른 값 가능 (Domain Randomization).

---

## 21. 미분 가능 시뮬레이션 (Differentiable Simulation)

### 기본 설정

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
        requires_grad=True,  # 그래디언트 활성화
    ),
)
scene.build(n_envs=2)  # 배치 환경 필수
```

### Forward + Backward 패턴

```python
# gs.tensor는 torch.Tensor 확장 — requires_grad 지원
v_list = [gs.tensor([[0.0, 1.0, 0.0]], requires_grad=True) for _ in range(horizon)]

scene.reset()
init_pos = gs.tensor([[0.3, 0.1, 0.28]], requires_grad=True)
entity.set_position(init_pos)

# Forward pass
loss = 0.0
for i, v_i in enumerate(v_list):
    entity.set_velocity(vel=v_i)
    scene.step()

    if i == horizon - 1:
        goal = gs.tensor([0.5, 0.8, 0.05])
        state = entity.get_state()
        loss += torch.pow(state.pos - goal, 2).sum()

# Backward pass — 그래디언트가 모든 입력 텐서로 역전파
loss.backward()

for v_i in v_list:
    print(v_i.grad)  # 각 스텝의 속도에 대한 그래디언트
    v_i.zero_grad()
```

> `loss.backward()` 호출 시 Genesis 내부적으로 `_step_grad()`를 역순으로 실행.
> `scene.reset()` 후에만 다시 forward 가능 (`_forward_ready` 플래그).

### MPC에 활용

```python
# Shooting-based MPC: 미분 가능 시뮬레이션으로 최적 제어 입력 탐색
optimizer = torch.optim.Adam(v_list, lr=0.01)

for mpc_iter in range(100):
    scene.reset()
    entity.set_position(current_pos)

    loss = 0.0
    for i, v_i in enumerate(v_list):
        entity.set_velocity(vel=v_i)
        scene.step()
    
    state = entity.get_state()
    loss = torch.pow(state.pos - goal, 2).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 22. 상태 저장/복원 (State Save/Restore)

```python
# 현재 상태 저장
state = scene.get_state()  # SimState 객체

# 상태 복원 (초기 상태로 리셋)
scene.reset()  # 초기 상태로

# 특정 상태로 복원
scene.reset(state=saved_state)

# 특정 환경만 리셋
scene.reset(envs_idx=[0, 2])
```

> `scene.reset(state=...)` 호출 시 해당 state가 새로운 초기 상태로 등록된다.
> MPC rollout에서 현재 상태 저장 → 여러 제어 입력 시뮬레이션 → 최적 입력 선택에 활용.

### Entity 레벨 상태

```python
# entity 상태 (MPM 파티클 등)
entity_state = entity.get_state()
# entity_state.pos — 위치
# entity_state.vel — 속도

# solver 레벨 상태 접근
scene_state = scene.get_state()
mpm_state = scene_state.solvers_state[scene.solvers.index(scene.mpm_solver)]
# mpm_state.pos — 모든 MPM 파티클 위치
# mpm_state.active — 활성 파티클 마스크
```

---

## 23. 중력/코리올리 항 계산 방법

Genesis에는 `C(q, q_dot) * q_dot + g(q)` 를 직접 반환하는 API가 없다. 대안:

### 방법 1: gravity_compensation 사용 (권장)

```python
robot = scene.add_entity(
    gs.morphs.MJCF(file="panda.xml"),
    material=gs.materials.Rigid(gravity_compensation=1.0),
)
# 중력이 자동 보상되므로 tau = M*q_ddot + C*q_dot만 고려
# 실질적으로 impedance 제어 시 중력 항 무시 가능
```

### 방법 2: 제로 가속도 토크 측정

```python
# h(q, q_dot) = 제로 가속도를 유지하는 데 필요한 토크
# 1. 현재 상태 저장
state = scene.get_state()
# 2. 제로 토크 인가 후 한 스텝
robot.control_dofs_force(np.zeros(n_dofs), motors_dof)
scene.step()
# 3. 가속도 측정 → h = -M * q_ddot_measured
q_ddot = (robot.get_dofs_velocity().cpu().numpy() - prev_vel) / dt
M = scene.sim.rigid_solver.get_mass_mat(dofs_idx=motors_dof).cpu().numpy()
h_estimated = -M @ q_ddot
# 4. 상태 복원
scene.reset(state=state)
```

### 방법 3: 미분 가능 시뮬레이션으로 자동 미분

```python
# requires_grad=True 설정 시 그래디언트를 통해 동역학 항 자동 계산
# 별도 구현 불필요 — loss.backward()가 모든 것을 처리
```

---

## 24. Impedance 제어 구현 가이드

### 기본 Task-space Impedance

```python
class TaskSpaceImpedanceController:
    def __init__(self, Kp, Kd):
        self.Kp = np.diag(Kp)  # (6, 6)
        self.Kd = np.diag(Kd)  # (6, 6)

    def compute(self, J, pos_err, rot_err, ee_vel, ee_ang):
        error = np.concatenate([pos_err, rot_err])     # (6,)
        vel   = np.concatenate([ee_vel, ee_ang])        # (6,)
        F = self.Kp @ error - self.Kd @ vel             # (6,)
        tau = J.T @ F                                    # (n_dofs,)
        return tau
```

### Operational Space Control (OSC)

```python
# tau = J^T * Lambda * (Kp*e - Kd*dx) + J^T * Lambda * J_dot * q_dot + h(q, q_dot)
# 간소화 (J_dot 무시, gravity_compensation 사용):
M = scene.sim.rigid_solver.get_mass_mat(dofs_idx=motors_dof).cpu().numpy()
J = robot.get_jacobian(ee_link).cpu().numpy()[:, :n_dofs]
M_inv = np.linalg.inv(M)
Lambda = np.linalg.inv(J @ M_inv @ J.T)  # operational space inertia

F_task = Kp * error - Kd * vel  # (6,)
tau = J.T @ Lambda @ F_task     # dynamically consistent
```

### Null-space 자세 유지

```python
# 주 작업 + 보조 작업 (관절 자세 유지)
J_bar = M_inv @ J.T @ Lambda
N = np.eye(n_dofs) - J_bar @ J

tau_task = J.T @ Lambda @ F_task
tau_posture = Kp_null * (q_desired - q_current) - Kd_null * q_dot
tau = tau_task + N.T @ tau_posture
```

---

## 25. MPC 구현 가이드

### 방법 1: 미분 가능 시뮬레이션 기반 (Shooting MPC)

```python
scene = gs.Scene(
    sim_options=gs.options.SimOptions(requires_grad=True, dt=5e-3, substeps=10),
)
scene.build(n_envs=1)

horizon = 20
u_seq = [gs.tensor([[0.0]*n_dofs], requires_grad=True) for _ in range(horizon)]
optimizer = torch.optim.Adam(u_seq, lr=0.01)

for iteration in range(max_iter):
    scene.reset(state=current_state)
    cost = 0.0
    
    for t in range(horizon):
        robot.control_dofs_force(u_seq[t], motors_dof)
        scene.step()
        # running cost
        state = robot.get_state()
        cost += running_cost(state, target)
    
    # terminal cost
    cost += terminal_cost(robot.get_state(), target)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# 첫 번째 제어 입력 적용
robot.control_dofs_force(u_seq[0].detach(), motors_dof)
```

### 방법 2: 상태 저장/복원 기반 (Sampling MPC)

```python
# CEM (Cross-Entropy Method) 또는 MPPI 스타일
state = scene.get_state()
best_cost = float('inf')
best_u = None

for sample in range(n_samples):
    scene.reset(state=state)
    u_candidate = sample_control_sequence()
    cost = 0.0
    
    for t in range(horizon):
        robot.control_dofs_force(u_candidate[t], motors_dof)
        scene.step()
        cost += evaluate_cost()
    
    if cost < best_cost:
        best_cost = cost
        best_u = u_candidate

# 최적 입력 적용
scene.reset(state=state)
robot.control_dofs_force(best_u[0], motors_dof)
scene.step()
```

---

## 26. 유용한 유틸리티 함수

### Quaternion 연산

```python
gs.transform_quat_by_quat(q1, q2)  # 쿼터니언 곱
gs.inv_quat(q)                       # 쿼터니언 역
gs.quat_to_rotvec(q)                 # 쿼터니언 → 회전 벡터 (3,)
```

### 좌표 변환

```python
link.get_pos()   # 월드 좌표 위치 (3,)
link.get_quat()  # 월드 좌표 자세 (4,) [w, x, y, z]
link.get_vel()   # 월드 좌표 선속도 (3,)
link.get_ang()   # 월드 좌표 각속도 (3,)
```

### 디버그 시각화

```python
scene.draw_debug_line(start, end, radius=0.002, color=(1, 0, 0, 0.5))
scene.draw_debug_sphere(pos, radius=0.01, color=(0, 1, 0, 0.5))
```

---

## 27. 제어 구현 시 주의사항 요약

1. `scene.t`는 스텝 카운트(정수). 실제 시간 = `scene.t * scene.dt`
2. `ctrl_mode`는 DOF별 배타적. 마지막 호출이 덮어씀
3. Genesis에 I 항 없음. PID는 `control_dofs_force`로 직접 구현
4. MPM + Rigid 혼합 시 Rigid entity를 먼저 추가 (센서 인덱스 정합)
5. 외력은 매 스텝 초기화됨 — 매 루프마다 재적용 필요
6. `get_jacobian`은 `requires_jac_and_IK=True` 필요
7. 질량 행렬은 매 스텝 자동 갱신
8. 미분 가능 시뮬레이션은 `n_envs >= 1` 필요
9. `scene.reset()` 후에만 새 forward pass 가능
10. `gravity_compensation=1.0` 사용 시 중력 항 별도 계산 불필요
