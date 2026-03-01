# Genesis FEM 상태 추출 가이드

SAP Coupling 환경에서 FEM entity의 상태 데이터를 추출하는 방법 정리.
QP / MPC 제어기 설계를 위한 입력 벡터 수집 목적.

---

## 환경 전제

```python
gs.init(backend=gs.gpu, precision="64")

scene = gs.Scene(
    fem_options=gs.options.FEMOptions(use_implicit_solver=True),
    coupler_options=gs.options.SAPCouplerOptions(),
    ...
)

franka = scene.add_entity(gs.morphs.MJCF(...), material=gs.materials.Rigid(...))
sphere = scene.add_entity(
    morph=gs.morphs.Sphere(radius=0.02, pos=(0.65, 0.0, 0.02)),
    material=gs.materials.FEM.Elastic(model="linear_corotated", E=1e5, nu=0.4),
)
scene.build()
```

---

## 1. 버텍스 위치 / 속도

```python
state = sphere.get_state()

pos = state.pos  # torch.Tensor (B, n_vertices, 3)
vel = state.vel  # torch.Tensor (B, n_vertices, 3)
```

- `scene.step()` 직후 호출
- `B`: 병렬 환경 수 (단일 환경이면 B=1)

---

## 2. 변형량 (Displacement)

초기 위치 대비 현재 위치 차이.

```python
# sphere.init_positions: (n_vertices, 3) - 브로드캐스트 가능
displacement = state.pos - sphere.init_positions  # (B, n_vertices, 3)

# 스칼라 변형 크기 (per vertex)
deformation_norm = displacement.norm(dim=-1)  # (B, n_vertices)

# 전체 평균 변형량
mean_deformation = displacement.norm(dim=-1).mean(dim=-1)  # (B,)
```

---

## 3. FEM 중심 위치 / 속도

버텍스 평균으로 rigid body처럼 취급할 때.

```python
com_pos = state.pos.mean(dim=1)  # (B, 3)
com_vel = state.vel.mean(dim=1)  # (B, 3)
```

---

## 4. 탄성 복원력 (Elastic Force)

implicit solver 수렴 후 각 버텍스에 누적된 내부 탄성력.

```python
forces = scene.sim.fem_solver.get_forces()  # (B, n_vertices, 3)

# 합력 (전체 물체에 작용하는 net force)
net_force = forces.sum(dim=1)  # (B, 3)
```

- `scene.step()` 이후에 호출해야 의미있는 값
- 외란 추정, 임피던스 제어의 환경 힘 항에 활용 가능

---

## 5. 에너지 / 응력 그래디언트 (Element-level)

```python
fem = scene.sim.fem_solver

# 에너지 밀도 (per element)
energy = fem.elements_el_energy.energy.to_numpy()    # (B, n_elements)

# ∂E/∂F : 1st Piola-Kirchhoff stress에 비례 (per element, 3x3 matrix)
grad_F = fem.elements_el_energy.gradient.to_numpy()  # (B, n_elements, 3, 3)
```

- `gradient`는 `∂E/∂F` 이므로 PK1 stress = `gradient / V` (V: rest volume)
- 직접 Cauchy stress가 필요하면 `F`와 `J`를 이용해 변환 필요

---

## 6. 변형 그래디언트 F (Deformation Gradient)

공식 getter 없음. solver 내부 필드에서 직접 계산.

```python
fem = scene.sim.fem_solver
f_idx = scene.sim.cur_substep_local

# element i, batch b의 F 계산
# elements_i[i].B : 초기 상태의 역변형 그래디언트 (3x3)
# elements_v[f, v, b].pos : 현재 버텍스 위치

# numpy로 꺼내서 수동 계산 예시
el2v   = fem.elements_i.el2v.to_numpy()   # (n_elements, 4)
B_rest = fem.elements_i.B.to_numpy()      # (n_elements, 3, 3)
pos_np = fem.elements_v.pos.to_numpy()[f_idx]  # (n_vertices, B, 3)

# element i, batch 0
i_e = 0
v0, v1, v2, v3 = el2v[i_e]
p = pos_np[:, 0, :]  # (n_vertices, 3)
D = np.column_stack([p[v0]-p[v3], p[v1]-p[v3], p[v2]-p[v3]])  # (3,3)
F = D @ B_rest[i_e]  # (3,3) deformation gradient
J = np.linalg.det(F)  # volume ratio
```

---

## 7. 접촉력 (SAP Contact Force)

SAP coupler가 계산한 contact impulse `gamma`에서 force로 변환.

```python
coupler = scene.coupler  # SAPCoupler

# rigid-FEM 접촉 핸들러
handler = coupler.rigid_fem_contact  # RigidFemTriTetContactHandler

n_pairs = handler.n_contact_pairs[None]  # 현재 활성 접촉 쌍 수

if n_pairs > 0:
    # gamma: contact frame 기준 impulse (B_contact, 3)
    # [0]: normal 방향, [1],[2]: tangential 방향
    gamma = handler.contact_pairs.sap_info.gamma.to_numpy()[:n_pairs]  # (n_pairs, 3)

    contact_force = gamma / scene.sim.dt  # impulse -> force

    normal_force     = contact_force[:, 0]   # (n_pairs,) - 법선 방향
    tangential_force = contact_force[:, 1:]  # (n_pairs, 2) - 마찰력
```

핸들러 종류:

| 핸들러 | 조건 |
|---|---|
| `coupler.rigid_fem_contact` | rigid-FEM 접촉 (`enable_rigid_fem_contact=True`) |
| `coupler.fem_floor_tet_contact` | FEM-바닥 접촉 (tet 방식) |
| `coupler.fem_floor_vert_contact` | FEM-바닥 접촉 (vertex 방식) |

---

## 8. Rigid 로봇 상태 (Franka)

```python
# end-effector 위치 / 자세
ee = franka.get_link("hand")
ee_pos  = ee.get_pos()   # (3,) or (B, 3)
ee_quat = ee.get_quat()  # (4,) or (B, 4)
ee_vel  = ee.get_vel()   # (B, 3)

# 관절 상태
qpos = franka.get_qpos()  # (B, n_dof)
qvel = franka.get_qvel()  # (B, n_dof)

# Jacobian (QP/MPC에서 핵심)
J = franka.get_jacobian(link=ee)  # (B, 6, n_dof)
```

---

## 9. QP / MPC 상태 벡터 구성 예시

```python
scene.step()

# FEM 상태
state      = sphere.get_state()
com_pos    = state.pos.mean(dim=1)          # (B, 3)
com_vel    = state.vel.mean(dim=1)          # (B, 3)
deform     = (state.pos - sphere.init_positions).norm(dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)
fem_forces = scene.sim.fem_solver.get_forces().sum(dim=1)  # (B, 3)

# Rigid 상태
qpos = franka.get_qpos()   # (B, n_dof)
qvel = franka.get_qvel()   # (B, n_dof)
ee   = franka.get_link("hand")
J    = franka.get_jacobian(link=ee)  # (B, 6, n_dof)

# 접촉력
handler = scene.coupler.rigid_fem_contact
n_pairs = handler.n_contact_pairs[None]
if n_pairs > 0:
    gamma = handler.contact_pairs.sap_info.gamma.to_numpy()[:n_pairs]
    f_contact = gamma / scene.sim.dt  # (n_pairs, 3)

# 상태 벡터 조합 (예시)
import torch
x = torch.cat([
    qpos,           # 관절 위치
    qvel,           # 관절 속도
    com_pos,        # FEM 중심 위치
    com_vel,        # FEM 중심 속도
    deform,         # 평균 변형량 (스칼라)
    fem_forces,     # 탄성 복원력 합
], dim=-1)  # (B, n_dof*2 + 3 + 3 + 1 + 3)
```

---

## 10. 주의사항

- `precision="64"` 필수 (SAPCoupler는 32bit 미지원)
- `get_forces()`는 implicit solver 수렴 후 유효 → `scene.step()` 이후 호출
- `elements_el_energy`는 `compute_ele_hessian_gradient` 커널 실행 후 채워짐 (implicit solver 전용)
- FEM 버텍스 수가 많을 경우 QP 차원 폭발 → PCA / modal decomposition으로 축소 권장
- `gamma`는 contact frame 기준이므로 world frame 변환이 필요할 수 있음
