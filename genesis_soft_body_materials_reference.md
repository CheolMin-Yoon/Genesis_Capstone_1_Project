# Genesis 소프트 바디 재질(Material) 레퍼런스

Genesis 시뮬레이션에서 사용 가능한 모든 소프트 바디 재질 클래스를 정리한 문서입니다.
각 솔버(MPM, FEM, PBD, SPH, SF)별로 재질 파라미터, 구성 모델(constitutive model) 수식, 실제 물질 참고값, 코드 예제를 포함합니다.

---

## 목차

1. [기본 물리량 및 수식](#1-기본-물리량-및-수식)
2. [MPM 재질](#2-mpm-재질)
   - 2.1 MPM.Elastic
   - 2.2 MPM.ElastoPlastic
   - 2.3 MPM.Snow
   - 2.4 MPM.Liquid
   - 2.5 MPM.Sand
   - 2.6 MPM.Muscle
3. [FEM 재질](#3-fem-재질)
   - 3.1 FEM.Elastic
   - 3.2 FEM.Muscle
   - 3.3 FEM.Cloth
4. [PBD 재질](#4-pbd-재질)
   - 4.1 PBD.Cloth
   - 4.2 PBD.Elastic
   - 4.3 PBD.Liquid
   - 4.4 PBD.Particle
5. [SPH 재질](#5-sph-재질)
   - 5.1 SPH.Liquid
6. [SF 재질](#6-sf-재질)
   - 6.1 SF.Smoke
7. [실제 물질별 파라미터 참고표](#7-실제-물질별-파라미터-참고표)
8. [솔버 선택 가이드](#8-솔버-선택-가이드)
9. [코드 예제 모음](#9-코드-예제-모음)

---

## 1. 기본 물리량 및 수식

### 1.1 핵심 재질 파라미터

| 기호 | 이름 | 단위 | 설명 |
|------|------|------|------|
| E | Young's modulus (영률) | Pa (N/m²) | 재질의 강성(stiffness). 클수록 단단함 |
| ν | Poisson's ratio (포아송비) | 무차원 | 횡방향 수축 / 축방향 신장 비율. 0~0.5 범위 |
| ρ | Density (밀도) | kg/m³ | 단위 부피당 질량 |
| λ | Lamé's 1st parameter | Pa | 체적 변형 저항 |
| μ | Lamé's 2nd parameter (전단 계수) | Pa | 전단 변형 저항 |

### 1.2 Lamé 파라미터 변환 공식

Genesis 내부에서 E, ν로부터 Lamé 파라미터를 자동 계산합니다:

```
μ = E / (2(1 + ν))          ... 전단 계수 (shear modulus, G와 동일)
λ = Eν / ((1 + ν)(1 - 2ν))  ... 체적 관련 계수
```

역변환:
```
E = μ(3λ + 2μ) / (λ + μ)
ν = λ / (2(λ + μ))
```

> Genesis에서는 `lam`, `mu`를 직접 지정할 수도 있습니다. 직접 지정하면 E, ν에서 계산된 값을 덮어씁니다.

### 1.3 포아송비(ν)의 물리적 의미

| ν 값 | 의미 | 예시 |
|-------|------|------|
| 0.0 | 횡방향 변형 없음 (코르크) | 코르크 |
| 0.2~0.3 | 일반 금속/세라믹 | 강철(0.3), 콘크리트(0.2) |
| 0.4~0.45 | 부드러운 재질 | 고무(0.45~0.5) |
| 0.5 | 비압축성 (부피 보존) | 이상적 고무, 물 |

> ⚠️ ν = 0.5이면 λ → ∞ (수치 불안정). Genesis에서는 0.49 이하를 권장합니다.

### 1.4 변형 기울기 (Deformation Gradient, F)

연속체 역학에서 변형을 기술하는 핵심 텐서:

```
F = ∂x / ∂X    (현재 위치 x의 기준 위치 X에 대한 기울기)
J = det(F)      (체적 변화율: J > 1 팽창, J < 1 압축, J = 1 부피 보존)
```

SVD 분해: `F = U · S · Vᵀ` (회전 U, V와 주축 신장 S로 분리)

---

## 2. MPM 재질

MPM (Material Point Method)은 입자 기반 연속체 시뮬레이션입니다. 대변형, 파괴, 유체-고체 전환 등에 적합합니다.

### 공통 파라미터 (MPM.Base)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `E` | 1e6 | Young's modulus (Pa) |
| `nu` | 0.2 | Poisson's ratio |
| `rho` | 1000.0 | 밀도 (kg/m³) |
| `lam` | None | Lamé 1st (None이면 E, ν에서 계산) |
| `mu` | None | Lamé 2nd (None이면 E, ν에서 계산) |
| `sampler` | 'pbs' (Linux x86) / 'random' | 입자 샘플러 ('pbs', 'regular', 'random') |

---

### 2.1 MPM.Elastic — 탄성체

가장 기본적인 탄성 재질. 변형 후 원래 형태로 복원됩니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `E` | 3e5 (기본 Base의 1e6이 아님!) | Young's modulus |
| `model` | 'corotation' | 구성 모델: 'corotation' 또는 'neohooken' |

**구성 모델 수식:**

**Corotation 모델** (기본값):
```
σ = 2μ(F - UVᵀ)Fᵀ + λJ(J-1)I
```
- `UVᵀ`: F의 회전 성분 (극분해의 R)
- 회전을 제거한 순수 변형만으로 응력 계산
- 대변형에서도 안정적

**Neo-Hookean 모델** (`model='neohooken'`):
```
σ = μ(FFᵀ) + (λ·ln(J) - μ)I
```
- 비선형 초탄성 모델
- 고무 같은 대변형 재질에 적합

**사용 예시:**
```python
# 부드러운 젤리
soft_jelly = gs.materials.MPM.Elastic(E=1e4, nu=0.3, rho=1000)

# 단단한 고무공
hard_rubber = gs.materials.MPM.Elastic(E=1e6, nu=0.45, rho=1100, model='neohooken')

# 매우 부드러운 스폰지
sponge = gs.materials.MPM.Elastic(E=5e3, nu=0.2, rho=300)
```

---

### 2.2 MPM.ElastoPlastic — 탄소성체

탄성 한계를 넘으면 영구 변형(소성 변형)이 발생하는 재질. 점토, 금속 등에 적합합니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `yield_lower` | 2.5e-2 | 항복 하한 (von Mises 미사용 시) |
| `yield_higher` | 4.5e-3 | 항복 상한 (von Mises 미사용 시) |
| `use_von_mises` | True | von Mises 항복 기준 사용 여부 |
| `von_mises_yield_stress` | 10000.0 | von Mises 항복 응력 (Pa) |

**항복 기준 (Yield Criterion):**

**von Mises 기준** (기본값, `use_von_mises=True`):
```
ε = [ln(S₀₀), ln(S₁₁), ln(S₂₂)]     ... 로그 변형률
ε̂ = ε - (Σεᵢ/3)                       ... 편차 변형률 (deviatoric)
Δγ = ||ε̂|| - σ_yield / (2μ)           ... 항복 판정

if Δγ > 0:  (항복 발생)
    ε ← ε - (Δγ / ||ε̂||) · ε̂         ... 소성 보정
    S_new = diag(exp(ε))
    F_new = U · S_new · Vᵀ
```

**Clamp 기준** (`use_von_mises=False`):
```
S_new[d,d] = clamp(S[d,d], 1 - yield_lower, 1 + yield_higher)
F_new = U · S_new · Vᵀ
```

**사용 예시:**
```python
# 점토 (쉽게 변형, 영구 변형)
clay = gs.materials.MPM.ElastoPlastic(
    E=5e4, nu=0.3, rho=1800,
    use_von_mises=True, von_mises_yield_stress=500
)

# 단단한 금속 (높은 항복 응력)
metal = gs.materials.MPM.ElastoPlastic(
    E=1e7, nu=0.3, rho=7800,
    use_von_mises=True, von_mises_yield_stress=50000
)
```

---

### 2.3 MPM.Snow — 눈

ElastoPlastic의 특수 케이스. 압축되면 단단해지는 경화(hardening) 특성이 있습니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `yield_lower` | 2.5e-2 | 항복 하한 |
| `yield_higher` | 4.5e-3 | 항복 상한 |

> von Mises는 사용하지 않습니다 (내부적으로 `use_von_mises=False`).

**경화 모델 (Hardening):**
```
Jp_new = Jp · Π(S[d,d] / S_new[d,d])   ... 소성 체적비 누적

h = exp(10 · (1 - Jp))                   ... 경화 계수
μ_eff = μ · h                             ... 유효 전단 계수
λ_eff = λ · h                             ... 유효 체적 계수

σ = 2μ_eff(F - R)Fᵀ + λ_eff · J(J-1)I
```

- `Jp < 1` (압축됨) → `h > 1` → 더 단단해짐
- `Jp > 1` (팽창됨) → `h < 1` → 더 부드러워짐
- Disney의 "Frozen" 눈 시뮬레이션 논문 기반

**사용 예시:**
```python
# 신선한 눈
fresh_snow = gs.materials.MPM.Snow(E=1.4e5, nu=0.2, rho=400)

# 다져진 눈 (더 단단)
packed_snow = gs.materials.MPM.Snow(E=5e5, nu=0.2, rho=600)
```

---

### 2.4 MPM.Liquid — 액체

유체 시뮬레이션용 재질. 전단 응력이 없는(또는 점성이 있는) 유체입니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `viscous` | False | 점성 유체 여부 (False면 μ=0) |

**변형 기울기 업데이트:**
```
F_new = J^(1/3) · I    ... 등방성 변형만 유지 (전단 제거)
```

**응력 계산:**
- 비점성 (`viscous=False`): `μ = 0`이므로 압력 항만 남음
  ```
  σ = λ · J(J-1) · I
  ```
- 점성 (`viscous=True`): 전단 응력 포함
  ```
  σ = 2μ(F - UVᵀ)Fᵀ + λ · J(J-1) · I
  ```

**사용 예시:**
```python
# 물
water = gs.materials.MPM.Liquid(E=4e5, nu=0.1, rho=1000, viscous=False)

# 꿀 (점성 유체)
honey = gs.materials.MPM.Liquid(E=4e5, nu=0.1, rho=1400, viscous=True)
```

---

### 2.5 MPM.Sand — 모래

Drucker-Prager 항복 기준을 사용하는 입상 재질(granular material)입니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `friction_angle` | 45 | 내부 마찰각 (도, degrees) |
| `sampler` | 'random' | 입자 샘플러 (기본값이 'random') |

**Drucker-Prager 모델:**
```
sin_φ = sin(friction_angle)
α = √(2/3) · 2sin_φ / (3 - sin_φ)     ... 마찰 계수

ε = [ln|S₀₀|, ln|S₁₁|, ln|S₂₂|]       ... 로그 변형률
tr = Σεᵢ + Jp                            ... 체적 변형률

if tr ≥ 0:  (인장 → 완전 소성)
    Jp_new = tr
    S_new = I
else:       (압축 → 마찰 기반 항복)
    ε̂ = ε - tr/3
    Δγ = ||ε̂|| + (3λ + 2μ)/(2μ) · tr · α
    S_new[d,d] = exp(ε[d] - max(0, Δγ)/||ε̂|| · ε̂[d])
```

**응력 (로그 변형률 기반):**
```
center[i,i] = 2μ · ln(S[i,i]) / S[i,i] + λ · Σln(S[j,j]) / S[i,i]
σ = U · center · Vᵀ · F_newᵀ
```

**사용 예시:**
```python
# 건조한 모래
dry_sand = gs.materials.MPM.Sand(E=1e6, nu=0.2, rho=1600, friction_angle=35)

# 자갈 (높은 마찰각)
gravel = gs.materials.MPM.Sand(E=5e6, nu=0.2, rho=2000, friction_angle=50)
```

---

### 2.6 MPM.Muscle — 근육

Elastic을 상속하며, 근육 수축(actuation) 기능이 추가된 재질입니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model` | 'neohooken' | 구성 모델 (Elastic과 동일) |
| `n_groups` | 1 | 근육 그룹 수 |

**근육 수축 응력:**
```
σ_total = σ_elastic + σ_muscle
σ_muscle = E · actu · F · (m_dir ⊗ m_dir) · Fᵀ
```
- `actu`: 활성화 값 (0~1, 외부에서 제어)
- `m_dir`: 근육 섬유 방향 벡터
- `E` (Young's modulus)가 근육 강성(stiffness)으로 사용됨

**사용 예시:**
```python
# 소프트 로봇 근육
muscle = gs.materials.MPM.Muscle(
    E=5e4, nu=0.45, rho=1000,
    model='neohooken', n_groups=4
)
```

---

## 3. FEM 재질

FEM (Finite Element Method)은 메시 기반 연속체 시뮬레이션입니다. 사면체(tetrahedra) 메시를 사용하며, 정밀한 변형 해석에 적합합니다.

### 공통 파라미터 (FEM.Base)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `E` | 1e6 | Young's modulus (Pa) |
| `nu` | 0.2 | Poisson's ratio |
| `rho` | 1000.0 | 밀도 (kg/m³) |
| `hydroelastic_modulus` | 1e7 | 수탄성 접촉 계수 |
| `friction_mu` | 0.1 | 마찰 계수 |
| `contact_resistance` | None | IPC 접촉 저항 (None이면 글로벌 기본값) |
| `hessian_invariant` | False | Hessian 1회 계산 여부 |

---

### 3.1 FEM.Elastic — 탄성체

FEM 기반 탄성 재질. 3가지 구성 모델을 지원합니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model` | 'linear' | 구성 모델: 'linear', 'stable_neohookean', 'linear_corotated' |

**구성 모델 수식:**

**Linear 모델** (기본값):
```
ε = ½(F + Fᵀ) - I                    ... 소변형 변형률 텐서
σ = 2με + λ·tr(ε)·I                  ... 선형 탄성 응력

에너지: Ψ = μ||ε||² + ½λ·tr(ε)²
```
- 소변형(small deformation) 가정
- Hessian이 상수 → `hessian_invariant = True` (빠름)
- 대변형에서는 부정확

**Stable Neo-Hookean 모델** (`model='stable_neohookean'`):
```
Ic = ||F||²                           ... F의 Frobenius 노름 제곱
α = 1 + 0.75μ/λ

σ = μ(1 - 1/(Ic+1))F + λ(J - α)·∂J/∂F

에너지: Ψ = ½(μ(Ic - 3) + λ̃(J - α)²)
        여기서 λ̃ = λ + μ
```
- 대변형에서도 수치적으로 안정
- 역전(inversion)에서도 안정적
- 고무, 생체 조직 등에 적합

**Linear Corotated 모델** (`model='linear_corotated'`):
```
R = polar_decompose(F)의 회전 부분
F̂ = RᵀF                              ... 회전 제거된 변형
ε = ½(F̂ + F̂ᵀ) - I

σ = 2μ·R·ε + λ·tr(ε)·R

에너지: Ψ = μ||ε||² + ½λ·tr(ε)²
```
- Linear의 대변형 확장
- 회전을 제거하고 선형 모델 적용
- Linear보다 정확하지만 Neo-Hookean보다 빠름

**사용 예시:**
```python
# 부드러운 젤리 (FEM)
fem_jelly = gs.materials.FEM.Elastic(E=1e4, nu=0.3, rho=1000, model='stable_neohookean')

# 단단한 고무 (FEM)
fem_rubber = gs.materials.FEM.Elastic(E=1e6, nu=0.45, rho=1100, model='stable_neohookean')

# 빠른 시뮬레이션용 (소변형)
fem_linear = gs.materials.FEM.Elastic(E=1e6, nu=0.3, rho=1000, model='linear')
```

---

### 3.2 FEM.Muscle — 근육

FEM.Elastic을 상속하며, 근육 수축 기능이 추가됩니다.

**추가 파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model` | 'linear' | 구성 모델 |
| `n_groups` | 1 | 근육 그룹 수 |
| `friction_mu` | 0.1 | 마찰 계수 |

**근육 수축 응력:**
```
l = ||F · m_dir||                     ... 현재 근육 길이
σ_total = σ_elastic + (E · actu / l) · F · (m_dir ⊗ m_dir)
```

- MPM.Muscle과 유사하지만 FEM 프레임워크 내에서 동작
- `actu`: 활성화 신호 (외부 제어)
- `m_dir`: 근육 섬유 방향

**사용 예시:**
```python
fem_muscle = gs.materials.FEM.Muscle(
    E=5e4, nu=0.45, rho=1000,
    model='stable_neohookean', n_groups=2
)
```

---

### 3.3 FEM.Cloth — 천 (IPC 전용)

IPC (Incremental Potential Contact) 솔버와 함께 사용하는 얇은 쉘/멤브레인 재질입니다.

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `E` | 1e4 | Young's modulus (Pa) |
| `nu` | 0.49 | Poisson's ratio (거의 비압축성) |
| `rho` | 200.0 | 밀도 (kg/m³, 일반 직물) |
| `thickness` | 0.001 | 쉘 두께 (m, 1mm) |
| `bending_stiffness` | None | 굽힘 강성 (None이면 굽힘 저항 없음) |
| `model` | 'stable_neohookean' | FEM 모델 (천에서는 미사용) |
| `friction_mu` | 0.1 | 마찰 계수 |

> ⚠️ IPC 솔버 필수, GPU 백엔드 필수, 표면 메시만 허용

**사용 예시:**
```python
cloth = gs.materials.FEM.Cloth(
    E=10e3, nu=0.49, rho=200,
    thickness=0.001, bending_stiffness=10.0
)
```

---

## 4. PBD 재질

PBD (Position Based Dynamics)는 위치 기반 시뮬레이션입니다. 물리적 정확도보다 안정성과 속도를 우선합니다. 실시간 시뮬레이션에 적합합니다.

> PBD는 E, ν 대신 compliance(유연도)와 relaxation(이완) 파라미터를 사용합니다.

### Compliance vs Stiffness

```
compliance = 1 / stiffness    (단위: m/N 또는 rad/N)
```
- compliance = 0 → 완전 강체 (무한 강성)
- compliance 클수록 → 더 유연

### Relaxation

```
relaxation ∈ (0, 1]
```
- relaxation = 1 → 제약 조건 완전 적용
- relaxation 작을수록 → 제약 조건 약화 (더 부드러운 동작)

---

### 4.1 PBD.Cloth — 천

PBD 기반 천 시뮬레이션 재질입니다.

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rho` | 4.0 | 밀도 (kg/m², 2D 면적 기준!) |
| `static_friction` | 0.15 | 정적 마찰 계수 |
| `kinetic_friction` | 0.15 | 동적 마찰 계수 |
| `stretch_compliance` | 1e-7 | 신축 유연도 (m/N). 0에 가까울수록 신축 안됨 |
| `bending_compliance` | 1e-5 | 굽힘 유연도 (rad/N). 0에 가까울수록 뻣뻣 |
| `stretch_relaxation` | 0.3 | 신축 이완 계수 |
| `bending_relaxation` | 0.1 | 굽힘 이완 계수 |
| `air_resistance` | 1e-3 | 공기 저항 (감쇠력) |

> ⚠️ `rho`의 단위가 kg/m² (면적 밀도)입니다. 질량 = rho × 표면적.

**사용 예시:**
```python
# 실크 (가볍고 유연)
silk = gs.materials.PBD.Cloth(
    rho=1.5, stretch_compliance=1e-6, bending_compliance=1e-4,
    stretch_relaxation=0.3, bending_relaxation=0.05
)

# 데님 (무겁고 뻣뻣)
denim = gs.materials.PBD.Cloth(
    rho=8.0, stretch_compliance=1e-8, bending_compliance=1e-6,
    stretch_relaxation=0.5, bending_relaxation=0.3
)

# 기본 천
cloth = gs.materials.PBD.Cloth()
```

---

### 4.2 PBD.Elastic — 3D 탄성체

PBD 기반 3D 볼륨 탄성체입니다. 사면체 메시를 사용합니다.

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rho` | 1000.0 | 밀도 (kg/m³) |
| `static_friction` | 0.15 | 정적 마찰 계수 |
| `kinetic_friction` | 0.15 | 동적 마찰 계수 |
| `stretch_compliance` | 0.0 | 신축 유연도 (m/N) |
| `bending_compliance` | 0.0 | 굽힘 유연도 (rad/N) |
| `volume_compliance` | 0.0 | 체적 유연도 (m³/N). 0이면 비압축성 |
| `stretch_relaxation` | 0.1 | 신축 이완 |
| `bending_relaxation` | 0.1 | 굽힘 이완 |
| `volume_relaxation` | 0.1 | 체적 이완 |

**사용 예시:**
```python
# 부드러운 젤리 (PBD)
pbd_jelly = gs.materials.PBD.Elastic(
    rho=1000, stretch_compliance=1e-4, volume_compliance=1e-5,
    stretch_relaxation=0.2, volume_relaxation=0.2
)

# 단단한 고무 (PBD)
pbd_rubber = gs.materials.PBD.Elastic(
    rho=1100, stretch_compliance=0.0, volume_compliance=0.0,
    stretch_relaxation=0.5, volume_relaxation=0.5
)
```

---

### 4.3 PBD.Liquid — 액체

PBD 기반 유체 시뮬레이션입니다. 밀도 제약 조건으로 비압축성을 유지합니다.

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rho` | 1000.0 | 정지 밀도 (kg/m³) |
| `sampler` | 'pbs' (Linux x86) / 'random' | 입자 샘플러 |
| `density_relaxation` | 0.2 | 밀도 제약 이완 계수. 클수록 빠른 수렴, 불안정 가능 |
| `viscosity_relaxation` | 0.01 | 점성 이완 계수. 클수록 점성 증가 |

**사용 예시:**
```python
# 물 (PBD)
pbd_water = gs.materials.PBD.Liquid(rho=1000, density_relaxation=0.2, viscosity_relaxation=0.01)

# 시럽 (점성 높음)
pbd_syrup = gs.materials.PBD.Liquid(rho=1300, density_relaxation=0.2, viscosity_relaxation=0.1)
```

---

### 4.4 PBD.Particle — 입자 (상호작용 없음)

입자 간 상호작용이 전혀 없는 순수 입자입니다. 외부 힘(중력 등)만 받습니다. 파티클 애니메이션용입니다.

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rho` | 1000.0 | 밀도 (kg/m³) |
| `sampler` | 'pbs' (Linux x86) / 'random' | 입자 샘플러 |

**사용 예시:**
```python
particles = gs.materials.PBD.Particle(rho=500)
```

---

## 5. SPH 재질

SPH (Smoothed Particle Hydrodynamics)는 입자 기반 유체 시뮬레이션입니다. MPM.Liquid보다 물리적으로 정확한 유체 시뮬레이션에 적합합니다.

### 5.1 SPH.Liquid — 액체

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `rho` | 1000.0 | 정지 밀도 (kg/m³) |
| `stiffness` | 50000.0 | 상태 강성 (N/m²). 압축에 대한 압력 증가율 |
| `exponent` | 7.0 | 상태 지수. 클수록 비선형적 압력 응답 |
| `mu` | 0.005 | 점성 계수. 유체 내부 마찰 |
| `gamma` | 0.01 | 표면 장력. 경계에서 입자 응집력 |
| `sampler` | 'regular' | 입자 샘플러 (SPH는 'regular' 권장!) |

**상태 방정식 (Equation of State):**
```
p = stiffness · ((ρ/ρ₀)^exponent - 1)
```
- `ρ₀`: 정지 밀도 (`rho`)
- `ρ`: 현재 밀도
- Tait 방정식 기반

> ⚠️ SPH는 초기 입자 분포에 매우 민감합니다. `sampler='regular'`를 사용해야 수치적으로 안정합니다.
> 'pbs'나 'random' 샘플러는 초기 밀도 변동을 일으켜 시뮬레이션이 발산할 수 있습니다.

**사용 예시:**
```python
# 물 (SPH)
sph_water = gs.materials.SPH.Liquid(
    rho=1000, stiffness=50000, exponent=7,
    mu=0.005, gamma=0.01
)

# 기름 (높은 점성)
sph_oil = gs.materials.SPH.Liquid(
    rho=900, stiffness=50000, exponent=7,
    mu=0.05, gamma=0.02
)
```

---

## 6. SF 재질

SF (Stable Fluids)는 격자 기반 유체 시뮬레이션입니다. 연기, 가스 등 기체 시뮬레이션에 사용됩니다.

### 6.1 SF.Smoke — 연기

파라미터가 없는 최소한의 재질 클래스입니다. 내부적으로 `sampler='regular'`를 사용합니다.

**사용 예시:**
```python
smoke = gs.materials.SF.Smoke()
```

---

## 7. 실제 물질별 파라미터 참고표

### 7.1 MPM/FEM용 (E, ν, ρ 기반)

| 물질 | E (Pa) | ν | ρ (kg/m³) | 권장 재질 클래스 |
|------|--------|---|-----------|-----------------|
| 매우 부드러운 젤리 | 1e3 ~ 1e4 | 0.3~0.45 | 1000 | MPM.Elastic / FEM.Elastic |
| 두부/마시멜로 | 1e4 ~ 5e4 | 0.3~0.4 | 300~500 | MPM.Elastic |
| 고무 (소프트) | 1e5 ~ 1e6 | 0.45~0.49 | 1000~1200 | MPM.Elastic(neohooken) / FEM.Elastic(stable_neohookean) |
| 고무 (하드) | 1e6 ~ 1e7 | 0.45~0.49 | 1100~1300 | MPM.Elastic(neohooken) / FEM.Elastic(stable_neohookean) |
| 점토 | 5e4 ~ 5e5 | 0.3~0.4 | 1500~2000 | MPM.ElastoPlastic (yield_stress=500~5000) |
| 플라스틱 | 1e8 ~ 3e9 | 0.3~0.4 | 900~1400 | MPM.ElastoPlastic (yield_stress=20000~50000) |
| 근육/생체조직 | 1e4 ~ 1e6 | 0.4~0.49 | 1000~1100 | MPM.Muscle / FEM.Muscle |
| 눈 (신선) | 1e4 ~ 2e5 | 0.1~0.3 | 100~400 | MPM.Snow |
| 눈 (다져진) | 2e5 ~ 1e6 | 0.2~0.3 | 400~700 | MPM.Snow |
| 모래 (건조) | 5e5 ~ 5e6 | 0.2~0.3 | 1500~1700 | MPM.Sand (friction_angle=30~40) |
| 자갈 | 5e6 ~ 1e7 | 0.2~0.3 | 1800~2200 | MPM.Sand (friction_angle=40~50) |
| 물 | 2e5 ~ 1e6 | 0.1~0.2 | 1000 | MPM.Liquid / SPH.Liquid / PBD.Liquid |
| 꿀 | 2e5 ~ 1e6 | 0.1~0.2 | 1400 | MPM.Liquid(viscous=True) |

### 7.2 PBD.Cloth용 (compliance 기반)

| 직물 | rho (kg/m²) | stretch_compliance | bending_compliance | 비고 |
|------|-------------|-------------------|-------------------|------|
| 실크 | 0.5~2.0 | 1e-6 ~ 1e-5 | 1e-4 ~ 1e-3 | 가볍고 유연 |
| 면 (cotton) | 2.0~5.0 | 1e-7 ~ 1e-6 | 1e-5 ~ 1e-4 | 중간 |
| 데님 | 5.0~10.0 | 1e-8 ~ 1e-7 | 1e-6 ~ 1e-5 | 무겁고 뻣뻣 |
| 가죽 | 5.0~8.0 | 1e-8 ~ 1e-7 | 1e-7 ~ 1e-6 | 매우 뻣뻣 |
| 나일론 | 1.0~3.0 | 1e-7 ~ 1e-6 | 1e-5 ~ 1e-4 | 탄성 있음 |

### 7.3 SPH.Liquid용

| 유체 | rho (kg/m³) | stiffness | mu (점성) | gamma (표면장력) |
|------|-------------|-----------|-----------|-----------------|
| 물 | 1000 | 50000 | 0.001~0.01 | 0.01~0.05 |
| 기름 | 800~900 | 50000 | 0.03~0.1 | 0.02~0.05 |
| 꿀 | 1400 | 50000 | 0.5~2.0 | 0.05~0.1 |
| 수은 | 13600 | 100000 | 0.001 | 0.4~0.5 |

---

## 8. 솔버 선택 가이드

### 어떤 솔버를 써야 할까?

| 시뮬레이션 대상 | 권장 솔버 | 이유 |
|----------------|-----------|------|
| 부드러운 물체 (젤리, 고무) | MPM 또는 FEM | MPM: 대변형, 파괴 가능. FEM: 정밀도 높음 |
| 점토, 소성 변형 | MPM (ElastoPlastic) | 소성 변형 모델 내장 |
| 눈 | MPM (Snow) | 경화 모델 내장 |
| 모래, 입상체 | MPM (Sand) | Drucker-Prager 모델 내장 |
| 액체 (정확한 물리) | SPH | 상태 방정식 기반, 점성/표면장력 지원 |
| 액체 (빠른 시뮬) | PBD.Liquid 또는 MPM.Liquid | 실시간에 가까운 속도 |
| 천/직물 | PBD.Cloth 또는 FEM.Cloth(IPC) | PBD: 빠름. FEM.Cloth: 정밀 접촉 |
| 연기/가스 | SF.Smoke | 격자 기반 유체 |
| 소프트 로봇 (근육) | MPM.Muscle 또는 FEM.Muscle | 근육 수축 모델 내장 |
| 파티클 효과 | PBD.Particle | 상호작용 없는 순수 입자 |

### MPM vs FEM 비교

| 특성 | MPM | FEM |
|------|-----|-----|
| 메시 | 입자 기반 (메시 불필요) | 사면체 메시 필요 |
| 대변형 | 매우 강함 | 가능 (Neo-Hookean) |
| 파괴/분리 | 자연스러움 | 어려움 |
| 정밀도 | 중간 | 높음 |
| 속도 | 빠름 | 중간 |
| 유체-고체 전환 | 가능 (Snow, Sand) | 불가 |
| 접촉 처리 | 기본 | IPC (정밀) |

### PBD vs 물리 기반 (MPM/FEM) 비교

| 특성 | PBD | MPM/FEM |
|------|-----|---------|
| 물리 정확도 | 낮음 (위치 기반) | 높음 (응력 기반) |
| 안정성 | 매우 높음 | 파라미터 의존 |
| 속도 | 매우 빠름 | 중간~느림 |
| 파라미터 | compliance/relaxation | E, ν, ρ |
| 용도 | 실시간, 게임 | 공학, 연구 |

---

## 9. 코드 예제 모음

### 9.1 MPM 탄성 큐브 (젤리)

```python
import genesis as gs

gs.init(backend=gs.cuda)

scene = gs.Scene(dt=1e-3, gravity=(0, 0, -9.81))
scene.add_entity(gs.morphs.Plane())

# 부드러운 젤리 큐브
jelly = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0, 0, 0.5), size=(0.2, 0.2, 0.2)
    ),
    material=gs.materials.MPM.Elastic(
        E=1e4, nu=0.3, rho=1000, model='corotation'
    ),
)

scene.build()
for i in range(1000):
    scene.step()
```

### 9.2 MPM 점토 (소성 변형)

```python
clay = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0, 0, 1.0), radius=0.1
    ),
    material=gs.materials.MPM.ElastoPlastic(
        E=5e4, nu=0.3, rho=1800,
        use_von_mises=True,
        von_mises_yield_stress=500,  # 낮은 항복 응력 → 쉽게 변형
    ),
)
```

### 9.3 PBD 천 시뮬레이션

```python
cloth = scene.add_entity(
    morph=gs.morphs.Mesh(
        file='cloth.obj',
        pos=(0, 0, 1.0),
    ),
    material=gs.materials.PBD.Cloth(
        rho=4.0,
        stretch_compliance=1e-7,
        bending_compliance=1e-5,
        stretch_relaxation=0.3,
        bending_relaxation=0.1,
    ),
)
```

### 9.4 SPH 물 시뮬레이션

```python
water = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0, 0, 0.5), size=(0.3, 0.3, 0.3)
    ),
    material=gs.materials.SPH.Liquid(
        rho=1000,
        stiffness=50000,
        exponent=7,
        mu=0.005,
        gamma=0.01,
        sampler='regular',  # SPH는 regular 필수!
    ),
)
```

### 9.5 FEM 탄성체 (IPC 접촉)

```python
fem_ball = scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0, 0, 0.5), radius=0.1
    ),
    material=gs.materials.FEM.Elastic(
        E=1e5, nu=0.4, rho=1000,
        model='stable_neohookean',
        friction_mu=0.3,
    ),
)
```

### 9.6 MPM 모래 + 물 커플링

```python
sand = scene.add_entity(
    morph=gs.morphs.Box(pos=(0, 0, 0.3), size=(0.3, 0.3, 0.1)),
    material=gs.materials.MPM.Sand(
        E=1e6, nu=0.2, rho=1600, friction_angle=35
    ),
)

water = scene.add_entity(
    morph=gs.morphs.Box(pos=(0, 0, 0.8), size=(0.2, 0.2, 0.3)),
    material=gs.materials.MPM.Liquid(
        E=4e5, nu=0.1, rho=1000, viscous=False
    ),
)
```

### 9.7 소프트 로봇 근육

```python
# MPM 근육 (worm 예제 참고)
worm_body = scene.add_entity(
    morph=gs.morphs.Mesh(file='worm.obj', pos=(0, 0, 0.1)),
    material=gs.materials.MPM.Muscle(
        E=5e4, nu=0.45, rho=1000,
        model='neohooken', n_groups=4
    ),
)

# 근육 활성화 (시뮬레이션 루프 내)
# worm_body.set_actuation(actu_signal)  # actu_signal: (n_groups,) 배열
```

---

## 부록: 파라미터 튜닝 팁

1. **E (Young's modulus) 조절**: 가장 직관적인 강성 조절. 10배씩 변경하며 테스트
2. **ν (Poisson's ratio)**: 0.3이 안전한 시작점. 고무는 0.45~0.49
3. **ν = 0.5 금지**: 수치적으로 λ → ∞. 최대 0.49까지
4. **MPM sampler**: 'pbs'가 가장 균일하지만 Linux x86 전용. 그 외 'random' 사용
5. **SPH sampler**: 반드시 'regular' 사용. 다른 샘플러는 발산 위험
6. **PBD compliance = 0**: 완전 강체 제약. 약간의 유연성이 필요하면 1e-8 ~ 1e-6
7. **dt (시간 간격)**: 재질이 단단할수록 작은 dt 필요. E > 1e6이면 dt ≤ 1e-3 권장
8. **MPM.ElastoPlastic yield_stress**: 작을수록 쉽게 소성 변형. 점토 ~500, 금속 ~50000
9. **MPM.Snow**: E를 너무 크게 하면 경화 효과가 과도해짐. 1e4~5e5 범위 권장
10. **FEM model 선택**: 소변형 → 'linear' (빠름), 대변형 → 'stable_neohookean' (안정), 중간 → 'linear_corotated'
