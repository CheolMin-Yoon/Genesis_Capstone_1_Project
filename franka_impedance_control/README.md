# Franka Impedance Control with RBF Neural Network

## 프로젝트 개요 (Project Overview)

이 프로젝트는 Genesis 물리 시뮬레이터에서 Franka Emika Panda 로봇을 위한 태스크 공간 임피던스 제어 시스템을 구현합니다. RBF 신경망을 사용한 적응형 접촉력 추정과 리아푸노프 안정성 보장을 포함합니다.

This project implements a task space impedance control system for the Franka Emika Panda robot in the Genesis physics simulator, featuring adaptive contact force estimation using an RBF neural network with Lyapunov stability guarantees.

## 주요 기능 (Key Features)

- **태스크 공간 임피던스 제어 (Task Space Impedance Control)**: 엔드 이펙터의 유연한 동작 제어
- **영공간 제어 (Null Space Control)**: 관절 한계 회피를 위한 부차적 목표 달성
- **RBF 신경망 (RBF Neural Network)**: 그리퍼 접촉력 추정
- **리아푸노프 학습 (Lyapunov Learning)**: 안정성이 보장된 온라인 학습
- **모듈형 아키텍처 (Modular Architecture)**: 완전히 독립적인 컴포넌트 설계

## 프로젝트 구조 (Project Structure)

```
examples/franka_impedance_control/
├── README.md                     # 프로젝트 문서 (This file)
├── interfaces.py                 # 모듈 간 인터페이스 정의 (Interface contracts)
├── config.py                     # 설정 데이터클래스 (Configuration dataclasses)
├── utils.py                      # 수학적 유틸리티 함수 (Mathematical utilities)
├── operational_space.py          # 작업 공간 동역학 (Operational space dynamics)
├── impedance_controller.py       # 임피던스 제어기 (Impedance controller)
├── null_space_controller.py      # 영공간 제어기 (Null space controller)
├── rbf_network.py                # RBF 신경망 (RBF neural network)
├── lyapunov_learning.py          # 리아푸노프 학습 (Lyapunov learning)
├── safety.py                     # 안전 메커니즘 (Safety mechanisms)
└── main.py                       # 메인 실행 스크립트 (Main execution script)
```

## 모듈형 아키텍처 (Modular Architecture)

### 설계 원칙 (Design Principles)

이 프로젝트는 **완전한 모듈 독립성**을 핵심 원칙으로 합니다:

1. **명확한 인터페이스 (Clear Interfaces)**: 모든 컴포넌트는 `interfaces.py`에 정의된 인터페이스를 통해서만 통신
2. **느슨한 결합 (Loose Coupling)**: 각 모듈은 다른 모듈의 내부 구현에 의존하지 않음
3. **독립적 테스트 (Isolated Testing)**: 각 모듈을 완전히 격리된 상태에서 테스트 가능
4. **재사용성 (Reusability)**: 모듈을 다른 프로젝트에서 쉽게 재사용 가능

### 컴포넌트 독립성 (Component Independence)

#### 1. **OperationalSpaceDynamics** (작업 공간 동역학)
- **입력**: Jacobian J, Mass matrix M
- **출력**: Operational space inertia Λ, Pseudoinverse J̄, Null space projector N
- **의존성**: 없음 (순수 수학 연산)

#### 2. **ImpedanceController** (임피던스 제어기)
- **입력**: Robot state, Desired pose
- **출력**: Task space torques τ_task
- **의존성**: IOperationalSpaceDynamics 인터페이스만 사용

#### 3. **NullSpaceController** (영공간 제어기)
- **입력**: Joint positions q, Joint velocities q̇
- **출력**: Null space torques τ_null
- **의존성**: 없음 (독립적 계산)

#### 4. **RBFNetwork** (RBF 신경망)
- **입력**: Gripper state [d, ḋ]
- **출력**: Estimated force F̂_ext
- **의존성**: 없음 (독립적 신경망)

#### 5. **LyapunovLearning** (리아푸노프 학습)
- **입력**: RBFNetwork instance, Prediction error
- **출력**: Weight updates
- **의존성**: IRBFNetwork 인터페이스만 사용

## 설정 (Configuration)

### 기본 설정 사용 (Using Default Configuration)

```python
from config import SystemConfig

# 기본 설정 생성
config = SystemConfig.create_default()
```

### 사용자 정의 설정 (Custom Configuration)

```python
from config import SystemConfig, ImpedanceControlConfig

# 높은 강성 설정 (정밀 제어용)
config = SystemConfig.create_high_stiffness()

# 또는 수동으로 설정
config = SystemConfig()
config.impedance.K_p_translation = 1500.0
config.impedance.K_d_translation = 150.0
config.rbf_network.n_centers = 20
config.rbf_network.learning_rate = 0.005
```

## 제어 주파수 (Control Frequency)

- **물리 시뮬레이션**: 10000 Hz (dt = 0.001s, substeps = 10)
- **제어 루프**: 1000 Hz (매 물리 스텝마다 실행)
- **RBF 학습**: 1000 Hz (제어 루프와 동기화)

## 안전 기능 (Safety Features)

1. **토크 제한 (Torque Limiting)**: Franka Panda의 최대 토크 한계 준수
2. **특이점 처리 (Singularity Handling)**: 자코비안 특이점 근처에서 감쇠 추가
3. **NaN/Inf 검출 (NaN/Inf Detection)**: 수치 오류 자동 감지 및 복구
4. **관절 한계 보호 (Joint Limit Protection)**: 관절 한계 근처에서 반발력 생성
5. **안전 모드 (Safe Mode)**: 오류 누적 시 안전 모드로 자동 전환

## 개발 가이드 (Development Guide)

### 새 컴포넌트 추가 (Adding New Components)

1. `interfaces.py`에 인터페이스 정의 추가
2. 새 모듈 파일 생성 (예: `my_controller.py`)
3. 인터페이스를 구현하는 클래스 작성
4. `config.py`에 설정 클래스 추가 (필요시)
5. 독립적인 단위 테스트 작성

### 테스트 작성 (Writing Tests)

각 모듈은 완전히 격리된 상태에서 테스트되어야 합니다:

```python
# 좋은 예: 독립적 테스트
def test_operational_space_inertia():
    # 테스트 데이터 생성
    J = np.random.randn(6, 7)
    M = np.eye(7)
    
    # 모듈 독립적으로 테스트
    dynamics = OperationalSpaceDynamics()
    Lambda = dynamics.compute_lambda(J, M)
    
    # 검증
    assert Lambda.shape == (6, 6)
```

## 참고 문헌 (References)

- Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators"
- Hogan, N. (1985). "Impedance Control: An Approach to Manipulation"
- Slotine, J.-J. E., & Li, W. (1991). "Applied Nonlinear Control"

## 라이선스 (License)

이 프로젝트는 Genesis 프로젝트의 라이선스를 따릅니다.

This project follows the license of the Genesis project.
