"""
영공간 제어기 모듈 (Null Space Controller Module)

이 모듈은 로봇의 영공간(null space)에서 부차적인 목표를 달성하는 독립적인 제어기입니다.
주요 목표는 관절 한계 회피이며, 태스크 공간 제어를 방해하지 않습니다.

This module provides an independent null space controller for achieving secondary objectives.
The primary goal is joint limit avoidance without interfering with task space control.

주요 기능 (Key Features):
- 관절 한계 포텐셜 계산: φ(q)
- 포텐셜 그래디언트 계산: ∇φ(q)
- 영공간 토크 계산: τ_null = -K_null·∇φ(q)
- 선택적 QP 솔버 통합 (OSQP)

모듈 독립성 (Module Independence):
이 모듈은 완전히 독립적이며 ImpedanceController나 RBFNetwork에 의존하지 않습니다.
명확한 입력/출력 인터페이스를 제공하여 격리된 테스트와 재사용이 가능합니다.

This module is completely independent and does not depend on ImpedanceController or RBFNetwork.
It provides clear input/output interfaces for isolated testing and reuse.
"""

import numpy as np
from typing import Optional, Tuple
from config import NullSpaceConfig


class NullSpaceController:
    """
    영공간 제어기 클래스
    
    Null space controller class for secondary objective achievement.
    
    이 클래스는 태스크 공간 제어를 방해하지 않으면서 부차적인 목표를 달성합니다.
    주요 응용은 관절 한계 회피이며, 관절이 한계에 가까워질수록 반발력을 생성합니다.
    
    This class achieves secondary objectives without interfering with task space control.
    Primary application is joint limit avoidance, generating repulsive forces as joints approach limits.
    
    영공간 제어 원리 (Null Space Control Principle):
    전체 토크를 태스크 공간 토크와 영공간 토크로 분해:
    τ_total = τ_task + N·τ_null
    
    여기서 N = I - J^T·J̄^T는 영공간 투영 행렬이며,
    N·τ_null은 태스크 공간 동작에 영향을 주지 않습니다 (N·J^T = 0).
    
    Decompose total torque into task space and null space components:
    τ_total = τ_task + N·τ_null
    
    where N = I - J^T·J̄^T is the null space projection matrix,
    and N·τ_null does not affect task space motion (N·J^T = 0).
    
    관절 한계 회피 (Joint Limit Avoidance):
    관절이 한계에 가까워질수록 증가하는 포텐셜 함수를 정의:
    φ(q) = Σ_i [1/(q_i - q_min_i) + 1/(q_max_i - q_i)]
    
    포텐셜의 그래디언트는 반발력 방향을 나타냅니다:
    ∇φ(q) = -∂φ/∂q
    
    영공간 토크는 이 그래디언트를 따라 관절을 한계로부터 멀어지게 합니다:
    τ_null = -K_null·∇φ(q)
    
    Define potential function that increases as joints approach limits:
    φ(q) = Σ_i [1/(q_i - q_min_i) + 1/(q_max_i - q_i)]
    
    Gradient of potential indicates repulsive force direction:
    ∇φ(q) = -∂φ/∂q
    
    Null space torque pushes joints away from limits:
    τ_null = -K_null·∇φ(q)
    """
    
    def __init__(self, config: NullSpaceConfig):
        """
        영공간 제어기 초기화
        
        Initialize null space controller.
        
        Args:
            config: 영공간 제어 설정 (NullSpaceConfig)
                   - K_null: 영공간 게인
                   - joint_limit_margin: 관절 한계 마진 (5%)
                   - q_min, q_max: 관절 한계
                   - use_qp_solver: QP 솔버 사용 여부
        """
        self.config = config
        self.K_null = config.K_null
        self.q_min = config.q_min
        self.q_max = config.q_max
        self.margin = config.joint_limit_margin
        self.use_qp_solver = config.use_qp_solver
        
        # 관절 범위 계산 (Compute joint ranges)
        self.q_range = self.q_max - self.q_min
        
        # 마진 거리 계산 (Compute margin distances)
        # 관절이 이 거리 이내로 한계에 접근하면 반발력이 활성화됩니다.
        # Repulsive forces activate when joints approach within this distance.
        self.margin_dist = self.margin * self.q_range
        
        # QP 솔버 초기화 (선택적) (Initialize QP solver - optional)
        self.qp_solver = None
        if self.use_qp_solver:
            try:
                import osqp
                self.qp_solver = osqp
            except ImportError:
                print(
                    "Warning: OSQP not installed. QP solver disabled. "
                    "Install with: pip install osqp"
                )
                self.use_qp_solver = False
    
    def compute_joint_limit_potential(self, q: np.ndarray) -> float:
        """
        관절 한계 포텐셜 계산
        
        Compute joint limit potential: φ(q)
        
        포텐셜 함수는 관절이 한계에 가까워질수록 증가합니다.
        마진 거리 이내로 접근하면 포텐셜이 급격히 증가하여 강한 반발력을 생성합니다.
        
        Potential function increases as joints approach limits.
        Within margin distance, potential increases rapidly to generate strong repulsive forces.
        
        포텐셜 정의 (Potential Definition):
        φ(q) = Σ_i [φ_lower_i(q_i) + φ_upper_i(q_i)]
        
        여기서:
        φ_lower_i = 1/(q_i - q_min_i)  if (q_i - q_min_i) < margin_dist_i
        φ_upper_i = 1/(q_max_i - q_i)  if (q_max_i - q_i) < margin_dist_i
        
        where:
        φ_lower_i = 1/(q_i - q_min_i)  if (q_i - q_min_i) < margin_dist_i
        φ_upper_i = 1/(q_max_i - q_i)  if (q_max_i - q_i) < margin_dist_i
        
        Args:
            q: 현재 관절 위치 (n,) - Current joint positions
        
        Returns:
            phi: 포텐셜 값 (스칼라) - Potential value (scalar)
        
        수학적 특성 (Mathematical Properties):
        1. φ(q) ≥ 0 (항상 양수)
        2. q → q_min 또는 q → q_max일 때 φ → ∞
        3. 마진 밖에서는 φ = 0 (반발력 없음)
        
        1. φ(q) ≥ 0 (always positive)
        2. φ → ∞ as q → q_min or q → q_max
        3. φ = 0 outside margin (no repulsive force)
        """
        # 하한 근접도 계산 (Compute distance to lower limits)
        dist_to_min = q - self.q_min
        
        # 상한 근접도 계산 (Compute distance to upper limits)
        dist_to_max = self.q_max - q
        
        # 하한 포텐셜 계산 (Compute lower limit potential)
        # 마진 거리 이내로 접근하면 포텐셜 활성화
        # Activate potential when within margin distance
        lower_potential = np.where(
            dist_to_min < self.margin_dist,
            1.0 / (dist_to_min + 1e-6),  # 1e-6은 0으로 나누기 방지
            0.0
        )
        
        # 상한 포텐셜 계산 (Compute upper limit potential)
        upper_potential = np.where(
            dist_to_max < self.margin_dist,
            1.0 / (dist_to_max + 1e-6),
            0.0
        )
        
        # 전체 포텐셜 = 모든 관절의 포텐셜 합
        # Total potential = sum of potentials for all joints
        phi = np.sum(lower_potential + upper_potential)
        
        return phi
    
    def compute_joint_limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """
        관절 한계 포텐셜 그래디언트 계산
        
        Compute joint limit potential gradient: ∇φ(q)
        
        포텐셜 그래디언트는 포텐셜이 가장 빠르게 증가하는 방향을 나타냅니다.
        음의 그래디언트 방향으로 토크를 가하면 관절이 한계로부터 멀어집니다.
        
        Potential gradient indicates direction of steepest potential increase.
        Applying torque in negative gradient direction pushes joints away from limits.
        
        그래디언트 계산 (Gradient Computation):
        ∇φ(q) = ∂φ/∂q = [∂φ/∂q_1, ∂φ/∂q_2, ..., ∂φ/∂q_n]^T
        
        각 관절에 대해:
        ∂φ/∂q_i = -1/(q_i - q_min_i)² + 1/(q_max_i - q_i)²
        
        For each joint:
        ∂φ/∂q_i = -1/(q_i - q_min_i)² + 1/(q_max_i - q_i)²
        
        Args:
            q: 현재 관절 위치 (n,) - Current joint positions
        
        Returns:
            grad_phi: 포텐셜 그래디언트 (n,) - Potential gradient
                     각 관절에 대한 포텐셜의 편미분
                     Partial derivatives of potential for each joint
        
        물리적 의미 (Physical Meaning):
        - grad_phi[i] > 0: 관절 i가 상한에 가까움 (양의 방향으로 밀림)
        - grad_phi[i] < 0: 관절 i가 하한에 가까움 (음의 방향으로 밀림)
        - grad_phi[i] = 0: 관절 i가 안전 영역에 있음 (반발력 없음)
        
        - grad_phi[i] > 0: joint i near upper limit (pushed in positive direction)
        - grad_phi[i] < 0: joint i near lower limit (pushed in negative direction)
        - grad_phi[i] = 0: joint i in safe region (no repulsive force)
        """
        # 하한 근접도 계산 (Compute distance to lower limits)
        dist_to_min = q - self.q_min
        
        # 상한 근접도 계산 (Compute distance to upper limits)
        dist_to_max = self.q_max - q
        
        # 하한 그래디언트 계산 (Compute lower limit gradient)
        # ∂φ_lower/∂q_i = -1/(q_i - q_min_i)²
        # 마진 거리 이내로 접근하면 그래디언트 활성화
        # Activate gradient when within margin distance
        lower_gradient = np.where(
            dist_to_min < self.margin_dist,
            -1.0 / (dist_to_min**2 + 1e-6),  # 음수: 하한으로부터 멀어지는 방향
            0.0
        )
        
        # 상한 그래디언트 계산 (Compute upper limit gradient)
        # ∂φ_upper/∂q_i = 1/(q_max_i - q_i)²
        upper_gradient = np.where(
            dist_to_max < self.margin_dist,
            1.0 / (dist_to_max**2 + 1e-6),  # 양수: 상한으로부터 멀어지는 방향
            0.0
        )
        
        # 전체 그래디언트 = 하한 그래디언트 + 상한 그래디언트
        # Total gradient = lower gradient + upper gradient
        grad_phi = lower_gradient + upper_gradient
        
        return grad_phi

    
    def compute_null_space_torque(self, q: np.ndarray) -> np.ndarray:
        """
        영공간 토크 계산
        
        Compute null space torque: τ_null = -K_null·∇φ(q)
        
        영공간 토크는 관절 한계 포텐셜의 음의 그래디언트 방향으로 생성됩니다.
        이 토크는 관절을 한계로부터 멀어지게 하며, 영공간 투영 행렬 N과 결합되어
        태스크 공간 제어를 방해하지 않습니다.
        
        Null space torque is generated in the negative gradient direction of joint limit potential.
        This torque pushes joints away from limits and, when combined with null space projector N,
        does not interfere with task space control.
        
        제어 법칙 (Control Law):
        τ_null = -K_null·∇φ(q)
        
        여기서:
        - K_null: 영공간 게인 (양수)
        - ∇φ(q): 포텐셜 그래디언트
        - 음수 부호: 포텐셜을 감소시키는 방향 (한계로부터 멀어짐)
        
        where:
        - K_null: null space gain (positive)
        - ∇φ(q): potential gradient
        - negative sign: direction that decreases potential (away from limits)
        
        Args:
            q: 현재 관절 위치 (n,) - Current joint positions
        
        Returns:
            tau_null: 영공간 토크 (n,) - Null space torques
                     관절 한계 회피를 위한 토크
                     Torques for joint limit avoidance
        
        사용 예시 (Usage Example):
        >>> controller = NullSpaceController(config)
        >>> q = np.array([0.5, 0.3, -0.2, -1.5, 0.8, 2.0, 0.1])
        >>> tau_null = controller.compute_null_space_torque(q)
        >>> # 영공간 투영 행렬 N과 결합하여 사용
        >>> # Combine with null space projector N
        >>> tau_total = tau_task + N @ tau_null
        
        안전성 (Safety):
        영공간 토크는 항상 관절을 안전 영역으로 이동시키는 방향으로 작용합니다.
        K_null 게인을 조정하여 반발력의 강도를 제어할 수 있습니다.
        
        Null space torque always acts to move joints toward safe region.
        Adjust K_null gain to control repulsive force strength.
        """
        # 포텐셜 그래디언트 계산 (Compute potential gradient)
        grad_phi = self.compute_joint_limit_gradient(q)
        
        # 영공간 토크 계산 (Compute null space torque)
        # τ_null = -K_null·∇φ(q)
        # 음의 그래디언트 방향으로 토크를 가하여 관절을 한계로부터 멀어지게 함
        # Apply torque in negative gradient direction to push joints away from limits
        tau_null = -self.K_null * grad_phi
        
        return tau_null
    
    def solve_qp(
        self,
        tau_desired: np.ndarray,
        tau_max: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray
    ) -> np.ndarray:
        """
        QP 솔버를 사용한 제약 조건 처리 (선택적)
        
        Solve constrained optimization using QP solver (optional).
        
        이 메서드는 OSQP 또는 qpOASES를 사용하여 제약 조건을 만족하는 최적 토크를 계산합니다.
        제약 조건에는 토크 한계, 관절 한계, 속도 한계 등이 포함될 수 있습니다.
        
        This method uses OSQP or qpOASES to compute optimal torques satisfying constraints.
        Constraints can include torque limits, joint limits, velocity limits, etc.
        
        최적화 문제 (Optimization Problem):
        minimize    (1/2)·||τ - τ_desired||²
        subject to  -τ_max ≤ τ ≤ τ_max
                    q_min ≤ q + dq·dt ≤ q_max  (선택적)
        
        minimize    (1/2)·||τ - τ_desired||²
        subject to  -τ_max ≤ τ ≤ τ_max
                    q_min ≤ q + dq·dt ≤ q_max  (optional)
        
        Args:
            tau_desired: 원하는 토크 (n,) - Desired torques
            tau_max: 토크 한계 (n,) - Torque limits
            q: 현재 관절 위치 (n,) - Current joint positions
            dq: 현재 관절 속도 (n,) - Current joint velocities
        
        Returns:
            tau_optimal: 최적화된 토크 (n,) - Optimized torques
                        제약 조건을 만족하면서 tau_desired에 가장 가까운 토크
                        Torques closest to tau_desired while satisfying constraints
        
        참고 (Note):
        QP 솔버가 설치되지 않았거나 use_qp_solver=False인 경우,
        간단한 토크 클램핑만 수행합니다.
        
        If QP solver is not installed or use_qp_solver=False,
        performs simple torque clamping only.
        """
        if not self.use_qp_solver or self.qp_solver is None:
            # QP 솔버가 비활성화된 경우 간단한 클램핑 수행
            # Perform simple clamping if QP solver is disabled
            tau_optimal = np.clip(tau_desired, -tau_max, tau_max)
            return tau_optimal
        
        # QP 문제 설정 (Setup QP problem)
        # minimize (1/2)·τ^T·P·τ + q^T·τ
        # subject to l ≤ A·τ ≤ u
        
        n = len(tau_desired)
        
        # 목적 함수: minimize ||τ - τ_desired||²
        # Objective: minimize ||τ - τ_desired||²
        # = (1/2)·τ^T·I·τ - τ_desired^T·τ + const
        P = np.eye(n)  # 2차 항 계수 (Quadratic term coefficient)
        q_vec = -tau_desired  # 1차 항 계수 (Linear term coefficient)
        
        # 제약 조건: -τ_max ≤ τ ≤ τ_max
        # Constraints: -τ_max ≤ τ ≤ τ_max
        A = np.eye(n)
        l = -tau_max  # 하한 (Lower bound)
        u = tau_max   # 상한 (Upper bound)
        
        try:
            # OSQP 문제 생성 및 해결 (Create and solve OSQP problem)
            import scipy.sparse as sp
            
            P_sparse = sp.csc_matrix(P)
            A_sparse = sp.csc_matrix(A)
            
            prob = self.qp_solver.OSQP()
            prob.setup(P_sparse, q_vec, A_sparse, l, u, verbose=False)
            result = prob.solve()
            
            if result.info.status == 'solved':
                tau_optimal = result.x
            else:
                # QP 해결 실패 시 클램핑으로 대체
                # Fall back to clamping if QP solve fails
                print(f"Warning: QP solver failed with status {result.info.status}. Using clamping.")
                tau_optimal = np.clip(tau_desired, -tau_max, tau_max)
        
        except Exception as e:
            # 오류 발생 시 클램핑으로 대체
            # Fall back to clamping on error
            print(f"Warning: QP solver error: {e}. Using clamping.")
            tau_optimal = np.clip(tau_desired, -tau_max, tau_max)
        
        return tau_optimal
    
    def compute_null_space_torque_with_projection(
        self,
        q: np.ndarray,
        N: np.ndarray
    ) -> np.ndarray:
        """
        영공간 투영이 적용된 영공간 토크 계산
        
        Compute null space torque with projection applied.
        
        이 메서드는 영공간 토크를 계산하고 영공간 투영 행렬 N을 적용합니다.
        결과 토크는 태스크 공간 제어를 방해하지 않습니다.
        
        This method computes null space torque and applies null space projector N.
        Resulting torque does not interfere with task space control.
        
        계산 과정 (Computation):
        1. τ_null_raw = -K_null·∇φ(q)  (원시 영공간 토크)
        2. τ_null_projected = N·τ_null_raw  (투영된 영공간 토크)
        
        1. τ_null_raw = -K_null·∇φ(q)  (raw null space torque)
        2. τ_null_projected = N·τ_null_raw  (projected null space torque)
        
        Args:
            q: 현재 관절 위치 (n,) - Current joint positions
            N: 영공간 투영 행렬 (n × n) - Null space projection matrix
        
        Returns:
            tau_null_projected: 투영된 영공간 토크 (n,)
                               Projected null space torques
        
        특성 (Properties):
        - J·M⁻¹·τ_null_projected = 0 (태스크 공간에 영향 없음)
        - τ_null_projected는 관절 한계 회피를 달성
        
        - J·M⁻¹·τ_null_projected = 0 (no effect on task space)
        - τ_null_projected achieves joint limit avoidance
        
        사용 예시 (Usage Example):
        >>> # 작업 공간 동역학 계산
        >>> # Compute operational space dynamics
        >>> Lambda, J_bar, N = dynamics.compute_all(J, M)
        >>> 
        >>> # 투영된 영공간 토크 계산
        >>> # Compute projected null space torque
        >>> tau_null = controller.compute_null_space_torque_with_projection(q, N)
        >>> 
        >>> # 전체 토크 = 태스크 토크 + 투영된 영공간 토크
        >>> # Total torque = task torque + projected null space torque
        >>> tau_total = tau_task + tau_null
        """
        # 원시 영공간 토크 계산 (Compute raw null space torque)
        tau_null_raw = self.compute_null_space_torque(q)
        
        # 영공간 투영 적용 (Apply null space projection)
        # N·τ_null은 태스크 공간에 영향을 주지 않음 (N·J^T = 0)
        # N·τ_null does not affect task space (N·J^T = 0)
        tau_null_projected = N @ tau_null_raw
        
        return tau_null_projected
    
    def get_joint_limit_status(self, q: np.ndarray) -> dict:
        """
        관절 한계 상태 정보 반환
        
        Return joint limit status information.
        
        이 메서드는 각 관절의 한계 근접도와 상태를 반환합니다.
        디버깅과 모니터링에 유용합니다.
        
        This method returns limit proximity and status for each joint.
        Useful for debugging and monitoring.
        
        Args:
            q: 현재 관절 위치 (n,) - Current joint positions
        
        Returns:
            status: 관절 한계 상태 정보 딕셔너리
                   - 'distances_to_min': 하한까지의 거리 (n,)
                   - 'distances_to_max': 상한까지의 거리 (n,)
                   - 'normalized_positions': 정규화된 위치 [0, 1] (n,)
                   - 'in_margin': 마진 내에 있는지 여부 (n,)
                   - 'potential': 전체 포텐셜 값
                   - 'gradient': 포텐셜 그래디언트 (n,)
        """
        # 한계까지의 거리 계산 (Compute distances to limits)
        dist_to_min = q - self.q_min
        dist_to_max = self.q_max - q
        
        # 정규화된 위치 계산 [0, 1] (Compute normalized positions)
        # 0 = 하한, 1 = 상한
        # 0 = lower limit, 1 = upper limit
        normalized_pos = (q - self.q_min) / self.q_range
        
        # 마진 내에 있는지 확인 (Check if within margin)
        in_margin = (dist_to_min < self.margin_dist) | (dist_to_max < self.margin_dist)
        
        # 포텐셜과 그래디언트 계산 (Compute potential and gradient)
        potential = self.compute_joint_limit_potential(q)
        gradient = self.compute_joint_limit_gradient(q)
        
        status = {
            'distances_to_min': dist_to_min,
            'distances_to_max': dist_to_max,
            'normalized_positions': normalized_pos,
            'in_margin': in_margin,
            'potential': potential,
            'gradient': gradient,
            'num_joints_in_margin': np.sum(in_margin)
        }
        
        return status
    
    def __repr__(self) -> str:
        """
        문자열 표현 반환
        
        Return string representation.
        """
        return (
            f"NullSpaceController(\n"
            f"  K_null={self.K_null},\n"
            f"  margin={self.margin * 100:.1f}%,\n"
            f"  n_joints={len(self.q_min)},\n"
            f"  use_qp_solver={self.use_qp_solver}\n"
            f")"
        )
