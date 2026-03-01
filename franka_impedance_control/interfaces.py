"""
인터페이스 정의 모듈 (Interfaces Module)

이 모듈은 임피던스 제어 시스템의 모든 컴포넌트 간 인터페이스 계약을 정의합니다.
각 컴포넌트는 이 인터페이스를 통해서만 통신하며, 내부 구현에 의존하지 않습니다.

This module defines clear interface contracts for all components in the impedance control system.
Components communicate only through these interfaces, ensuring loose coupling and modularity.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple
import numpy as np


# ============================================================================
# 데이터 구조 (Data Structures)
# ============================================================================

@dataclass
class RobotState:
    """
    로봇 상태 데이터 구조
    
    Robot state data structure containing all necessary information
    about the robot's current configuration and dynamics.
    """
    q: np.ndarray          # 관절 위치 (n,) - Joint positions
    dq: np.ndarray         # 관절 속도 (n,) - Joint velocities
    tau_measured: np.ndarray  # 측정된 토크 (n,) - Measured torques
    tau_commanded: np.ndarray # 명령된 토크 (n,) - Commanded torques
    
    ee_pos: np.ndarray     # 엔드 이펙터 위치 (3,) - End-effector position
    ee_quat: np.ndarray    # 엔드 이펙터 자세 (4,) - End-effector orientation (quaternion)
    ee_vel: np.ndarray     # 엔드 이펙터 속도 (6,) [linear, angular] - End-effector velocity
    
    gripper_gap: float     # 그리퍼 간격 - Gripper gap distance
    gripper_vel: float     # 그리퍼 속도 - Gripper velocity


@dataclass
class ControlOutput:
    """
    제어 출력 데이터 구조
    
    Control output data structure containing computed torques and forces.
    """
    tau_task: np.ndarray   # 태스크 공간 토크 (n,) - Task space torques
    tau_null: np.ndarray   # 영공간 토크 (n,) - Null space torques
    tau_total: np.ndarray  # 총 토크 (n,) - Total torques
    
    F_task: np.ndarray     # 태스크 공간 힘 (6,) - Task space force
    F_ext_estimated: float # 추정된 외부 접촉력 - Estimated external contact force
    
    task_error: np.ndarray # 태스크 공간 오차 (6,) - Task space error


# ============================================================================
# 컴포넌트 인터페이스 (Component Interfaces)
# ============================================================================

class IOperationalSpaceDynamics(Protocol):
    """
    작업 공간 동역학 계산 인터페이스
    
    Interface for operational space dynamics computations.
    Defines methods for computing operational space inertia, pseudoinverse, and null space projector.
    """
    
    def compute_lambda(self, J: np.ndarray, M: np.ndarray, damping: float = 0.01) -> np.ndarray:
        """
        작업 공간 관성 행렬 계산: Λ = (J·M⁻¹·J^T)⁻¹
        
        Compute operational space inertia matrix.
        
        Args:
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
            M: 질량 행렬 (n × n) - Mass matrix
            damping: 정규화 파라미터 - Regularization parameter
            
        Returns:
            Λ: 작업 공간 관성 행렬 (6 × 6) - Operational space inertia
        """
        ...
    
    def compute_dynamically_consistent_pseudoinverse(
        self, J: np.ndarray, M: np.ndarray, Lambda: np.ndarray
    ) -> np.ndarray:
        """
        동역학적으로 일관된 의사역행렬 계산: J̄ = M⁻¹·J^T·Λ
        
        Compute dynamically consistent pseudoinverse.
        
        Args:
            J: 자코비안 행렬 (6 × n)
            M: 질량 행렬 (n × n)
            Lambda: 작업 공간 관성 행렬 (6 × 6)
            
        Returns:
            J̄: 동역학적으로 일관된 의사역행렬 (n × 6)
        """
        ...
    
    def compute_null_space_projector(self, J: np.ndarray, J_bar: np.ndarray) -> np.ndarray:
        """
        영공간 투영 행렬 계산: N = I - J^T·J̄^T
        
        Compute null space projection matrix.
        
        Args:
            J: 자코비안 행렬 (6 × n)
            J_bar: 동역학적으로 일관된 의사역행렬 (n × 6)
            
        Returns:
            N: 영공간 투영 행렬 (n × n)
        """
        ...


class IImpedanceController(Protocol):
    """
    임피던스 제어기 인터페이스
    
    Interface for task space impedance controller.
    Defines methods for computing task space forces and joint torques.
    """
    
    def compute_task_space_error(
        self, x_desired: np.ndarray, x_current: np.ndarray
    ) -> np.ndarray:
        """
        태스크 공간 오차 계산
        
        Compute task space error (position and orientation).
        
        Args:
            x_desired: 목표 위치/자세 (7,) [pos(3), quat(4)]
            x_current: 현재 위치/자세 (7,)
            
        Returns:
            error: 태스크 공간 오차 (6,) [position(3), orientation(3)]
        """
        ...
    
    def compute_task_space_force(
        self, error: np.ndarray, d_error: np.ndarray, Lambda: np.ndarray
    ) -> np.ndarray:
        """
        태스크 공간 힘 계산: F_task = Λ·(K_p·e + K_d·ė)
        
        Compute task space force using impedance control law.
        
        Args:
            error: 태스크 공간 오차 (6,)
            d_error: 태스크 공간 오차 미분 (6,)
            Lambda: 작업 공간 관성 행렬 (6 × 6)
            
        Returns:
            F_task: 태스크 공간 힘 (6,)
        """
        ...
    
    def compute_joint_torques(
        self, F_task: np.ndarray, J: np.ndarray, tau_null: np.ndarray, N: np.ndarray
    ) -> np.ndarray:
        """
        관절 토크 계산: τ = J^T·F_task + N·τ_null
        
        Compute joint torques from task space forces and null space torques.
        
        Args:
            F_task: 태스크 공간 힘 (6,)
            J: 자코비안 행렬 (6 × n)
            tau_null: 영공간 토크 (n,)
            N: 영공간 투영 행렬 (n × n)
            
        Returns:
            tau: 관절 토크 (n,)
        """
        ...


class INullSpaceController(Protocol):
    """
    영공간 제어기 인터페이스
    
    Interface for null space controller.
    Defines methods for joint limit avoidance in the null space.
    """
    
    def compute_joint_limit_gradient(self, q: np.ndarray) -> np.ndarray:
        """
        관절 한계 회피를 위한 포텐셜 그래디언트 계산
        
        Compute potential gradient for joint limit avoidance.
        
        Args:
            q: 현재 관절 위치 (n,)
            
        Returns:
            grad_phi: 포텐셜 그래디언트 (n,)
        """
        ...
    
    def compute_null_space_torque(self, q: np.ndarray) -> np.ndarray:
        """
        영공간 토크 계산: τ_null = -K_null·∇φ(q)
        
        Compute null space torques for secondary objectives.
        
        Args:
            q: 현재 관절 위치 (n,)
            
        Returns:
            tau_null: 영공간 토크 (n,)
        """
        ...


class IRBFNetwork(Protocol):
    """
    RBF 신경망 인터페이스
    
    Interface for RBF neural network force estimator.
    Defines methods for forward propagation and residual force computation.
    """
    
    def forward(self, gripper_state: np.ndarray) -> float:
        """
        순전파: 접촉력 추정
        
        Forward propagation to estimate external contact force.
        
        Args:
            gripper_state: [d, ḋ] 그리퍼 간격과 속도
            
        Returns:
            F_ext_hat: 추정된 외부 접촉력
        """
        ...
    
    def compute_residual_force(
        self, tau_measured: np.ndarray, tau_commanded: np.ndarray
    ) -> float:
        """
        잔차 힘 계산: F_residual = τ_measured - τ_commanded
        
        Compute residual force from measured and commanded torques.
        
        Args:
            tau_measured: 측정된 토크 (n,)
            tau_commanded: 명령된 토크 (n,)
            
        Returns:
            F_residual: 잔차 힘
        """
        ...


class ILyapunovLearning(Protocol):
    """
    리아푸노프 학습 알고리즘 인터페이스
    
    Interface for Lyapunov-based adaptive learning.
    Defines methods for weight updates with stability guarantees.
    """
    
    def compute_prediction_error(self, F_residual: float, F_estimated: float) -> float:
        """
        예측 오차 계산: e = F_residual - F̂_ext
        
        Compute prediction error.
        
        Args:
            F_residual: 잔차 힘
            F_estimated: 추정된 힘
            
        Returns:
            error: 예측 오차
        """
        ...
    
    def update_weights(self, error: float, phi: np.ndarray) -> np.ndarray:
        """
        리아푸노프 기반 가중치 업데이트: Δw = -γ·φ(x)·e
        
        Update network weights using Lyapunov-based learning law.
        
        Args:
            error: 예측 오차
            phi: 기저 함수 출력
            
        Returns:
            delta_w: 가중치 변화량
        """
        ...
    
    def verify_lyapunov_decrease(self, V_current: float, V_previous: float) -> bool:
        """
        리아푸노프 함수 감소 검증
        
        Verify that Lyapunov function is decreasing (stability check).
        
        Args:
            V_current: 현재 리아푸노프 함수 값
            V_previous: 이전 리아푸노프 함수 값
            
        Returns:
            is_stable: V_current <= V_previous인지 여부
        """
        ...
