"""
작업 공간 동역학 모듈 (Operational Space Dynamics Module)

이 모듈은 로봇의 작업 공간(태스크 공간) 동역학을 계산하는 독립적인 모듈입니다.
자코비안과 질량 행렬을 입력으로 받아 작업 공간 관성, 의사역행렬, 영공간 투영 행렬을 계산합니다.

This module provides independent operational space dynamics computations.
It takes Jacobian and mass matrix as inputs and computes operational space inertia,
dynamically consistent pseudoinverse, and null space projector.

주요 기능 (Key Features):
- 작업 공간 관성 행렬 계산: Λ = (J·M⁻¹·J^T)⁻¹
- 동역학적으로 일관된 의사역행렬: J̄ = M⁻¹·J^T·Λ
- 영공간 투영 행렬: N = I - J^T·J̄^T
- 수치 안정성 보장 (정규화, 특이점 검출)

모듈 독립성 (Module Independence):
이 모듈은 완전히 독립적이며 다른 제어기 모듈에 의존하지 않습니다.
순수한 수학 연산만을 수행하며, 명확한 입력/출력 인터페이스를 제공합니다.

This module is completely independent and does not depend on any controller modules.
It performs pure mathematical operations with clear input/output interfaces.
"""

import numpy as np
from typing import Tuple


class OperationalSpaceDynamics:
    """
    작업 공간 동역학 계산 클래스
    
    Operational space dynamics computation class.
    
    이 클래스는 로봇의 태스크 공간 동역학을 계산하는 정적 메서드들을 제공합니다.
    Khatib의 작업 공간 제어 이론을 기반으로 구현되었습니다.
    
    This class provides static methods for computing robot task space dynamics.
    Implementation is based on Khatib's operational space control theory.
    
    참고문헌 (Reference):
    Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators"
    IEEE Journal on Robotics and Automation, 3(1), 43-53.
    """
    
    @staticmethod
    def compute_lambda(J: np.ndarray, M: np.ndarray, damping: float = 0.01) -> np.ndarray:
        """
        작업 공간 관성 행렬 계산
        
        Compute operational space inertia matrix: Λ = (J·M⁻¹·J^T)⁻¹
        
        작업 공간 관성 행렬 Λ는 태스크 공간에서의 로봇 관성을 나타냅니다.
        이 행렬은 태스크 공간 힘과 가속도 사이의 관계를 정의합니다: F = Λ·ẍ
        
        The operational space inertia matrix Λ represents robot inertia in task space.
        It defines the relationship between task space force and acceleration: F = Λ·ẍ
        
        수치 안정성 (Numerical Stability):
        특이점 근처에서 수치 불안정성을 방지하기 위해 정규화 항 λI를 추가합니다.
        Λ = (J·M⁻¹·J^T + λI)⁻¹
        
        To prevent numerical instability near singularities, we add regularization λI:
        Λ = (J·M⁻¹·J^T + λI)⁻¹
        
        Args:
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
               6차원 태스크 공간 [위치(3), 회전(3)]을 n차원 관절 공간으로 매핑
               Maps 6D task space [position(3), rotation(3)] to n-DOF joint space
               
            M: 질량 행렬 (n × n) - Mass matrix
               관절 공간에서의 로봇 관성 행렬
               Robot inertia matrix in joint space
               
            damping: 정규화 파라미터 (기본값: 0.01) - Regularization parameter
                    특이점 근처에서 수치 안정성을 위해 추가되는 감쇠
                    Damping added for numerical stability near singularities
        
        Returns:
            Lambda: 작업 공간 관성 행렬 (6 × 6) - Operational space inertia matrix
                   태스크 공간에서의 로봇 관성을 나타내는 대칭 양정부호 행렬
                   Symmetric positive definite matrix representing robot inertia in task space
        
        Raises:
            ValueError: 입력 행렬의 차원이 올바르지 않은 경우
            ValueError: 계산 결과에 NaN이나 Inf가 포함된 경우
        
        수학적 배경 (Mathematical Background):
        관절 공간 동역학: τ = M·q̈ + C·q̇ + g
        태스크 공간 동역학: F = Λ·ẍ + μ + p
        여기서 Λ = (J·M⁻¹·J^T)⁻¹는 태스크 공간 관성입니다.
        
        Joint space dynamics: τ = M·q̈ + C·q̇ + g
        Task space dynamics: F = Λ·ẍ + μ + p
        where Λ = (J·M⁻¹·J^T)⁻¹ is the task space inertia.
        """
        # 입력 검증 (Input validation)
        if J.ndim != 2 or J.shape[0] != 6:
            raise ValueError(f"Jacobian must be (6 × n), got {J.shape}")
        
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Mass matrix must be square (n × n), got {M.shape}")
        
        n_dofs = M.shape[0]
        if J.shape[1] != n_dofs:
            raise ValueError(
                f"Jacobian columns ({J.shape[1]}) must match mass matrix size ({n_dofs})"
            )
        
        # 질량 행렬 역행렬 계산 (Compute mass matrix inverse)
        # M⁻¹을 계산합니다. 질량 행렬은 대칭 양정부호이므로 안정적으로 역행렬을 구할 수 있습니다.
        # Compute M⁻¹. Mass matrix is symmetric positive definite, so inversion is stable.
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix is singular and cannot be inverted")
        
        # J·M⁻¹·J^T 계산 (Compute J·M⁻¹·J^T)
        # 이 행렬은 태스크 공간에서의 관성을 나타내는 중간 결과입니다.
        # This matrix is an intermediate result representing inertia in task space.
        JMinvJT = J @ M_inv @ J.T
        
        # 정규화 추가 (Add regularization)
        # 특이점 근처에서 수치 안정성을 위해 대각 행렬 λI를 추가합니다.
        # Add diagonal matrix λI for numerical stability near singularities.
        JMinvJT_reg = JMinvJT + damping * np.eye(6)
        
        # 작업 공간 관성 행렬 계산 (Compute operational space inertia)
        # Λ = (J·M⁻¹·J^T + λI)⁻¹
        try:
            Lambda = np.linalg.inv(JMinvJT_reg)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Failed to compute operational space inertia. "
                "Jacobian may be near singularity. Try increasing damping parameter."
            )
        
        # 결과 검증 (Validate result)
        if np.any(np.isnan(Lambda)) or np.any(np.isinf(Lambda)):
            raise ValueError("Operational space inertia contains NaN or Inf values")
        
        return Lambda
    
    @staticmethod
    def compute_dynamically_consistent_pseudoinverse(
        J: np.ndarray, M: np.ndarray, Lambda: np.ndarray
    ) -> np.ndarray:
        """
        동역학적으로 일관된 의사역행렬 계산
        
        Compute dynamically consistent pseudoinverse: J̄ = M⁻¹·J^T·Λ
        
        동역학적으로 일관된 의사역행렬 J̄는 태스크 공간 힘을 관절 토크로 변환할 때 사용됩니다.
        일반적인 의사역행렬과 달리, 이 의사역행렬은 로봇의 동역학을 고려하여 계산됩니다.
        
        The dynamically consistent pseudoinverse J̄ is used to transform task space forces to joint torques.
        Unlike the standard pseudoinverse, this considers the robot's dynamics.
        
        특성 (Properties):
        1. J̄·J·M = Λ (동역학적 일관성)
        2. J·J̄ = I (태스크 공간에서 항등 변환)
        3. 영공간 투영: N = I - J^T·J̄^T는 태스크 공간에 영향을 주지 않음
        
        1. J̄·J·M = Λ (dynamical consistency)
        2. J·J̄ = I (identity in task space)
        3. Null space projection: N = I - J^T·J̄^T does not affect task space
        
        Args:
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
            M: 질량 행렬 (n × n) - Mass matrix
            Lambda: 작업 공간 관성 행렬 (6 × 6) - Operational space inertia matrix
        
        Returns:
            J_bar: 동역학적으로 일관된 의사역행렬 (n × 6)
                  Dynamically consistent pseudoinverse
        
        Raises:
            ValueError: 입력 행렬의 차원이 올바르지 않은 경우
            ValueError: 계산 결과에 NaN이나 Inf가 포함된 경우
        
        수학적 배경 (Mathematical Background):
        태스크 공간 힘 F를 관절 토크 τ로 변환:
        τ = J^T·F (일반적인 변환)
        τ = J̄^T·Λ·F (동역학적으로 일관된 변환)
        
        Transform task space force F to joint torque τ:
        τ = J^T·F (standard transformation)
        τ = J̄^T·Λ·F (dynamically consistent transformation)
        """
        # 입력 검증 (Input validation)
        if J.ndim != 2 or J.shape[0] != 6:
            raise ValueError(f"Jacobian must be (6 × n), got {J.shape}")
        
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Mass matrix must be square (n × n), got {M.shape}")
        
        if Lambda.shape != (6, 6):
            raise ValueError(f"Lambda must be (6 × 6), got {Lambda.shape}")
        
        n_dofs = M.shape[0]
        if J.shape[1] != n_dofs:
            raise ValueError(
                f"Jacobian columns ({J.shape[1]}) must match mass matrix size ({n_dofs})"
            )
        
        # 질량 행렬 역행렬 계산 (Compute mass matrix inverse)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix is singular and cannot be inverted")
        
        # 동역학적으로 일관된 의사역행렬 계산 (Compute dynamically consistent pseudoinverse)
        # J̄ = M⁻¹·J^T·Λ
        # 이 의사역행렬은 태스크 공간과 관절 공간 사이의 힘 변환에 사용됩니다.
        # This pseudoinverse is used for force transformation between task and joint spaces.
        J_bar = M_inv @ J.T @ Lambda
        
        # 결과 검증 (Validate result)
        if np.any(np.isnan(J_bar)) or np.any(np.isinf(J_bar)):
            raise ValueError(
                "Dynamically consistent pseudoinverse contains NaN or Inf values"
            )
        
        return J_bar
    
    @staticmethod
    def compute_null_space_projector(J: np.ndarray, J_bar: np.ndarray) -> np.ndarray:
        """
        영공간 투영 행렬 계산
        
        Compute null space projection matrix: N = I - J^T·J̄^T
        
        영공간 투영 행렬 N은 태스크 공간에 영향을 주지 않는 관절 공간 방향을 나타냅니다.
        이 행렬을 사용하여 부차적인 목표(예: 관절 한계 회피)를 달성할 수 있습니다.
        
        The null space projection matrix N represents joint space directions that do not affect task space.
        This matrix enables achieving secondary objectives (e.g., joint limit avoidance).
        
        특성 (Properties):
        1. N·J^T = 0 (영공간 특성: 태스크 공간에 영향 없음)
        2. N^2 = N (멱등성: 투영 행렬의 특성)
        3. τ_null = N·τ는 태스크 공간 동작에 영향을 주지 않음
        
        1. N·J^T = 0 (null space property: no effect on task space)
        2. N^2 = N (idempotency: property of projection matrices)
        3. τ_null = N·τ does not affect task space motion
        
        Args:
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
            J_bar: 동역학적으로 일관된 의사역행렬 (n × 6)
                  Dynamically consistent pseudoinverse
        
        Returns:
            N: 영공간 투영 행렬 (n × n) - Null space projection matrix
               태스크 공간에 영향을 주지 않는 관절 공간 방향을 투영
               Projects onto joint space directions that do not affect task space
        
        Raises:
            ValueError: 입력 행렬의 차원이 올바르지 않은 경우
            ValueError: 계산 결과에 NaN이나 Inf가 포함된 경우
        
        수학적 배경 (Mathematical Background):
        전체 관절 토크를 태스크 공간 토크와 영공간 토크로 분해:
        τ_total = τ_task + τ_null
        τ_total = J^T·F + N·τ_0
        
        여기서 N·τ_0는 태스크 공간 동작에 영향을 주지 않으면서
        부차적인 목표(관절 한계 회피, 특이점 회피 등)를 달성합니다.
        
        Decompose total joint torque into task space and null space components:
        τ_total = τ_task + τ_null
        τ_total = J^T·F + N·τ_0
        
        where N·τ_0 achieves secondary objectives (joint limit avoidance, singularity avoidance)
        without affecting task space motion.
        
        응용 (Applications):
        - 관절 한계 회피 (Joint limit avoidance)
        - 특이점 회피 (Singularity avoidance)
        - 장애물 회피 (Obstacle avoidance)
        - 에너지 최적화 (Energy optimization)
        """
        # 입력 검증 (Input validation)
        if J.ndim != 2 or J.shape[0] != 6:
            raise ValueError(f"Jacobian must be (6 × n), got {J.shape}")
        
        n_dofs = J.shape[1]
        
        if J_bar.shape != (n_dofs, 6):
            raise ValueError(
                f"J_bar must be ({n_dofs} × 6) to match Jacobian, got {J_bar.shape}"
            )
        
        # 항등 행렬 생성 (Create identity matrix)
        I = np.eye(n_dofs)
        
        # 영공간 투영 행렬 계산 (Compute null space projection matrix)
        # N = I - J^T·J̄^T
        # 이 행렬은 태스크 공간의 영공간(null space)으로 투영합니다.
        # This matrix projects onto the null space of the task space.
        N = I - J.T @ J_bar.T
        
        # 결과 검증 (Validate result)
        if np.any(np.isnan(N)) or np.any(np.isinf(N)):
            raise ValueError("Null space projector contains NaN or Inf values")
        
        return N
    
    @staticmethod
    def compute_all(
        J: np.ndarray, M: np.ndarray, damping: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        모든 작업 공간 동역학 행렬을 한 번에 계산
        
        Compute all operational space dynamics matrices at once.
        
        이 메서드는 Λ, J̄, N을 한 번에 계산하여 효율성을 높입니다.
        개별 메서드를 여러 번 호출하는 것보다 계산이 효율적입니다.
        
        This method computes Λ, J̄, N at once for efficiency.
        More efficient than calling individual methods multiple times.
        
        Args:
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
            M: 질량 행렬 (n × n) - Mass matrix
            damping: 정규화 파라미터 (기본값: 0.01) - Regularization parameter
        
        Returns:
            Lambda: 작업 공간 관성 행렬 (6 × 6) - Operational space inertia
            J_bar: 동역학적으로 일관된 의사역행렬 (n × 6) - Dynamically consistent pseudoinverse
            N: 영공간 투영 행렬 (n × n) - Null space projection matrix
        
        Example:
            >>> dynamics = OperationalSpaceDynamics()
            >>> Lambda, J_bar, N = dynamics.compute_all(J, M)
            >>> # 이제 Lambda, J_bar, N을 사용하여 제어 계산 수행
            >>> # Now use Lambda, J_bar, N for control computations
        """
        # 작업 공간 관성 행렬 계산 (Compute operational space inertia)
        Lambda = OperationalSpaceDynamics.compute_lambda(J, M, damping)
        
        # 동역학적으로 일관된 의사역행렬 계산 (Compute dynamically consistent pseudoinverse)
        J_bar = OperationalSpaceDynamics.compute_dynamically_consistent_pseudoinverse(
            J, M, Lambda
        )
        
        # 영공간 투영 행렬 계산 (Compute null space projection matrix)
        N = OperationalSpaceDynamics.compute_null_space_projector(J, J_bar)
        
        return Lambda, J_bar, N
