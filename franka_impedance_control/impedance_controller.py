"""
임피던스 제어기 모듈 (Impedance Controller Module)

이 모듈은 태스크 공간 임피던스 제어를 구현하는 독립적인 모듈입니다.
로봇의 엔드 이펙터를 태스크 공간에서 제어하며, 환경과의 유연한 상호작용을 가능하게 합니다.

This module implements task space impedance control as an independent module.
It controls the robot's end-effector in task space, enabling compliant interaction with the environment.

주요 기능 (Key Features):
- 태스크 공간 오차 계산 (위치 및 자세)
- 임피던스 제어 법칙: F_task = Λ·(K_p·e + K_d·ė)
- 관절 토크 변환: τ = J^T·F_task + N·τ_null
- 수치 안정성 보장

모듈 독립성 (Module Independence):
이 모듈은 NullSpaceController나 RBFNetwork에 의존하지 않습니다.
명확한 입력/출력 인터페이스를 통해 독립적으로 동작하며 테스트 가능합니다.

This module does not depend on NullSpaceController or RBFNetwork.
It operates independently through clear input/output interfaces and is testable in isolation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import ImpedanceControlConfig
from .operational_space import OperationalSpaceDynamics
from .utils import quaternion_error, validate_array


class ImpedanceController:
    """
    태스크 공간 임피던스 제어기
    
    Task space impedance controller.
    
    이 클래스는 로봇의 엔드 이펙터를 태스크 공간에서 임피던스 제어합니다.
    임피던스 제어는 로봇이 환경과 상호작용할 때 유연한 동작을 가능하게 하며,
    강성(stiffness)과 감쇠(damping)를 조절하여 원하는 동적 특성을 구현합니다.
    
    This class implements task space impedance control for the robot's end-effector.
    Impedance control enables compliant robot behavior during environmental interaction,
    achieving desired dynamic characteristics by adjusting stiffness and damping.
    
    제어 법칙 (Control Law):
    F_task = Λ·(K_p·e + K_d·ė) + μ
    τ = J^T·F_task + N·τ_null
    
    여기서:
    - Λ: 작업 공간 관성 행렬 (Operational space inertia)
    - K_p: 강성 행렬 (Stiffness matrix)
    - K_d: 감쇠 행렬 (Damping matrix)
    - e: 태스크 공간 오차 (Task space error)
    - ė: 태스크 공간 오차 미분 (Task space error derivative)
    - μ: 피드포워드 힘 (Feedforward force)
    - J: 자코비안 행렬 (Jacobian matrix)
    - N: 영공간 투영 행렬 (Null space projector)
    - τ_null: 영공간 토크 (Null space torque)
    
    참고문헌 (References):
    - Hogan, N. (1985). "Impedance Control: An Approach to Manipulation"
    - Khatib, O. (1987). "A unified approach for motion and force control"
    """
    
    def __init__(
        self,
        robot,
        ee_link: str,
        config: Optional[ImpedanceControlConfig] = None
    ):
        """
        임피던스 제어기 초기화
        
        Initialize impedance controller.
        
        Args:
            robot: Genesis 로봇 엔티티 - Genesis robot entity
                  로봇의 상태를 읽고 제어 명령을 보내는 데 사용됩니다.
                  Used to read robot state and send control commands.
                  
            ee_link: 엔드 이펙터 링크 이름 - End-effector link name
                    제어할 엔드 이펙터의 링크 이름 (예: "panda_hand")
                    Name of the end-effector link to control (e.g., "panda_hand")
                    
            config: 제어 설정 - Control configuration
                   임피던스 제어 파라미터 (K_p, K_d, 주파수 등)
                   Impedance control parameters (K_p, K_d, frequencies, etc.)
                   None인 경우 기본 설정 사용
                   Uses default configuration if None
        """
        self.robot = robot
        self.ee_link = ee_link
        self.config = config if config is not None else ImpedanceControlConfig()
        
        # 작업 공간 동역학 계산기 (Operational space dynamics computer)
        self.op_space = OperationalSpaceDynamics()
        
        # 강성 및 감쇠 행렬 생성 (Generate stiffness and damping matrices)
        self.K_p = self.config.get_K_p_matrix()  # (6 × 6)
        self.K_d = self.config.get_K_d_matrix()  # (6 × 6)
        
        # 이전 오차 저장 (오차 미분 계산용)
        # Store previous error (for error derivative computation)
        self.prev_error: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        
        # 제어 주기 계산 (Compute control period)
        self.dt = 1.0 / self.config.control_freq  # seconds
    
    def compute_task_space_error(
        self,
        x_desired: np.ndarray,
        x_current: np.ndarray
    ) -> np.ndarray:
        """
        태스크 공간 오차 계산
        
        Compute task space error (position and orientation).
        
        태스크 공간 오차는 목표 위치/자세와 현재 위치/자세의 차이입니다.
        위치 오차는 단순 벡터 차이로 계산하고, 자세 오차는 쿼터니언 오차로 계산합니다.
        
        Task space error is the difference between desired and current pose.
        Position error is computed as simple vector difference,
        orientation error is computed as quaternion error.
        
        Args:
            x_desired: 목표 위치/자세 (7,) [pos(3), quat(4)]
                      Desired pose [position(3), quaternion(4)]
                      쿼터니언 형식: [w, x, y, z]
                      Quaternion format: [w, x, y, z]
                      
            x_current: 현재 위치/자세 (7,) [pos(3), quat(4)]
                      Current pose [position(3), quaternion(4)]
        
        Returns:
            error: 태스크 공간 오차 (6,) [position_error(3), orientation_error(3)]
                  Task space error [position error(3), orientation error(3)]
                  
                  위치 오차: e_pos = pos_desired - pos_current
                  자세 오차: e_ori = 2·sign(w)·[x, y, z] (쿼터니언 오차를 각속도 벡터로 변환)
                  
                  Position error: e_pos = pos_desired - pos_current
                  Orientation error: e_ori = 2·sign(w)·[x, y, z] (quaternion error as angular velocity)
        
        Raises:
            ValueError: 입력 배열의 크기가 올바르지 않은 경우
        """
        # 입력 검증 (Input validation)
        if x_desired.shape != (7,):
            raise ValueError(f"x_desired must be (7,), got {x_desired.shape}")
        if x_current.shape != (7,):
            raise ValueError(f"x_current must be (7,), got {x_current.shape}")
        
        # 위치 오차 계산 (Compute position error)
        # e_pos = pos_desired - pos_current
        pos_desired = x_desired[:3]
        pos_current = x_current[:3]
        pos_error = pos_desired - pos_current
        
        # 자세 오차 계산 (Compute orientation error)
        # 쿼터니언 오차를 각속도 벡터로 변환
        # Convert quaternion error to angular velocity vector
        quat_desired = x_desired[3:7]
        quat_current = x_current[3:7]
        ori_error = quaternion_error(quat_desired, quat_current)
        
        # 전체 태스크 공간 오차 결합 (Combine into full task space error)
        error = np.concatenate([pos_error, ori_error])
        
        # 결과 검증 (Validate result)
        validate_array(error, "task_space_error")
        
        return error
    
    def compute_task_space_error_derivative(
        self,
        error: np.ndarray,
        ee_vel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        태스크 공간 오차 미분 계산
        
        Compute task space error derivative.
        
        오차 미분은 두 가지 방법으로 계산할 수 있습니다:
        1. 엔드 이펙터 속도 사용: ė = -ẋ_current (목표가 정지 상태인 경우)
        2. 수치 미분: ė ≈ (e_current - e_previous) / dt
        
        Error derivative can be computed in two ways:
        1. Using end-effector velocity: ė = -ẋ_current (when target is stationary)
        2. Numerical differentiation: ė ≈ (e_current - e_previous) / dt
        
        Args:
            error: 현재 태스크 공간 오차 (6,) - Current task space error
            ee_vel: 엔드 이펙터 속도 (6,) [linear(3), angular(3)] (선택적)
                   End-effector velocity (optional)
                   None인 경우 수치 미분 사용
                   Uses numerical differentiation if None
        
        Returns:
            d_error: 태스크 공간 오차 미분 (6,)
                    Task space error derivative
        """
        # 방법 1: 엔드 이펙터 속도 사용 (Method 1: Use end-effector velocity)
        # 목표가 정지 상태라고 가정: ė = -ẋ_current
        # Assume target is stationary: ė = -ẋ_current
        if ee_vel is not None:
            d_error = -ee_vel
            validate_array(d_error, "task_space_error_derivative")
            return d_error
        
        # 방법 2: 수치 미분 (Method 2: Numerical differentiation)
        # ė ≈ (e_current - e_previous) / dt
        if self.prev_error is not None:
            d_error = (error - self.prev_error) / self.dt
        else:
            # 첫 번째 스텝: 오차 미분을 0으로 초기화
            # First step: Initialize error derivative to zero
            d_error = np.zeros(6)
        
        # 현재 오차 저장 (Store current error for next iteration)
        self.prev_error = error.copy()
        
        validate_array(d_error, "task_space_error_derivative")
        return d_error
    
    def compute_task_space_force(
        self,
        error: np.ndarray,
        d_error: np.ndarray,
        Lambda: np.ndarray,
        feedforward_force: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        태스크 공간 힘 계산
        
        Compute task space force using impedance control law.
        
        임피던스 제어 법칙:
        F_task = Λ·(K_p·e + K_d·ė) + μ
        
        Impedance control law:
        F_task = Λ·(K_p·e + K_d·ė) + μ
        
        여기서:
        - Λ: 작업 공간 관성 행렬 (로봇의 태스크 공간 관성)
        - K_p: 강성 행렬 (위치 오차에 대한 복원력)
        - K_d: 감쇠 행렬 (속도 오차에 대한 감쇠력)
        - e: 태스크 공간 오차
        - ė: 태스크 공간 오차 미분
        - μ: 피드포워드 힘 (중력 보상, 외부 힘 등)
        
        where:
        - Λ: Operational space inertia (robot inertia in task space)
        - K_p: Stiffness matrix (restoring force for position error)
        - K_d: Damping matrix (damping force for velocity error)
        - e: Task space error
        - ė: Task space error derivative
        - μ: Feedforward force (gravity compensation, external forces, etc.)
        
        물리적 의미 (Physical Interpretation):
        이 제어 법칙은 로봇 엔드 이펙터를 가상의 스프링-댐퍼 시스템으로 만듭니다.
        K_p는 스프링 강성, K_d는 댐퍼 계수에 해당합니다.
        
        This control law makes the robot end-effector behave like a virtual spring-damper system.
        K_p corresponds to spring stiffness, K_d corresponds to damper coefficient.
        
        Args:
            error: 태스크 공간 오차 (6,) - Task space error
            d_error: 태스크 공간 오차 미분 (6,) - Task space error derivative
            Lambda: 작업 공간 관성 행렬 (6 × 6) - Operational space inertia
            feedforward_force: 피드포워드 힘 (6,) (선택적) - Feedforward force (optional)
        
        Returns:
            F_task: 태스크 공간 힘 (6,) [force(3), torque(3)]
                   Task space force [force(3), torque(3)]
        
        Raises:
            ValueError: 입력 배열의 크기가 올바르지 않은 경우
        """
        # 입력 검증 (Input validation)
        if error.shape != (6,):
            raise ValueError(f"error must be (6,), got {error.shape}")
        if d_error.shape != (6,):
            raise ValueError(f"d_error must be (6,), got {d_error.shape}")
        if Lambda.shape != (6, 6):
            raise ValueError(f"Lambda must be (6 × 6), got {Lambda.shape}")
        
        # PD 제어 힘 계산 (Compute PD control force)
        # F_pd = K_p·e + K_d·ė
        F_pd = self.K_p @ error + self.K_d @ d_error
        
        # 작업 공간 관성으로 스케일링 (Scale by operational space inertia)
        # F_task = Λ·F_pd
        # 이는 로봇의 동역학을 고려하여 힘을 조정합니다.
        # This adjusts the force considering robot dynamics.
        F_task = Lambda @ F_pd
        
        # 피드포워드 힘 추가 (Add feedforward force)
        if feedforward_force is not None:
            if feedforward_force.shape != (6,):
                raise ValueError(
                    f"feedforward_force must be (6,), got {feedforward_force.shape}"
                )
            F_task += feedforward_force
        
        # 결과 검증 (Validate result)
        validate_array(F_task, "task_space_force")
        
        return F_task
    
    def compute_joint_torques(
        self,
        F_task: np.ndarray,
        J: np.ndarray,
        tau_null: Optional[np.ndarray] = None,
        N: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        관절 토크 계산
        
        Compute joint torques from task space forces and null space torques.
        
        관절 토크 계산 법칙:
        τ = J^T·F_task + N·τ_null
        
        Joint torque computation law:
        τ = J^T·F_task + N·τ_null
        
        여기서:
        - J^T: 자코비안 전치 (태스크 공간 힘을 관절 토크로 변환)
        - F_task: 태스크 공간 힘
        - N: 영공간 투영 행렬 (태스크 공간에 영향을 주지 않는 방향)
        - τ_null: 영공간 토크 (부차적인 목표 달성용)
        
        where:
        - J^T: Jacobian transpose (transforms task space force to joint torque)
        - F_task: Task space force
        - N: Null space projector (directions that don't affect task space)
        - τ_null: Null space torque (for secondary objectives)
        
        물리적 의미 (Physical Interpretation):
        J^T·F_task는 태스크 공간 목표를 달성하기 위한 주요 토크입니다.
        N·τ_null은 태스크 공간 동작을 방해하지 않으면서 부차적인 목표
        (예: 관절 한계 회피, 특이점 회피)를 달성하기 위한 토크입니다.
        
        J^T·F_task is the primary torque for achieving task space objectives.
        N·τ_null is the torque for secondary objectives (e.g., joint limit avoidance,
        singularity avoidance) without interfering with task space motion.
        
        Args:
            F_task: 태스크 공간 힘 (6,) - Task space force
            J: 자코비안 행렬 (6 × n) - Jacobian matrix
            tau_null: 영공간 토크 (n,) (선택적) - Null space torque (optional)
            N: 영공간 투영 행렬 (n × n) (선택적) - Null space projector (optional)
        
        Returns:
            tau: 관절 토크 (n,) - Joint torques
        
        Raises:
            ValueError: 입력 배열의 크기가 올바르지 않은 경우
        
        Note:
            tau_null과 N은 함께 제공되거나 둘 다 None이어야 합니다.
            tau_null and N must be provided together or both be None.
        """
        # 입력 검증 (Input validation)
        if F_task.shape != (6,):
            raise ValueError(f"F_task must be (6,), got {F_task.shape}")
        if J.ndim != 2 or J.shape[0] != 6:
            raise ValueError(f"J must be (6 × n), got {J.shape}")
        
        n_dofs = J.shape[1]
        
        # 태스크 공간 토크 계산 (Compute task space torque)
        # τ_task = J^T·F_task
        # 자코비안 전치를 사용하여 태스크 공간 힘을 관절 토크로 변환합니다.
        # Use Jacobian transpose to transform task space force to joint torque.
        tau_task = J.T @ F_task
        
        # 영공간 토크 추가 (Add null space torque)
        if tau_null is not None and N is not None:
            # 입력 검증 (Input validation)
            if tau_null.shape != (n_dofs,):
                raise ValueError(f"tau_null must be ({n_dofs},), got {tau_null.shape}")
            if N.shape != (n_dofs, n_dofs):
                raise ValueError(f"N must be ({n_dofs} × {n_dofs}), got {N.shape}")
            
            # 영공간 투영 적용 (Apply null space projection)
            # τ_total = τ_task + N·τ_null
            # N·τ_null은 태스크 공간에 영향을 주지 않습니다.
            # N·τ_null does not affect task space motion.
            tau = tau_task + N @ tau_null
        elif tau_null is None and N is None:
            # 영공간 토크 없음 (No null space torque)
            tau = tau_task
        else:
            raise ValueError("tau_null and N must be provided together or both be None")
        
        # 결과 검증 (Validate result)
        validate_array(tau, "joint_torques")
        
        return tau
    
    def step(
        self,
        x_desired: np.ndarray,
        tau_null: Optional[np.ndarray] = None,
        feedforward_force: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        제어 루프 한 스텝 실행
        
        Execute one control loop iteration.
        
        이 메서드는 임피던스 제어의 전체 파이프라인을 실행합니다:
        1. 로봇 상태 읽기 (자코비안, 질량 행렬, 엔드 이펙터 위치/속도)
        2. 작업 공간 동역학 계산 (Λ, J̄, N)
        3. 태스크 공간 오차 계산
        4. 태스크 공간 힘 계산
        5. 관절 토크 계산
        6. 토크 명령 전송
        
        This method executes the complete impedance control pipeline:
        1. Read robot state (Jacobian, mass matrix, end-effector pose/velocity)
        2. Compute operational space dynamics (Λ, J̄, N)
        3. Compute task space error
        4. Compute task space force
        5. Compute joint torques
        6. Send torque commands
        
        Args:
            x_desired: 목표 위치/자세 (7,) [pos(3), quat(4)]
                      Desired pose [position(3), quaternion(4)]
                      
            tau_null: 영공간 토크 (n,) (선택적) - Null space torque (optional)
                     NullSpaceController에서 계산된 토크
                     Torque computed by NullSpaceController
                     
            feedforward_force: 피드포워드 힘 (6,) (선택적) - Feedforward force (optional)
                             중력 보상이나 외부 힘 보상
                             Gravity compensation or external force compensation
        
        Returns:
            tau: 계산된 관절 토크 (n,) - Computed joint torques
            info: 디버깅 정보 딕셔너리 - Debugging information dictionary
                 {
                     'error': 태스크 공간 오차,
                     'd_error': 태스크 공간 오차 미분,
                     'F_task': 태스크 공간 힘,
                     'Lambda': 작업 공간 관성 행렬,
                     'manipulability': 가조작도
                 }
        
        Example:
            >>> controller = ImpedanceController(robot, "panda_hand")
            >>> x_desired = np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
            >>> tau, info = controller.step(x_desired)
            >>> print(f"Task space error: {info['error']}")
        """
        # 1. 로봇 상태 읽기 (Read robot state)
        # Genesis API를 사용하여 로봇의 현재 상태를 읽습니다.
        # Read current robot state using Genesis API.
        
        # 자코비안 행렬 (Jacobian matrix)
        J = self.robot.get_jacobian(link=self.ee_link)  # (6 × n)
        
        # 질량 행렬 (Mass matrix)
        M = self.robot.get_mass_mat(decompose=False)  # (n × n)
        
        # 엔드 이펙터 위치 (End-effector position)
        ee_pos = self.robot.get_link_pos(link=self.ee_link)  # (3,)
        
        # 엔드 이펙터 자세 (쿼터니언) (End-effector orientation as quaternion)
        ee_quat = self.robot.get_link_quat(link=self.ee_link)  # (4,) [w, x, y, z]
        
        # 현재 위치/자세 결합 (Combine current pose)
        x_current = np.concatenate([ee_pos, ee_quat])  # (7,)
        
        # 엔드 이펙터 속도 (End-effector velocity)
        ee_vel = self.robot.get_link_vel(link=self.ee_link)  # (6,) [linear, angular]
        
        # 2. 작업 공간 동역학 계산 (Compute operational space dynamics)
        Lambda, J_bar, N = self.op_space.compute_all(
            J, M, damping=self.config.damping_lambda
        )
        
        # 3. 태스크 공간 오차 계산 (Compute task space error)
        error = self.compute_task_space_error(x_desired, x_current)
        
        # 4. 태스크 공간 오차 미분 계산 (Compute task space error derivative)
        d_error = self.compute_task_space_error_derivative(error, ee_vel)
        
        # 5. 태스크 공간 힘 계산 (Compute task space force)
        F_task = self.compute_task_space_force(
            error, d_error, Lambda, feedforward_force
        )
        
        # 6. 관절 토크 계산 (Compute joint torques)
        tau = self.compute_joint_torques(F_task, J, tau_null, N)
        
        # 7. 디버깅 정보 수집 (Collect debugging information)
        info = {
            'error': error,
            'd_error': d_error,
            'F_task': F_task,
            'Lambda': Lambda,
            'J_bar': J_bar,
            'N': N,
            'manipulability': np.sqrt(np.linalg.det(J @ J.T)),
            'x_current': x_current,
            'ee_vel': ee_vel
        }
        
        return tau, info
    
    def reset(self):
        """
        제어기 상태 초기화
        
        Reset controller state.
        
        이전 오차와 시간 정보를 초기화합니다.
        새로운 제어 시퀀스를 시작할 때 호출하세요.
        
        Resets previous error and time information.
        Call this when starting a new control sequence.
        """
        self.prev_error = None
        self.prev_time = None
    
    # ========================================================================
    # 그리퍼 임피던스 제어 (Gripper Impedance Control)
    # ========================================================================
    # 
    # 이 섹션은 그리퍼 제어를 위한 독립적인 서브모듈입니다.
    # 메인 임피던스 제어 로직과 분리되어 있으며, 그리퍼 손가락의
    # 위치 제어와 힘 제한을 담당합니다.
    #
    # This section is an independent submodule for gripper control.
    # It is isolated from the main impedance control logic and handles
    # gripper finger position control and force limiting.
    #
    # ========================================================================
    
    def compute_gripper_error(
        self,
        gripper_gap_desired: float,
        gripper_gap_current: float
    ) -> np.ndarray:
        """
        그리퍼 손가락 위치 오차 계산
        
        Compute gripper finger position errors.
        
        그리퍼는 두 개의 손가락으로 구성되며, 각 손가락은 독립적으로 제어됩니다.
        간격(gap) 목표값이 주어지면, 각 손가락이 중심에서 얼마나 이동해야 하는지 계산합니다.
        
        The gripper consists of two fingers, each controlled independently.
        Given a desired gap, we compute how much each finger should move from the center.
        
        간격 정의 (Gap definition):
        - gap = 0: 완전히 닫힘 (fully closed)
        - gap > 0: 손가락 사이 거리 (distance between fingers)
        
        각 손가락의 목표 위치 (Target position for each finger):
        - finger_1_desired = -gap/2 (왼쪽 손가락, left finger)
        - finger_2_desired = +gap/2 (오른쪽 손가락, right finger)
        
        Args:
            gripper_gap_desired: 목표 그리퍼 간격 (m) - Desired gripper gap
            gripper_gap_current: 현재 그리퍼 간격 (m) - Current gripper gap
        
        Returns:
            errors: 각 손가락의 위치 오차 (2,) [finger_1_error, finger_2_error]
                   Position errors for each finger
                   
                   e_i = q_desired_i - q_current_i
                   
                   여기서 q_i는 각 손가락의 관절 위치입니다.
                   where q_i is the joint position of each finger.
        
        Example:
            >>> # 그리퍼를 5cm 간격으로 열기
            >>> # Open gripper to 5cm gap
            >>> errors = controller.compute_gripper_error(0.05, 0.02)
            >>> print(errors)  # [0.015, 0.015] (각 손가락이 1.5cm씩 더 열려야 함)
        """
        # 간격 오차 계산 (Compute gap error)
        gap_error = gripper_gap_desired - gripper_gap_current
        
        # 각 손가락의 오차는 간격 오차의 절반
        # Each finger error is half of the gap error
        # (두 손가락이 대칭적으로 움직이므로)
        # (since both fingers move symmetrically)
        finger_error = gap_error / 2.0
        
        # 두 손가락 모두 같은 크기의 오차를 가짐
        # Both fingers have the same magnitude of error
        errors = np.array([finger_error, finger_error])
        
        return errors
    
    def compute_gripper_torques(
        self,
        gripper_errors: np.ndarray,
        gripper_velocities: np.ndarray,
        K_p_gripper: float,
        K_d_gripper: float
    ) -> np.ndarray:
        """
        그리퍼 임피던스 토크 계산
        
        Compute gripper impedance torques using PD control law.
        
        그리퍼 제어 법칙 (Gripper control law):
        τ_gripper = K_p·e + K_d·ė
        
        여기서:
        - K_p: 그리퍼 강성 (gripper stiffness)
        - K_d: 그리퍼 감쇠 (gripper damping)
        - e: 손가락 위치 오차 (finger position error)
        - ė: 손가락 속도 (finger velocity, 목표 속도가 0이므로 -v_current)
        
        where:
        - K_p: gripper stiffness
        - K_d: gripper damping
        - e: finger position error
        - ė: finger velocity (equals -v_current since target velocity is 0)
        
        물리적 의미 (Physical interpretation):
        이 제어 법칙은 각 손가락을 가상의 스프링-댐퍼로 만듭니다.
        K_p는 목표 위치로 복원하는 힘을, K_d는 진동을 감쇠시키는 힘을 제공합니다.
        
        This control law makes each finger behave like a virtual spring-damper.
        K_p provides restoring force toward target position,
        K_d provides damping force to reduce oscillations.
        
        Args:
            gripper_errors: 손가락 위치 오차 (2,) - Finger position errors
            gripper_velocities: 손가락 속도 (2,) - Finger velocities
            K_p_gripper: 그리퍼 강성 - Gripper stiffness
            K_d_gripper: 그리퍼 감쇠 - Gripper damping
        
        Returns:
            tau_gripper: 그리퍼 토크 (2,) [finger_1_torque, finger_2_torque]
                        Gripper torques for each finger
        
        Raises:
            ValueError: 입력 배열의 크기가 올바르지 않은 경우
        
        Example:
            >>> errors = np.array([0.01, 0.01])  # 1cm 오차
            >>> velocities = np.array([0.05, 0.05])  # 5cm/s 속도
            >>> tau = controller.compute_gripper_torques(errors, velocities, 50.0, 5.0)
            >>> print(tau)  # [0.25, 0.25] N (각 손가락에 0.25N 힘)
        """
        # 입력 검증 (Input validation)
        if gripper_errors.shape != (2,):
            raise ValueError(f"gripper_errors must be (2,), got {gripper_errors.shape}")
        if gripper_velocities.shape != (2,):
            raise ValueError(f"gripper_velocities must be (2,), got {gripper_velocities.shape}")
        
        # PD 제어 법칙 적용 (Apply PD control law)
        # τ = K_p·e + K_d·ė
        # 여기서 ė = -v (목표 속도가 0이므로)
        # where ė = -v (since target velocity is 0)
        tau_gripper = K_p_gripper * gripper_errors - K_d_gripper * gripper_velocities
        
        # 결과 검증 (Validate result)
        validate_array(tau_gripper, "gripper_torques")
        
        return tau_gripper
    
    def clamp_gripper_force(
        self,
        tau_gripper: np.ndarray,
        max_gripper_force: float
    ) -> np.ndarray:
        """
        그리퍼 힘 제한
        
        Clamp gripper forces to prevent excessive grasping force.
        
        그리퍼가 물체를 파손하지 않도록 최대 힘을 제한합니다.
        각 손가락의 토크를 독립적으로 제한하여 안전한 파지를 보장합니다.
        
        Limits maximum force to prevent damage to grasped objects.
        Each finger's torque is clamped independently to ensure safe grasping.
        
        제한 법칙 (Clamping law):
        τ_clamped_i = clip(τ_i, -F_max, F_max)
        
        여기서:
        - τ_i: 각 손가락의 계산된 토크
        - F_max: 최대 허용 힘
        
        where:
        - τ_i: computed torque for each finger
        - F_max: maximum allowed force
        
        안전 고려사항 (Safety considerations):
        - 닫는 힘(양수)과 여는 힘(음수) 모두 제한됩니다.
        - 제한이 발생하면 경고 메시지를 출력할 수 있습니다.
        
        - Both closing force (positive) and opening force (negative) are limited.
        - A warning message can be printed when clamping occurs.
        
        Args:
            tau_gripper: 계산된 그리퍼 토크 (2,) - Computed gripper torques
            max_gripper_force: 최대 그리퍼 힘 (N) - Maximum gripper force
        
        Returns:
            tau_clamped: 제한된 그리퍼 토크 (2,) - Clamped gripper torques
                        ||τ_clamped_i|| ≤ max_gripper_force for all i
        
        Raises:
            ValueError: 입력 배열의 크기가 올바르지 않거나 max_force가 음수인 경우
        
        Example:
            >>> tau = np.array([25.0, 30.0])  # 과도한 힘
            >>> tau_safe = controller.clamp_gripper_force(tau, 20.0)
            >>> print(tau_safe)  # [20.0, 20.0] (최대값으로 제한됨)
        """
        # 입력 검증 (Input validation)
        if tau_gripper.shape != (2,):
            raise ValueError(f"tau_gripper must be (2,), got {tau_gripper.shape}")
        if max_gripper_force <= 0:
            raise ValueError(f"max_gripper_force must be positive, got {max_gripper_force}")
        
        # 토크 제한 (Clamp torques)
        # 각 손가락의 토크를 [-F_max, F_max] 범위로 제한
        # Clamp each finger's torque to [-F_max, F_max] range
        tau_clamped = np.clip(tau_gripper, -max_gripper_force, max_gripper_force)
        
        # 제한이 발생했는지 확인 (Check if clamping occurred)
        if not np.allclose(tau_gripper, tau_clamped):
            # 경고: 그리퍼 힘이 제한되었습니다
            # Warning: Gripper force was clamped
            # (실제 구현에서는 로깅 시스템 사용 가능)
            # (In actual implementation, could use logging system)
            pass  # 조용히 제한 (silently clamp)
        
        # 결과 검증 (Validate result)
        validate_array(tau_clamped, "clamped_gripper_torques")
        
        # 제한 조건 확인 (Verify clamping constraint)
        assert np.all(np.abs(tau_clamped) <= max_gripper_force), \
            "Clamped torques exceed maximum force"
        
        return tau_clamped
