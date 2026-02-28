"""
설정 데이터클래스 모듈 (Configuration Dataclasses Module)

이 모듈은 임피던스 제어 시스템의 모든 설정 파라미터를 정의합니다.
각 컴포넌트는 독립적인 설정 클래스를 가지며, 파라미터 검증 기능을 포함합니다.

This module defines configuration dataclasses for all components in the impedance control system.
Each component has its own independent configuration with parameter validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ============================================================================
# 임피던스 제어 설정 (Impedance Control Configuration)
# ============================================================================

@dataclass
class ImpedanceControlConfig:
    """
    임피던스 제어 설정
    
    Configuration for task space impedance controller.
    Defines stiffness, damping, and control parameters.
    """
    
    # 태스크 공간 강성 (Task space stiffness)
    K_p_translation: float = 1000.0  # N/m - 위치 강성
    K_p_rotation: float = 100.0      # Nm/rad - 회전 강성
    
    # 태스크 공간 감쇠 (Task space damping)
    K_d_translation: float = 100.0   # N·s/m - 위치 감쇠
    K_d_rotation: float = 10.0       # Nm·s/rad - 회전 감쇠
    
    # 영공간 제어 게인 (Null space control gain)
    K_null: float = 10.0
    
    # 제어 주파수 (Control frequency)
    control_freq: float = 1000.0     # Hz (1000Hz control loop)
    physics_freq: float = 10000.0    # Hz (10000Hz physics simulation)
    
    # 수치 안정성 파라미터 (Numerical stability parameters)
    damping_lambda: float = 0.01     # Jacobian 정규화 - Jacobian regularization
    singularity_threshold: float = 1e-4  # 특이점 임계값 - Singularity threshold
    
    def __post_init__(self):
        """
        설정 파라미터 검증
        
        Validate configuration parameters after initialization.
        """
        # 강성과 감쇠는 양수여야 함 (Stiffness and damping must be positive)
        if self.K_p_translation <= 0 or self.K_p_rotation <= 0:
            raise ValueError("Stiffness parameters must be positive")
        
        if self.K_d_translation <= 0 or self.K_d_rotation <= 0:
            raise ValueError("Damping parameters must be positive")
        
        if self.K_null <= 0:
            raise ValueError("Null space gain must be positive")
        
        # 주파수는 양수여야 함 (Frequencies must be positive)
        if self.control_freq <= 0 or self.physics_freq <= 0:
            raise ValueError("Frequencies must be positive")
        
        # 제어 주파수는 물리 주파수보다 작거나 같아야 함
        # Control frequency must be less than or equal to physics frequency
        if self.control_freq > self.physics_freq:
            raise ValueError("Control frequency cannot exceed physics frequency")
    
    def get_K_p_matrix(self) -> np.ndarray:
        """
        강성 행렬 생성 (6×6 대각 행렬)
        
        Generate stiffness matrix (6×6 diagonal).
        
        Returns:
            K_p: 강성 행렬 [translation(3), rotation(3)]
        """
        K_p = np.diag([
            self.K_p_translation, self.K_p_translation, self.K_p_translation,
            self.K_p_rotation, self.K_p_rotation, self.K_p_rotation
        ])
        return K_p
    
    def get_K_d_matrix(self) -> np.ndarray:
        """
        감쇠 행렬 생성 (6×6 대각 행렬)
        
        Generate damping matrix (6×6 diagonal).
        
        Returns:
            K_d: 감쇠 행렬 [translation(3), rotation(3)]
        """
        K_d = np.diag([
            self.K_d_translation, self.K_d_translation, self.K_d_translation,
            self.K_d_rotation, self.K_d_rotation, self.K_d_rotation
        ])
        return K_d


# ============================================================================
# RBF 네트워크 설정 (RBF Network Configuration)
# ============================================================================

@dataclass
class RBFNetworkConfig:
    """
    RBF 네트워크 설정
    
    Configuration for RBF neural network force estimator.
    Defines network architecture and learning parameters.
    """
    
    n_centers: int = 15              # RBF 중심점 개수 - Number of RBF centers
    input_dim: int = 2               # 입력 차원 [d, ḋ] - Input dimension
    output_dim: int = 1              # 출력 차원 (접촉력) - Output dimension (contact force)
    learning_rate: float = 0.01      # 학습률 γ - Learning rate
    initial_width: float = 0.1       # 초기 RBF 폭 - Initial RBF width
    
    def __post_init__(self):
        """
        설정 파라미터 검증
        
        Validate configuration parameters after initialization.
        """
        # 중심점 개수는 양의 정수여야 함 (Number of centers must be positive integer)
        if self.n_centers <= 0:
            raise ValueError("Number of RBF centers must be positive")
        
        # 입력/출력 차원은 양의 정수여야 함 (Dimensions must be positive integers)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("Input and output dimensions must be positive")
        
        # 학습률은 양수여야 함 (리아푸노프 안정성 보장)
        # Learning rate must be positive (for Lyapunov stability)
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive for Lyapunov stability (γ > 0)")
        
        # RBF 폭은 양수여야 함 (RBF width must be positive)
        if self.initial_width <= 0:
            raise ValueError("RBF width must be positive")


# ============================================================================
# 그리퍼 제어 설정 (Gripper Control Configuration)
# ============================================================================

@dataclass
class GripperConfig:
    """
    그리퍼 제어 설정
    
    Configuration for gripper impedance control.
    Defines gripper-specific control parameters.
    """
    
    K_p_gripper: float = 50.0        # 그리퍼 강성 - Gripper stiffness
    K_d_gripper: float = 5.0         # 그리퍼 감쇠 - Gripper damping
    max_gripper_force: float = 20.0  # 최대 그리퍼 힘 (N) - Maximum gripper force
    gripper_dof_indices: List[int] = field(default_factory=lambda: [7, 8])  # 그리퍼 DOF 인덱스
    
    def __post_init__(self):
        """
        설정 파라미터 검증
        
        Validate configuration parameters after initialization.
        """
        # 강성과 감쇠는 양수여야 함 (Stiffness and damping must be positive)
        if self.K_p_gripper <= 0 or self.K_d_gripper <= 0:
            raise ValueError("Gripper stiffness and damping must be positive")
        
        # 최대 힘은 양수여야 함 (Maximum force must be positive)
        if self.max_gripper_force <= 0:
            raise ValueError("Maximum gripper force must be positive")
        
        # DOF 인덱스는 2개여야 함 (두 손가락)
        # DOF indices must be 2 (two fingers)
        if len(self.gripper_dof_indices) != 2:
            raise ValueError("Gripper must have exactly 2 DOF indices (two fingers)")


# ============================================================================
# 영공간 제어 설정 (Null Space Control Configuration)
# ============================================================================

@dataclass
class NullSpaceConfig:
    """
    영공간 제어 설정
    
    Configuration for null space controller.
    Defines joint limit avoidance parameters.
    """
    
    K_null: float = 10.0             # 영공간 게인 - Null space gain
    joint_limit_margin: float = 0.05 # 관절 한계 마진 (5%) - Joint limit margin
    use_qp_solver: bool = False      # QP 솔버 사용 여부 - Whether to use QP solver
    
    # Franka Panda 관절 한계 (Joint limits for Franka Panda)
    # 단위: radians
    q_min: np.ndarray = field(default_factory=lambda: np.array([
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
    ]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([
        2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
    ]))
    
    def __post_init__(self):
        """
        설정 파라미터 검증
        
        Validate configuration parameters after initialization.
        """
        # 영공간 게인은 양수여야 함 (Null space gain must be positive)
        if self.K_null <= 0:
            raise ValueError("Null space gain must be positive")
        
        # 마진은 0과 1 사이여야 함 (Margin must be between 0 and 1)
        if not (0 < self.joint_limit_margin < 1):
            raise ValueError("Joint limit margin must be between 0 and 1")
        
        # 관절 한계 검증 (Validate joint limits)
        if len(self.q_min) != len(self.q_max):
            raise ValueError("q_min and q_max must have the same length")
        
        if np.any(self.q_min >= self.q_max):
            raise ValueError("q_min must be less than q_max for all joints")


# ============================================================================
# 안전 설정 (Safety Configuration)
# ============================================================================

@dataclass
class SafetyConfig:
    """
    안전 설정
    
    Configuration for safety monitoring and error handling.
    """
    
    # Franka Panda 최대 토크 한계 (Maximum torque limits for Franka Panda)
    # 단위: Nm
    tau_max: np.ndarray = field(default_factory=lambda: np.array([
        87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0
    ]))
    
    # 안전 모니터링 파라미터 (Safety monitoring parameters)
    max_error_count: int = 10        # 최대 오류 횟수 - Maximum error count
    singularity_threshold: float = 1e-4  # 특이점 임계값 - Singularity threshold
    nan_inf_check: bool = True       # NaN/Inf 검사 활성화 - Enable NaN/Inf checking
    
    def __post_init__(self):
        """
        설정 파라미터 검증
        
        Validate configuration parameters after initialization.
        """
        # 토크 한계는 양수여야 함 (Torque limits must be positive)
        if np.any(self.tau_max <= 0):
            raise ValueError("Torque limits must be positive")
        
        # 최대 오류 횟수는 양의 정수여야 함 (Max error count must be positive integer)
        if self.max_error_count <= 0:
            raise ValueError("Maximum error count must be positive")
        
        # 특이점 임계값은 양수여야 함 (Singularity threshold must be positive)
        if self.singularity_threshold <= 0:
            raise ValueError("Singularity threshold must be positive")


# ============================================================================
# 전체 시스템 설정 (Complete System Configuration)
# ============================================================================

@dataclass
class SystemConfig:
    """
    전체 시스템 설정
    
    Complete system configuration combining all component configurations.
    This provides a single entry point for configuring the entire control system.
    """
    
    impedance: ImpedanceControlConfig = field(default_factory=ImpedanceControlConfig)
    rbf_network: RBFNetworkConfig = field(default_factory=RBFNetworkConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    null_space: NullSpaceConfig = field(default_factory=NullSpaceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    def __post_init__(self):
        """
        전체 시스템 설정 검증
        
        Validate complete system configuration.
        """
        # 각 컴포넌트의 설정이 유효한지 확인
        # Verify that each component configuration is valid
        # (각 컴포넌트의 __post_init__에서 이미 검증됨)
        # (Already validated in each component's __post_init__)
        pass
    
    @classmethod
    def create_default(cls) -> "SystemConfig":
        """
        기본 설정 생성
        
        Create default system configuration with reasonable parameters
        for the Franka Panda robot.
        
        Returns:
            SystemConfig: 기본 설정 객체
        """
        return cls()
    
    @classmethod
    def create_high_stiffness(cls) -> "SystemConfig":
        """
        높은 강성 설정 생성 (정밀 위치 제어용)
        
        Create high stiffness configuration for precise position control.
        
        Returns:
            SystemConfig: 높은 강성 설정 객체
        """
        config = cls()
        config.impedance.K_p_translation = 2000.0
        config.impedance.K_p_rotation = 200.0
        config.impedance.K_d_translation = 200.0
        config.impedance.K_d_rotation = 20.0
        return config
    
    @classmethod
    def create_low_stiffness(cls) -> "SystemConfig":
        """
        낮은 강성 설정 생성 (유연한 상호작용용)
        
        Create low stiffness configuration for compliant interaction.
        
        Returns:
            SystemConfig: 낮은 강성 설정 객체
        """
        config = cls()
        config.impedance.K_p_translation = 500.0
        config.impedance.K_p_rotation = 50.0
        config.impedance.K_d_translation = 50.0
        config.impedance.K_d_rotation = 5.0
        return config
