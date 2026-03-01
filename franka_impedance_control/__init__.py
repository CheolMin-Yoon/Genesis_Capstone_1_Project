"""
Franka Impedance Control with RBF Neural Network

태스크 공간 임피던스 제어 시스템 (Task Space Impedance Control System)

이 패키지는 Genesis 물리 시뮬레이터에서 Franka Emika Panda 로봇을 위한
완전한 임피던스 제어 시스템을 제공합니다.

This package provides a complete impedance control system for the Franka Emika Panda robot
in the Genesis physics simulator.

주요 컴포넌트 (Main Components):
- ImpedanceController: 태스크 공간 임피던스 제어
- NullSpaceController: 영공간 제어 (관절 한계 회피)
- RBFNetwork: RBF 신경망 기반 접촉력 추정
- LyapunovLearning: 리아푸노프 안정성 보장 학습

모듈형 아키텍처 (Modular Architecture):
각 컴포넌트는 완전히 독립적이며 명확한 인터페이스를 통해 통신합니다.
"""

__version__ = "0.1.0"
__author__ = "Genesis AI"

# 인터페이스 및 데이터 구조 (Interfaces and Data Structures)
from .interfaces import (
    RobotState,
    ControlOutput,
    IOperationalSpaceDynamics,
    IImpedanceController,
    INullSpaceController,
    IRBFNetwork,
    ILyapunovLearning,
)

# 설정 (Configuration)
from .config import (
    ImpedanceControlConfig,
    RBFNetworkConfig,
    GripperConfig,
    NullSpaceConfig,
    SafetyConfig,
    SystemConfig,
)

# 유틸리티 (Utilities)
from .utils import (
    safe_inverse,
    is_positive_definite,
    validate_array,
    clamp_array,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_error,
    skew_symmetric,
    compute_manipulability,
    is_near_singularity,
    compute_damping_factor,
    compute_joint_range_percentage,
    is_near_joint_limit,
)

# 작업 공간 동역학 (Operational Space Dynamics)
from .operational_space import OperationalSpaceDynamics

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Interfaces
    "RobotState",
    "ControlOutput",
    "IOperationalSpaceDynamics",
    "IImpedanceController",
    "INullSpaceController",
    "IRBFNetwork",
    "ILyapunovLearning",
    
    # Configuration
    "ImpedanceControlConfig",
    "RBFNetworkConfig",
    "GripperConfig",
    "NullSpaceConfig",
    "SafetyConfig",
    "SystemConfig",
    
    # Operational Space Dynamics
    "OperationalSpaceDynamics",
    
    # Utilities
    "safe_inverse",
    "is_positive_definite",
    "validate_array",
    "clamp_array",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "quaternion_error",
    "skew_symmetric",
    "compute_manipulability",
    "is_near_singularity",
    "compute_damping_factor",
    "compute_joint_range_percentage",
    "is_near_joint_limit",
]
