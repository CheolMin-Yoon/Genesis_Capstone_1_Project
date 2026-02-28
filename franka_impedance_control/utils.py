"""
유틸리티 함수 모듈 (Utility Functions Module)

이 모듈은 임피던스 제어 시스템에서 사용되는 수학적 헬퍼 함수들을 제공합니다.
모든 함수는 독립적이며 다른 모듈에 의존하지 않습니다.

This module provides mathematical helper functions used throughout the impedance control system.
All functions are independent and do not depend on other modules.
"""

import numpy as np
from typing import Tuple


# ============================================================================
# 행렬 연산 유틸리티 (Matrix Operation Utilities)
# ============================================================================

def safe_inverse(matrix: np.ndarray, damping: float = 1e-6) -> np.ndarray:
    """
    안전한 행렬 역행렬 계산
    
    Compute matrix inverse with regularization for numerical stability.
    Adds damping to diagonal before inversion to avoid singularities.
    
    Args:
        matrix: 역행렬을 계산할 행렬 (n × n)
        damping: 정규화 파라미터 (기본값: 1e-6)
        
    Returns:
        matrix_inv: 역행렬 (n × n)
    """
    n = matrix.shape[0]
    regularized = matrix + damping * np.eye(n)
    return np.linalg.inv(regularized)


def is_positive_definite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    행렬이 양정부호인지 확인
    
    Check if a matrix is positive definite by verifying all eigenvalues are positive.
    
    Args:
        matrix: 확인할 행렬 (n × n)
        tol: 허용 오차 (기본값: 1e-8)
        
    Returns:
        is_pd: 양정부호 여부
    """
    try:
        # 대칭 행렬인지 확인 (Check if symmetric)
        if not np.allclose(matrix, matrix.T, atol=tol):
            return False
        
        # 고유값 계산 (Compute eigenvalues)
        eigenvalues = np.linalg.eigvalsh(matrix)
        
        # 모든 고유값이 양수인지 확인 (Check if all eigenvalues are positive)
        return np.all(eigenvalues > tol)
    except np.linalg.LinAlgError:
        return False


def validate_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    """
    배열에 NaN이나 Inf가 없는지 검증
    
    Validate that array contains no NaN or Inf values.
    Raises ValueError if invalid values are found.
    
    Args:
        array: 검증할 배열
        name: 배열 이름 (오류 메시지용)
        
    Returns:
        array: 검증된 배열 (입력과 동일)
        
    Raises:
        ValueError: NaN이나 Inf가 발견된 경우
    """
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains Inf values")
    return array


def clamp_array(array: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    배열 값을 범위 내로 제한
    
    Clamp array values to specified range element-wise.
    
    Args:
        array: 제한할 배열
        min_val: 최소값
        max_val: 최대값
        
    Returns:
        clamped: 제한된 배열
    """
    return np.clip(array, min_val, max_val)


# ============================================================================
# 자세 및 회전 유틸리티 (Pose and Rotation Utilities)
# ============================================================================

def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    쿼터니언을 회전 행렬로 변환
    
    Convert quaternion to rotation matrix.
    
    Args:
        quat: 쿼터니언 [w, x, y, z] (4,)
        
    Returns:
        R: 회전 행렬 (3 × 3)
    """
    w, x, y, z = quat
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    회전 행렬을 쿼터니언으로 변환
    
    Convert rotation matrix to quaternion.
    
    Args:
        R: 회전 행렬 (3 × 3)
        
    Returns:
        quat: 쿼터니언 [w, x, y, z] (4,)
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def quaternion_error(q_desired: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """
    쿼터니언 오차를 각속도 벡터로 계산
    
    Compute orientation error as angular velocity vector from quaternion difference.
    
    Args:
        q_desired: 목표 쿼터니언 [w, x, y, z] (4,)
        q_current: 현재 쿼터니언 [w, x, y, z] (4,)
        
    Returns:
        error: 각속도 오차 벡터 (3,)
    """
    # 쿼터니언 정규화 (Normalize quaternions)
    q_desired = q_desired / np.linalg.norm(q_desired)
    q_current = q_current / np.linalg.norm(q_current)
    
    # 오차 쿼터니언 계산: q_error = q_desired * q_current^(-1)
    # Compute error quaternion: q_error = q_desired * q_current^(-1)
    q_current_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    
    # 쿼터니언 곱셈 (Quaternion multiplication)
    w1, x1, y1, z1 = q_desired
    w2, x2, y2, z2 = q_current_inv
    
    q_error = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    
    # 오차 벡터 추출 (Extract error vector)
    # error = 2 * sign(w) * [x, y, z]
    sign = 1.0 if q_error[0] >= 0 else -1.0
    error = 2.0 * sign * q_error[1:4]
    
    return error


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    벡터를 반대칭 행렬로 변환
    
    Convert vector to skew-symmetric matrix for cross product.
    
    Args:
        v: 벡터 (3,)
        
    Returns:
        S: 반대칭 행렬 (3 × 3)
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


# ============================================================================
# 동역학 유틸리티 (Dynamics Utilities)
# ============================================================================

def compute_manipulability(J: np.ndarray) -> float:
    """
    매니퓰레이터 가조작도 계산
    
    Compute manipulability measure (Yoshikawa's manipulability index).
    
    Args:
        J: 자코비안 행렬 (6 × n)
        
    Returns:
        manipulability: 가조작도 (0에 가까우면 특이점)
    """
    JJT = J @ J.T
    det = np.linalg.det(JJT)
    return np.sqrt(max(0, det))


def is_near_singularity(J: np.ndarray, threshold: float = 1e-4) -> bool:
    """
    자코비안이 특이점 근처인지 확인
    
    Check if Jacobian is near singularity.
    
    Args:
        J: 자코비안 행렬 (6 × n)
        threshold: 특이점 임계값
        
    Returns:
        near_singularity: 특이점 근처 여부
    """
    manipulability = compute_manipulability(J)
    return manipulability < threshold


def compute_damping_factor(J: np.ndarray, threshold: float = 1e-4) -> float:
    """
    특이점 근처에서 감쇠 계수 계산
    
    Compute damping factor near singularities for numerical stability.
    
    Args:
        J: 자코비안 행렬 (6 × n)
        threshold: 특이점 임계값
        
    Returns:
        damping: 감쇠 계수 (특이점에 가까울수록 큼)
    """
    manipulability = compute_manipulability(J)
    
    if manipulability < threshold:
        # 특이점에 가까우면 감쇠 추가
        # Add damping near singularity
        damping = threshold - manipulability
    else:
        damping = 0.0
    
    return damping


# ============================================================================
# 관절 한계 유틸리티 (Joint Limit Utilities)
# ============================================================================

def compute_joint_range_percentage(
    q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray
) -> np.ndarray:
    """
    각 관절의 범위 내 위치를 백분율로 계산
    
    Compute percentage of joint range for each joint (0 = min, 1 = max).
    
    Args:
        q: 현재 관절 위치 (n,)
        q_min: 관절 최소값 (n,)
        q_max: 관절 최대값 (n,)
        
    Returns:
        percentage: 범위 백분율 (n,) [0, 1]
    """
    q_range = q_max - q_min
    percentage = (q - q_min) / q_range
    return np.clip(percentage, 0.0, 1.0)


def is_near_joint_limit(
    q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray, margin: float = 0.05
) -> np.ndarray:
    """
    각 관절이 한계 근처인지 확인
    
    Check if each joint is near its limits (within margin percentage).
    
    Args:
        q: 현재 관절 위치 (n,)
        q_min: 관절 최소값 (n,)
        q_max: 관절 최대값 (n,)
        margin: 한계 마진 (기본값: 5%)
        
    Returns:
        near_limit: 각 관절의 한계 근처 여부 (n,) [bool]
    """
    q_range = q_max - q_min
    margin_dist = margin * q_range
    
    near_min = (q - q_min) < margin_dist
    near_max = (q_max - q) < margin_dist
    
    return near_min | near_max


# ============================================================================
# 로깅 및 디버깅 유틸리티 (Logging and Debugging Utilities)
# ============================================================================

def print_matrix_info(matrix: np.ndarray, name: str = "Matrix"):
    """
    행렬 정보 출력 (디버깅용)
    
    Print matrix information for debugging.
    
    Args:
        matrix: 출력할 행렬
        name: 행렬 이름
    """
    print(f"\n{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Dtype: {matrix.dtype}")
    print(f"  Min: {np.min(matrix):.6f}")
    print(f"  Max: {np.max(matrix):.6f}")
    print(f"  Mean: {np.mean(matrix):.6f}")
    print(f"  Std: {np.std(matrix):.6f}")
    
    if np.any(np.isnan(matrix)):
        print(f"  WARNING: Contains NaN values!")
    if np.any(np.isinf(matrix)):
        print(f"  WARNING: Contains Inf values!")
    
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        try:
            det = np.linalg.det(matrix)
            print(f"  Determinant: {det:.6e}")
            
            eigenvalues = np.linalg.eigvalsh(matrix)
            print(f"  Eigenvalues: min={np.min(eigenvalues):.6e}, max={np.max(eigenvalues):.6e}")
        except np.linalg.LinAlgError:
            print(f"  WARNING: Could not compute determinant/eigenvalues")


def format_vector(v: np.ndarray, precision: int = 4) -> str:
    """
    벡터를 읽기 쉬운 문자열로 포맷
    
    Format vector as readable string.
    
    Args:
        v: 벡터
        precision: 소수점 자릿수
        
    Returns:
        formatted: 포맷된 문자열
    """
    return "[" + ", ".join([f"{x:.{precision}f}" for x in v]) + "]"
