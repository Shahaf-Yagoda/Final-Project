"""
Shared MediaPipe Pose Detection Factory
Centralizes MediaPipe pose detector initialization to avoid code duplication.
"""
import mediapipe as mp
from typing import Optional


class PoseDetectorConfig:
    """Configuration class for pose detector settings."""
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence


class PoseDetectorFactory:
    """Factory class for creating MediaPipe pose detectors with consistent configuration."""
    
    # Default configurations for different use cases
    DEFAULT_CONFIG = PoseDetectorConfig()
    
    LIVE_EXERCISE_CONFIG = PoseDetectorConfig(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    VIDEO_ANALYSIS_CONFIG = PoseDetectorConfig(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3  # Lower tracking for video analysis
    )
    
    HIGH_ACCURACY_CONFIG = PoseDetectorConfig(
        static_image_mode=False,
        model_complexity=2,  # Higher accuracy
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    LOW_PERFORMANCE_CONFIG = PoseDetectorConfig(
        static_image_mode=False,
        model_complexity=0,  # Lower complexity for performance
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    @classmethod
    def create_pose_detector(cls, config: Optional[PoseDetectorConfig] = None) -> mp.solutions.pose.Pose:
        """
        Create a MediaPipe pose detector with the specified configuration.
        
        Args:
            config: PoseDetectorConfig object. If None, uses DEFAULT_CONFIG.
            
        Returns:
            Configured MediaPipe Pose detector instance.
        """
        if config is None:
            config = cls.DEFAULT_CONFIG
            
        mp_pose = mp.solutions.pose
        return mp_pose.Pose(
            static_image_mode=config.static_image_mode,
            model_complexity=config.model_complexity,
            enable_segmentation=config.enable_segmentation,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
    
    @classmethod
    def create_live_exercise_detector(cls) -> mp.solutions.pose.Pose:
        """Create pose detector optimized for live exercise tracking."""
        return cls.create_pose_detector(cls.LIVE_EXERCISE_CONFIG)
    
    @classmethod
    def create_video_analysis_detector(cls) -> mp.solutions.pose.Pose:
        """Create pose detector optimized for video analysis."""
        return cls.create_pose_detector(cls.VIDEO_ANALYSIS_CONFIG)
    
    @classmethod
    def create_high_accuracy_detector(cls) -> mp.solutions.pose.Pose:
        """Create pose detector with highest accuracy settings."""
        return cls.create_pose_detector(cls.HIGH_ACCURACY_CONFIG)
    
    @classmethod
    def create_performance_detector(cls) -> mp.solutions.pose.Pose:
        """Create pose detector optimized for performance on low-end hardware."""
        return cls.create_pose_detector(cls.LOW_PERFORMANCE_CONFIG)


# Convenience functions for backward compatibility
def create_default_pose_detector() -> mp.solutions.pose.Pose:
    """Create pose detector with default settings."""
    return PoseDetectorFactory.create_pose_detector()


def get_mp_pose_solutions():
    """Get MediaPipe pose solutions for drawing utilities."""
    return mp.solutions.pose


def get_mp_drawing_utils():
    """Get MediaPipe drawing utilities."""
    return mp.solutions.drawing_utils