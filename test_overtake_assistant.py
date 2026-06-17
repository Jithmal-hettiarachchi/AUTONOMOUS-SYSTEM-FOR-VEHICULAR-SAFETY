"""
Unit tests for Overtake Assistant module.

Tests state machine transitions, clearance zone geometry, and advisory generation.
"""

import pytest
from src.overtake.types import OvertakeStatus, OvertakeAdvisory
from src.overtake.state import StateTracker
from src.overtake.clearance import (
    calculate_clearance_zone,
    is_zone_valid,
    point_in_clearance_zone,
    bbox_intersects_zone,
)
from src.overtake.line_analysis import is_broken_line, estimate_line_confidence
from src.overtake.assistant import OvertakeAssistant, OvertakeConfig
from src.lane.result import LanePolynomial, LaneResult
from src.detection.result import Detection, DetectionLabel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default overtake assistant configuration."""
    return OvertakeConfig(
        enabled=True,
        min_lane_confidence=0.6,
        stability_frames=3,
        zone_width_ratio=1.0,
        zone_y_top_ratio=0.65,
        safe_frames_required=5,
        broken_line_density_threshold=0.5,
    )


@pytest.fixture
def mock_left_lane():
    """Create a mock left lane polynomial."""
    # x = ay² + by + c
    # Centered lanes for 640x480 frame
    # Left lane at ~100-120 pixels from left edge
    return LanePolynomial(
        coefficients=(0.0001, -0.05, 150),
        y_range=(312, 479),
        confidence=0.85,
        point_count=50,
    )


@pytest.fixture
def mock_right_lane():
    """Create a mock right lane polynomial."""
    # Right lane at ~400-450 pixels, leaving room for clearance zone on right
    return LanePolynomial(
        coefficients=(0.0001, 0.05, 400),
        y_range=(312, 479),
        confidence=0.85,
        point_count=55,
    )


@pytest.fixture
def valid_lane_result(mock_left_lane, mock_right_lane):
    """Create a valid lane result with both lanes."""
    return LaneResult(
        left_lane=mock_left_lane,
        right_lane=mock_right_lane,
        valid=True,
        partial=False,
        timestamp=0.0,
        latency_ms=10.0,
    )


@pytest.fixture
def invalid_lane_result():
    """Create an invalid lane result."""
    return LaneResult(
        left_lane=None,
        right_lane=None,
        valid=False,
        partial=False,
        timestamp=0.0,
        latency_ms=10.0,
    )


@pytest.fixture
def vehicle_detection():
    """Create a vehicle detection in the clearance zone.
    
    For left-hand traffic (overtake on right), clearance zone is to the right
    of the right lane. With right lane at ~400-450px, clearance zone is ~450-640.
    """
    return Detection(
        bbox=(480.0, 350.0, 560.0, 420.0),  # In right-side clearance zone
        confidence=0.8,
        label=DetectionLabel.VEHICLE,
        timestamp=0.0,
    )


# =============================================================================
# StateTracker Tests
# =============================================================================

class TestStateTracker:
    """Tests for the StateTracker state machine."""
    
    def test_starts_disabled(self):
        """Tracker should start in DISABLED state."""
        tracker = StateTracker()
        assert tracker.current_state == OvertakeStatus.DISABLED
        assert tracker.safe_frame_count == 0
        assert tracker.lane_stable_count == 0
    
    def test_requires_stable_lanes_before_evaluation(self):
        """Should stay DISABLED until lanes are stable for required frames."""
        tracker = StateTracker(required_stable_frames=3, required_safe_frames=2)
        
        # First 2 frames with valid lanes - still DISABLED
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.DISABLED
        
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.DISABLED
        
        # Third frame - now can evaluate (but not yet SAFE due to safe_frames)
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.UNSAFE  # Still counting safe frames
    
    def test_requires_consecutive_safe_frames_for_safe(self):
        """Should require N consecutive safe frames before SAFE status."""
        # Note: safe frame counting starts from when lanes become stable
        # With required_stable_frames=1, first update makes us stable AND counts as first safe frame
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=3)
        
        # First update: becomes stable + safe_count=1
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.UNSAFE  # count=1, need 3
        
        # Second update: safe_count=2
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.UNSAFE  # count=2, need 3
        
        # Third update: safe_count=3 -> SAFE
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.SAFE
    
    def test_vehicle_in_zone_resets_safe_count(self):
        """Vehicle entering zone should reset safe frame count."""
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=3)
        
        # Get to counting state
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.safe_frame_count == 2
        
        # Vehicle enters
        result = tracker.update(lanes_valid=True, zone_clear=False, broken_line=True)
        assert result == OvertakeStatus.UNSAFE
        assert tracker.safe_frame_count == 0
    
    def test_solid_line_resets_safe_count(self):
        """Solid line detection should reset safe frame count."""
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=3)
        
        # Get to counting state
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.safe_frame_count == 2
        
        # Solid line detected
        result = tracker.update(lanes_valid=True, zone_clear=True, broken_line=False)
        assert result == OvertakeStatus.UNSAFE
        assert tracker.safe_frame_count == 0
    
    def test_lane_loss_resets_everything(self):
        """Lost lanes should reset to DISABLED immediately."""
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=2)
        
        # Get to SAFE
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.current_state == OvertakeStatus.SAFE
        
        # Lose lanes
        result = tracker.update(lanes_valid=False, zone_clear=True, broken_line=True)
        assert result == OvertakeStatus.DISABLED
        assert tracker.lane_stable_count == 0
        assert tracker.safe_frame_count == 0
    
    def test_reset_clears_all_state(self):
        """Reset should clear all internal state."""
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=2)
        
        # Build up some state
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.safe_frame_count > 0
        
        # Reset
        tracker.reset()
        assert tracker.current_state == OvertakeStatus.DISABLED
        assert tracker.safe_frame_count == 0
        assert tracker.lane_stable_count == 0
    
    def test_frames_until_safe_property(self):
        """Should correctly report frames remaining until SAFE."""
        tracker = StateTracker(required_stable_frames=1, required_safe_frames=5)
        
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.frames_until_safe == 4
        
        tracker.update(lanes_valid=True, zone_clear=True, broken_line=True)
        assert tracker.frames_until_safe == 3


# =============================================================================
# Clearance Zone Geometry Tests
# =============================================================================

class TestClearanceZone:
    """Tests for clearance zone geometry calculations."""
    
    def test_calculate_clearance_zone_basic(self, mock_left_lane, mock_right_lane):
        """Should calculate reasonable clearance zone polygon."""
        zone = calculate_clearance_zone(
            left_lane=mock_left_lane,
            right_lane=mock_right_lane,
            frame_width=640,
            frame_height=480,
            zone_y_top_ratio=0.65,
            zone_width_ratio=1.0,
        )
        
        # Should have 4 vertices
        assert len(zone) == 4
        
        # Zone should be to the LEFT of left lane (smaller x values)
        top_left, top_right, bottom_right, bottom_left = zone
        
        # Top right should be approximately at left lane position at y_top
        assert top_right[1] == int(480 * 0.65)  # y = 312
        
        # All x values should be non-negative
        assert all(x >= 0 for x, y in zone)
    
    def test_is_zone_valid_rejects_too_narrow(self):
        """Should reject zones that are too narrow."""
        narrow_zone = [
            (10, 100),
            (15, 100),  # Only 5 pixels wide
            (15, 200),
            (10, 200),
        ]
        assert not is_zone_valid(narrow_zone, frame_width=640, min_width_px=30)
    
    def test_is_zone_valid_accepts_valid_zone(self):
        """Should accept properly sized zones."""
        valid_zone = [
            (0, 100),
            (100, 100),  # 100 pixels wide
            (120, 200),
            (0, 200),
        ]
        assert is_zone_valid(valid_zone, frame_width=640, min_width_px=30)
    
    def test_point_in_clearance_zone(self):
        """Should correctly detect points inside/outside zone."""
        zone = [
            (0, 100),
            (100, 100),
            (100, 200),
            (0, 200),
        ]
        
        # Point inside
        assert point_in_clearance_zone(50, 150, zone)
        
        # Point outside (right of zone)
        assert not point_in_clearance_zone(150, 150, zone)
        
        # Point outside (above zone)
        assert not point_in_clearance_zone(50, 50, zone)
    
    def test_bbox_intersects_zone_center_detection(self):
        """Should detect bbox intersection using center point."""
        zone = [
            (0, 100),
            (100, 100),
            (100, 200),
            (0, 200),
        ]
        
        # Bbox with center inside zone
        bbox_inside = (40, 140, 60, 160)  # center at (50, 150)
        assert bbox_intersects_zone(bbox_inside, zone)
        
        # Bbox with center outside zone
        bbox_outside = (140, 140, 160, 160)  # center at (150, 150)
        assert not bbox_intersects_zone(bbox_outside, zone)


# =============================================================================
# Line Analysis Tests
# =============================================================================

class TestLineAnalysis:
    """Tests for broken/solid line detection."""
    
    def test_is_broken_line_low_density(self):
        """Low point density should indicate broken line."""
        lane = LanePolynomial(
            coefficients=(0, 0, 100),
            y_range=(100, 300),  # 200 pixels
            confidence=0.8,
            point_count=50,  # 0.25 points per pixel
        )
        
        assert is_broken_line(lane, density_threshold=0.5)
    
    def test_is_broken_line_high_density(self):
        """High point density should indicate solid line."""
        lane = LanePolynomial(
            coefficients=(0, 0, 100),
            y_range=(100, 300),  # 200 pixels
            confidence=0.8,
            point_count=150,  # 0.75 points per pixel
        )
        
        assert not is_broken_line(lane, density_threshold=0.5)
    
    def test_estimate_line_confidence(self, mock_left_lane):
        """Should combine polynomial confidence with density."""
        confidence = estimate_line_confidence(mock_left_lane)
        
        # Should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
        
        # Should be influenced by polynomial's base confidence
        assert confidence > 0.5  # Base confidence is 0.85


# =============================================================================
# OvertakeAssistant Integration Tests
# =============================================================================

class TestOvertakeAssistant:
    """Integration tests for the full OvertakeAssistant."""
    
    def test_disabled_when_config_disabled(self):
        """Should return DISABLED when disabled in config."""
        config = OvertakeConfig(enabled=False)
        assistant = OvertakeAssistant(config)
        
        advisory = assistant.evaluate(
            lane_result=LaneResult(None, None, False, False, 0, 0),
            detections=[],
            frame_width=640,
            frame_height=480,
        )
        
        assert advisory.status == OvertakeStatus.DISABLED
        assert "disabled in config" in advisory.reason.lower()
    
    def test_disabled_without_valid_lanes(self, default_config, invalid_lane_result):
        """Should return DISABLED when lanes not detected."""
        assistant = OvertakeAssistant(default_config)
        
        advisory = assistant.evaluate(
            lane_result=invalid_lane_result,
            detections=[],
            frame_width=640,
            frame_height=480,
        )
        
        assert advisory.status == OvertakeStatus.DISABLED
        assert advisory.clearance_zone is None
    
    def test_unsafe_when_vehicle_in_zone(
        self, default_config, valid_lane_result, vehicle_detection
    ):
        """Should return UNSAFE when vehicle detected in clearance zone."""
        assistant = OvertakeAssistant(default_config)
        
        # Get past stability requirement
        for _ in range(default_config.stability_frames + 1):
            advisory = assistant.evaluate(
                lane_result=valid_lane_result,
                detections=[vehicle_detection],
                frame_width=640,
                frame_height=480,
            )
        
        assert advisory.status == OvertakeStatus.UNSAFE
        assert advisory.vehicles_in_zone >= 1
    
    def test_clearance_zone_calculated_when_enabled(
        self, default_config, valid_lane_result
    ):
        """Should calculate clearance zone when conditions met."""
        assistant = OvertakeAssistant(default_config)
        
        # Get past stability requirement
        for _ in range(default_config.stability_frames + 1):
            advisory = assistant.evaluate(
                lane_result=valid_lane_result,
                detections=[],
                frame_width=640,
                frame_height=480,
            )
        
        # Should have clearance zone calculated
        assert advisory.clearance_zone is not None
        assert len(advisory.clearance_zone) == 4
    
    def test_advisory_validation(self):
        """OvertakeAdvisory should validate its fields."""
        # Invalid confidence
        with pytest.raises(ValueError):
            OvertakeAdvisory(
                status=OvertakeStatus.SAFE,
                reason="test",
                clearance_zone=None,
                confidence=1.5,  # Invalid
                vehicles_in_zone=0,
            )
        
        # Negative vehicles count
        with pytest.raises(ValueError):
            OvertakeAdvisory(
                status=OvertakeStatus.SAFE,
                reason="test",
                clearance_zone=None,
                confidence=0.8,
                vehicles_in_zone=-1,  # Invalid
            )
    
    def test_reset_clears_state(self, default_config, valid_lane_result):
        """Reset should clear all internal state."""
        assistant = OvertakeAssistant(default_config)
        
        # Build up state
        for _ in range(5):
            assistant.evaluate(
                lane_result=valid_lane_result,
                detections=[],
                frame_width=640,
                frame_height=480,
            )
        
        # Reset
        assistant.reset()
        
        # Should be back to DISABLED (needs stability again)
        advisory = assistant.evaluate(
            lane_result=valid_lane_result,
            detections=[],
            frame_width=640,
            frame_height=480,
        )
        assert advisory.status == OvertakeStatus.DISABLED


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_detections_list(self, default_config, valid_lane_result):
        """Should handle empty detections list."""
        assistant = OvertakeAssistant(default_config)
        
        advisory = assistant.evaluate(
            lane_result=valid_lane_result,
            detections=[],
            frame_width=640,
            frame_height=480,
        )
        
        assert advisory.vehicles_in_zone == 0
    
    def test_non_vehicle_detections_ignored(
        self, default_config, valid_lane_result
    ):
        """Should ignore non-vehicle detections in clearance zone."""
        assistant = OvertakeAssistant(default_config)
        
        pedestrian = Detection(
            bbox=(50.0, 350.0, 100.0, 400.0),
            confidence=0.9,
            label=DetectionLabel.PEDESTRIAN,
            timestamp=0.0,
        )
        
        # Get past stability
        for _ in range(default_config.stability_frames + 1):
            advisory = assistant.evaluate(
                lane_result=valid_lane_result,
                detections=[pedestrian],
                frame_width=640,
                frame_height=480,
            )
        
        # Pedestrian should not count as vehicle in zone
        assert advisory.vehicles_in_zone == 0
    
    def test_very_small_frame(self, default_config, valid_lane_result):
        """Should handle very small frame dimensions."""
        assistant = OvertakeAssistant(default_config)
        
        # Small frame shouldn't crash
        advisory = assistant.evaluate(
            lane_result=valid_lane_result,
            detections=[],
            frame_width=100,
            frame_height=100,
        )
        
        # Should return something (likely DISABLED due to invalid geometry)
        assert advisory.status in OvertakeStatus


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
