"""
Unit tests for IP camera adapter.

Tests cover:
- Initialization and connection
- Frame capture
- Auto-reconnection logic
- Telemetry metrics
- URL masking for credentials
- Error handling
"""

import time
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from src.capture.ip_camera import IPCameraAdapter, IPCameraMetrics
from src.capture.adapter import CaptureConfig
from src.capture.frame import Frame, FrameSource


# =============================================================================
# IPCameraMetrics Tests
# =============================================================================

class TestIPCameraMetrics:
    """Tests for IP camera telemetry metrics."""
    
    def test_initial_state(self):
        """Test metrics start at zero."""
        metrics = IPCameraMetrics()
        
        assert metrics.acquisition_latency_ms == 0.0
        assert metrics.reconnect_count == 0
        assert metrics.total_downtime_ms == 0.0
        assert metrics.connection_lost_time is None
    
    def test_record_frame(self):
        """Test recording successful frame acquisition."""
        metrics = IPCameraMetrics()
        
        metrics.record_frame(15.5)
        
        assert metrics.acquisition_latency_ms == 15.5
        assert metrics.last_frame_time > 0
    
    def test_record_disconnect(self):
        """Test recording connection loss."""
        metrics = IPCameraMetrics()
        
        metrics.record_disconnect()
        
        assert metrics.connection_lost_time is not None
    
    def test_downtime_calculation(self):
        """Test that downtime is calculated when reconnecting."""
        metrics = IPCameraMetrics()
        
        # Simulate disconnect
        metrics.record_disconnect()
        disconnect_time = metrics.connection_lost_time
        
        # Wait a bit
        time.sleep(0.05)  # 50ms
        
        # Reconnect via frame
        metrics.record_frame(10.0)
        
        # Downtime should be recorded
        assert metrics.total_downtime_ms >= 45  # At least 45ms
        assert metrics.connection_lost_time is None
    
    def test_record_reconnect(self):
        """Test reconnection counter."""
        metrics = IPCameraMetrics()
        
        metrics.record_reconnect()
        metrics.record_reconnect()
        metrics.record_reconnect()
        
        assert metrics.reconnect_count == 3
    
    def test_get_metrics_thread_safe(self):
        """Test thread-safe metric retrieval."""
        metrics = IPCameraMetrics()
        metrics.record_frame(20.0)
        metrics.record_reconnect()
        
        latency, reconnects, downtime = metrics.get_metrics()
        
        assert latency == 20.0
        assert reconnects == 1
        assert downtime == 0.0
    
    def test_reset(self):
        """Test metrics reset."""
        metrics = IPCameraMetrics()
        metrics.record_frame(25.0)
        metrics.record_reconnect()
        metrics.record_disconnect()
        
        metrics.reset()
        
        assert metrics.acquisition_latency_ms == 0.0
        assert metrics.reconnect_count == 0
        assert metrics.total_downtime_ms == 0.0
        assert metrics.connection_lost_time is None


# =============================================================================
# IPCameraAdapter Tests
# =============================================================================

class TestIPCameraAdapter:
    """Tests for IP camera adapter."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CaptureConfig(
            resolution=(640, 480),
            target_fps=15,
            timeout_ms=100,
            source=FrameSource.IP_CAMERA,
            ip_url="http://192.168.1.100:8080/video",
        )
    
    @pytest.fixture
    def adapter(self, config):
        """Create adapter instance."""
        return IPCameraAdapter(config)
    
    def test_initialization_without_url(self):
        """Test that initialization fails without URL."""
        config = CaptureConfig(
            source=FrameSource.IP_CAMERA,
            ip_url=None,
        )
        adapter = IPCameraAdapter(config)
        
        result = adapter.initialize()
        
        assert result is False
    
    @patch('cv2.VideoCapture')
    def test_successful_initialization(self, mock_capture_class, adapter):
        """Test successful connection to IP camera."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        
        # Mock successful connection
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: {
            3: 640,  # CAP_PROP_FRAME_WIDTH
            4: 480,  # CAP_PROP_FRAME_HEIGHT
            5: 15.0,  # CAP_PROP_FPS
        }.get(prop, 0)
        
        result = adapter.initialize()
        
        assert result is True
        assert adapter.is_initialized is True
        mock_capture_class.assert_called_once_with("http://192.168.1.100:8080/video")
    
    @patch('cv2.VideoCapture')
    def test_failed_connection(self, mock_capture_class, adapter):
        """Test handling failed connection."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        result = adapter.initialize()
        
        assert result is False
        assert adapter.is_initialized is False
    
    @patch('cv2.VideoCapture')
    def test_connection_open_but_no_frames(self, mock_capture_class, adapter):
        """Test connection opens but no frames received."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        
        result = adapter.initialize()
        
        assert result is False
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_capture_frame(self, mock_capture_class, adapter):
        """Test successful frame capture."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        
        # Setup for initialization
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100, 100] = [255, 128, 64]  # Add some data
        mock_cap.read.return_value = (True, test_frame.copy())
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 15)
        
        adapter.initialize()
        
        frame = adapter.capture()
        
        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.source == FrameSource.IP_CAMERA
        assert frame.data.shape == (480, 640, 3)
        assert frame.sequence == 0
    
    @patch('cv2.VideoCapture')
    def test_capture_increments_sequence(self, mock_capture_class, adapter):
        """Test frame sequence increments."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 15)
        
        adapter.initialize()
        
        frame1 = adapter.capture()
        frame2 = adapter.capture()
        frame3 = adapter.capture()
        
        assert frame1.sequence == 0
        assert frame2.sequence == 1
        assert frame3.sequence == 2
    
    @patch('cv2.VideoCapture')
    def test_capture_stale_frame_returns_none(self, mock_capture_class, adapter):
        """Test that stale frames (too old) are rejected."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 15)
        
        adapter.initialize()
        
        # Simulate stale frame by setting old timestamp
        with adapter._grab_lock:
            adapter._latest_frame_time = 0  # Very old
        
        # Capture should fail due to stale frame
        frame = adapter.capture()
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_grab_thread_updates_latest_frame(self, mock_capture_class, adapter):
        """Test that grab thread continuously updates the frame buffer."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 15)
        
        adapter.initialize()
        
        # Wait a moment for grab thread to run
        import time
        time.sleep(0.05)
        
        # Should have a recent frame
        with adapter._grab_lock:
            assert adapter._latest_frame is not None
            assert adapter._latest_frame_time > 0
        
        adapter.release()
    
    @patch('cv2.VideoCapture')
    def test_release(self, mock_capture_class, adapter):
        """Test resource release."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 15
        
        adapter.initialize()
        adapter.release()
        
        assert adapter.is_initialized is False
        mock_cap.release.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_is_healthy_when_connected(self, mock_capture_class, adapter):
        """Test health check when connected."""
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else 15
        
        adapter.initialize()
        adapter.capture()  # Update last_successful_read
        
        assert adapter.is_healthy() is True
    
    def test_is_healthy_when_not_initialized(self, adapter):
        """Test health check when not initialized."""
        assert adapter.is_healthy() is False
    
    @patch('cv2.VideoCapture')
    def test_frame_resize(self, mock_capture_class):
        """Test frame is resized to target resolution."""
        config = CaptureConfig(
            resolution=(320, 240),
            source=FrameSource.IP_CAMERA,
            ip_url="http://test:8080/video",
        )
        adapter = IPCameraAdapter(config)
        
        mock_cap = MagicMock()
        mock_capture_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        # Return larger frame
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 15)
        
        adapter.initialize()
        frame = adapter.capture()
        
        assert frame.data.shape == (240, 320, 3)
    
    def test_url_masking_with_credentials(self):
        """Test URL masking hides passwords."""
        url = "rtsp://admin:secret123@camera.local:554/stream"
        masked = IPCameraAdapter._mask_url(url)
        
        assert "secret123" not in masked
        assert "admin" in masked
        assert "***" in masked
    
    def test_url_masking_without_credentials(self):
        """Test URL masking for URLs without credentials."""
        url = "http://camera.local:8080/video"
        masked = IPCameraAdapter._mask_url(url)
        
        assert masked == url
    
    def test_url_masking_with_user_only(self):
        """Test URL masking with username but no password."""
        url = "rtsp://admin@camera.local:554/stream"
        masked = IPCameraAdapter._mask_url(url)
        
        # Should return original since no password
        assert masked == url


# =============================================================================
# Integration Tests
# =============================================================================

class TestIPCameraIntegration:
    """Integration tests for IP camera with factory."""
    
    def test_factory_creates_ip_adapter(self):
        """Test factory creates IPCameraAdapter for IP source."""
        from src.capture.factory import create_camera_adapter
        
        config = CaptureConfig(
            source=FrameSource.IP_CAMERA,
            ip_url="http://test:8080/video",
        )
        
        adapter = create_camera_adapter(config)
        
        assert isinstance(adapter, IPCameraAdapter)
    
    def test_factory_raises_without_url(self):
        """Test factory raises error without IP URL."""
        from src.capture.factory import create_camera_adapter
        
        config = CaptureConfig(
            source=FrameSource.IP_CAMERA,
            ip_url=None,
        )
        
        with pytest.raises(ValueError, match="ip_url required"):
            create_camera_adapter(config)
    
    def test_frame_source_enum(self):
        """Test IP_CAMERA is in FrameSource enum."""
        assert FrameSource.IP_CAMERA.value == "ip"
    
    def test_capture_config_has_ip_url(self):
        """Test CaptureConfig has ip_url field."""
        config = CaptureConfig(
            ip_url="http://test:8080/video"
        )
        assert config.ip_url == "http://test:8080/video"


# =============================================================================
# Telemetry Integration Tests
# =============================================================================

class TestIPCameraTelemetry:
    """Tests for IP camera telemetry integration."""
    
    def test_frame_metrics_has_ip_fields(self):
        """Test FrameMetrics has IP-specific fields."""
        from src.telemetry.metrics import FrameMetrics
        
        metrics = FrameMetrics()
        
        assert hasattr(metrics, 'ip_acquisition_latency_ms')
        assert hasattr(metrics, 'ip_reconnect_count')
        assert hasattr(metrics, 'ip_downtime_ms')
    
    def test_frame_metrics_to_dict_includes_ip(self):
        """Test IP metrics appear in dict when set."""
        from src.telemetry.metrics import FrameMetrics
        
        metrics = FrameMetrics(
            ip_acquisition_latency_ms=25.5,
            ip_reconnect_count=2,
            ip_downtime_ms=1500.0,
        )
        
        result = metrics.to_dict()
        
        assert result['ip_acquisition_latency_ms'] == 25.5
        assert result['ip_reconnect_count'] == 2
        assert result['ip_downtime_ms'] == 1500.0
    
    def test_frame_metrics_to_dict_omits_none_ip(self):
        """Test IP metrics omitted from dict when None."""
        from src.telemetry.metrics import FrameMetrics
        
        metrics = FrameMetrics()  # All IP fields are None
        
        result = metrics.to_dict()
        
        assert 'ip_acquisition_latency_ms' not in result
        assert 'ip_reconnect_count' not in result
        assert 'ip_downtime_ms' not in result
    
    def test_telemetry_record_has_ip_fields(self):
        """Test TelemetryRecord has IP-specific fields."""
        from src.telemetry.logger import TelemetryRecord
        
        record = TelemetryRecord(
            timestamp="2024-01-01T00:00:00Z",
            frame_seq=1,
            capture_fps=15.0,
            capture_latency_ms=10.0,
            lane_latency_ms=5.0,
            yolo_latency_ms=None,
            yolo_skipped=True,
            decision_latency_ms=1.0,
            alert_type=None,
            alert_latency_ms=None,
            cpu_temperature_c=None,
            dropped_frames=0,
            lane_valid=True,
            detections_count=0,
            collision_risks=0,
            ip_acquisition_latency_ms=20.0,
            ip_reconnect_count=1,
            ip_downtime_ms=500.0,
        )
        
        assert record.ip_acquisition_latency_ms == 20.0
        assert record.ip_reconnect_count == 1
        assert record.ip_downtime_ms == 500.0
    
    def test_telemetry_record_to_json_includes_ip(self):
        """Test IP metrics appear in JSON when set."""
        from src.telemetry.logger import TelemetryRecord
        import json
        
        record = TelemetryRecord(
            timestamp="2024-01-01T00:00:00Z",
            frame_seq=1,
            capture_fps=15.0,
            capture_latency_ms=10.0,
            lane_latency_ms=5.0,
            yolo_latency_ms=None,
            yolo_skipped=True,
            decision_latency_ms=1.0,
            alert_type=None,
            alert_latency_ms=None,
            cpu_temperature_c=None,
            dropped_frames=0,
            lane_valid=True,
            detections_count=0,
            collision_risks=0,
            ip_acquisition_latency_ms=20.0,
            ip_reconnect_count=1,
            ip_downtime_ms=500.0,
        )
        
        json_str = record.to_json()
        data = json.loads(json_str)
        
        assert 'ip_acquisition_latency_ms' in data
        assert data['ip_acquisition_latency_ms'] == 20.0
    
    def test_telemetry_record_to_json_omits_none_ip(self):
        """Test IP metrics omitted from JSON when None."""
        from src.telemetry.logger import TelemetryRecord
        import json
        
        record = TelemetryRecord(
            timestamp="2024-01-01T00:00:00Z",
            frame_seq=1,
            capture_fps=15.0,
            capture_latency_ms=10.0,
            lane_latency_ms=5.0,
            yolo_latency_ms=None,
            yolo_skipped=True,
            decision_latency_ms=1.0,
            alert_type=None,
            alert_latency_ms=None,
            cpu_temperature_c=None,
            dropped_frames=0,
            lane_valid=True,
            detections_count=0,
            collision_risks=0,
        )
        
        json_str = record.to_json()
        data = json.loads(json_str)
        
        assert 'ip_acquisition_latency_ms' not in data
        assert 'ip_reconnect_count' not in data
        assert 'ip_downtime_ms' not in data
