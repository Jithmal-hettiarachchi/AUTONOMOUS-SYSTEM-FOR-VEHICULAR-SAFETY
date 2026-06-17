#!/usr/bin/env python3
"""
Vehicle Safety Alert System - Entry Point (Windows Compatible)
"""

import sys
import platform
from pathlib import Path

# --- CROSS-PLATFORM BYPASS ---
# If running on a Windows PC, safely mock the Raspberry Pi hardware
# so the AI and display can run without crashing.
if platform.system() == "Windows":
    from unittest.mock import MagicMock
    sys.modules['RPi'] = MagicMock()
    sys.modules['RPi.GPIO'] = MagicMock()
    sys.modules['picamera2'] = MagicMock()
    sys.modules['smbus'] = MagicMock() # Bypasses I2C/LiDAR errors
    print("[INFO] Windows PC detected. Bypassing Pi hardware.")
# -----------------------------

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())