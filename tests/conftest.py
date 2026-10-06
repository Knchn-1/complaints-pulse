"""
Pytest configuration and session-wide initialization.
"""

import os
# Ensure torch DLLs initialize first on Windows
try:
    import torch
except Exception:
    pass
