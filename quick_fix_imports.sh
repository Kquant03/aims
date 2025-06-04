#!/bin/bash
# Quick fix for import issues

echo "Fixing import issues..."

# Fix 1: Add List to websocket_server.py imports
sed -i 's/from typing import Set, Dict, Any, Optional/from typing import Set, Dict, List, Any, Optional/' src/api/websocket_server.py

# Fix 2: Fix NVML decode issue in gpu_optimizer.py
sed -i "s/pynvml.nvmlDeviceGetName(handle).decode()/str(pynvml.nvmlDeviceGetName(handle))/" src/utils/gpu_optimizer.py

# Fix 3: Also fix Tuple import if needed
sed -i 's/from typing import Dict, List, Optional, Any$/from typing import Dict, List, Optional, Any, Tuple/' src/core/memory_manager.py

echo "✅ Fixed import issues!"
echo ""
echo "Starting AIMS..."
./venv/bin/python -m src.main