# state_serializer.py - Efficient state serialization (Fixed)
import json
import pickle
import gzip
from typing import Any, Dict
from datetime import datetime
import numpy as np
import logging
import msgpack

logger = logging.getLogger(__name__)

class StateSerializer:
    """Handles efficient serialization of complex AI states"""
    
    # Fix for serialize_msgpack method in state_serializer.py

@staticmethod
def serialize_msgpack(data: Any, compress: bool = True) -> bytes:
    """Serialize data using MessagePack for better performance"""
    try:
        def default(obj):
            if isinstance(obj, np.ndarray):
                return {"_type": "numpy.ndarray", "dtype": str(obj.dtype),
                       "shape": obj.shape, "data": obj.tolist()}
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, datetime):
                return {"_type": "datetime", "value": obj.isoformat()}
            raise TypeError(f"Object of type {type(obj)} is not serializable")
        
        # Pack the data
        packed = msgpack.packb(data, default=default, use_bin_type=True)
        
        # Ensure packed is bytes (for type checker)
        if packed is None:
            raise ValueError("msgpack.packb returned None")
        if not isinstance(packed, bytes):
            # Convert to bytes if somehow it's not
            packed = bytes(packed) if hasattr(packed, '__bytes__') else str(packed).encode('utf-8')
        
        # Now packed is guaranteed to be bytes
        if compress:
            compressed = gzip.compress(packed, compresslevel=6)
            # compressed is guaranteed to be bytes by gzip.compress
            result = b"msgpack_gz:" + compressed
        else:
            result = b"msgpack:" + packed
        
        return result
            
    except Exception as e:
        logger.error(f"Error serializing with msgpack: {e}")
        raise
