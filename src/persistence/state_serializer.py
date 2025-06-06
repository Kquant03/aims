# state_serializer.py - Efficient state serialization
import json
import pickle
import gzip
from typing import Any, Dict
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StateSerializer:
    """Handles efficient serialization of complex AI states"""
    
    @staticmethod
    def serialize_msgpack(data: Any, compress: bool = True) -> bytes:
        """Serialize data using MessagePack for better performance"""
        try:
            import msgpack
            
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
            
            if packed is None:
                raise ValueError("msgpack.packb returned None")
            
            # Ensure packed is bytes
            if not isinstance(packed, bytes):
                packed = bytes(packed) if hasattr(packed, '__bytes__') else str(packed).encode('utf-8')
            
            # Compress if requested
            if compress:
                compressed = gzip.compress(packed, compresslevel=6)
                result = b"msgpack_gz:" + compressed
            else:
                result = b"msgpack:" + packed
            
            return result
                
        except Exception as e:
            logger.error(f"Error serializing with msgpack: {e}")
            raise
    
    @staticmethod
    def serialize_json(data: Any, compress: bool = True) -> bytes:
        """Serialize data as JSON"""
        json_str = json.dumps(data, default=str)
        json_bytes = json_str.encode('utf-8')
        
        if compress:
            return b"json_gz:" + gzip.compress(json_bytes)
        return b"json:" + json_bytes
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize data based on format prefix"""
        if data.startswith(b"msgpack_gz:"):
            import msgpack
            compressed = data[11:]
            decompressed = gzip.decompress(compressed)
            return msgpack.unpackb(decompressed, raw=False)
        elif data.startswith(b"msgpack:"):
            import msgpack
            return msgpack.unpackb(data[8:], raw=False)
        elif data.startswith(b"json_gz:"):
            compressed = data[8:]
            decompressed = gzip.decompress(compressed)
            return json.loads(decompressed.decode('utf-8'))
        elif data.startswith(b"json:"):
            return json.loads(data[5:].decode('utf-8'))
        else:
            # Try pickle as fallback
            return pickle.loads(data)
