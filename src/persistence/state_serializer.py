# state_serializer.py - Efficient state serialization
import json
import pickle
import gzip
import base64
from typing import Any, Dict, Union
from datetime import datetime
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class StateSerializer:
    """Handles efficient serialization of complex AI states"""
    
    @staticmethod
    def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
        """Serialize PyTorch tensor"""
        return {
            'type': 'torch_tensor',
            'dtype': str(tensor.dtype),
            'shape': list(tensor.shape),
            'device': str(tensor.device),
            'data': tensor.cpu().numpy().tolist()
        }
    
    @staticmethod
    def deserialize_tensor(data: Dict[str, Any]) -> torch.Tensor:
        """Deserialize PyTorch tensor"""
        numpy_array = np.array(data['data'])
        tensor = torch.from_numpy(numpy_array)
        
        # Convert dtype if needed
        if 'float' in data['dtype']:
            tensor = tensor.float()
        elif 'int' in data['dtype']:
            tensor = tensor.long()
        
        # Move to device if specified and available
        if 'cuda' in data['device'] and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor
    
    @staticmethod
    def serialize_state(state: Any, compress: bool = True) -> bytes:
        """Serialize any state object to bytes"""
        try:
            # Try JSON first (most compatible)
            if hasattr(state, 'to_dict'):
                state_dict = state.to_dict()
                json_str = json.dumps(state_dict, default=str)
                data = json_str.encode('utf-8')
                format_type = 'json'
            else:
                # Fall back to pickle
                data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                format_type = 'pickle'
            
            # Compress if requested
            if compress:
                data = gzip.compress(data, compresslevel=6)
                format_type += '_gz'
            
            # Add format header
            header = f"{format_type}:".encode('utf-8')
            return header + data
            
        except Exception as e:
            logger.error(f"Error serializing state: {e}")
            raise
    
    @staticmethod
    def deserialize_state(data: bytes) -> Any:
        """Deserialize state from bytes"""
        try:
            # Parse format header
            header_end = data.find(b':')
            if header_end == -1:
                raise ValueError("Invalid serialized data format")
            
            format_type = data[:header_end].decode('utf-8')
            payload = data[header_end + 1:]
            
            # Decompress if needed
            if format_type.endswith('_gz'):
                payload = gzip.decompress(payload)
                format_type = format_type[:-3]
            
            # Deserialize based on format
            if format_type == 'json':
                json_str = payload.decode('utf-8')
                return json.loads(json_str)
            elif format_type == 'pickle':
                return pickle.loads(payload)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Error deserializing state: {e}")
            raise
    
    @staticmethod
    def create_state_diff(old_state: Dict, new_state: Dict) -> Dict[str, Any]:
        """Create a diff between two states for efficient storage"""
        diff = {
            'timestamp': datetime.now().isoformat(),
            'changes': {}
        }
        
        # Find changed keys
        all_keys = set(old_state.keys()) | set(new_state.keys())
        
        for key in all_keys:
            if key not in old_state:
                diff['changes'][key] = {'action': 'added', 'value': new_state[key]}
            elif key not in new_state:
                diff['changes'][key] = {'action': 'removed'}
            elif old_state[key] != new_state[key]:
                diff['changes'][key] = {
                    'action': 'modified',
                    'old_value': old_state[key],
                    'new_value': new_state[key]
                }
        
        return diff
    
    @staticmethod
    def apply_state_diff(base_state: Dict, diff: Dict[str, Any]) -> Dict:
        """Apply a diff to reconstruct a state"""
        result_state = base_state.copy()
        
        for key, change in diff['changes'].items():
            if change['action'] == 'added':
                result_state[key] = change['value']
            elif change['action'] == 'removed':
                result_state.pop(key, None)
            elif change['action'] == 'modified':
                result_state[key] = change['new_value']
        
        return result_state