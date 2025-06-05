# state_manager.py - State Persistence and Backup Management (Fixed)
import os
import json
import pickle
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)

class StateManager:
    """Manages state persistence, backup, and recovery"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = Path(config.get('state_path', 'data/states'))
        self.backup_path = Path(config.get('backup_path', 'data/backups'))
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup settings
        self.backup_interval_hours = config.get('backup_interval_hours', 6)
        self.max_local_backups = config.get('max_local_backups', 7)
        self.compression_enabled = config.get('compression', True)
        
    async def save_complete_state(self, claude_interface) -> str:
        """Save complete system state"""
        timestamp = datetime.now()
        state_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Import ConsciousnessState here to avoid circular imports
        from core.living_consciousness import ConsciousnessState
        
        # Gather all state components
        complete_state = {
            'state_id': state_id,
            'timestamp': timestamp.isoformat(),
            'version': '1.0',
            
            # Consciousness state
            'consciousness': {
                'state': claude_interface.consciousness.state.to_dict(),
                'memory_buffer': list(claude_interface.consciousness.memory_buffer),
                'coherence_history': []  # Could track history if needed
            },
            
            # Personality state
            'personality': {
                'traits': claude_interface.personality.profile.get_traits(),
                'trait_bounds': claude_interface.personality.profile.trait_bounds,
                'interaction_history': claude_interface.personality.interaction_history[-100:]  # Last 100
            },
            
            # Emotional state
            'emotions': {
                'current_state': claude_interface.emotions.current_state.__dict__,
                'baseline_state': claude_interface.emotions.baseline_state.__dict__,
                'state_history': [s.__dict__ for s in claude_interface.emotions.state_history]
            },
            
            # Active sessions
            'sessions': {
                session_id: {
                    'user_id': context.user_id,
                    'interaction_count': context.consciousness_state.interaction_count,
                    'last_interaction': context.consciousness_state.last_interaction.isoformat() 
                        if context.consciousness_state.last_interaction else None
                }
                for session_id, context in claude_interface.active_sessions.items()
            },
            
            # Memory statistics
            'memory_stats': claude_interface.memory_manager.get_statistics()
        }
        
        # Save state
        state_file = self.base_path / f"complete_state_{state_id}.json"
        
        if self.compression_enabled:
            # Save compressed
            with gzip.open(state_file.with_suffix('.json.gz'), 'wt') as f:
                json.dump(complete_state, f, indent=2)
        else:
            # Save uncompressed
            with open(state_file, 'w') as f:
                json.dump(complete_state, f, indent=2)
        
        logger.info(f"Saved complete state: {state_id}")
        
        # Trigger backup if needed
        await self._check_and_backup(state_id)
        
        return state_id
    
    async def load_complete_state(self, state_id: str, claude_interface) -> bool:
        """Load a complete system state"""
        try:
            # Import ConsciousnessState here to avoid circular imports
            from core.living_consciousness import ConsciousnessState
            
            # Find state file
            state_file = self._find_state_file(state_id)
            if not state_file:
                logger.error(f"State file not found: {state_id}")
                return False
            
            # Load state data
            if state_file.suffix == '.gz':
                with gzip.open(state_file, 'rt') as f:
                    state_data = json.load(f)
            else:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
            
            # Restore consciousness state
            claude_interface.consciousness.state = ConsciousnessState.from_dict(
                state_data['consciousness']['state']
            )
            claude_interface.consciousness.memory_buffer.clear()
            claude_interface.consciousness.memory_buffer.extend(
                state_data['consciousness']['memory_buffer']
            )
            
            # Restore personality
            personality_traits = state_data['personality']['traits']
            for trait, value in personality_traits.items():
                setattr(claude_interface.personality.profile, trait, value)
            
            # Restore emotions
            emotional_state = state_data['emotions']['current_state']
            claude_interface.emotions.current_state.pleasure = emotional_state['pleasure']
            claude_interface.emotions.current_state.arousal = emotional_state['arousal']
            claude_interface.emotions.current_state.dominance = emotional_state['dominance']
            
            logger.info(f"Loaded complete state: {state_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state {state_id}: {e}")
            return False
    
    def _find_state_file(self, state_id: str) -> Optional[Path]:
        """Find a state file by ID"""
        # Check for compressed version first
        compressed = self.base_path / f"complete_state_{state_id}.json.gz"
        if compressed.exists():
            return compressed
        
        # Check for uncompressed version
        uncompressed = self.base_path / f"complete_state_{state_id}.json"
        if uncompressed.exists():
            return uncompressed
        
        # Check in backups
        for backup_file in self.backup_path.glob(f"*{state_id}*"):
            return backup_file
        
        return None
    
    async def _check_and_backup(self, state_id: str):
        """Check if backup is needed and perform it"""
        # Get list of existing backups
        backups = sorted(self.backup_path.glob("backup_*.tar.gz"))
        
        # Check if backup is needed
        if not backups or (datetime.now() - datetime.fromtimestamp(
            backups[-1].stat().st_mtime
        )) > timedelta(hours=self.backup_interval_hours):
            await self.create_backup(state_id)
    
    async def create_backup(self, state_id: str) -> str:
        """Create a backup of all states"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{backup_timestamp}.tar.gz"
        backup_file = self.backup_path / backup_name
        
        try:
            # Create tar.gz of states directory
            import tarfile
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(self.base_path, arcname="states")
            
            logger.info(f"Created backup: {backup_name}")
            
            # Clean old backups
            await self._cleanup_old_backups()
            
            return backup_name
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            if backup_file.exists():
                backup_file.unlink()
            raise
    
    async def _cleanup_old_backups(self):
        """Remove old local backups"""
        backups = sorted(self.backup_path.glob("backup_*.tar.gz"))
        
        if len(backups) > self.max_local_backups:
            for old_backup in backups[:-self.max_local_backups]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")
    
    async def list_available_states(self) -> List[Dict[str, Any]]:
        """List all available states"""
        states = []
        
        # List local states
        for state_file in self.base_path.glob("complete_state_*.json*"):
            stat = state_file.stat()
            state_id = state_file.stem.replace('complete_state_', '').replace('.json', '')
            
            states.append({
                'state_id': state_id,
                'location': 'local',
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'compressed': state_file.suffix == '.gz'
            })
        
        return sorted(states, key=lambda x: x['modified'], reverse=True)
    
    async def export_user_data(self, user_id: str, output_path: str):
        """Export all data for a specific user"""
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'states': [],
            'memories': []
        }
        
        # Find all states for this user
        for state_file in self.base_path.glob("*_state.json*"):
            if user_id in state_file.stem:
                with open(state_file, 'r') as f:
                    export_data['states'].append(json.load(f))
        
        # Save export
        with gzip.open(output_path, 'wt') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported user data for {user_id} to {output_path}")
    
    async def automatic_backup_loop(self, claude_interface):
        """Run automatic backups periodically"""
        while True:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)
                state_id = await self.save_complete_state(claude_interface)
                await self.create_backup(state_id)
            except Exception as e:
                logger.error(f"Error in automatic backup: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes