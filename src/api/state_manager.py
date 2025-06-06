# state_manager.py - State Persistence and Backup Management (FIXED)
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
import tarfile

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
                'interaction_history': claude_interface.personality.interaction_history[-100:] if hasattr(claude_interface.personality, 'interaction_history') else []
            },
            
            # Emotional state
            'emotions': {
                'current_state': claude_interface.emotions.current_state.__dict__,
                'baseline_state': claude_interface.emotions.baseline_state.__dict__,
                'state_history': [s.__dict__ for s in claude_interface.emotions.state_history] if hasattr(claude_interface.emotions, 'state_history') else []
            },
            
            # Active sessions
            'sessions': {
                session_id: {
                    'user_id': context.user_id,
                    'interaction_count': context.consciousness_state.interaction_count,
                    'last_interaction': context.consciousness_state.last_interaction.isoformat() 
                        if context.consciousness_state.last_interaction else None,
                    'recent_memories': context.recent_memories
                }
                for session_id, context in claude_interface.active_sessions.items()
            },
            
            # Attention patterns if available
            'attention_patterns': dict(list(claude_interface.attention_agent.attention_patterns.items())[-50:]) if hasattr(claude_interface.attention_agent, 'attention_patterns') else {},
            
            # Memory statistics
            'memory_stats': claude_interface.memory_system.get_statistics() if hasattr(claude_interface.memory_system, 'get_statistics') else {}
        }
        
        # Save state
        state_file = self.base_path / f"complete_state_{state_id}.json"
        
        if self.compression_enabled:
            # Save compressed
            with gzip.open(state_file.with_suffix('.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(complete_state, f, indent=2)
        else:
            # Save uncompressed
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(complete_state, f, indent=2)
        
        logger.info(f"Saved complete state: {state_id}")
        
        # Trigger backup if needed
        await self._check_and_backup(state_id)
        
        return state_id
    
    async def load_complete_state(self, state_id: str, claude_interface) -> bool:
        """Load a complete system state"""
        try:
            # Import ConsciousnessState from the correct module
            from src.core.consciousness_state import ConsciousnessState
            
            # Find state file
            state_file = self._find_state_file(state_id)
            if not state_file:
                logger.error(f"State file not found: {state_id}")
                return False
            
            # Load state data
            if state_file.suffix == '.gz':
                with gzip.open(state_file, 'rt', encoding='utf-8') as f:
                    state_data = json.load(f)
            else:
                with open(state_file, 'r', encoding='utf-8') as f:
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
                if hasattr(claude_interface.personality.profile, trait):
                    setattr(claude_interface.personality.profile, trait, value)
            
            # Restore emotions
            emotional_state = state_data['emotions']['current_state']
            claude_interface.emotions.current_state.pleasure = emotional_state['pleasure']
            claude_interface.emotions.current_state.arousal = emotional_state['arousal']
            claude_interface.emotions.current_state.dominance = emotional_state['dominance']
            
            # Restore baseline emotional state
            baseline_state = state_data['emotions']['baseline_state']
            claude_interface.emotions.baseline_state.pleasure = baseline_state['pleasure']
            claude_interface.emotions.baseline_state.arousal = baseline_state['arousal']
            claude_interface.emotions.baseline_state.dominance = baseline_state['dominance']
            
            # Restore attention patterns if available
            if 'attention_patterns' in state_data and hasattr(claude_interface.attention_agent, 'attention_patterns'):
                claude_interface.attention_agent.attention_patterns.update(state_data['attention_patterns'])
            
            logger.info(f"Loaded complete state: {state_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state {state_id}: {e}", exc_info=True)
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
            'memories': [],
            'sessions': []
        }
        
        # Find all states for this user
        for state_file in self.base_path.glob("*_state.json*"):
            if user_id in state_file.stem:
                try:
                    if state_file.suffix == '.gz':
                        with gzip.open(state_file, 'rt', encoding='utf-8') as f:
                            state_content = json.load(f)
                    else:
                        with open(state_file, 'r', encoding='utf-8') as f:
                            state_content = json.load(f)
                    export_data['states'].append(state_content)
                except Exception as e:
                    logger.error(f"Error reading state file {state_file}: {e}")
        
        # Find complete states that include this user's sessions
        for state_file in self.base_path.glob("complete_state_*.json*"):
            try:
                if state_file.suffix == '.gz':
                    with gzip.open(state_file, 'rt', encoding='utf-8') as f:
                        complete_state = json.load(f)
                else:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        complete_state = json.load(f)
                        
                # Check if user has sessions in this state
                for session_id, session_data in complete_state.get('sessions', {}).items():
                    if session_data.get('user_id') == user_id:
                        export_data['sessions'].append({
                            'state_id': complete_state['state_id'],
                            'session_id': session_id,
                            'session_data': session_data,
                            'timestamp': complete_state['timestamp']
                        })
            except Exception as e:
                logger.error(f"Error reading complete state file {state_file}: {e}")
        
        # Save export
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported user data for {user_id} to {output_path}")
        return output_path
    
    async def import_user_data(self, import_path: str, claude_interface) -> bool:
        """Import user data from an export file"""
        try:
            with gzip.open(import_path, 'rt', encoding='utf-8') as f:
                import_data = json.load(f)
            
            user_id = import_data['user_id']
            
            # Import user states
            for state_data in import_data.get('states', []):
                try:
                    # Save the state
                    state_file = self.base_path / f"{user_id}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Error importing state: {e}")
            
            logger.info(f"Imported user data for {user_id} from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing user data: {e}")
            return False
    
    async def automatic_backup_loop(self, claude_interface):
        """Run automatic backups periodically"""
        while True:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)
                state_id = await self.save_complete_state(claude_interface)
                await self.create_backup(state_id)
                logger.info(f"Automatic backup completed: {state_id}")
            except Exception as e:
                logger.error(f"Error in automatic backup: {e}", exc_info=True)
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def get_state_summary(self, state_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a saved state without loading it"""
        try:
            state_file = self._find_state_file(state_id)
            if not state_file:
                return None
            
            # Load state data
            if state_file.suffix == '.gz':
                with gzip.open(state_file, 'rt', encoding='utf-8') as f:
                    state_data = json.load(f)
            else:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
            
            # Create summary
            summary = {
                'state_id': state_id,
                'timestamp': state_data['timestamp'],
                'version': state_data.get('version', 'unknown'),
                'consciousness': {
                    'coherence': state_data['consciousness']['state']['global_coherence'],
                    'interaction_count': state_data['consciousness']['state']['interaction_count'],
                    'working_memory_size': len(state_data['consciousness']['memory_buffer'])
                },
                'sessions': {
                    'count': len(state_data.get('sessions', {})),
                    'users': list(set(s['user_id'] for s in state_data.get('sessions', {}).values()))
                },
                'emotional_state': state_data['emotions']['current_state']
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting state summary: {e}")
            return None