# backup_manager.py - Fixed local backup implementation
import os
import json
import pickle
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class BackupManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_root = Path(config.get('backup_path', 'data/backups'))
        self.state_root = Path(config.get('state_path', 'data/states'))
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Backup settings
        self.compression_level = config.get('compression_level', 6)
        self.max_backups = config.get('max_backups', 10)
    
    async def create_backup(self, claude_interface, backup_name: Optional[str] = None) -> str:
        """Create a local backup of the system state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_id = f"backup_{timestamp}"
        if backup_name:
            backup_id += f"_{backup_name}"
        
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(exist_ok=True)
        
        try:
            # Save consciousness state
            with open(backup_dir / "consciousness.pkl", 'wb') as f:
                pickle.dump({
                    'state': claude_interface.consciousness.state,
                    'memory_buffer': list(claude_interface.consciousness.memory_buffer)
                }, f)
            
            # Save personality state
            with open(backup_dir / "personality.json", 'w') as f:
                json.dump({
                    'profile': claude_interface.personality.profile.get_traits(),
                    'interaction_history': claude_interface.personality.interaction_history[-100:]
                }, f, indent=2)
            
            # Save emotional state
            with open(backup_dir / "emotions.json", 'w') as f:
                json.dump({
                    'current': claude_interface.emotions.current_state.__dict__,
                    'baseline': claude_interface.emotions.baseline_state.__dict__
                }, f, indent=2)
            
            # Create compressed archive
            archive_path = self.backup_root / f"{backup_id}.tar.gz"
            with tarfile.open(archive_path, 'w:gz', compresslevel=self.compression_level) as tar:
                tar.add(backup_dir, arcname=backup_id)
            
            # Remove uncompressed directory
            shutil.rmtree(backup_dir)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            logger.info(f"Created backup: {backup_id}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise
    
    def _cleanup_old_backups(self):
        """Remove old backups keeping only the most recent ones"""
        backups = sorted(self.backup_root.glob("backup_*.tar.gz"))
        
        if len(backups) > self.max_backups:
            for old_backup in backups[:-self.max_backups]:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")