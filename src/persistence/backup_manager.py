# backup_manager.py - Comprehensive backup and recovery system
import os
import json
import pickle
import shutil
import tarfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
import aioboto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class BackupManager:
    """Manages system backups with versioning and integrity checks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_root = Path(config.get('backup_path', 'data/backups'))
        self.state_root = Path(config.get('state_path', 'data/states'))
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Backup settings
        self.compression_level = config.get('compression_level', 6)
        self.verify_backups = config.get('verify_backups', True)
        self.encryption_enabled = config.get('encryption_enabled', False)

    async def _init_s3_client(self):
        """Initialize S3 client if configured"""
        if self.s3_config.get('enabled'):
            self.s3_session = aioboto3.Session()
            return True
        return False
    
    async def upload_to_s3(self, local_path: Path, s3_key: str, 
                          metadata: Optional[Dict[str, str]] = None):
        """Upload file to S3 with retry logic"""
        if not self.s3_config.get('enabled'):
            logger.warning("S3 backup not enabled")
            return False
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.s3_session.client(
                    's3',
                    aws_access_key_id=self.s3_config['access_key'],
                    aws_secret_access_key=self.s3_config['secret_key'],
                    region_name=self.s3_config.get('region', 'us-east-1')
                ) as s3:
                    # Add metadata
                    extra_args = {}
                    if metadata:
                        extra_args['Metadata'] = metadata
                    
                    # Upload with progress callback
                    file_size = local_path.stat().st_size
                    
                    with open(local_path, 'rb') as f:
                        await s3.upload_fileobj(
                            f,
                            self.s3_config['bucket'],
                            s3_key,
                            ExtraArgs=extra_args,
                            Callback=self._upload_progress_callback(file_size)
                        )
                    
                    logger.info(f"Successfully uploaded {local_path.name} to S3: {s3_key}")
                    return True
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchBucket':
                    logger.error(f"S3 bucket {self.s3_config['bucket']} does not exist")
                    return False
                elif attempt < max_retries - 1:
                    logger.warning(f"S3 upload failed (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to upload to S3 after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error during S3 upload: {e}")
                return False
    
    async def download_from_s3(self, s3_key: str, local_path: Path) -> bool:
        """Download file from S3"""
        if not self.s3_config.get('enabled'):
            return False
        
        try:
            async with self.s3_session.client(
                's3',
                aws_access_key_id=self.s3_config['access_key'],
                aws_secret_access_key=self.s3_config['secret_key'],
                region_name=self.s3_config.get('region', 'us-east-1')
            ) as s3:
                # Get object size first
                response = await s3.head_object(
                    Bucket=self.s3_config['bucket'],
                    Key=s3_key
                )
                file_size = response['ContentLength']
                
                # Download with progress
                with open(local_path, 'wb') as f:
                    await s3.download_fileobj(
                        self.s3_config['bucket'],
                        s3_key,
                        f,
                        Callback=self._download_progress_callback(file_size)
                    )
                
                logger.info(f"Successfully downloaded {s3_key} from S3")
                return True
                
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"S3 object not found: {s3_key}")
            else:
                logger.error(f"S3 download error: {e}")
            return False
    
    async def list_s3_backups(self) -> List[Dict[str, Any]]:
        """List all backups in S3"""
        if not self.s3_config.get('enabled'):
            return []
        
        backups = []
        
        try:
            async with self.s3_session.client(
                's3',
                aws_access_key_id=self.s3_config['access_key'],
                aws_secret_access_key=self.s3_config['secret_key'],
                region_name=self.s3_config.get('region', 'us-east-1')
            ) as s3:
                paginator = s3.get_paginator('list_objects_v2')
                
                async for page in paginator.paginate(
                    Bucket=self.s3_config['bucket'],
                    Prefix='aims-backups/'
                ):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            backups.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'storage_class': obj.get('StorageClass', 'STANDARD')
                            })
        
        except Exception as e:
            logger.error(f"Error listing S3 backups: {e}")
        
        return backups
    
    def _upload_progress_callback(self, file_size: int):
        """Create upload progress callback"""
        uploaded = 0
        
        def callback(bytes_amount):
            nonlocal uploaded
            uploaded += bytes_amount
            percentage = (uploaded / file_size) * 100
            logger.debug(f"Upload progress: {percentage:.1f}%")
        
        return callback
    
    def _download_progress_callback(self, file_size: int):
        """Create download progress callback"""
        downloaded = 0
        
        def callback(bytes_amount):
            nonlocal downloaded
            downloaded += bytes_amount
            percentage = (downloaded / file_size) * 100
            logger.debug(f"Download progress: {percentage:.1f}%")
        
        return callback
        
    async def create_checkpoint(self, claude_interface, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a complete system checkpoint"""
        timestamp = datetime.now()
        checkpoint_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        if checkpoint_name:
            checkpoint_id = f"{checkpoint_id}_{checkpoint_name}"
        
        checkpoint_dir = self.backup_root / f"checkpoint_{checkpoint_id}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        try:
            # Save all components
            components_saved = []
            
            # 1. Consciousness state
            consciousness_file = checkpoint_dir / "consciousness_state.pkl"
            with open(consciousness_file, 'wb') as f:
                pickle.dump({
                    'state': claude_interface.consciousness.state,
                    'memory_buffer': list(claude_interface.consciousness.memory_buffer),
                    'device': str(claude_interface.consciousness.device)
                }, f)
            components_saved.append('consciousness')
            
            # 2. Personality state
            personality_file = checkpoint_dir / "personality_state.json"
            with open(personality_file, 'w') as f:
                json.dump({
                    'profile': claude_interface.personality.profile.get_traits(),
                    'trait_bounds': claude_interface.personality.profile.trait_bounds,
                    'interaction_history': claude_interface.personality.interaction_history[-1000:]
                }, f, indent=2)
            components_saved.append('personality')
            
            # 3. Emotional state
            emotion_file = checkpoint_dir / "emotional_state.json"
            with open(emotion_file, 'w') as f:
                json.dump({
                    'current': claude_interface.emotions.current_state.__dict__,
                    'baseline': claude_interface.emotions.baseline_state.__dict__,
                    'history': [s.__dict__ for s in claude_interface.emotions.state_history]
                }, f, indent=2)
            components_saved.append('emotions')
            
            # 4. Active sessions
            sessions_file = checkpoint_dir / "active_sessions.json"
            with open(sessions_file, 'w') as f:
                sessions_data = {}
                for session_id, context in claude_interface.active_sessions.items():
                    sessions_data[session_id] = {
                        'user_id': context.user_id,
                        'consciousness_state': context.consciousness_state.to_dict(),
                        'recent_memories': context.recent_memories,
                        'personality_modifiers': context.personality_modifiers,
                        'emotional_context': context.emotional_context
                    }
                json.dump(sessions_data, f, indent=2)
            components_saved.append('sessions')
            
            # 5. Memory index snapshot
            memory_snapshot = await self._create_memory_snapshot(claude_interface.memory_manager)
            memory_file = checkpoint_dir / "memory_snapshot.json"
            with open(memory_file, 'w') as f:
                json.dump(memory_snapshot, f, indent=2)
            components_saved.append('memory_snapshot')
            
            # Create metadata
            metadata = {
                'checkpoint_id': checkpoint_id,
                'timestamp': timestamp.isoformat(),
                'components': components_saved,
                'aims_version': self.config.get('version', '1.0.0'),
                'total_interactions': claude_interface.consciousness.state.interaction_count,
                'checksum': await self._calculate_checkpoint_checksum(checkpoint_dir)
            }
            
            metadata_file = checkpoint_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create compressed archive
            if self.config.get('compress_checkpoints', True):
                archive_path = await self._compress_checkpoint(checkpoint_dir, checkpoint_id)
                # Remove uncompressed directory
                shutil.rmtree(checkpoint_dir)
                metadata['archive_path'] = str(archive_path)
            
            logger.info(f"Created checkpoint: {checkpoint_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
            # Cleanup on error
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str, claude_interface) -> bool:
        """Restore system from a checkpoint"""
        try:
            # Find checkpoint
            checkpoint_path = self._find_checkpoint(checkpoint_id)
            if not checkpoint_path:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            # Extract if compressed
            if checkpoint_path.suffix == '.tar.gz':
                checkpoint_dir = await self._extract_checkpoint(checkpoint_path)
            else:
                checkpoint_dir = checkpoint_path
            
            # Verify integrity
            if self.verify_backups:
                if not await self._verify_checkpoint_integrity(checkpoint_dir):
                    logger.error("Checkpoint integrity check failed")
                    return False
            
            # Restore components
            # 1. Consciousness state
            consciousness_file = checkpoint_dir / "consciousness_state.pkl"
            with open(consciousness_file, 'rb') as f:
                consciousness_data = pickle.load(f)
                claude_interface.consciousness.state = consciousness_data['state']
                claude_interface.consciousness.memory_buffer.clear()
                claude_interface.consciousness.memory_buffer.extend(consciousness_data['memory_buffer'])
            
            # 2. Personality state
            personality_file = checkpoint_dir / "personality_state.json"
            with open(personality_file, 'r') as f:
                personality_data = json.load(f)
                for trait, value in personality_data['profile'].items():
                    setattr(claude_interface.personality.profile, trait, value)
                claude_interface.personality.interaction_history = personality_data['interaction_history']
            
            # 3. Emotional state
            emotion_file = checkpoint_dir / "emotional_state.json"
            with open(emotion_file, 'r') as f:
                emotion_data = json.load(f)
                for key, value in emotion_data['current'].items():
                    setattr(claude_interface.emotions.current_state, key, value)
                for key, value in emotion_data['baseline'].items():
                    setattr(claude_interface.emotions.baseline_state, key, value)
            
            # 4. Sessions (optional - may want to start fresh)
            if self.config.get('restore_sessions', False):
                sessions_file = checkpoint_dir / "active_sessions.json"
                if sessions_file.exists():
                    with open(sessions_file, 'r') as f:
                        sessions_data = json.load(f)
                        # Restore sessions logic here
            
            logger.info(f"Successfully restored checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring checkpoint: {e}")
            return False
    
    async def _create_memory_snapshot(self, memory_manager) -> Dict[str, Any]:
        """Create a snapshot of memory statistics and key memories"""
        stats = memory_manager.get_statistics()
        
        # Get sample of important memories
        important_memories = await memory_manager.retrieve_memories(
            query="important user interactions",
            k=100
        )
        
        snapshot = {
            'statistics': stats,
            'important_memory_ids': [m.id for m in important_memories],
            'memory_count': stats.get('total_memories', 0),
            'snapshot_time': datetime.now().isoformat()
        }
        
        return snapshot
    
    async def _calculate_checkpoint_checksum(self, checkpoint_dir: Path) -> str:
        """Calculate checksum for checkpoint verification"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(checkpoint_dir.rglob('*')):
            if file_path.is_file() and file_path.name != 'metadata.json':
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    async def _compress_checkpoint(self, checkpoint_dir: Path, checkpoint_id: str) -> Path:
        """Compress checkpoint directory"""
        archive_path = self.backup_root / f"{checkpoint_id}.tar.gz"
        
        with tarfile.open(archive_path, 'w:gz', compresslevel=self.compression_level) as tar:
            tar.add(checkpoint_dir, arcname=checkpoint_id)
        
        return archive_path
    
    async def _extract_checkpoint(self, archive_path: Path) -> Path:
        """Extract compressed checkpoint"""
        extract_dir = self.backup_root / 'temp_extract'
        extract_dir.mkdir(exist_ok=True)
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        # Find the checkpoint directory
        checkpoint_dirs = list(extract_dir.iterdir())
        if checkpoint_dirs:
            return checkpoint_dirs[0]
        
        raise ValueError("No checkpoint directory found in archive")
    
    async def _verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
        """Verify checkpoint integrity using checksum"""
        metadata_file = checkpoint_dir / "metadata.json"
        
        if not metadata_file.exists():
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        stored_checksum = metadata.get('checksum')
        if not stored_checksum:
            return True  # No checksum to verify
        
        calculated_checksum = await self._calculate_checkpoint_checksum(checkpoint_dir)
        return calculated_checksum == stored_checksum
    
    def _find_checkpoint(self, checkpoint_id: str) -> Optional[Path]:
        """Find a checkpoint by ID"""
        # Check for compressed version
        archive_path = self.backup_root / f"{checkpoint_id}.tar.gz"
        if archive_path.exists():
            return archive_path
        
        # Check for directory
        checkpoint_dir = self.backup_root / f"checkpoint_{checkpoint_id}"
        if checkpoint_dir.exists():
            return checkpoint_dir
        
        # Search by partial ID
        for path in self.backup_root.glob(f"*{checkpoint_id}*"):
            return path
        
        return None
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for path in self.backup_root.glob("checkpoint_*"):
            if path.is_dir():
                metadata_file = path / "metadata.json"
            elif path.suffix == '.tar.gz':
                # Would need to extract metadata from archive
                continue
            else:
                continue
            
            if metadata_file and metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append(metadata)
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    async def cleanup_old_checkpoints(self, keep_days: int = 30, keep_count: int = 10):
        """Clean up old checkpoints based on age and count"""
        checkpoints = await self.list_checkpoints()
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Keep minimum count
        to_keep = set(c['checkpoint_id'] for c in checkpoints[:keep_count])
        
        # Keep recent checkpoints
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        for checkpoint in checkpoints:
            checkpoint_time = datetime.fromisoformat(checkpoint['timestamp'])
            if checkpoint_time > cutoff_date:
                to_keep.add(checkpoint['checkpoint_id'])
        
        # Remove old checkpoints
        removed_count = 0
        for checkpoint in checkpoints:
            if checkpoint['checkpoint_id'] not in to_keep:
                checkpoint_path = self._find_checkpoint(checkpoint['checkpoint_id'])
                if checkpoint_path:
                    if checkpoint_path.is_dir():
                        shutil.rmtree(checkpoint_path)
                    else:
                        checkpoint_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_id']}")
        
        return removed_count