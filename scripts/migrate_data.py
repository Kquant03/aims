#!/usr/bin/env python3
# scripts/migrate_data.py - Data migration utilities
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging

logger = setup_logging('INFO', 'logs/migration.log')

class DataMigrator:
    """Handles data migrations between versions"""
    
    def __init__(self):
        self.migrations = {
            '1.0_to_1.1': self.migrate_1_0_to_1_1,
            '1.1_to_1.2': self.migrate_1_1_to_1_2,
        }
    
    async def migrate_1_0_to_1_1(self, data_path: Path):
        """Migrate from version 1.0 to 1.1"""
        logger.info("Migrating from 1.0 to 1.1...")
        
        # Example: Add new fields to existing state files
        for state_file in data_path.glob("*_state.json"):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Add new fields if missing
            if 'version' not in state:
                state['version'] = '1.1'
            
            if 'consciousness' in state and 'phi_score' not in state['consciousness']:
                state['consciousness']['phi_score'] = 0.0
            
            # Save updated state
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        logger.info("Migration 1.0 to 1.1 complete")
    
    async def migrate_1_1_to_1_2(self, data_path: Path):
        """Migrate from version 1.1 to 1.2"""
        logger.info("Migrating from 1.1 to 1.2...")
        
        # Add your migration logic here
        
        logger.info("Migration 1.1 to 1.2 complete")
    
    async def run_migration(self, from_version: str, to_version: str, data_path: Path):
        """Run migration from one version to another"""
        migration_key = f"{from_version}_to_{to_version}"
        
        if migration_key not in self.migrations:
            raise ValueError(f"No migration path from {from_version} to {to_version}")
        
        # Backup data before migration
        backup_path = data_path.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating backup at {backup_path}")
        
        import shutil
        shutil.copytree(data_path, backup_path)
        
        # Run migration
        await self.migrations[migration_key](data_path)
        
        logger.info(f"Migration complete. Backup saved at {backup_path}")

async def restore_from_backup(backup_path: Path, target_path: Path):
    """Restore data from backup"""
    logger.info(f"Restoring from {backup_path} to {target_path}")
    
    import shutil
    
    # Remove existing data
    if target_path.exists():
        shutil.rmtree(target_path)
    
    # Copy backup
    shutil.copytree(backup_path, target_path)
    
    logger.info("Restore complete")

async def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIMS Data Migration Tool')
    parser.add_argument('action', choices=['migrate', 'restore'], 
                       help='Action to perform')
    parser.add_argument('--from-version', help='Source version (for migrate)')
    parser.add_argument('--to-version', help='Target version (for migrate)')
    parser.add_argument('--backup-path', help='Backup path (for restore)')
    parser.add_argument('--data-path', default='data/states', 
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    if args.action == 'migrate':
        if not args.from_version or not args.to_version:
            parser.error("migrate requires --from-version and --to-version")
        
        migrator = DataMigrator()
        await migrator.run_migration(args.from_version, args.to_version, data_path)
        
    elif args.action == 'restore':
        if not args.backup_path:
            parser.error("restore requires --backup-path")
        
        backup_path = Path(args.backup_path)
        await restore_from_backup(backup_path, data_path)

if __name__ == "__main__":
    asyncio.run(main())