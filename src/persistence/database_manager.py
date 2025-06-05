# database_manager.py - Fixed version
import asyncio
import asyncpg
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import numpy as np
from typing import Optional, Dict, Any, List
import uuid
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                host=self.config.get('pg_host', 'localhost'),
                port=self.config.get('pg_port', 5432),
                database=self.config.get('pg_database', 'aims_db'),
                user=self.config.get('pg_user', 'aims_user'),
                password=self.config.get('pg_password', ''),
                min_size=10,
                max_size=50,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password', ''),
                decode_responses=True,
                max_connections=100
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            # Qdrant connection
            self.qdrant_client = QdrantClient(
                host=self.config.get('qdrant_host', 'localhost'),
                port=self.config.get('qdrant_port', 6333)
            )
            
            self.initialized = True
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise
    
    async def close(self):
        """Close all database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("All database connections closed")