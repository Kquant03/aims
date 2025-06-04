# AIMS Production Deployment Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Deployment Options](#deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Manual Deployment](#manual-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Configuration](#configuration)
8. [Monitoring Setup](#monitoring-setup)
9. [Scaling Strategies](#scaling-strategies)
10. [Backup and Recovery](#backup-and-recovery)
11. [Security Hardening](#security-hardening)
12. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

#### Minimum (CPU-only mode)
- CPU: 8-core modern processor (Intel i7/AMD Ryzen 7 or better)
- RAM: 32GB
- Storage: 100GB SSD
- Network: 100Mbps

#### Recommended (GPU-accelerated)
- GPU: NVIDIA RTX 3090 (24GB VRAM) or better
- CPU: 16-core processor
- RAM: 64GB
- Storage: 500GB NVMe SSD
- Network: 1Gbps

#### Production (High-load)
- GPU: 2x NVIDIA RTX 3090 or A100
- CPU: 32-core processor
- RAM: 128GB
- Storage: 2TB NVMe SSD RAID
- Network: 10Gbps

### Software Requirements
- OS: Ubuntu 20.04/22.04 LTS or RHEL 8+
- Docker: 20.10+
- Docker Compose: 2.0+
- Python: 3.10+
- CUDA: 12.1+ (for GPU)
- PostgreSQL: 14+
- Redis: 7+

## Pre-deployment Checklist

- [ ] Verify system requirements met
- [ ] Obtain Anthropic API key
- [ ] Configure firewall rules
- [ ] Set up SSL certificates
- [ ] Create backup storage location
- [ ] Configure monitoring tools
- [ ] Set up log aggregation
- [ ] Create deployment user account
- [ ] Configure environment variables
- [ ] Test database connections
- [ ] Verify GPU drivers (if applicable)

## Deployment Options

### 1. Single Server Deployment
Best for: Development, small teams, <100 concurrent users

### 2. Multi-Server Deployment
Best for: Production, 100-1000 concurrent users

### 3. Kubernetes Deployment
Best for: Enterprise, 1000+ concurrent users, high availability

## Docker Deployment

### 1. Create Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # PostgreSQL with replication
  postgres-primary:
    image: pgvector/pgvector:pg16
    container_name: aims_postgres_primary
    environment:
      POSTGRES_DB: aims_memory
      POSTGRES_USER: aims
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aims"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Redis with persistence
  redis:
    image: redis:7-alpine
    container_name: aims_redis
    command: >
      redis-server 
      --maxmemory 8gb 
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec
      --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: always

  # AIMS Application
  aims:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: aims_app
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://aims:${POSTGRES_PASSWORD}@postgres-primary:5432/aims_memory
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"
      - "8765:8765"
    depends_on:
      - postgres-primary
      - redis
    restart: always
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: aims_nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - aims
    restart: always

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: aims_prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: always

  grafana:
    image: grafana/grafana:latest
    container_name: aims_grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    restart: always

volumes:
  postgres_primary_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: aims_network
    driver: bridge
```

### 2. Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash aims

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Flash Attention
RUN pip3 install flash-attn --no-build-isolation

# Copy application
COPY --chown=aims:aims . .

# Create necessary directories
RUN mkdir -p logs data/states data/backups data/uploads && \
    chown -R aims:aims /app

# Switch to app user
USER aims

# Expose ports
EXPOSE 8000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python3", "-m", "src.main"]
```

### 3. Deploy with Docker

```bash
# Clone repository
git clone https://github.com/yourusername/aims.git
cd aims

# Create environment file
cp .env.example .env.prod
# Edit .env.prod with production values

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale if needed
docker-compose -f docker-compose.prod.yml up -d --scale aims=3
```

## Manual Deployment

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip \
    postgresql-14 postgresql-contrib-14 redis-server \
    nginx certbot python3-certbot-nginx \
    build-essential git curl

# Install NVIDIA drivers (if GPU)
sudo apt install -y nvidia-driver-525
sudo apt install -y cuda-12-1

# Create application user
sudo useradd -m -s /bin/bash aims
sudo usermod -aG docker aims
```

### 2. PostgreSQL Setup

```bash
# Configure PostgreSQL
sudo -u postgres psql <<EOF
CREATE USER aims WITH PASSWORD 'secure_password';
CREATE DATABASE aims_memory OWNER aims;
\c aims_memory
CREATE EXTENSION vector;
EOF

# Edit PostgreSQL config for performance
sudo nano /etc/postgresql/14/main/postgresql.conf
# Set:
# shared_buffers = 8GB
# effective_cache_size = 24GB
# work_mem = 256MB
# maintenance_work_mem = 2GB
# max_connections = 200

sudo systemctl restart postgresql
```

### 3. Application Deployment

```bash
# Switch to aims user
sudo su - aims

# Clone repository
git clone https://github.com/yourusername/aims.git
cd aims

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Set up environment
cp .env.example .env
# Edit .env with production values

# Run database migrations
python scripts/setup_databases.py

# Create systemd service
sudo nano /etc/systemd/system/aims.service
```

### 4. Systemd Service

```ini
[Unit]
Description=AIMS Consciousness AI
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=aims
Group=aims
WorkingDirectory=/home/aims/aims
Environment="PATH=/home/aims/aims/venv/bin"
ExecStart=/home/aims/aims/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /home/aims/aims/logs/access.log \
    --error-logfile /home/aims/aims/logs/error.log \
    src.ui.web_interface:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 5. Nginx Configuration

```nginx
# /etc/nginx/sites-available/aims
upstream aims_backend {
    server 127.0.0.1:8000;
}

upstream aims_websocket {
    server 127.0.0.1:8765;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 100M;

    location / {
        proxy_pass http://aims_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    location /ws {
        proxy_pass http://aims_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    location /static {
        alias /home/aims/aims/src/ui/static;
        expires 30d;
    }
}
```

## Cloud Deployment

### AWS Deployment

```bash
# Create EC2 instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type g4dn.xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxxx \
    --subnet-id subnet-xxxxxx \
    --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=aims-prod}]'

# Create RDS PostgreSQL
aws rds create-db-instance \
    --db-instance-identifier aims-db \
    --db-instance-class db.m6g.xlarge \
    --engine postgres \
    --engine-version 14.7 \
    --master-username aims \
    --master-user-password $DB_PASSWORD \
    --allocated-storage 100 \
    --backup-retention-period 30

# Create ElastiCache Redis
aws elasticache create-replication-group \
    --replication-group-id aims-cache \
    --replication-group-description "AIMS Redis Cache" \
    --engine redis \
    --cache-node-type cache.r6g.large \
    --num-cache-clusters 2
```

### Google Cloud Deployment

```bash
# Create GCE instance with GPU
gcloud compute instances create aims-prod \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE

# Create Cloud SQL PostgreSQL
gcloud sql instances create aims-db \
    --database-version=POSTGRES_14 \
    --tier=db-n1-highmem-4 \
    --region=us-central1 \
    --backup-start-time=03:00 \
    --enable-bin-log

# Create Memorystore Redis
gcloud redis instances create aims-cache \
    --size=5 \
    --region=us-central1 \
    --redis-version=redis_6_x
```

## Configuration

### Environment Variables

```bash
# .env.prod
# API Keys
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=optional_key_here

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aims_memory
POSTGRES_USER=aims
POSTGRES_PASSWORD=secure_password_here

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secure_redis_password

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
SESSION_SECRET=generate_strong_secret_here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# GPU Settings
USE_GPU=true
GPU_MEMORY_FRACTION=0.9

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
```

### Performance Tuning

```yaml
# config/production.yaml
consciousness:
  cycle_frequency: 2.0
  working_memory_size: 7
  coherence_threshold: 0.7

memory:
  max_memories: 1000000
  consolidation_threshold: 0.3
  embedding_dim: 768
  use_gpu: true

api:
  rate_limit: 60
  max_message_length: 10000
  timeout: 300

cache:
  ttl: 3600
  max_size: 10000

performance:
  batch_size: 32
  num_workers: 4
  prefetch_factor: 2
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aims'
    static_configs:
      - targets: ['aims:8000']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['nvidia-gpu-exporter:9445']
```

### 2. Grafana Dashboards

Import these dashboard IDs:
- System Overview: 1860
- PostgreSQL: 9628
- Redis: 763
- NVIDIA GPU: 14574

Custom AIMS dashboard available in `monitoring/grafana-dashboard.json`

### 3. Alerting Rules

```yaml
# alerts.yml
groups:
  - name: aims_alerts
    rules:
      - alert: HighMemoryUsage
        expr: node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1
        for: 5m
        annotations:
          summary: "High memory usage detected"

      - alert: LowConsciousnessCoherence
        expr: aims_consciousness_coherence < 0.5
        for: 10m
        annotations:
          summary: "Consciousness coherence below threshold"

      - alert: HighGPUTemperature
        expr: nvidia_gpu_temperature_celsius > 85
        for: 5m
        annotations:
          summary: "GPU temperature critical"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        annotations:
          summary: "PostgreSQL database is down"
```

## Scaling Strategies

### Horizontal Scaling

1. **Database Scaling**
   - PostgreSQL read replicas
   - PgBouncer connection pooling
   - Partitioning large tables

2. **Redis Scaling**
   - Redis Cluster for sharding
   - Redis Sentinel for HA
   - Separate cache instances by type

3. **Application Scaling**
   - Multiple AIMS instances
   - Load balancer (HAProxy/Nginx)
   - Sticky sessions for WebSocket

### Vertical Scaling

1. **GPU Upgrade Path**
   - RTX 3090 → RTX 4090
   - RTX 4090 → A100 40GB
   - Single GPU → Multi-GPU

2. **Memory Optimization**
   - Increase shared_buffers
   - Tune work_mem
   - Optimize query plans

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backup/aims"
S3_BUCKET="s3://your-bucket/aims-backups"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR/$(date +%Y%m%d)
cd $BACKUP_DIR/$(date +%Y%m%d)

# Backup PostgreSQL
pg_dump -h localhost -U aims aims_memory | gzip > postgres_$(date +%H%M%S).sql.gz

# Backup Redis
redis-cli --rdb redis_$(date +%H%M%S).rdb

# Backup application data
tar czf data_$(date +%H%M%S).tar.gz /home/aims/aims/data

# Sync to S3
aws s3 sync . $S3_BUCKET/$(date +%Y%m%d)/

# Cleanup old backups
find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;
```

### Recovery Procedures

```bash
# Restore PostgreSQL
gunzip -c postgres_backup.sql.gz | psql -h localhost -U aims aims_memory

# Restore Redis
redis-cli --pipe < redis_backup.rdb

# Restore application data
tar xzf data_backup.tar.gz -C /
```

## Security Hardening

### 1. Firewall Rules

```bash
# UFW configuration
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL internal
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis internal
sudo ufw enable
```

### 2. SSL/TLS Configuration

```bash
# Generate SSL certificate with Let's Encrypt
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### 3. Application Security

```python
# Additional security headers in Nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall drivers if needed
sudo apt install --reinstall nvidia-driver-525
```

2. **High Memory Usage**
```bash
# Check memory consumers
ps aux --sort=-%mem | head

# Clear Python cache
python -c "import gc; gc.collect()"

# Restart services
sudo systemctl restart aims
```

3. **Database Connection Issues**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Increase connection limit if needed
sudo -u postgres psql -c "ALTER SYSTEM SET max_connections = 500;"
```

4. **WebSocket Connection Failures**
```bash
# Check if port is open
sudo netstat -tlnp | grep 8765

# Test WebSocket
wscat -c ws://localhost:8765

# Check Nginx proxy
tail -f /var/log/nginx/error.log
```

### Performance Diagnostics

```bash
# System performance
htop
iotop
nvidia-smi dmon

# Application profiling
python -m cProfile -o profile.stats src/main.py
python -m pstats profile.stats

# Database slow queries
sudo -u postgres psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

## Maintenance Schedule

### Daily
- Check system logs
- Monitor resource usage
- Verify backup completion

### Weekly
- Update dependencies
- Run security scans
- Performance analysis

### Monthly
- OS updates
- Database maintenance (VACUUM, ANALYZE)
- Review and rotate logs

### Quarterly
- Security audit
- Capacity planning
- Disaster recovery test

## Support

For deployment support:
- Documentation: https://github.com/yourusername/aims/wiki
- Issues: https://github.com/yourusername/aims/issues
- Community: Discord/Slack channel

Remember to always test deployment changes in a staging environment before applying to production!