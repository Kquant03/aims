# scripts/daily_backup.sh
#!/bin/bash
# Daily backup script for AIMS

BACKUP_DIR="/app/data/backups"
STATE_DIR="/app/data/states"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/daily_backup_$TIMESTAMP.tar.gz"

echo "Starting daily backup at $TIMESTAMP"

# Create backup
tar -czf "$BACKUP_FILE" -C "$STATE_DIR" .

# Upload to S3 if configured
if [ "$S3_BACKUP_ENABLED" = "true" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_FILE" "s3://$S3_BUCKET/aims-backups/"
fi

# Clean old local backups (keep last 7)
find "$BACKUP_DIR" -name "daily_backup_*.tar.gz" -mtime +7 -delete

echo "Backup complete: $BACKUP_FILE"