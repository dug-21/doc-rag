#!/bin/bash
# Production Backup Script for Doc-RAG System
# Performs comprehensive backup of all data and configurations

set -euo pipefail

# Configuration
BACKUP_ROOT="/opt/docrag/backups"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
BACKUP_DIR="${BACKUP_ROOT}/${TIMESTAMP}"
LOG_FILE="${BACKUP_ROOT}/backup-${TIMESTAMP}.log"
RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root or with sudo privileges"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
mkdir -p "${BACKUP_ROOT}/logs"

log "Starting production backup to: $BACKUP_DIR"

# Function to check service availability
check_service() {
    local service=$1
    local port=$2
    if ! timeout 10 bash -c "</dev/tcp/localhost/$port"; then
        warning "Service $service on port $port is not responding"
        return 1
    fi
    return 0
}

# Function to backup PostgreSQL
backup_postgresql() {
    log "Backing up PostgreSQL database..."
    
    if ! check_service "PostgreSQL" 5432; then
        error "PostgreSQL is not accessible"
    fi
    
    local pg_backup_dir="${BACKUP_DIR}/postgresql"
    mkdir -p "$pg_backup_dir"
    
    # Backup main database
    docker exec doc-rag-postgres-prod pg_dump \
        -U docrag \
        -d docrag \
        --format=custom \
        --compress=9 \
        --verbose > "${pg_backup_dir}/docrag.dump" 2>> "$LOG_FILE"
    
    # Backup all databases (including system databases)
    docker exec doc-rag-postgres-prod pg_dumpall \
        -U docrag \
        --clean \
        --verbose > "${pg_backup_dir}/all_databases.sql" 2>> "$LOG_FILE"
    
    # Export database schema only
    docker exec doc-rag-postgres-prod pg_dump \
        -U docrag \
        -d docrag \
        --schema-only \
        --verbose > "${pg_backup_dir}/schema.sql" 2>> "$LOG_FILE"
    
    # Compress SQL files
    gzip "${pg_backup_dir}/all_databases.sql"
    gzip "${pg_backup_dir}/schema.sql"
    
    success "PostgreSQL backup completed"
}

# Function to backup MongoDB
backup_mongodb() {
    log "Backing up MongoDB database..."
    
    if ! check_service "MongoDB" 27017; then
        error "MongoDB is not accessible"
    fi
    
    local mongo_backup_dir="${BACKUP_DIR}/mongodb"
    mkdir -p "$mongo_backup_dir"
    
    # Create MongoDB dump
    docker exec doc-rag-mongodb mongodump \
        --db docrag \
        --out /tmp/mongodb_backup 2>> "$LOG_FILE"
    
    # Copy backup from container
    docker cp doc-rag-mongodb:/tmp/mongodb_backup/docrag "${mongo_backup_dir}/"
    
    # Compress backup
    cd "$mongo_backup_dir"
    tar -czf "mongodb-docrag-${TIMESTAMP}.tar.gz" docrag/
    rm -rf docrag/
    
    success "MongoDB backup completed"
}

# Function to backup Redis
backup_redis() {
    log "Backing up Redis database..."
    
    if ! check_service "Redis" 6379; then
        error "Redis is not accessible"
    fi
    
    local redis_backup_dir="${BACKUP_DIR}/redis"
    mkdir -p "$redis_backup_dir"
    
    # Force Redis to save current state
    docker exec doc-rag-redis-prod redis-cli \
        -a "${REDIS_PASSWORD:-redis_secret_2024}" \
        BGSAVE 2>> "$LOG_FILE"
    
    # Wait for background save to complete
    sleep 5
    while [[ $(docker exec doc-rag-redis-prod redis-cli \
        -a "${REDIS_PASSWORD:-redis_secret_2024}" \
        LASTSAVE 2>/dev/null) == $(docker exec doc-rag-redis-prod redis-cli \
        -a "${REDIS_PASSWORD:-redis_secret_2024}" \
        LASTSAVE 2>/dev/null) ]]; do
        sleep 1
    done
    
    # Copy RDB file
    docker cp doc-rag-redis-prod:/data/dump.rdb "${redis_backup_dir}/"
    
    # Also backup AOF if enabled
    if docker exec doc-rag-redis-prod test -f /data/appendonly.aof 2>/dev/null; then
        docker cp doc-rag-redis-prod:/data/appendonly.aof "${redis_backup_dir}/"
    fi
    
    success "Redis backup completed"
}

# Function to backup Qdrant vector database
backup_qdrant() {
    log "Backing up Qdrant vector database..."
    
    if ! check_service "Qdrant" 6333; then
        error "Qdrant is not accessible"
    fi
    
    local qdrant_backup_dir="${BACKUP_DIR}/qdrant"
    mkdir -p "$qdrant_backup_dir"
    
    # Create snapshots for all collections
    local collections=$(curl -s http://localhost:6333/collections | jq -r '.result.collections[].name')
    
    for collection in $collections; do
        log "Creating snapshot for collection: $collection"
        
        # Create snapshot
        snapshot_info=$(curl -s -X POST "http://localhost:6333/collections/$collection/snapshots")
        snapshot_name=$(echo "$snapshot_info" | jq -r '.result.name')
        
        if [[ "$snapshot_name" != "null" ]]; then
            # Download snapshot
            curl -s "http://localhost:6333/collections/$collection/snapshots/$snapshot_name" \
                -o "${qdrant_backup_dir}/${collection}-${snapshot_name}"
            log "Snapshot created for collection $collection: $snapshot_name"
        else
            warning "Failed to create snapshot for collection: $collection"
        fi
    done
    
    success "Qdrant backup completed"
}

# Function to backup MinIO object storage
backup_minio() {
    log "Backing up MinIO object storage..."
    
    if ! check_service "MinIO" 9000; then
        error "MinIO is not accessible"
    fi
    
    local minio_backup_dir="${BACKUP_DIR}/minio"
    mkdir -p "$minio_backup_dir"
    
    # Copy MinIO data directory
    docker exec doc-rag-minio find /data -type f -name "*.* " | while read -r file; do
        docker cp "doc-rag-minio:$file" "$minio_backup_dir/"
    done
    
    # Alternative: Use mc (MinIO Client) if available
    if command -v mc &> /dev/null; then
        mc alias set backup-minio http://localhost:9000 \
            "${MINIO_ACCESS_KEY:-docrag_access}" \
            "${MINIO_SECRET_KEY:-docrag_secret_key_2024}"
        
        mc cp --recursive backup-minio/ "${minio_backup_dir}/buckets/"
    fi
    
    success "MinIO backup completed"
}

# Function to backup configuration files
backup_configs() {
    log "Backing up configuration files..."
    
    local config_backup_dir="${BACKUP_DIR}/configs"
    mkdir -p "$config_backup_dir"
    
    # Copy configuration directories
    cp -r /workspaces/doc-rag/config "${config_backup_dir}/"
    cp -r /workspaces/doc-rag/k8s "${config_backup_dir}/"
    
    # Copy important files
    cp /workspaces/doc-rag/docker-compose*.yml "${config_backup_dir}/"
    cp /workspaces/doc-rag/.env.production "${config_backup_dir}/" 2>/dev/null || true
    
    # Copy SSL certificates
    if [[ -d "/workspaces/doc-rag/config/ssl" ]]; then
        cp -r /workspaces/doc-rag/config/ssl "${config_backup_dir}/"
    fi
    
    success "Configuration backup completed"
}

# Function to backup application data and logs
backup_application_data() {
    log "Backing up application data and logs..."
    
    local app_backup_dir="${BACKUP_DIR}/application"
    mkdir -p "$app_backup_dir"
    
    # Backup uploaded documents and processed data
    if [[ -d "/opt/docrag/storage" ]]; then
        cp -r /opt/docrag/storage "${app_backup_dir}/"
    fi
    
    # Backup logs (last 7 days)
    if [[ -d "/opt/docrag/logs" ]]; then
        find /opt/docrag/logs -name "*.log*" -mtime -7 \
            -exec cp {} "${app_backup_dir}/logs/" \; 2>/dev/null || true
    fi
    
    # Backup model cache
    docker cp doc-rag-embedder:/models "${app_backup_dir}/" 2>/dev/null || true
    
    success "Application data backup completed"
}

# Function to create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="${BACKUP_DIR}/MANIFEST.txt"
    
    cat > "$manifest_file" << EOF
Doc-RAG Production Backup Manifest
==================================

Backup Timestamp: $TIMESTAMP
Backup Location: $BACKUP_DIR
Created By: $(whoami)@$(hostname)
System: $(uname -a)

Components Backed Up:
- PostgreSQL Database (docrag)
- MongoDB Database (docrag)  
- Redis Cache
- Qdrant Vector Database
- MinIO Object Storage
- Configuration Files
- Application Data & Logs

Backup Size: $(du -sh "$BACKUP_DIR" | cut -f1)

File Listing:
$(find "$BACKUP_DIR" -type f -exec ls -lh {} \; | sort -k9)

Checksums:
$(find "$BACKUP_DIR" -type f -exec md5sum {} \;)
EOF
    
    success "Backup manifest created"
}

# Function to compress backup
compress_backup() {
    log "Compressing backup archive..."
    
    cd "$BACKUP_ROOT"
    tar -czf "${TIMESTAMP}.tar.gz" "$TIMESTAMP/"
    
    if [[ $? -eq 0 ]]; then
        rm -rf "$TIMESTAMP/"
        success "Backup compressed to: ${BACKUP_ROOT}/${TIMESTAMP}.tar.gz"
    else
        error "Failed to compress backup"
    fi
}

# Function to upload to cloud storage (optional)
upload_to_cloud() {
    if [[ -n "${AWS_S3_BUCKET:-}" ]]; then
        log "Uploading backup to AWS S3..."
        
        aws s3 cp "${BACKUP_ROOT}/${TIMESTAMP}.tar.gz" \
            "s3://${AWS_S3_BUCKET}/doc-rag/backups/${TIMESTAMP}.tar.gz" \
            --storage-class STANDARD_IA
        
        success "Backup uploaded to S3"
    fi
    
    if [[ -n "${AZURE_STORAGE_ACCOUNT:-}" ]]; then
        log "Uploading backup to Azure Storage..."
        
        az storage blob upload \
            --account-name "$AZURE_STORAGE_ACCOUNT" \
            --container-name "doc-rag-backups" \
            --name "${TIMESTAMP}.tar.gz" \
            --file "${BACKUP_ROOT}/${TIMESTAMP}.tar.gz" \
            --tier Hot
        
        success "Backup uploaded to Azure Storage"
    fi
}

# Function to cleanup old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."
    
    find "$BACKUP_ROOT" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_ROOT" -name "backup-*.log" -mtime +$RETENTION_DAYS -delete
    
    success "Old backup cleanup completed"
}

# Main execution
main() {
    log "=== Doc-RAG Production Backup Started ==="
    
    # Check disk space (need at least 10GB free)
    available_space=$(df "$BACKUP_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        error "Insufficient disk space. Need at least 10GB free in $BACKUP_ROOT"
    fi
    
    # Load environment variables
    if [[ -f "/workspaces/doc-rag/.env.production" ]]; then
        export $(cat /workspaces/doc-rag/.env.production | xargs)
    fi
    
    # Perform backups
    backup_postgresql
    backup_mongodb
    backup_redis
    backup_qdrant
    backup_minio
    backup_configs
    backup_application_data
    
    # Create manifest and compress
    create_manifest
    compress_backup
    
    # Optional cloud upload
    upload_to_cloud
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "=== Doc-RAG Production Backup Completed ==="
    success "Backup successfully created: ${BACKUP_ROOT}/${TIMESTAMP}.tar.gz"
    
    # Display backup information
    echo
    echo "Backup Summary:"
    echo "==============="
    echo "Location: ${BACKUP_ROOT}/${TIMESTAMP}.tar.gz"
    echo "Size: $(ls -lh "${BACKUP_ROOT}/${TIMESTAMP}.tar.gz" | awk '{print $5}')"
    echo "Checksum: $(md5sum "${BACKUP_ROOT}/${TIMESTAMP}.tar.gz" | awk '{print $1}')"
    echo
}

# Handle script termination
trap 'error "Backup interrupted"' INT TERM

# Run main function
main "$@"