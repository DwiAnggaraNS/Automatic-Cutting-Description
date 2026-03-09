#!/bin/bash
# ============================================================
#  CVAT + SAM — STOP SCRIPT (tanpa hapus container)
#  Pause semua service, SAM tetap terdaftar di Nuclio
# ============================================================

CVAT_DIR="$HOME/cvat"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}[CVAT] Menghentikan semua service (tanpa hapus container)...${NC}"

cd "$CVAT_DIR" || { echo -e "${RED}[ERROR] Folder $CVAT_DIR tidak ditemukan!${NC}"; exit 1; }

# Stop container SAM juga
SAM_CONTAINER=$(sudo docker ps --format "{{.Names}}" | grep "nuclio-nuclio")
if [ -n "$SAM_CONTAINER" ]; then
    echo -e "${YELLOW}[SAM]  Menghentikan SAM: $SAM_CONTAINER${NC}"
    sudo docker stop $SAM_CONTAINER
fi

# Stop semua container CVAT + Nuclio (tanpa down, config tetap tersimpan)
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    stop

# Verifikasi
RUNNING=$(sudo docker ps --format "{{.Names}}" | grep -E "cvat|nuclio")
if [ -z "$RUNNING" ]; then
    echo -e "${GREEN}[OK]   Semua service berhenti. Resource aman.${NC}"
    echo -e "${GREEN}       Jalankan cvat-start.sh untuk melanjutkan anotasi.${NC}"
else
    echo -e "${RED}[WARN] Masih ada container berjalan: $RUNNING${NC}"
fi
