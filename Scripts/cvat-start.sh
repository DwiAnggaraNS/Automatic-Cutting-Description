#!/bin/bash
# ============================================================
#  CVAT + SAM — START SCRIPT
#  Nyalakan CVAT, Nuclio, lalu deploy SAM GPU
# ============================================================

CVAT_DIR="$HOME/cvat"
SAM_YAML="serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml"
SAM_PATH="serverless/pytorch/facebookresearch/sam/nuclio"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}   CVAT + SAM — Starting All Services          ${NC}"
echo -e "${CYAN}================================================${NC}"

if [ ! -d "$CVAT_DIR" ]; then
    echo -e "${RED}[ERROR] Folder $CVAT_DIR tidak ditemukan!${NC}"
    exit 1
fi

cd "$CVAT_DIR"

# ── 1. Jalankan CVAT + Nuclio ──────────────────────────────
echo -e "\n${YELLOW}[1/4] Menjalankan CVAT + Nuclio...${NC}"
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d

# ── 2. Tunggu Nuclio siap ─────────────────────────────────
echo -e "\n${YELLOW}[2/4] Menunggu Nuclio siap...${NC}"
for i in {1..12}; do
    STATUS=$(sudo docker inspect --format='{{.State.Health.Status}}' nuclio 2>/dev/null)
    if [ "$STATUS" = "healthy" ]; then
        echo -e "${GREEN}[OK]  Nuclio sudah healthy!${NC}"
        break
    fi
    echo -e "      Menunggu... ($i/12)"
    sleep 5
done

STATUS=$(sudo docker inspect --format='{{.State.Health.Status}}' nuclio 2>/dev/null)
if [ "$STATUS" != "healthy" ]; then
    echo -e "${RED}[WARN] Nuclio belum healthy setelah 60 detik. Cek: sudo docker logs nuclio${NC}"
fi

# ── 3. Buat project Nuclio jika belum ada ─────────────────
echo -e "\n${YELLOW}[3/4] Mempersiapkan Nuclio project...${NC}"
PROJECT_EXISTS=$(sudo nuctl get projects --platform local 2>/dev/null | grep "cvat")
if [ -z "$PROJECT_EXISTS" ]; then
    sudo nuctl create project cvat --platform local
    echo -e "${GREEN}[OK]  Project 'cvat' berhasil dibuat.${NC}"
else
    echo -e "${GREEN}[OK]  Project 'cvat' sudah ada.${NC}"
fi

# ── 4. Start atau Deploy SAM GPU ──────────────────────────
echo -e "\n${YELLOW}[4/4] Menjalankan SAM (GPU)...${NC}"

# Cek apakah container SAM sudah ada (stopped) — tinggal start ulang
SAM_STOPPED=$(sudo docker ps -a --format "{{.Names}}\t{{.Status}}" | grep "nuclio-nuclio")
SAM_RUNNING=$(sudo docker ps --format "{{.Names}}" | grep "nuclio-nuclio")

if [ -n "$SAM_RUNNING" ]; then
    # Sudah jalan
    echo -e "${GREEN}[OK]  SAM sudah berjalan: $SAM_RUNNING${NC}"

elif [ -n "$SAM_STOPPED" ]; then
    # Container ada tapi stopped → tinggal start, TIDAK perlu deploy ulang
    SAM_NAME=$(echo "$SAM_STOPPED" | awk '{print $1}')
    echo -e "${GREEN}[OK]  Container SAM ditemukan (stopped), menjalankan ulang...${NC}"
    sudo docker start $SAM_NAME
    echo -e "${GREEN}[OK]  SAM berhasil dijalankan kembali (tanpa deploy ulang).${NC}"

else
    # Belum ada sama sekali → deploy (build jika perlu)
    SAM_IMAGE=$(sudo docker images --format "{{.Repository}}:{{.Tag}}" | grep "sam.*gpu")
    if [ -n "$SAM_IMAGE" ]; then
        echo -e "${YELLOW}      Image SAM ada tapi container hilang, deploy ulang (~30 detik)...${NC}"
    else
        echo -e "${YELLOW}      Image SAM belum ada, build dari awal (~15 menit)...${NC}"
    fi

    sudo nuctl deploy \
        --project-name cvat \
        --path "$SAM_PATH" \
        --file "$SAM_YAML" \
        --platform local

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK]  SAM berhasil di-deploy!${NC}"
    else
        echo -e "${RED}[ERROR] Gagal deploy SAM. Jalankan: sudo docker logs nuclio${NC}"
    fi
fi

# ── Ringkasan Status ──────────────────────────────────────
echo -e "\n${CYAN}================================================${NC}"
echo -e "${CYAN}   STATUS AKHIR                                 ${NC}"
echo -e "${CYAN}================================================${NC}"
sudo docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null
echo ""
SAM_STATUS=$(sudo docker ps --format "{{.Names}}\t{{.Status}}" | grep "nuclio-nuclio")
if [ -n "$SAM_STATUS" ]; then
    echo -e "${GREEN}[SAM]  $SAM_STATUS${NC}"
else
    echo -e "${RED}[SAM]  Tidak berjalan!${NC}"
fi

echo -e "\n${GREEN}[DONE] CVAT siap digunakan di: http://localhost:8080${NC}"
echo -e "${GREEN}       Nuclio dashboard      : http://localhost:8070${NC}"
