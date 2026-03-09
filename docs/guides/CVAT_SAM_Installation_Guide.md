# Panduan Instalasi CVAT + SAM di Ubuntu (Lokal)

> Dokumen ini dibuat berdasarkan pengalaman instalasi nyata beserta semua kesalahan yang ditemui,
> agar menjadi referensi yang akurat untuk instalasi berikutnya.

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Instalasi CVAT](#2-instalasi-cvat)
3. [Konfigurasi function-gpu.yaml (PENTING)](#3-konfigurasi-function-gpuyaml-penting)
4. [Deploy SAM Function](#4-deploy-sam-function)
5. [Verifikasi Instalasi](#5-verifikasi-instalasi)
6. [Script Utilitas](#6-script-utilitas)
7. [Troubleshooting](#7-troubleshooting)
8. [Pelajaran dari Kesalahan](#8-pelajaran-dari-kesalahan)

---

## 1. Prasyarat

### Hardware
- GPU NVIDIA (disarankan ≥8GB VRAM untuk SAM ViT-H)
- RAM ≥ 16GB
- Storage ≥ 50GB (image SAM ~10GB)

### Software
```bash
# Docker Engine (bukan Docker Desktop)
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# NVIDIA Container Toolkit (agar Docker bisa akses GPU)
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# nuctl CLI (Nuclio CLI)
# Cek versi Nuclio yang dipakai di docker-compose.serverless.yml terlebih dahulu
# lalu download nuctl dengan versi yang SAMA PERSIS
wget https://github.com/nuclio/nuclio/releases/download/<VERSI>/nuctl-<VERSI>-linux-amd64
chmod +x nuctl-<VERSI>-linux-amd64
sudo mv nuctl-<VERSI>-linux-amd64 /usr/local/bin/nuctl
```

> ⚠️ **PENTING:** Versi `nuctl` CLI harus sama dengan versi image Nuclio di docker-compose.
> Ketidakcocokan versi menyebabkan perintah `nuctl deploy` gagal.

---

## 2. Instalasi CVAT

```bash
# Clone repository CVAT
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# Jalankan CVAT + Nuclio (serverless)
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d

# Buat superuser
sudo docker exec -it cvat_server python manage.py createsuperuser
```

### Buat Project di Nuclio
```bash
sudo nuctl create project cvat --platform local
```

---

## 3. Konfigurasi function-gpu.yaml (PENTING)

> ⚠️ **Ini adalah langkah yang SERING TERLEWAT dan menjadi sumber error utama.**

File yang perlu diedit **sebelum deploy:**
```
~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml
```

### Tambahkan Port Statis pada Triggers

Tanpa port statis, Docker akan assign port secara **dinamis** setiap container di-restart.
Akibatnya, Nuclio masih menyimpan port lama → CVAT kirim request ke port yang salah → **504 Timeout**.

```bash
nano ~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml
```

Cari bagian `triggers` lalu tambahkan `port`:

```yaml
spec:
  triggers:
    myHttpTrigger:
      kind: http
      name: myHttpTrigger
      numWorkers: 1
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        port: 32769          # ← WAJIB DITAMBAHKAN: port statis
        maxRequestBodySize: 33554432
  
  # Naikkan timeout karena model SAM perlu waktu load
  eventTimeout: 120s         # ← default 30s, terlalu pendek untuk SAM
  readinessTimeoutSeconds: 180
```

> ℹ️ **Kenapa harus statis?** Docker container bersifat immutable. Setelah container berjalan,
> port yang terdaftar di Nuclio tidak otomatis update saat container restart dengan port baru.
> Port statis memastikan konsistensi antara metadata Nuclio dan binding Docker.

---

## 4. Deploy SAM Function

Setelah `function-gpu.yaml` dikonfigurasi, lakukan deploy **dari scratch**:

```bash
cd ~/cvat

# Deploy SAM GPU (proses ini ~10-15 menit pertama kali karena download model ~2.4GB)
sudo nuctl deploy \
    --project-name cvat \
    --path serverless/pytorch/facebookresearch/sam/nuclio \
    --file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
    --platform local
```

> ℹ️ Proses build image SAM memakan waktu lama karena:
> - Install dependencies CUDA/PyTorch
> - Download model `sam_vit_h_4b8939.pth` (~2.4GB) dari Facebook AI

---

## 5. Verifikasi Instalasi

Setelah deploy selesai, **verifikasi port konsisten** antara Nuclio API dan Docker:

```bash
# Cek port yang tercatat di Nuclio
curl -s http://localhost:8070/api/functions | python3 -m json.tool | grep -E "(httpPort|externalInvocation)"

# Cek port aktual di Docker
sudo docker ps | grep sam
```

**Hasil yang benar:** kedua output harus menunjukkan port yang SAMA (misal `32769`).

```bash
# Test invoke SAM secara langsung
curl -X POST http://localhost:32769/ \
    -H "Content-Type: application/json" \
    -d '{"image": "test"}' \
    --max-time 30 -v
```

---

## 6. Script Utilitas

Salin script berikut ke `~/cvat/` untuk memudahkan operasional harian:

### cvat-start.sh
Menjalankan CVAT + Nuclio + SAM sekaligus dengan deteksi otomatis apakah perlu deploy ulang atau cukup `docker start`.

### cvat-stop.sh
Menghentikan semua service **tanpa menghapus container**, sehingga data dan konfigurasi tetap tersimpan.

```bash
# Beri izin eksekusi
chmod +x ~/cvat/cvat-start.sh
chmod +x ~/cvat/cvat-stop.sh

# Penggunaan harian
~/cvat/cvat-start.sh   # nyalakan
~/cvat/cvat-stop.sh    # matikan
```

---

## 7. Troubleshooting

### 504 Gateway Timeout saat menggunakan SAM

**Penyebab paling umum:**

| Penyebab | Ciri | Solusi |
|---|---|---|
| Port mismatch | Port Nuclio ≠ port Docker | Paksa port statis di `function-gpu.yaml` |
| Model belum selesai load | Error hanya di request pertama | Tunggu ~60 detik setelah container healthy |
| `eventTimeout` terlalu pendek | Timeout konsisten setiap request | Naikkan ke `120s` atau `300s` |
| SAM container tidak di network yang benar | `docker network inspect` tidak tampil `cvat_cvat` | Pastikan `network: cvat_cvat` di platform spec |

### Cek Diagnosa Cepat

```bash
# 1. Semua container hidup?
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(cvat|nuclio|sam)"

# 2. SAM container ada di network yang benar?
sudo docker inspect nuclio-nuclio-pth-facebookresearch-sam-vit-h | grep -A5 "Networks"

# 3. Log error SAM
sudo docker logs nuclio-nuclio-pth-facebookresearch-sam-vit-h --tail 50

# 4. Port konsisten?
curl -s http://localhost:8070/api/functions | python3 -m json.tool | grep httpPort
sudo docker ps | grep sam
```

### Redeploy SAM dari Awal (jika semua fail)

```bash
# Step 1: Stop semua
~/cvat/cvat-stop.sh

# Step 2: Hapus container SAM lama
sudo docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h 2>/dev/null
sudo docker rm nuclio-nuclio-pth-facebookresearch-sam-vit-h 2>/dev/null

# Step 3: Hapus function dari registry Nuclio
curl -X DELETE http://localhost:8070/api/functions/pth-facebookresearch-sam-vit-h \
    -H "x-nuclio-function-namespace: nuclio"

# Step 4: Hapus image lama (opsional, paksa rebuild bersih)
sudo docker rmi cvat.pth.facebookresearch.sam.vit_h:latest-gpu 2>/dev/null

# Step 5: Down seluruh docker compose
cd ~/cvat
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    down

# Step 6: Edit function-gpu.yaml (pastikan port statis sudah ada)
nano ~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml

# Step 7: Up ulang & deploy
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d

# Tunggu Nuclio healthy
sleep 30

# Step 8: Deploy SAM
sudo nuctl deploy \
    --project-name cvat \
    --path serverless/pytorch/facebookresearch/sam/nuclio \
    --file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
    --platform local
```

---

## 8. Pelajaran dari Kesalahan

Bagian ini merangkum semua kesalahan yang ditemui selama proses instalasi.

---

### ❌ Kesalahan 1: Tidak Menset Port Statis di function-gpu.yaml

**Apa yang terjadi:**
Setiap kali container SAM di-restart (misal setelah `cvat-stop.sh` + `cvat-start.sh`),
Docker mengassign port berbeda secara dinamis (32769 → 32770 → 32771 dst).
Nuclio tetap menyimpan metadata port lama, sehingga CVAT mengirim request ke port yang
sudah tidak valid → **504 Gateway Timeout**.

**Bukti:**
```
# Nuclio API bilang port 32770
"httpPort": 32770

# Tapi Docker assign port 32771
0.0.0.0:32771->8080/tcp   nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

**Pelajaran:** Selalu set `port` statis di bagian `triggers.myHttpTrigger.attributes` pada `function-gpu.yaml` **sebelum** pertama kali deploy.

---

### ❌ Kesalahan 2: Menambahkan NUCLIO_INVOKE_DIRECT ke docker-compose.yml

**Apa yang terjadi:**
Percobaan menambahkan environment variable `NUCLIO_INVOKE_DIRECT: "true"` ke service
`backend-dev` di `docker-compose.yml` sebagai workaround, tapi tidak berhasil
dan justru menambah kebingungan karena bukan root cause yang sebenarnya.

**Pelajaran:** Selalu identifikasi root cause sebelum menambahkan konfigurasi workaround.
Tambahan konfigurasi yang tidak diperlukan bisa menyulitkan debugging selanjutnya.

---

### ❌ Kesalahan 3: Edit File di Host Berharap Langsung Berlaku di Container

**Apa yang terjadi:**
Mengedit `function-gpu.yaml` di host, tapi tidak melakukan `docker compose down` + deploy ulang.
Container yang sudah berjalan bersifat **immutable** — file di dalam container tidak berubah
hanya karena file sumbernya di host diedit.

**Pelajaran:** Setiap perubahan pada file konfigurasi function Nuclio **wajib** diikuti dengan:
1. `docker compose down` (bukan hanya `stop`)
2. Deploy ulang via `nuctl deploy`

---

### ❌ Kesalahan 4: Menggunakan docker start untuk Container SAM yang Stopped

**Apa yang terjadi:**
Script `cvat-start.sh` melakukan `docker start` pada container SAM yang stopped.
Ini menyebabkan container jalan ulang dengan **port baru** (dinamis), sementara Nuclio
masih mencatat port lama → mismatch → timeout.

**Pelajaran:** Jangan gunakan `docker start` untuk container SAM jika port tidak statis.
Dengan port statis di `function-gpu.yaml`, masalah ini tidak akan terjadi karena
port selalu konsisten meski container di-restart.

---

### ❌ Kesalahan 5: eventTimeout Terlalu Pendek (Default 30s)

**Apa yang terjadi:**
Default `eventTimeout` di Nuclio adalah `30s`. Model SAM ViT-H adalah model besar
yang butuh waktu lebih lama untuk memproses, terutama pada request pertama
(cold start + load model ke VRAM).

**Pelajaran:** Selalu set `eventTimeout` ke minimal `120s` untuk SAM di `function-gpu.yaml`.

---

### ✅ Ringkasan Checklist Sebelum Deploy

```
[ ] nuctl versi == versi Nuclio di docker-compose.serverless.yml
[ ] function-gpu.yaml: port statis ditambahkan di triggers
[ ] function-gpu.yaml: eventTimeout dinaikkan ke 120s+
[ ] docker compose down (bukan hanya stop) sebelum deploy ulang
[ ] Semua container di network cvat_cvat yang sama
[ ] Verifikasi httpPort Nuclio == port Docker setelah deploy
```
