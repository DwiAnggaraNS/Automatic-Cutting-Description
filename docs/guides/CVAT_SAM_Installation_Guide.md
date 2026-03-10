# CVAT + SAM Installation Guide on Ubuntu (Local)

> This document was written based on a real installation experience, including all errors encountered,
> to serve as an accurate reference for future installations.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [CVAT Installation](#2-cvat-installation)
3. [Configure function-gpu.yaml (IMPORTANT)](#3-configure-function-gpuyaml-important)
4. [Deploy SAM Function](#4-deploy-sam-function)
5. [Verify Installation](#5-verify-installation)
6. [Utility Scripts](#6-utility-scripts)
7. [Troubleshooting](#7-troubleshooting)
8. [Lessons Learned](#8-lessons-learned)

---

## 1. Prerequisites

### Hardware
- NVIDIA GPU (recommended ≥8GB VRAM for SAM ViT-H)
- RAM ≥ 16GB
- Storage ≥ 50GB (SAM image ~10GB)

### Software
```bash
# Docker Engine (not Docker Desktop)
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# NVIDIA Container Toolkit (to allow Docker to access the GPU)
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker

# nuctl CLI (Nuclio CLI)
# First check the Nuclio version used in docker-compose.serverless.yml
# then download nuctl with the EXACT SAME version
wget https://github.com/nuclio/nuclio/releases/download/<VERSION>/nuctl-<VERSION>-linux-amd64
chmod +x nuctl-<VERSION>-linux-amd64
sudo mv nuctl-<VERSION>-linux-amd64 /usr/local/bin/nuctl
```

> ⚠️ **IMPORTANT:** The `nuctl` CLI version must match the Nuclio image version in docker-compose.
> Version mismatch causes `nuctl deploy` to fail.

---

## 2. CVAT Installation

```bash
# Clone the CVAT repository
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# Start CVAT + Nuclio (serverless)
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d

# Create a superuser
sudo docker exec -it cvat_server python manage.py createsuperuser
```

### Create a Project in Nuclio
```bash
sudo nuctl create project cvat --platform local
```

---

## 3. Configure function-gpu.yaml (IMPORTANT)

> ⚠️ **This is the step most commonly skipped and is the primary source of errors.**

The file to edit **before deploying:**
```
~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml
```

### Add a Static Port to Triggers

Without a static port, Docker will assign ports **dynamically** every time the container is restarted.
As a result, Nuclio retains the old port → CVAT sends requests to the wrong port → **504 Timeout**.

```bash
nano ~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml
```

Locate the `triggers` section and add `port`:

```yaml
spec:
  triggers:
    myHttpTrigger:
      kind: http
      name: myHttpTrigger
      numWorkers: 1
      workerAvailabilityTimeoutMilliseconds: 10000
      attributes:
        port: 32769          # ← REQUIRED: static port
        maxRequestBodySize: 33554432
  
  # Increase timeout because the SAM model takes time to load
  eventTimeout: 120s         # ← default 30s is too short for SAM
  readinessTimeoutSeconds: 180
```

> ℹ️ **Why must it be static?** Docker containers are immutable. Once a container is running,
> the port registered in Nuclio does not automatically update when the container restarts with a new port.
> A static port ensures consistency between Nuclio metadata and the Docker binding.

---

## 4. Deploy SAM Function

After configuring `function-gpu.yaml`, deploy from scratch:

```bash
cd ~/cvat

# Deploy SAM GPU (first run takes ~10-15 minutes due to ~2.4GB model download)
sudo nuctl deploy \
    --project-name cvat \
    --path serverless/pytorch/facebookresearch/sam/nuclio \
    --file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
    --platform local
```

> ℹ️ The SAM image build takes a long time because:
> - Installing CUDA/PyTorch dependencies
> - Downloading model `sam_vit_h_4b8939.pth` (~2.4GB) from Facebook AI

---

## 5. Verify Installation

Once deployment is complete, **verify that the port is consistent** between the Nuclio API and Docker:

```bash
# Check the port registered in Nuclio
curl -s http://localhost:8070/api/functions | python3 -m json.tool | grep -E "(httpPort|externalInvocation)"

# Check the actual port in Docker
sudo docker ps | grep sam
```

**Expected result:** both outputs should show the SAME port (e.g., `32769`).

```bash
# Test invoking SAM directly
curl -X POST http://localhost:32769/ \
    -H "Content-Type: application/json" \
    -d '{"image": "test"}' \
    --max-time 30 -v
```

---

## 6. Utility Scripts

Copy the following scripts to `~/cvat/` for easier day-to-day operations:

### cvat-start.sh
Starts CVAT + Nuclio + SAM all at once, with automatic detection of whether a full redeploy is needed or a simple `docker start` is sufficient.

### cvat-stop.sh
Stops all services **without removing containers**, preserving all data and configuration.

```bash
# Grant execution permission
chmod +x ~/cvat/cvat-start.sh
chmod +x ~/cvat/cvat-stop.sh

# Daily usage
~/cvat/cvat-start.sh   # start
~/cvat/cvat-stop.sh    # stop
```

---

## 7. Troubleshooting

### 504 Gateway Timeout when using SAM

**Most common causes:**

| Cause | Symptom | Solution |
|---|---|---|
| Port mismatch | Nuclio port ≠ Docker port | Force static port in `function-gpu.yaml` |
| Model not finished loading | Error only on first request | Wait ~60 seconds after container is healthy |
| `eventTimeout` too short | Consistent timeout on every request | Increase to `120s` or `300s` |
| SAM container not on the correct network | `docker network inspect` doesn't show `cvat_cvat` | Ensure `network: cvat_cvat` in platform spec |

### Quick Diagnostic Check

```bash
# 1. Are all containers running?
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(cvat|nuclio|sam)"

# 2. Is the SAM container on the correct network?
sudo docker inspect nuclio-nuclio-pth-facebookresearch-sam-vit-h | grep -A5 "Networks"

# 3. SAM error logs
sudo docker logs nuclio-nuclio-pth-facebookresearch-sam-vit-h --tail 50

# 4. Port consistent?
curl -s http://localhost:8070/api/functions | python3 -m json.tool | grep httpPort
sudo docker ps | grep sam
```

### Redeploy SAM from Scratch (if everything fails)

```bash
# Step 1: Stop everything
~/cvat/cvat-stop.sh

# Step 2: Remove old SAM container
sudo docker stop nuclio-nuclio-pth-facebookresearch-sam-vit-h 2>/dev/null
sudo docker rm nuclio-nuclio-pth-facebookresearch-sam-vit-h 2>/dev/null

# Step 3: Remove the function from Nuclio registry
curl -X DELETE http://localhost:8070/api/functions/pth-facebookresearch-sam-vit-h \
    -H "x-nuclio-function-namespace: nuclio"

# Step 4: Remove old image (optional, forces a clean rebuild)
sudo docker rmi cvat.pth.facebookresearch.sam.vit_h:latest-gpu 2>/dev/null

# Step 5: Bring down the entire docker compose
cd ~/cvat
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    down

# Step 6: Edit function-gpu.yaml (ensure static port is already set)
nano ~/cvat/serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml

# Step 7: Bring up and deploy
sudo docker compose \
    -f docker-compose.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d

# Wait for Nuclio to be healthy
sleep 30

# Step 8: Deploy SAM
sudo nuctl deploy \
    --project-name cvat \
    --path serverless/pytorch/facebookresearch/sam/nuclio \
    --file serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml \
    --platform local
```

---

## 8. Lessons Learned

This section summarizes all errors encountered during the installation process.

---

### ❌ Mistake 1: Not Setting a Static Port in function-gpu.yaml

**What happened:**
Every time the SAM container was restarted (e.g., after `cvat-stop.sh` + `cvat-start.sh`),
Docker dynamically assigned a different port (32769 → 32770 → 32771, etc.).
Nuclio still stored the old port metadata, so CVAT sent requests to an invalid port → **504 Gateway Timeout**.

**Evidence:**
```
# Nuclio API reported port 32770
"httpPort": 32770

# But Docker assigned port 32771
0.0.0.0:32771->8080/tcp   nuclio-nuclio-pth-facebookresearch-sam-vit-h
```

**Lesson:** Always set a static `port` in the `triggers.myHttpTrigger.attributes` section of `function-gpu.yaml` **before** the first deployment.

---

### ❌ Mistake 2: Adding NUCLIO_INVOKE_DIRECT to docker-compose.yml

**What happened:**
An attempt was made to add the environment variable `NUCLIO_INVOKE_DIRECT: "true"` to the
`backend-dev` service in `docker-compose.yml` as a workaround, but it did not work
and only added confusion because it was not the actual root cause.

**Lesson:** Always identify the root cause before adding workaround configurations.
Unnecessary configuration additions can complicate future debugging.

---

### ❌ Mistake 3: Editing Files on the Host and Expecting Them to Take Effect in the Container

**What happened:**
Editing `function-gpu.yaml` on the host without running `docker compose down` + redeploying.
Already-running containers are **immutable** — files inside the container do not change
just because the source file on the host was edited.

**Lesson:** Every change to a Nuclio function configuration file **must** be followed by:
1. `docker compose down` (not just `stop`)
2. Redeploy via `nuctl deploy`

---

### ❌ Mistake 4: Using docker start for a Stopped SAM Container

**What happened:**
The `cvat-start.sh` script ran `docker start` on a stopped SAM container.
This caused the container to restart with a **new (dynamic) port**, while Nuclio
still recorded the old port → mismatch → timeout.

**Lesson:** Do not use `docker start` for the SAM container if the port is not static.
With a static port in `function-gpu.yaml`, this issue will not occur because
the port remains consistent even after container restarts.

---

### ❌ Mistake 5: eventTimeout Too Short (Default 30s)

**What happened:**
The default Nuclio `eventTimeout` is `30s`. The SAM ViT-H model is large
and requires more processing time, especially on the first request
(cold start + loading the model into VRAM).

**Lesson:** Always set `eventTimeout` to at least `120s` for SAM in `function-gpu.yaml`.

---

### ✅ Pre-Deploy Checklist

```
[ ] nuctl version == Nuclio version in docker-compose.serverless.yml
[ ] function-gpu.yaml: static port added to triggers
[ ] function-gpu.yaml: eventTimeout increased to 120s+
[ ] docker compose down (not just stop) before redeploying
[ ] All containers on the same cvat_cvat network
[ ] Verify Nuclio httpPort == Docker port after deployment
```
