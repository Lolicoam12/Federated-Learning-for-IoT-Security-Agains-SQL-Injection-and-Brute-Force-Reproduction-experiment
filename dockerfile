FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN mkdir -p /app/data /app/cache

# GPU torch first
RUN python3 -m pip install --no-cache-dir \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY ./src /app/src
COPY ./Data/data.csv /app/data/data.csv

ENV DATA_PATH=/app/data/data.csv
ENV CACHE_DIR=/app/cache
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]

#設定視窗版的 Docker buildx
# minikube -p minikube docker-env --shell powershell | Invoke-Expression

# docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
# minikube start --driver=docker --gpus=all --memory=16384 --cpus=8 --addons=ingress
#啟用GPU Plugin(不確定下面程式是否剛需)
# minikube addons enable nvidia-device-plugin
#驗證Node GPU
#kubectl describe node minikube | findstr /I nvidia

#修改後重新建置映像檔
# cd .\descriptors\
# kubectl delete deploy flower-client flower-server
# kubectl delete svc service-server
# kubectl delete pvc my-pvc
# cd D:\DockerKubernetes\minikube_Use_FL\minikube_Use_FL_re\minikubeFlower
# docker build -t kubeflower:latest .
# kubectl apply -f descriptors/copier.yaml
# kubectl apply -f descriptors/serverService.yaml
# kubectl apply -f descriptors/serverDeploy.yaml
# kubectl apply -f descriptors/clientDeploy.yaml
# kubectl get pods -owide
#另外開一個終端機視窗
# kubectl port-forward svc/service-server 9093:9093 9092:9092
# flwr run . k8s-local --stream
#查看日誌
#顯示flower-client的Pod名稱
# kubectl get pods -l app=flower-client
# kubectl logs -f pod/<flower-client-pod名稱>

#原cpu版
# FROM python:3.9.8-slim-bullseye
# \WORKDIR /app
# RUN /usr/local/bin/python -m pip install --upgrade pip
# COPY ./requirements.txt .
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
# COPY ./src ./src #測試PVC
# COPY ./Data/data.csv ./data/data.csv # 統一給 client.py 使用的資料路徑
# ENV DATA_PATH=/app/data
# ENV CACHE_DIR=/app/cache # 預設什麼都不做，讓 K8s 的 command 去覆蓋成
# # - flower-superlink（server） # - flower-supernode（client）
# CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
