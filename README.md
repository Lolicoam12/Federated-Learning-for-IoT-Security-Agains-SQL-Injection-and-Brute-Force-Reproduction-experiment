# KubeFlower: Kubernetes-based Federated Learning

*** The extension of this work can be found in [KubeFlower-Operator](https://github.com/REASON-6G/kubeflower-operator/tree/main) ***

# 要求
Verify GPU works in WSL2
check:
'''
nvidia-smi
'''
Minikube
Docker

## 專案概述
KubeFlower 是一個基於 Kubernetes 的聯邦學習 (Federated Learning) 系統，結合：

Flower Framework (v1.25.0) - 聯邦學習協調框架
Kubernetes/Minikube - 容器編排
Docker - 容器化 (支援 NVIDIA CUDA GPU)
PyTorch - 深度學習模型訓練
Pandas/Scikit-learn - 資料預處理
# 專案目錄結構

minikubeFlower/
├── .claude/                      # Claude Code agent 設定
│   └── agents/
│       └── federated-learning-architect.md
├── .flwr/                        # Flower 框架快取
├── .vscode/                      # VS Code 設定
├── Data/                         # 資料集目錄
│   └── data.csv                  # IDS 表格資料 (386MB, 110萬筆)
├── descriptors/                  # Kubernetes 部署清單
│   ├── serverDeploy.yaml        # Server 部署
│   ├── serverService.yaml       # Server 服務
│   ├── clientDeploy.yaml        # Client 部署 (單節點)
│   ├── clientDeploy_multinode.yaml  # Client 部署 (多節點 + Pod 反親性)
│   ├── volumeClaim.yaml         # PVC 儲存配置
│   └── copier.yaml              # 儲存輔助配置
├── src/                          # 應用程式原始碼
│   ├── server.py                # Flower Server (SuperLink)
│   └── client.py                # Flower Client (SuperNode)
├── dockerfile                    # Docker 映像定義 (CUDA GPU 支援)
├── pyproject.toml               # Flower 應用程式配置
├── requirements.txt             # Python 依賴套件
└── README.md                     # 專案文件

# 資源配置摘要
|元件	|CPU Request|	CPU Limit|	Memory Request	Memory Limit	GPU
|Server|	500m|	2 cores|	512Mi|	2Gi|	|無|
|Client (x2)|	2 cores|	4 cores|	4Gi	|8Gi	|待啟用|

# 關鍵配置參數
|參數|	值|	位置|	說明|
|num_rounds|	5|	pyproject.toml|	聯邦學習輪數|
|num_clients|	2|	pyproject.toml|	預期 Client 數量|
|min_available|	2|	pyproject.toml|	最少可用 Client|
|local_epochs|	1|	client.py:697|	每輪本地訓練週期|
|batch_size|	32|	client.py:546|	批次大小|
|learning_rate|	0.001|	client.py:114|	Adam 學習率|
|input_dim|	110|	資料衍生|	輸入特徵維度|
|hidden_layers|	[8, 8]|	client.py:683|	隱藏層維度|
|test_size|	0.2|	client.py:509|	測試集比例|
|scaling|	"standard"|	client.py:491|	數值縮放方法|
|encoding|	"onehot"|	client.py:492|	類別編碼方法|

# 啟動流程
設定視窗版的 Docker buildx
'''
minikube -p minikube docker-env --shell powershell | Invoke-Expression
'''
啟動minikube並配置硬體(須注意)
'''
minikube start --driver=docker --gpus=all --memory=16384 --cpus=8 --addons=ingress
'''
啟用GPU Plugin
'''
minikube addons enable nvidia-device-plugin
'''
驗證Node GPU
'''
kubectl describe node minikube | findstr /I nvidia
'''
另外開一個終端機視窗
'''
kubectl port-forward svc/service-server 9093:9093 9092:9092
'''
'''
flwr run . k8s-local --stream
'''
Go to the folder that contains Kubeflower
'''
docker build -t kubeflower:latest .
'''
kubectl apply -f descriptors/copier.yaml
kubectl apply -f descriptors/serverService.yaml
kubectl apply -f descriptors/serverDeploy.yaml
kubectl apply -f descriptors/clientDeploy.yaml
kubectl get pods -owide

查看日誌
顯示flower-client的Pod名稱
kubectl get pods -l app=flower-client
kubectl logs -f pod/(flower-client-pod名稱)

cd .\descriptors\
kubectl delete deploy flower-client flower-server
kubectl delete svc service-server
kubectl delete pvc my-pvc