import warnings  # 用於忽略警告訊息
from collections import OrderedDict  # 用於有序字典，幫助模型參數管理
import flwr as fl  # Flower 聯邦學習框架
import torch  # PyTorch 深度學習框架
import torch.nn as nn  # PyTorch 神經網絡模組
import torch.nn.functional as F  # PyTorch 函數模組（如激活函數）
from torch.utils.data import DataLoader, TensorDataset  # PyTorch 資料載入器與資料集
from tqdm import tqdm  # 進度條顯示工具
# import argparse  # 命令列參數解析器
import pandas as pd  # 資料處理庫，用於讀取 CSV 等
import os  # 作業系統相關操作，如路徑處理
import numpy as np  # 數值計算庫
from sklearn.pipeline import Pipeline  # sklearn 管道，用於資料預處理流程
from sklearn.impute import SimpleImputer  # 缺失值填補器
from sklearn.preprocessing import ( StandardScaler, MinMaxScaler,
     OneHotEncoder, OrdinalEncoder, LabelEncoder)  # 資料預處理工具（如縮放、編碼）
from sklearn.compose import ColumnTransformer  # 欄位轉換器
from sklearn.model_selection import train_test_split  # 資料切分工具
from sklearn.metrics import f1_score  # F1 分數評估指標
from sklearn.utils.class_weight import compute_class_weight  # 計算類別權重，用於不平衡資料
from flwr.app import Context
from flwr.clientapp import ClientApp
import json
import hashlib
from pathlib import Path
import os, time

#cache 路徑工具
def _get_cache_paths(datapath: str) -> tuple[Path, Path, Path]:
    cache_root = Path(os.environ.get("CACHE_DIR", "/app/cache"))
    pod = os.environ.get("POD_NAME", "pod-unknown")
    cache_dir = cache_root / pod
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = cache_dir / "dataset.npz"
    meta_path = cache_dir / "meta.json"
    lock_path = cache_dir / ".lock"
    return npz_path, meta_path, lock_path

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# 已廢棄（DEPRECATED）：此函數不應直接呼叫。
# 它依賴尚未初始化的全域變數（scaling、encoding、drop_empty_rows），
# 獨立呼叫會觸發 NameError。主流程已不使用此函數；
# DataCache 內部有等效且正確的 meta 機制。
def _current_cache_meta(csv_path: str) -> dict:
    return {
        "csv_sha256": _file_sha256(csv_path),
        "scaling": scaling,
        "encoding": encoding,
        "drop_empty_rows": drop_empty_rows,
        "split": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
        "version": 1,
    }

def _meta_matches(meta_path: Path, current: dict) -> bool:
    try:
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta == current
    except Exception:
        return False

# ========= cache (per-pod / per-process) =========
_DATA_CACHE = None          # tuple
_DATA_CACHE_PATH = None     # str

# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader

warnings.filterwarnings("ignore", category=UserWarning)  # 忽略使用者警告
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 設定裝置為 GPU（如果可用）否則 CPU

# 模型定義：帶 BatchNorm + Dropout 的 MLP，用於 IDS 表格特徵
class Net(nn.Module):
    """MLP with BatchNorm and Dropout for IDS tabular features"""

    def __init__(self, input_dim=110, hidden=[128, 64, 32], n_classes=1, dropout: float = 0.3):
        super(Net, self).__init__()  # 呼叫父類初始化
        layers = []  # 層列表
        prev = input_dim  # 前一層輸出維度初始化為輸入維度

        # 隱藏層：Linear → BatchNorm1d → ReLU → Dropout
        for h in hidden:  # 依隱藏層大小迴圈
            layers.append(nn.Linear(prev, h))       # 全連接層
            layers.append(nn.BatchNorm1d(h))        # 批次正規化，穩定訓練並加速收斂
            layers.append(nn.ReLU())                # ReLU 激活
            layers.append(nn.Dropout(p=dropout))    # Dropout 防止過擬合
            prev = h  # 更新前一層輸出維度

        # 輸出層
        layers.append(nn.Linear(prev, n_classes))  # 添加輸出全連接層
        self.net = nn.Sequential(*layers)  # 將層列表轉為 Sequential 模型

        # 根據任務決定輸出激活
        self.is_binary = n_classes == 1  # 判斷是否為二元分類

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向傳播
        return self.net(x)  # 通過模型計算輸出

class FocalLoss(nn.Module):
    """二元 Focal Loss，用於 IDS 不平衡分類。

    gamma=2 為標準設定，alpha 等同 BCEWithLogitsLoss 的 pos_weight，
    用於偏重少數類（SQLi）的 loss。
    """
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # SQLi 類別的 pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.alpha, reduction='none'
        )
        p_t = torch.exp(-bce)                        # 等同 sigmoid(logit)（正確類別機率）
        focal_weight = (1.0 - p_t) ** self.gamma     # easy examples 降低權重
        return (focal_weight * bce).mean()


# 訓練函數：訓練模型並在最後一個 epoch 後驗證
def train(
    net,
    trainloader,
    valloader,
    epochs,
    class_weights,
    is_binary,
    proximal_mu: float = 0.0,
    global_params=None,
    current_round: int = 0,
):
    """
    Train the model on the training set with optional FedProx proximal term.

    若 proximal_mu > 0 且 global_params 不為 None，則在每個 batch 的 task loss 後
    加入近端項：(proximal_mu / 2) * ||w - w_global||^2，
    限制本地模型不要偏離全局模型太遠，改善 non-IID 下的收斂穩定性。
    """
    # if is_binary:  # 二元分類
    #     pos_weight = (
    #         torch.tensor([class_weights[0] / class_weights[1]]).to(DEVICE)
    #         if class_weights is not None
    #         else None)  # 計算正類權重
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # BCE 損失（帶 logits）
    if is_binary:
        # Focal Loss：強化 hard examples（SQLi 攻擊），改善 Precision/Recall 雙低問題
        # alpha 等同 BCEWithLogitsLoss 的 pos_weight，偏重少數類（SQLi）
        alpha = (
            torch.tensor([class_weights[1] / class_weights[0]]).to(DEVICE)
            if class_weights is not None
            else None
        )
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
    else:  # 多類分類
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE)
            if class_weights is not None
            else None)  # 交叉熵損失
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Adam 最佳化器

    # 若啟用 FedProx，將全局參數移至 DEVICE，用於近端項計算
    # global_params 已由 fit() 傳入 tensor list（僅含可學習參數，對應 net.parameters()）
    global_tensors = None
    if proximal_mu > 0.0 and global_params is not None:
        global_tensors = [p.to(DEVICE) for p in global_params]

    for epoch in range(epochs):  # 每個 epoch 迴圈
        t0 = time.time()
        epoch_loss = 0.0
        net.train()  # 設定模型為訓練模式
        for inputs, labels in tqdm(trainloader, disable=True):  # 遍歷訓練資料（K8s 非 TTY 禁用進度條）
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # 移到裝置
            optimizer.zero_grad()  # 清零梯度
            outputs = net(inputs)  # 前向計算
            if is_binary:  # 二元損失計算
                loss = criterion(outputs, labels.float())
            else:  # 多類損失計算
                loss = criterion(outputs, labels)

            # FedProx 近端項：(proximal_mu / 2) * ||w - w_global||^2
            if global_tensors is not None:
                proximal_term = sum(
                    torch.sum((w - w_g) ** 2)
                    for w, w_g in zip(net.parameters(), global_tensors)
                )
                loss = loss + (proximal_mu / 2.0) * proximal_term

            loss.backward()  # 反向傳播
            optimizer.step()  # 更新參數
            epoch_loss += loss.item()
        print(f"[TRAIN] round={current_round} epoch={epoch+1}/{epochs} "
              f"loss={epoch_loss/len(trainloader):.4f} elapsed={time.time()-t0:.1f}s")

# 測試函數：評估模型在測試集上的表現
def test(net, testloader, is_binary, debug=False):
    """Validate the model on the test set."""
    net.eval()  # 切換為評估模式：停用 BatchNorm 的 running stats 更新與 Dropout
    if is_binary:  # 二元分類損失
        criterion = nn.BCEWithLogitsLoss()
    else:  # 多類分類損失
        criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0  # 初始化正確數、總數、損失
    all_preds, all_labels = [], []  # 收集所有預測與標籤
    with torch.no_grad():  # 無梯度模式
        for inputs, labels in tqdm(testloader):  # 遍歷測試資料
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # 移到裝置
            outputs = net(inputs)  # 前向計算
            if is_binary:  # 二元處理
                batch_loss = criterion(outputs, labels.float()).item()  # 批次損失
                pred = (torch.sigmoid(outputs) > 0.3).float().cpu().numpy()  # 降低閾值提升 Recall（入侵偵測漏報代價高）
                labels_np = labels.cpu().numpy()  # 標籤轉 numpy
            else:  # 多類處理
                batch_loss = criterion(outputs, labels).item()  # 批次損失
                pred = torch.argmax(outputs, dim=1).cpu().numpy()  # argmax 預測
                labels_np = labels.cpu().numpy()  # 標籤轉 numpy
            loss += batch_loss * labels.size(0)  # 累加損失
            total += labels.size(0)  # 累加總數
            correct += (pred == labels_np).sum()  # 累加正確數
            all_preds.extend(pred)  # 收集預測
            all_labels.extend(labels_np)  # 收集標籤
    avg_loss = loss / total  # 平均損失
    acc = correct / total  # 準確率

    if is_binary:  # 二元 F1
        f1 = f1_score(all_labels, all_preds, average='binary')
    else:  # 多類 F1 (macro)
        f1 = f1_score(all_labels, all_preds, average='macro')

    if debug:
        # [DEBUG] 預測分佈：看模型是不是全部猜 0（Benign）
        unique, counts = np.unique(all_preds, return_counts=True)
        print("[DEBUG] pred distribution:", dict(zip(unique, counts)))
    
    return avg_loss, acc, f1  # 返回損失、準確率、F1

#  工具：欄名清理/標籤偵測
def clean_col(c: str) -> str:  # 清理欄位名稱
    c = str(c).strip()  # 移除前後空格
    c = "".join(ch for ch in c if ch.isprintable())  # 保留可列印字元
    c = c.replace("\u200b", "").replace("\ufeff", "")  # 移除特定 Unicode
    c = "_".join(c.split())  # 空格轉下劃線
    return c

def auto_pick_label(df: pd.DataFrame) -> str:  # 自動選擇標籤欄
    candidates = ["label","Label","target","Target","class","Class","attack_cat","Attack","is_attack","y"]  # 候選欄名
    for c in candidates:  # 檢查是否存在
        if c in df.columns:
            return c
    raise ValueError("找不到標籤欄，請將變數 label_col 設為你的標籤欄名。")

#  前處理（fit 在訓練集）
def fit_preprocessor(X_train: pd.DataFrame):  # 擬合預處理器
    # 類別欄：object 或 category；其餘視為數值
    cat_cols = [
        c
        for c in X_train.columns
        if (X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category"))]  # 類別欄列表
    num_cols = [c for c in X_train.columns if c not in cat_cols]  # 數值欄列表

    # 數值：中位數 → 縮放
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # 中位數填補
            ("scaler", StandardScaler() if scaling == "standard" else MinMaxScaler())  # 標準化或 MinMax 縮放
        ]
    )

    # 類別：眾數 → 編碼
    if encoding == "onehot":  # OneHot 編碼
        cat_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False)
    else:  # Ordinal 編碼
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # 眾數填補
        ("encoder", cat_encoder)  # 編碼器
        ]
    )

    pre = ColumnTransformer(  # 欄位轉換器
        transformers=[
            ("num", num_pipe, num_cols),  # 數值管道
            ("cat", cat_pipe, cat_cols),  # 類別管道
        ],
        remainder="drop"  # 忽略其餘
    )
    pre.fit(X_train)  # 擬合訓練集
    return pre, num_cols, cat_cols  # 返回預處理器與欄列表

def transform_with_preprocessor(pre, X_df: pd.DataFrame) -> np.ndarray:  # 使用預處理器轉換資料
    return pre.transform(X_df)  # 轉換並返回 numpy 陣列

#  資料讀取與切分
def load_and_split(
    csv_path: str,
    label_col: str = None,
    partition_id: int = 0,
    num_partitions: int = 1,
):
    """
    讀取 CSV 並切分資料。
    若 num_partitions > 1，則對 train 集進行 IID 切片，
    每個 partition 持有不同的子集；val/test 保持全量以確保評估一致性。
    """
    df = pd.read_csv(csv_path, low_memory=False)  # 讀取 CSV
    df.columns = [clean_col(c) for c in df.columns]  # 清理欄名
    if drop_empty_rows:  # 若設定，移除全空行
        df = df.dropna(how="all")

    if label_col is None:  # 自動選擇標籤
        label = auto_pick_label(df)
    else:
        label = label_col
        if label not in df.columns:  # 檢查是否存在
            raise ValueError(f"指定的標籤欄 `{label}` 不在 CSV 欄位中。")

    # 要篩選的三種類別（小寫比對）(tag)
    # TARGETS = {"dictionarybruteforce", "sqlinjection", "benigntraffic"}  # 定義目標類別
    TARGETS = {"sqlinjection", "benigntraffic"}  # 定義目標類別
    df[label] = df[label].str.lower().str.replace(" ", "")  # 將標籤轉小寫並移除空格，以確保比對一致

    # 過濾只保留指定的類別
    df = df[df[label].isin(TARGETS)]  # 過濾資料框，只保留目標類別

    y_raw = df[label]  # 原始標籤
    X_raw = df.drop(columns=[label])  # 特徵

    # 先切 train/test，避免資料外洩
    y_for_split = y_raw.astype(str) if (y_raw.dtype == object or str(y_raw.dtype).startswith("category")) else y_raw  # 用於分層的標籤
    X_train_full, X_test, y_train_full, y_test_raw = train_test_split(
        X_raw, y_for_split, test_size=0.2, random_state=42, stratify=y_for_split  # 切分 80/20
    )
    # 從 train_full 再切 val (20% of train_full)
    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42,
        stratify=y_train_full  # 從訓練集切 80/20 為 train/val
    )

    # IID Partition 切片：各 client 持有不同的訓練子集（val/test 保持全量）
    if num_partitions > 1:
        n = len(X_train)
        # 使用固定 seed 確保每個 pod 的索引排列相同，再依 partition_id 取對應片段
        rng = np.random.default_rng(seed=42 + partition_id)
        indices = rng.permutation(n)
        # 計算此 partition 的起止索引（等分切片）
        chunk = n // num_partitions
        start = partition_id * chunk
        end = start + chunk if partition_id < num_partitions - 1 else n
        selected = indices[start:end]
        X_train = X_train.iloc[selected].reset_index(drop=True)
        y_train_raw = y_train_raw.iloc[selected].reset_index(drop=True)
        print(
            f"[load_and_split] IID partition {partition_id}/{num_partitions} "
            f"-> train_n={len(X_train)} (全量={n})"
        )

        # Val 集同樣切片，使本地 val 分布與 train partition 一致
        n_val = len(X_val)
        rng_val = np.random.default_rng(seed=99 + partition_id)  # 不同 seed 避免與 train 相關
        val_indices = rng_val.permutation(n_val)
        val_chunk = n_val // num_partitions
        val_start = partition_id * val_chunk
        val_end = val_start + val_chunk if partition_id < num_partitions - 1 else n_val
        val_selected = val_indices[val_start:val_end]
        X_val = X_val.iloc[val_selected].reset_index(drop=True)
        y_val_raw = y_val_raw.iloc[val_selected].reset_index(drop=True)
        print(
            f"[load_and_split] IID partition {partition_id}/{num_partitions} "
            f"-> val_n={len(X_val)} (全量={n_val})"
        )

    return X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw, label  # 返回切分結果

#  標籤編碼（二元 / 多類）
def encode_labels(y_train_raw, y_val_raw, y_test_raw):  # 編碼標籤
    le = LabelEncoder()  # 標籤編碼器
    y_train_idx = le.fit_transform(np.asarray(y_train_raw).astype(str))  # 擬合並轉換訓練標籤
    y_val_idx = le.transform(np.asarray(y_val_raw).astype(str))  # 轉換驗證標籤
    y_test_idx = le.transform(np.asarray(y_test_raw).astype(str))  # 轉換測試標籤
    classes = le.classes_  # 類別列表
    num_classes = len(classes)  # 類別數
    is_binary = (num_classes == 2)  # 是否二元
    return y_train_idx, y_val_idx, y_test_idx, num_classes, is_binary, le  # 返回編碼結果


class DataCache:
    """
    把 load_data() 的「前處理輸出」落盤到 cache_dir
    - X_train.npy, X_val.npy, X_test.npy
    - y_train.npy, y_val.npy, y_test.npy
    - class_weights.npy
    - meta.json
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 固定檔名（方案 A）
        self.x_train = self.cache_dir / "X_train.npy"
        self.x_val   = self.cache_dir / "X_val.npy"
        self.x_test  = self.cache_dir / "X_test.npy"
        self.y_train = self.cache_dir / "y_train.npy"
        self.y_val   = self.cache_dir / "y_val.npy"
        self.y_test  = self.cache_dir / "y_test.npy"
        self.cw      = self.cache_dir / "class_weights.npy"
        self.meta    = self.cache_dir / "meta.json"

        # 寫入鎖（避免 2 個 client 同時建 cache）
        self.lock = self.cache_dir / ".build.lock"

        # 前處理版本號
        self.preprocessing_version = "v1"

    def _hash_file(self, path: Path, algo="sha256", chunk_size=1024 * 1024) -> str:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _make_meta(self, csv_path: str, config: dict, extra: dict) -> dict:
        p = Path(csv_path)
        meta = {
            "preprocessing_version": self.preprocessing_version,
            "csv_path": str(p),
            "csv_hash": self._hash_file(p) if p.exists() else None,
            "csv_stat": {
                "size": p.stat().st_size if p.exists() else None,
                "mtime": p.stat().st_mtime if p.exists() else None,
            },
            "config": config,   # 前處理的關鍵設定（會拿來比對）
            "extra": extra,     # 例如：classes、input_dim、n_classes、is_binary
        }
        return meta

    def _read_meta(self) -> dict:
        with open(self.meta, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_meta_atomic(self, meta: dict):
        tmp = self.cache_dir / "meta.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.meta)

    def _all_cache_files_exist(self) -> bool:
        return all([
            self.x_train.exists(), self.x_val.exists(), self.x_test.exists(),
            self.y_train.exists(), self.y_val.exists(), self.y_test.exists(),
            self.cw.exists(),
            self.meta.exists(),
        ])

    def is_valid(self, csv_path: str, config: dict) -> tuple[bool, str]:
        # 檢查 cache 是否存在且 meta 匹配
        if not self._all_cache_files_exist():
            return False, "cache files missing"

        try:
            meta = self._read_meta()
        except Exception:
            return False, "meta.json unreadable"

        if meta.get("preprocessing_version") != self.preprocessing_version:
            return False, "preprocessing_version mismatch"

        # 比對 config
        if meta.get("config") != config:
            return False, "config mismatch"

        # 比對 CSV hash（確保原始資料沒變）
        p = Path(csv_path)
        if not p.exists():
            return False, "csv missing"

        current_hash = self._hash_file(p)
        if meta.get("csv_hash") != current_hash:
            return False, "csv hash mismatch"

        return True, "cache hit"

    def load(self):
        # 從 cache 讀回 numpy arrays（再由外層轉 torch）
        X_train = np.load(self.x_train, allow_pickle=False)
        X_val   = np.load(self.x_val, allow_pickle=False)
        X_test  = np.load(self.x_test, allow_pickle=False)
        y_train = np.load(self.y_train, allow_pickle=False)
        y_val   = np.load(self.y_val, allow_pickle=False)
        y_test  = np.load(self.y_test, allow_pickle=False)
        class_weights = np.load(self.cw, allow_pickle=False)

        meta = self._read_meta()
        extra = meta.get("extra", {})

        return X_train, X_val, X_test, y_train, y_val, y_test, class_weights, extra

    def _acquire_lock(self, timeout_sec=1800, poll=0.5):
        start = time.time()
        printed = False
        while True:
            try:
                fd = os.open(str(self.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                print(f"[CACHE][LOCK][ACQUIRED] lock={self.lock}")
                return
            except FileExistsError:
                if not printed:
                    print(f"[CACHE][LOCK][WAIT] lock={self.lock} timeout={timeout_sec}s poll={poll}s")
                    printed = True
                if time.time() - start > timeout_sec:
                    raise TimeoutError("Timeout waiting for cache build lock")
                time.sleep(poll)

    def _release_lock(self):
        try:
            if self.lock.exists():
                self.lock.unlink()
        except Exception:
            pass

    def save(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        class_weights,
        csv_path: str,
        config: dict,
        extra: dict,
    ):
        # 原子性寫入：先寫 tmp 再 replace

        self._acquire_lock()
        try:
            # 再確認一次別人已經建好了
            ok, reason = self.is_valid(csv_path, config)
            if ok:
                print(f"[DataCache] another process built cache already ({reason})")
                return

            def _atomic_save(arr, path: Path):
                tmp = Path(str(path) + ".tmp")
                with open(tmp, "wb") as f:
                    np.save(f, arr, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)


            # 強制 dtype
            X_train = np.asarray(X_train, dtype=np.float32)
            X_val   = np.asarray(X_val, dtype=np.float32)
            X_test  = np.asarray(X_test, dtype=np.float32)

            # y：二元是 float32 shape (N,1)，多類是 int64 shape (N,)
            y_train = np.asarray(y_train)
            y_val   = np.asarray(y_val)
            y_test  = np.asarray(y_test)

            class_weights = np.asarray(class_weights, dtype=np.float32)

            _atomic_save(X_train, self.x_train)
            _atomic_save(X_val,   self.x_val)
            _atomic_save(X_test,  self.x_test)
            _atomic_save(y_train, self.y_train)
            _atomic_save(y_val,   self.y_val)
            _atomic_save(y_test,  self.y_test)
            _atomic_save(class_weights, self.cw)

            meta = self._make_meta(csv_path, config=config, extra=extra)
            self._write_meta_atomic(meta)

        finally:
            self._release_lock()

# 主資料載入函數（含快取）
def load_data_with_cache(datapath: str, num_clients: int = 1):
    t0 = time.time()
    """Load tabular data with disk cache (DataCache) under CACHE_DIR (PVC)."""
    # print(f"Loading data from {datapath}")
    # 統一 log 格式
    pod = os.environ.get("POD_NAME", "pod-unknown")
    pid = os.getpid()

    def _clog(event: str, **kv):
        # event: HIT/MISS/REBUILD/LOAD/SAVE/LOCK-WAIT...
        base = {
            "pod": pod,
            "pid": pid,
        }
        base.update(kv)
        extras = " ".join([f"{k}={v}" for k, v in base.items()])
        print(f"[CACHE][{event}] {extras}")

    _clog("ENTER", datapath=datapath)

    # 從 POD_NAME 末尾數字解析 partition_id（e.g. flower-client-3 -> 3）
    num_partitions = max(1, num_clients)
    try:
        partition_id = int(pod.rsplit("-", 1)[-1])
        # 防止 partition_id 超出範圍
        partition_id = partition_id % num_partitions
    except (ValueError, IndexError):
        partition_id = 0
    _clog("PARTITION", partition_id=partition_id, num_partitions=num_partitions)

    global scaling, encoding, drop_empty_rows
    scaling = "standard"
    encoding = "onehot"
    drop_empty_rows = True

    csv_path = os.path.join(datapath, "data.csv")

    #每個 Pod 用自己的子目錄，避免兩個 client pod 同時寫壞同一份 cache
    cache_root = os.environ.get("CACHE_DIR", "/app/cache")
    # pod = os.environ.get("POD_NAME", "pod-unknown")
    cache_dir = str(Path(cache_root) / pod)

    cache = DataCache(cache_dir)

    # config：只要改了這些設定，就會自動判定 cache invalid ,重建
    config = {
        "scaling": scaling,
        "encoding": encoding,
        "drop_empty_rows": drop_empty_rows,
        "split": {"test_size": 0.2, "val_from_train": 0.2, "random_state": 42, "stratify": True},
        "targets": ["sqlinjection", "benigntraffic"], #變更目標標籤會重建(tag)
        "label_col": None,  # 現在是 auto_pick_label
        "partition_id": partition_id,       # partition 編號，不同 pod 持有不同訓練子集
        "num_partitions": num_partitions,   # 總 partition 數，改變時觸發 cache 重建
    }

    ok, reason = cache.is_valid(csv_path, config)
    _clog("CHECK", cache_dir=cache_dir, csv=csv_path, ok=ok, reason=reason)
    if ok:
        _clog("HIT", cache_dir=cache_dir, reason=reason)
        X_train, X_val, X_test, y_train, y_val, y_test, cw_np, extra = cache.load()
        _clog("LOAD", sec=f"{time.time()-t0:.2f}", input_dim=extra.get("input_dim"), is_binary=extra.get("is_binary"))
        input_dim = int(extra["input_dim"])
        n_classes = int(extra["n_classes"])
        is_binary = bool(extra["is_binary"])

        class_weights = torch.tensor(cw_np, dtype=torch.float32)

        X_train_t = torch.from_numpy(X_train).float()
        X_val_t   = torch.from_numpy(X_val).float()
        X_test_t  = torch.from_numpy(X_test).float()

        if is_binary:
            # cache 裡已經是 (N,1) float32
            y_train_t = torch.from_numpy(y_train).float()
            y_val_t   = torch.from_numpy(y_val).float()
            y_test_t  = torch.from_numpy(y_test).float()
        else:
            y_train_t = torch.from_numpy(y_train).long()
            y_val_t   = torch.from_numpy(y_val).long()
            y_test_t  = torch.from_numpy(y_test).long()

        trainset = TensorDataset(X_train_t, y_train_t)
        valset   = TensorDataset(X_val_t, y_val_t)
        testset  = TensorDataset(X_test_t, y_test_t)

        _clog("RETURN", mode="HIT", train_n=len(trainset), val_n=len(valset), test_n=len(testset))
        return (
            DataLoader(trainset, batch_size=512, shuffle=True),
            DataLoader(valset, batch_size=512, shuffle=False),
            DataLoader(testset, batch_size=512, shuffle=False),
            input_dim,
            n_classes,
            is_binary,
            class_weights,
        )

    # print(f"[load_data_with_cache] cache MISS ({reason}) -> rebuild at {cache_dir}")
    _clog("MISS", cache_dir=cache_dir, reason=reason)
    t_build = time.time()

    # cache miss -> 重新建立（傳入 partition 參數進行 IID 切片）
    (
        X_train,
        X_val,
        X_test,
        y_train_raw,
        y_val_raw,
        y_test_raw,
        label,
    ) = load_and_split(csv_path, partition_id=partition_id, num_partitions=num_partitions)

    (
        y_train_idx,
        y_val_idx,
        y_test_idx,
        num_classes,
        is_binary,
        le,
    ) = encode_labels(y_train_raw, y_val_raw, y_test_raw)

    # ===== DEBUG: label mapping & distribution =====
    print("[DEBUG] LabelEncoder classes_:", le.classes_)
    for i, c in enumerate(le.classes_):
        print(f"[DEBUG] label index {i} -> class '{c}'")

    uniq, cnt = np.unique(y_train_idx, return_counts=True)
    print(
        "[DEBUG] y_train_idx distribution:",
        dict(zip(uniq.tolist(), cnt.tolist()))
    )
    # =============================================

    pre, num_cols, cat_cols = fit_preprocessor(X_train)
    X_train_trans = transform_with_preprocessor(pre, X_train)
    X_val_trans   = transform_with_preprocessor(pre, X_val)
    X_test_trans  = transform_with_preprocessor(pre, X_test)

    input_dim = int(X_train_trans.shape[1])
    n_classes = 1 if is_binary else int(num_classes)

    # class weights
    cw_np = compute_class_weight("balanced", classes=np.unique(y_train_idx), y=y_train_idx).astype(np.float32)

    # y 存成 numpy（binary: float32 (N,1)，multi: int64 (N,)）
    if is_binary:
        y_train_np = np.asarray(y_train_idx, dtype=np.float32).reshape(-1, 1)
        y_val_np   = np.asarray(y_val_idx, dtype=np.float32).reshape(-1, 1)
        y_test_np  = np.asarray(y_test_idx, dtype=np.float32).reshape(-1, 1)
    else:
        y_train_np = np.asarray(y_train_idx, dtype=np.int64)
        y_val_np   = np.asarray(y_val_idx, dtype=np.int64)
        y_test_np  = np.asarray(y_test_idx, dtype=np.int64)

    extra = {"input_dim": input_dim, "n_classes": n_classes, "is_binary": bool(is_binary)}
    _clog("REBUILD", phase="save_begin", cache_dir=cache_dir)
    cache.save(
        X_train_trans, X_val_trans, X_test_trans,
        y_train_np, y_val_np, y_test_np,
        cw_np,
        csv_path=csv_path,
        config=config,
        extra=extra,
    )
    _clog("REBUILD", phase="save_done", sec=f"{time.time()-t_build:.2f}", input_dim=input_dim, is_binary=is_binary)

    # 回傳 dataloader（直接用剛算好的結果，不再從檔案讀一次）
    class_weights = torch.tensor(cw_np, dtype=torch.float32)

    X_train_t = torch.tensor(X_train_trans, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_trans, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_trans, dtype=torch.float32)

    if is_binary:
        y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
        y_val_t   = torch.tensor(y_val_np, dtype=torch.float32)
        y_test_t  = torch.tensor(y_test_np, dtype=torch.float32)
    else:
        y_train_t = torch.tensor(y_train_np, dtype=torch.long)
        y_val_t   = torch.tensor(y_val_np, dtype=torch.long)
        y_test_t  = torch.tensor(y_test_np, dtype=torch.long)

    trainset = TensorDataset(X_train_t, y_train_t)
    valset   = TensorDataset(X_val_t, y_val_t)
    testset  = TensorDataset(X_test_t, y_test_t)

    _clog("RETURN", mode="MISS->REBUILD", train_n=len(trainset), val_n=len(valset), test_n=len(testset))
    return (
        DataLoader(trainset, batch_size=512, shuffle=True),
        DataLoader(valset, batch_size=512, shuffle=False),
        DataLoader(testset, batch_size=512, shuffle=False),
        input_dim,
        n_classes,
        is_binary,
        class_weights,
    )


def client_fn(context: Context) -> fl.client.Client:
    """
    給 Flower SuperNode / ClientApp 使用的工廠函數。
    目標：避免每個 round 都重新讀 369MB CSV、重新切分、重新 fit preprocessor。
    同一個 Pod 內（同一個 Python process）只 load 一次，後續 round 用快取。
    """
    node_cfg = context.node_config or {}

    datapath = (
        os.environ.get("DATA_PATH")
        or node_cfg.get("datapath")
        or "./data"
    )
    print(f"[client_fn] node_id={context.node_id}, datapath={datapath}")

    #只載入一次資料（快取）
    global _DATA_CACHE
    if _DATA_CACHE is None:
        # print(f"[client_fn] cache miss > loading data from {datapath}")
        print(f"[client_fn] init -> loading (disk-cache may HIT) datapath={datapath}")

        # 讀取 NUM_CLIENTS 環境變數，用於 IID partition 分割
        num_clients = int(os.environ.get("NUM_CLIENTS", "1"))
        _DATA_CACHE = load_data_with_cache(datapath, num_clients=num_clients)
        print("[client_fn] cache filled")
    else:
        print("[client_fn] cache hit -> reuse preloaded data")

    (
        trainloader,
        valloader,
        testloader,
        input_dim,
        n_classes,
        is_binary,
        class_weights,
    ) = _DATA_CACHE

    # 初始化模型（每次 client_fn 都重建模型是可以；重點是資料不要重讀）
    # hidden 層擴大為 [128, 64, 32]，搭配 BatchNorm + Dropout 提升表達能力與泛化性
    net = Net(input_dim=input_dim, hidden=[128, 64, 32], n_classes=n_classes).to(DEVICE)

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            # 圈數設定epochs(輪)
            local_epochs = int(config.get("local_epochs", 1))
            # 讀取 FedProx 近端項係數（預設 0.0 代表不啟用近端項，行為等同 FedAvg）
            proximal_mu = float(config.get("proximal_mu", 0.0))
            current_round = int(config.get("current_round", 0))

            # set_parameters() 已將全局參數載入 net，
            # 此時從 net.parameters() 擷取 tensor，確保只含可學習參數
            # （排除 BatchNorm 的 running_mean / running_var / num_batches_tracked buffers）
            # 若直接傳入 numpy arrays（來自 state_dict），buffer 會混入導致 zip 錯位崩潰
            global_param_tensors = (
                [p.detach().clone() for p in net.parameters()]
                if proximal_mu > 0.0 else None
            )

            train(
                net,
                trainloader,
                valloader,
                epochs=local_epochs,
                class_weights=class_weights,
                is_binary=is_binary,
                proximal_mu=proximal_mu,          # FedProx 近端項係數
                global_params=global_param_tensors,  # 僅含可學習參數的 tensor list
                current_round=current_round,      # 傳入圈數供 epoch 計時日誌使用
            )

            # 本地訓練後，對本地模型評估 val set（訓練監控用）
            val_loss, val_acc, val_f1 = test(
                net,
                valloader,
                is_binary,
                debug=(current_round % 5 == 0)  # 每 5 輪印一次 pred 分布，降低日誌噪音
            )

            return (
                self.get_parameters({}),
                len(trainloader.dataset),
                {
                    "val_loss":      float(val_loss),
                    "val_accuracy":  float(val_acc),
                    "val_f1":        float(val_f1),
                    # test_* 由 evaluate() 負責（History distributed evaluate 區段）
                },
            )


        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy, f1 = test(net, testloader, is_binary)
            # 使用 test_ 前綴，與 fit() 回傳的 val_* 指標對齊，方便在 server History 中識別
            return float(loss), len(testloader.dataset), {
                "test_accuracy": float(accuracy),
                "test_f1": float(f1),
            }

    return FlowerClient().to_client()


# 這個 app 會被 SuperNode 找到並啟動
app = ClientApp(client_fn=client_fn)