# server.py
from typing import List, Tuple, Dict
import flwr as fl
from flwr.common import Metrics
from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from collections.abc import Mapping
import numpy as np

# return averaged
def weighted_avg_all(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    metrics: List[(num_examples, {"loss":..., "accuracy":..., "f1":...})]
    return: weighted average dict
    """
    totals: Dict[str, float] = {}
    weights_sum = 0

    if not metrics:
        print("Warning: weighted_avg_all called with empty metrics list.")
        return {}

    for num_examples, m in metrics:
        # 不要只判斷 dict，改用 Mapping（可吃 dict / OrderedDict / 其他 mapping）
        if not isinstance(m, Mapping):
            print(f"Warning: skipping malformed metric entry (metrics not mapping): ({num_examples}, {m})")
            continue

        #num_examples 允許 numpy int
        try:
            n = int(num_examples)
        except Exception:
            print(f"Warning: skipping malformed metric entry (num_examples not int-castable): ({num_examples}, {m})")
            continue

        if n <= 0:
            continue

        weights_sum += n

        for k, v in m.items():
            # 允許 numpy float
            if isinstance(v, (int, float, np.number)):
                totals[k] = totals.get(k, 0.0) + n * float(v)

    if weights_sum == 0:
        print("Warning: no examples reported by clients for metric aggregation (weights_sum == 0).")
        return {}

    return {k: tot / weights_sum for k, tot in totals.items()}


#ServerApp 用的 server_fn
def server_fn(context: Context) -> ServerAppComponents:
    """
    建立並回傳 ServerAppComponents，給 Flower ServerApp 使用。

    之後可以在 flwr run 時用 --run-config 傳參數，例如：
      flwr run . k8s-local --run-config='{"num_rounds": 5, "num_clients": 2}'
    """
    run_cfg = context.run_config or {}

    # 這幾個就是原本 argparse 的參數，現在從 run_config 讀，沒給就用預設值
    num_clients = int(run_cfg.get("num_clients", 2))
    min_available = int(run_cfg.get("min_available", 2))
    num_rounds = int(run_cfg.get("num_rounds", 5))
    local_epochs = int(run_cfg.get("local_epochs", run_cfg.get("epochs", 1)))
    print(
        f"[server_fn] num_clients={num_clients}, "
        f"min_available={min_available}, num_rounds={num_rounds}"
        f"local_epochs={local_epochs}"
    )

    # 每輪訓練的 config
    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "epochs": local_epochs,          # 兼容另一種 key
            "current_round": server_round,
        }
    
    # FedAvg 策略（沿用原本 server.py 的設定）
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,                  # 每輪抽樣比例（這裡全抽）
        fraction_evaluate=1.0,             # 評估抽樣
        min_fit_clients=num_clients,       # 每輪最少參與訓練的客戶端
        min_evaluate_clients=num_clients,  # 每輪最少參與評估的客戶端
        min_available_clients=min_available,  # 集群內最少可用客戶端
        fit_metrics_aggregation_fn=weighted_avg_all,
        evaluate_metrics_aggregation_fn=weighted_avg_all,
        on_fit_config_fn=fit_config, # 每輪訓練的 config
    )

    # ServerConfig：設定總共要跑幾輪
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


# 這個 app 會被 Flower 找到並啟動
app = ServerApp(server_fn=server_fn)
