"""
SuperLink 啟動 wrapper。
在 run_superlink() 前 patch store_traffic，
修復 Flower 1.25.0 pull_object preregister→put 視窗觸發 bytes=0 ValueError 的 bug。

根本原因：monkey-patch 寫在 server.py（ServerApp 程序）無法影響
         flower-superlink（SuperLink 程序），兩者是獨立 OS process。
此 wrapper 確保 patch 在 SuperLink 程序的 Python interpreter 內執行。
"""
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState

_orig_in_memory = InMemoryLinkState.store_traffic
def _patched_in_memory(self, run_id, *, bytes_sent, bytes_recv):
    if bytes_sent == 0 and bytes_recv == 0:
        return  # preregister→put 視窗邊界情況，靜默跳過
    _orig_in_memory(self, run_id, bytes_sent=bytes_sent, bytes_recv=bytes_recv)
InMemoryLinkState.store_traffic = _patched_in_memory

# SqliteLinkState 有相同 bug，防禦性 patch（目前 serverDeploy.yaml 未傳 --database，不會觸發）
try:
    from flwr.server.superlink.linkstate.sqlite_linkstate import SqliteLinkState
    _orig_sqlite = SqliteLinkState.store_traffic
    def _patched_sqlite(self, run_id, *, bytes_sent, bytes_recv):
        if bytes_sent == 0 and bytes_recv == 0:
            return
        _orig_sqlite(self, run_id, bytes_sent=bytes_sent, bytes_recv=bytes_recv)
    SqliteLinkState.store_traffic = _patched_sqlite
except ImportError:
    pass

print("[WRAPPER] store_traffic patch applied (Flower 1.25.0 bytes=0 workaround). Starting SuperLink...")

from flwr.server.app import run_superlink
run_superlink()
