# 高效 GPU 推理加速指南

本指南說明在 LLM 評測與推理場景中如何高效使用 GPU，涵蓋推理引擎選擇、服務化部署、多節點分散推理、資料分片策略，以及結果收集與合併的最佳實踐。

---

## 一、推理引擎選擇

### 核心堆疊

| 層次 | 推薦工具 | 為什麼 |
|------|---------|--------|
| 推理引擎 | **vLLM** | PagedAttention 大幅提升 KV-Cache 記憶體利用率，支援 continuous batching，吞吐遠高於原生 HuggingFace `generate()` |
| API 介面 | **OpenAI-compatible API** | vLLM 內建 `openai.api_server`，客戶端可用任何 OpenAI SDK 呼叫，零改動即可對接 |
| 模型格式 | **HuggingFace format** | vLLM 直接讀 HuggingFace 模型目錄，不需額外轉換 |
| 量化 | **AWQ / GPTQ / BitsAndBytes** | 4-bit / 8-bit 量化可大幅降低推理 VRAM 需求，代價是微幅精度損失 |

### 推理 vs 訓練的關鍵差異

| 面向 | 訓練 | 推理 |
|------|------|------|
| VRAM 瓶頸 | 模型參數 + 優化器狀態 + 梯度 | 模型參數 + **KV-Cache** |
| 並行策略 | Data Parallel（DDP） | **Tensor Parallel（TP）** / Pipeline Parallel（PP） |
| 吞吐提升手段 | 加大 batch + 梯度累積 | continuous batching + 提高並行請求數 |
| 精度要求 | bf16 / fp16 常見 | 量化至 4-bit 對多數評測影響極小 |

> [!TIP]
> **原則**：推理的瓶頸通常是 KV-Cache 而非計算本身。模型越大、context length 越長，KV-Cache 佔用的 VRAM 越多。選擇推理引擎時，KV-Cache 管理效率（如 PagedAttention）比純計算速度更重要。

---

## 二、vLLM 服務化部署

### 2.1 基本啟動

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768
```

關鍵參數說明：

| 參數 | 說明 |
|------|------|
| `--tensor-parallel-size` | 模型拆到幾張 GPU 上（TP），通常設為節點內 GPU 數的因數 |
| `--pipeline-parallel-size` | 模型按層拆到幾組 GPU 上（PP），跨節點時使用 |
| `--gpu-memory-utilization` | GPU 記憶體使用比例，預留一些給系統（建議 0.85–0.95） |
| `--max-model-len` | 最大 context length，直接影響 KV-Cache 佔用量 |

### 2.2 健康檢查

啟動後需等待模型載入完成，透過 API 確認就緒：

```bash
WAIT_DEADLINE=$(($(date +%s) + 600))  # 最多等 10 分鐘
while [ $(date +%s) -lt $WAIT_DEADLINE ]; do
    if curl -s "http://localhost:8000/v1/models" > /dev/null 2>&1; then
        echo "vLLM 就緒"
        break
    fi
    sleep 5
done
```

### 2.3 啟動失敗重試

vLLM 啟動時可能因 NCCL 初始化、port 搶佔等原因失敗，建議加入重試機制：

```bash
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "第 ${attempt} 次嘗試啟動 vLLM..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --tensor-parallel-size $TP_SIZE \
        --port $PORT &
    VLLM_PID=$!

    # 等待就緒或超時
    if wait_for_ready "http://localhost:$PORT/v1/models" 600; then
        echo "啟動成功"
        break
    fi

    kill $VLLM_PID 2>/dev/null
    echo "啟動失敗，30 秒後重試..."
    sleep 30
done
```

> [!IMPORTANT]
> **每次重試前務必清理上一次的殘留進程**，否則 GPU 記憶體不會釋放，導致下一次啟動必定 OOM。

---

## 三、分散推理架構

### 3.1 單機多實例（Data Parallel 推理）

當模型可以放進少量 GPU（例如 2 張），而節點有 8 張 GPU 時，可以在同一節點上啟動多個 vLLM 實例，每個實例處理不同的資料分片：

```text
Node (8 GPUs)
├── vLLM Instance 0 (GPU 0,1)  → 處理資料分片 0
├── vLLM Instance 1 (GPU 2,3)  → 處理資料分片 1
├── vLLM Instance 2 (GPU 4,5)  → 處理資料分片 2
└── vLLM Instance 3 (GPU 6,7)  → 處理資料分片 3
```

```bash
GPUS_PER_INSTANCE=2
INSTANCES_PER_NODE=$((8 / GPUS_PER_INSTANCE))

for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
    GPU_START=$((i * GPUS_PER_INSTANCE))
    GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
    CUDA_VISIBLE_DEVICES=$(seq -s, $GPU_START $GPU_END)
    PORT=$((8000 + i))

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --tensor-parallel-size $GPUS_PER_INSTANCE \
        --port $PORT &
done
```

### 3.2 多節點分散推理（SLURM + srun）

本專案的多節點推理模式：**每個節點獨立啟動 vLLM 服務，各自處理資料分片，最後合併結果**。這與訓練的 all-reduce 模式完全不同——推理不需要跨節點通訊。

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

# 每個節點的 wrapper 會：
# 1. 啟動 N 個 vLLM 實例（依 TP 需求分配 GPU）
# 2. 每個實例執行自己那份資料分片的推理
# 3. 結果寫入各自的 JSONL 檔
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    bash run_inference_wrapper.sh
```

關鍵差異：

| 面向 | 分散訓練 | 分散推理 |
|------|---------|---------|
| 跨節點通訊 | 必要（梯度同步） | **不需要**（各 rank 獨立推理） |
| MASTER_ADDR | 必須正確設定 | 不需要 |
| 結果同步 | 模型參數自動同步 | **需手動合併結果檔** |
| 擴展效率 | 受限於通訊頻寬 | **近乎線性擴展** |

> [!TIP]
> 推理的分散化比訓練簡單得多——本質上是 embarrassingly parallel。每個 rank 獨立處理自己的資料分片，不需要任何跨節點通訊，因此擴展效率接近 100%。

### 3.3 Tensor Parallel vs Pipeline Parallel

| 並行模式 | 適用場景 | 限制 |
|---------|---------|------|
| **Tensor Parallel（TP）** | 單節點內多 GPU | GPU 間需要高頻寬（NVLink 最佳）；TP 數通常不超過 8 |
| **Pipeline Parallel（PP）** | 模型太大放不進單節點 | 引入 pipeline bubble，會降低單一請求的延遲 |
| **TP + PP 混合** | 超大模型跨節點 | TP 在節點內（走 NVLink），PP 跨節點（走 InfiniBand） |

```text
# 例：70B 模型部署在 2 節點 × 8 GPU
--tensor-parallel-size 8    # 每個節點 8 GPU 做 TP
--pipeline-parallel-size 2  # 2 個節點做 PP
# 總共使用 16 GPU
```

> [!WARNING]
> TP 要求所有參與的 GPU 在每次 forward pass 都互傳 activation，因此**只適合在高頻寬互聯（NVLink）的節點內使用**。跨節點做 TP 會因網路延遲導致嚴重的吞吐下降。

---

## 四、資料分片與並行請求

### 4.1 資料分片策略

每個推理 rank 只處理資料集的一個子集，避免重複計算：

```python
# 均勻分片：rank i 取第 i 份
chunk_size = (total_size + world_size - 1) // world_size
start_idx = rank * chunk_size
end_idx = min(start_idx + chunk_size, total_size)
dataset.data = dataset.data[start_idx:end_idx]
```

這種分片方式保證：
- 每筆資料只會被一個 rank 處理（無重複）
- 所有資料都會被處理（無遺漏）
- 不需要跨 rank 通訊

### 4.2 並行請求（Concurrent API Calls）

vLLM 的 continuous batching 能高效處理並行請求。客戶端應使用 thread pool 同時發送多個請求，充分利用 GPU：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor() as executor:
    future_tasks = []
    for idx, question in enumerate(dataset):
        rate_limiter.wait()  # 可選：控制每秒請求數
        future = executor.submit(llm.call, question_text, prompt_lang)
        future_tasks.append((idx, future))

    for idx, future in tqdm(as_completed(dict(future_tasks).values())):
        result = future.result()
        # 處理結果...
```

> [!TIP]
> vLLM 內部已有請求排隊機制，客戶端不需要手動控制 batch size。盡量一次提交所有請求，讓 vLLM 的 scheduler 決定何時、如何 batch——這比客戶端手動分批效率更高。

### 4.3 Rate Limiting

對外部 API（如 OpenAI、Anthropic）推理時需要限流，但對本地 vLLM 通常設為無限制：

```python
class RateLimiter:
    def __init__(self, calls_per_second):
        self.no_limit = (calls_per_second == -1)  # -1 = 不限速
        self.interval = 1.0 / calls_per_second if not self.no_limit else 0
```

```yaml
llm_api:
  api_rate_limit: -1    # 本地 vLLM: -1（不限速）；外部 API: 依配額設定
```

---

## 五、結果收集與合併

### 5.1 Per-Rank 結果寫入

分散推理時，每個 rank 獨立寫入自己的結果檔，避免寫入競爭：

```python
# 檔名包含 node_id 和 rank，保證不同 rank 不會衝突
if world_size > 1:
    results_path = f"eval_results_{timestamp}_node{node_id}_rank{rank}.jsonl"
else:
    results_path = f"eval_results_{timestamp}.jsonl"

# Append mode 寫入
with open(results_path, "a", encoding="utf-8") as f:
    for detail in detailed_results:
        f.write(json.dumps(detail, ensure_ascii=False) + "\n")
```

### 5.2 結果合併（Finalization）

所有 rank 完成後，需要將分散的結果合併為一份完整報告：

```text
結果合併流程：
1. 掃描所有分片檔：results_{timestamp}_node*_rank*.jsonl
2. 依 run 分組，將同一 run 的所有分片合併為單一 JSONL
3. 重新計算跨分片的正確率統計
4. 輸出合併後的最終 JSON 報告
5. 清理個別分片檔（可選）
```

> [!IMPORTANT]
> **合併時必須確保所有 rank 都已完成寫入**。在 SLURM 環境下，`srun` 會等待所有 task 結束才返回，因此在 `srun` 之後執行合併是安全的。但若使用背景進程，需自行加入同步機制。

### 5.3 統一時間戳

多節點推理需要一個統一的時間戳來關聯同一次實驗的所有分片：

```bash
# 在 srun 前生成，傳給所有節點
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export TWINKLE_EVAL_RUN_TIMESTAMP=$TIMESTAMP

srun ... bash -c "
    # 每個節點都用同一個 TIMESTAMP
    uv run twinkle-eval --config $CONFIG --export json
"
```

---

## 六、環境與資源管理

### 6.1 暫存目錄隔離

多實例在同一節點運行時，務必隔離暫存目錄以避免衝突：

```bash
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${GLOBAL_RANK}"
export TRITON_CACHE_DIR="/tmp/triton_${GLOBAL_RANK}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${GLOBAL_RANK}"
export TIKTOKEN_RS_CACHE_DIR="/tmp/tiktoken_rs_${GLOBAL_RANK}"
```

### 6.2 啟動前清理

每次作業啟動前清理上一次的殘留暫存，避免佔用 `/tmp` 或 `/dev/shm`：

```bash
rm -rf /tmp/vllm_cache_* /tmp/inductor_* /tmp/triton_* \
       /tmp/tiktoken_rs_* /tmp/xdg_cache_* /tmp/vllm_*.log
rm -f /dev/shm/nccl-* /dev/shm/torch_*
```

### 6.3 HuggingFace Cache 管理

在 HPC 環境中，`$HOME` 通常有容量限制，應將 cache 指向工作目錄：

```bash
export HF_HOME="/work/${USER}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
```

### 6.4 Port 搶佔防範

多實例啟動時，錯開啟動時間避免 port 衝突：

```bash
for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
    PORT=$((8000 + i))
    sleep $((i * 10))  # 每個實例間隔 10 秒啟動
    start_vllm_instance $i $PORT &
done
```

> [!WARNING]
> NCCL 在初始化時會自動選擇通訊 port。多個 vLLM 實例同時啟動可能搶佔同一個 NCCL port，導致 hang 或啟動失敗。**錯開啟動時間是最簡單有效的解法。**

---

## 七、推理效能調優

### 7.1 context length 與 VRAM 的關係

KV-Cache 的 VRAM 佔用與 `max_model_len` 成正比。過長的 context 會擠壓可用於 batch 的記憶體，降低吞吐：

```text
KV-Cache VRAM ≈ 2 × num_layers × hidden_size × max_model_len × dtype_size × batch_slots
```

| max_model_len | 適用場景 | 影響 |
|---------------|---------|------|
| 4096 | 短文本分類 / 選擇題 | 可容納大量並行請求 |
| 32768 | 長文本理解 / CoT 推理 | 並行請求數大幅下降 |
| 131072 | 超長上下文 | 單請求即可能佔滿 VRAM |

> [!TIP]
> 將 `max_model_len` 設為**實際需要的最大長度**，而非模型支援的最大長度。評測選擇題時，prompt + response 通常不超過 4096 tokens，設為 32768 只會浪費 KV-Cache 空間。

### 7.2 gpu-memory-utilization 調整

```bash
--gpu-memory-utilization 0.90  # 預設建議值
```

- **設太高（> 0.95）**：系統記憶體不足時可能 OOM
- **設太低（< 0.80）**：無法充分利用 GPU，batch 容量受限
- **建議**：0.85–0.90，留 10–15% 給 CUDA context 和系統開銷

### 7.3 吞吐 vs 延遲的取捨

| 目標 | 策略 |
|------|------|
| **最大吞吐**（評測場景） | 一次提交大量請求，讓 vLLM continuous batching 充分運作 |
| **最低延遲**（互動場景） | 減少並行請求數，確保每個請求獨佔更多計算資源 |

評測場景通常追求吞吐——同時提交所有問題，讓 vLLM 自行排程。

---

## 八、常見問題與排查

| 問題 | 症狀 | 解法 |
|------|------|------|
| vLLM 啟動後 GPU util 為 0 | 服務正常但沒收到請求 | 確認客戶端 `base_url` 和 `port` 正確 |
| OOM（Out of Memory） | CUDA out of memory 錯誤 | 降低 `--max-model-len`、`--gpu-memory-utilization`，或增加 TP 數 |
| 推理速度極慢 | 吞吐遠低於預期 | 確認 TP GPU 間有 NVLink；檢查是否只發了單一請求（未利用 batching） |
| 多實例啟動 hang | 進程卡在 NCCL 初始化 | 錯開啟動時間（`sleep $((i * 10))`）；清理 `/dev/shm/nccl-*` |
| 結果檔案行數不對 | 合併後資料少於預期 | 檢查是否所有 rank 都完成；確認分片邏輯無重疊或遺漏 |
| 模型載入緩慢 | 啟動需要 10+ 分鐘 | 確認模型已在本地 cache（`HF_HUB_CACHE`），避免每次從遠端下載 |
| `from_pretrained` 多 rank 競爭 | 多個進程同時下載模型 | 先用單一進程下載到 shared cache，再啟動所有實例 |

---

## 九、快速 Checklist

在提交 SLURM 多節點推理作業前確認以下事項：

- [ ] 單實例推理正常，結果格式正確
- [ ] `--tensor-parallel-size` × `--pipeline-parallel-size` = 實例使用的 GPU 總數
- [ ] `--max-model-len` 設為實際所需，非模型最大值
- [ ] 每個實例的 port 不同（如 `8000 + i`）
- [ ] 多實例啟動有錯開時間（`sleep $((i * 10))`）
- [ ] 暫存目錄已按 rank 隔離（`/tmp/*_${RANK}`）
- [ ] 啟動前已清理上一次的殘留暫存（`/tmp/vllm_cache_*`、`/dev/shm/nccl-*`）
- [ ] HuggingFace cache 指向共享儲存，模型已預先下載
- [ ] 結果檔路徑在 shared filesystem 上，所有節點都看得到
- [ ] 結果檔名包含 `node_id` 和 `rank`，避免寫入衝突
- [ ] 有合併腳本在所有 rank 完成後執行結果整合
- [ ] 統一時間戳（`TWINKLE_EVAL_RUN_TIMESTAMP`）在 `srun` 前生成並傳遞給所有節點
