# 高效 GPU 加速運算指南

本指南說明在深度學習專案中如何高效使用 GPU 加速運算，涵蓋框架選擇、資源監測、多節點擴展、並行模式選擇，以及多進程讀取資料的搶佔問題處理。

---

## 一、框架選擇

### 核心堆疊

| 層次 | 推薦工具 | 為什麼 |
|------|---------|-------|
| 深度學習 | **PyTorch** | 動態圖、生態最完整，幾乎是 LLM / CV 研究的首選 |
| 模型管理 | **HuggingFace Transformers** / **LlamaFactory** | Transformers 提供統一的模型載入、tokenizer、推理介面；LlamaFactory 則封裝更高層的訓練流程，支援多種模型架構與訓練方式，適合快速實驗與零程式碼微調 |
| 參數高效微調 | **PEFT / LoRA** | 大部分下游任務**完全不需要全參數微調**——LoRA 只更新極少數 adapter 權重（通常 < 1% 參數），便能達到接近全量微調的效果，同時大幅降低 VRAM 需求、縮短訓練時間、減少過擬合風險 |
| 多 GPU / 多節點 | **Accelerate** | 透明封裝 DDP，幾乎不改原始訓練邏輯即可橫向擴充；亦可與 DeepSpeed、FSDP 整合 |
| 量化 | **BitsAndBytes** | `load_in_4bit` / `load_in_8bit` 可大幅壓縮模型佔用的 VRAM，代價是略微降低計算速度與精度，適合 VRAM 受限的場景 |

### 典型用法範例

```python
# 以 HuggingFace + PEFT 為例：載入基礎模型並套上 LoRA adapter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,  # bf16 比 fp32 省一半 VRAM，精度損失極小
)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    # 只微調注意力層的投影矩陣，其他權重保持不動
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 確認可訓練參數佔比
```

```python
# 以 Accelerate 包裝訓練迴圈，自動支援單卡 / 多卡 / 多節點
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=grad_accum,
    mixed_precision="bf16",
)
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)
```

> [!TIP]
> **原則**：量化是記憶體換速度的手段；增加 GPU 數量是提升吞吐的主要途徑。兩者可以並存，但需注意 4-bit 量化與 DDP 的相容性——建議搭配 DeepSpeed ZeRO 而非原生 DDP。

---

## 二、小規模監測 GPU 使用率

在提交大規模作業前，**務必先在單節點小規模確認 GPU 真的有充分利用**。GPU util 偏低通常代表資料讀取、CPU 預處理或通訊才是瓶頸，而非算力本身不足。

### 使用 `nvnodetop`

```bash
pip install nvnodetop
nvnodetop        # 啟動後按 q 離開
```

> [!IMPORTANT]
> **重要**：`nvnodetop` 是即時輪詢工具，長時間掛著會持續佔用系統資源並可能干擾 GPU 核間通訊效率。**確認完畢後立刻按 `q` 離開，不要讓它持續在背景運行。**

### 其他快速指令

```bash
# 查看 Slurm 規則、集群狀態與用戶工作負載狀態
pip install sltop
sltop             # 打開檢視器，一樣用完按 q 離開
```

> [!TIP]
> 有不少 cluster 會有設置特定規範，務必檢查 cluster 的使用規則（如 GPU 時限、作業優先級、資源限制等），以免提交後才發現不符合規定而被取消。
> 像是 nano4 超級電腦有限制 Partition `dev` 最多使用 2 小時的 GPU 時數，而 partition `normal` 有規定最低要啟用 16 張 GPU 才能提交，這些都會影響實驗是否能順利進行。

### 觀察重點

- **Util%**：訓練期間應持續 > 90%；長時間低於 50% 代表資料讀取、CPU 預處理或跨節點通訊是瓶頸。
- **MEM Used**：接近上限時縮小 `batch_size` 或啟用梯度累積，而非降低模型精度。
- **Power**：接近 GPU 的 TDP（熱設計功耗）才代表算力真正被充分使用。

---

## 三、從單節點擴展到多節點

### 3.1 節點內擴展（單機多卡）

**建議先在 `dev` partition 觀測資源用量並測試程式碼**，確認無誤後再到 `normal` partition 正式 scale up。`dev` partition 通常時限短、排隊快，適合用小規模資料跑完整流程、確認 GPU util 正常、debug 程式錯誤，避免浪費 `normal` partition 的大量 GPU 時數。

在正式提交多節點作業前，先單節點確認訓練程式沒有錯誤：

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --mixed_precision=bf16 \
    main.py --config configs/zero_shot.yaml
# --num_processes: 節點上有幾張卡
```

### 3.2 跨節點擴展（SLURM + srun）

本專案 `run_slurm.sh` 的核心模式：

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # 每個節點只跑 1 個 srun task（Accelerate 在內部 fork 進程）
#SBATCH --gres=gpu:8

MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500

srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    bash -c "
        accelerate launch \
            --multi_gpu \
            --num_machines=$SLURM_NNODES \
            --num_processes=$NUM_PROCESSES \
            --machine_rank=\$SLURM_NODEID \
            --main_process_ip=$MASTER_ADDR \
            --main_process_port=$MASTER_PORT \
            main.py
    "
```

關鍵變數說明：

| 變數 | 說明 |
|------|------|
| `--num_machines` | 節點總數（`$SLURM_NNODES`） |
| `--machine_rank` | 當前節點的 rank（`$SLURM_NODEID`，0-based） |
| `--main_process_ip` | 第一個節點的 IP，所有 worker 用它做 rendezvous |
| `--main_process_port` | 確保沒有衝突的空閒 port |

### 3.3 有效 batch size 計算

```text
有效 batch = batch_size × grad_accumulation_steps × num_gpus
           = 48         × 1                       × (2 nodes × 8 GPUs)
           = 768
```

> [!TIP]
> 像是微調 CoT 等更長的 context 需求，須將參數改為 `batch_size=24, grad_accum=2`，保持相同有效 batch。
> **這說明一個重要原則：context length 上升時，每筆樣本佔用的 VRAM 隨之線性增長，必須等比例降低 `batch_size` 才能避免 OOM；再透過增加 `grad_accumulation_steps` 補回有效 batch size，維持訓練穩定性。**

---

## 四、節點內並行 vs 跨節點並行

### 節點內並行（Intra-node）

- 通訊走 **NVLink / PCIe**（頻寬高，延遲低）
- all-reduce 梯度幾乎是免費的
- 適用場景：模型能放進單節點 VRAM、單節點算力已是瓶頸

**優先選節點內並行**，因為網路通訊開銷幾乎可忽略。

### 跨節點並行（Inter-node）

- 通訊走 **InfiniBand / Ethernet**（頻寬比 NVLink 低 1–2 個量級）
- 每個 all-reduce 都需要跨網路傳送梯度 → 通訊是主要瓶頸
- 有效策略：
  - 增大 `batch_size` 降低 step 數，減少通訊頻次
  - 搭配 **梯度累積（gradient accumulation）** 讓每次通訊換來更多計算
  - 考慮 **DeepSpeed ZeRO-3** 在超多節點場景下分片 optimizer state

### 判斷標準

| 情境 | 建議 |
|------|------|
| 單節點 8 GPU 已不夠快 | 擴跨節點 |
| VRAM 不夠裝模型 | 先考慮 ZeRO / offload，再考慮跨節點 tensor parallel |
| 通訊比計算慢 | 加大 batch / grad_accum，或升 InfiniBand |

> [!TIP]
> 在跨節點訓練中，通訊延遲可能成為瓶頸，並非越多 GPU 就越快。**確保每次通訊換來足夠的計算量**（透過增大 batch size 和梯度累積）是關鍵。

---

## 五、多進程讀取資料的搶佔問題

這是擴節點最容易踩坑的地方。

### 5.1 問題根源

多個 GPU rank（進程）同時存取同一份資料或同一個路徑，可能造成：

- **磁碟 I/O 競爭**：多 rank 同時讀大型模型 checkpoint 或資料集
- **檔案寫入競爭**：多 rank 同時嘗試寫 cache 或 checkpoint，最終檔案損壞
- **HuggingFace cache 競爭**：多節點的所有 rank 同時呼叫 `from_pretrained`，共享 NFS 上的同一個 `.cache/huggingface`

### 5.2 本專案的處理方式

#### 模式一：只讓 main process 寫，其他 rank 等待

```python
# main.py — CoT rationale 生成與快取
if accelerator.is_main_process:
    os.makedirs(_cache_dir, exist_ok=True)
    with open(_cache_file, "w") as _f:
        json.dump({"train": rationales_train}, _f)
    print(f"CoT rationales cached → {_cache_file}")

# src/train.py — model checkpoint 儲存
if accelerator.is_main_process:
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
```

#### 模式二：建目錄後讓所有 rank 等待同步

```python
# src/train.py
if accelerator.is_main_process:
    os.makedirs(save_dir, exist_ok=True)
accelerator.wait_for_everyone()   # ← 所有 rank 在此等到目錄建好再繼續
```

#### 模式三：rank 分片資料，避免重複計算

```python
# src/data.py — generate_cot_rationales
# 每個 rank 只處理自己那份，最後 gather 合併
local_indices = list(range(rank, len(df), num_processes))
# ...
all_results = gather_object(local_results)
all_results.sort(key=lambda x: x["idx"])
```

### 5.3 常見問題與解法

| 問題 | 症狀 | 解法 |
|------|------|------|
| 多 rank 同時 `from_pretrained`，卡在 NFS | 程式 hang，GPU idle | rank 0 先下載 → `wait_for_everyone()` → 其餘 rank 讀 local cache |
| 多個 DataLoader worker 同時讀同一 CSV | I/O 100%，GPU util 低 | 預先 tokenize 並快取為 `.pt`（`torch.save`），或減少 `num_workers` |
| 多 rank 同時寫 JSONL 輸出 | 行順序錯亂 / 重複 / 遺失 | 只讓 `is_main_process` 寫；或依 rank 命名暫存檔，最後合併 |
| TMPDIR 跨節點指向不同路徑 | 節點 A 找不到節點 B 的暫存檔 | `TMPDIR` 指向 shared filesystem（如 NFS），或不跨 rank 共享暫存 |

### 5.4 DataLoader 的 `num_workers` 設定

本專案的 `DataLoader` 沒有顯式設定 `num_workers`，預設為 0（主進程讀取）。
對於小到中型 CSV + 即時 tokenize 的場景，這已夠用，但若資料量大：

```python
# 在 DataLoader 建立時加入
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,          # 通常設為 CPU core 數的一半
    pin_memory=True,        # 加速 CPU→GPU 傳輸
    persistent_workers=True # 避免每個 epoch 重新 fork
)
```

> [!WARNING]
> `num_workers > 0` 時，每個 worker 是獨立進程，同樣需要避免對共享資源的競爭寫入（只讀通常沒問題）。

---

## 六、快速 Checklist

在提交 SLURM 多節點作業前確認以下事項：

- [ ] 單節點 GPU util > 90%（用 `nvnodetop` 確認，確認完立刻按 `q`）
- [ ] `MASTER_ADDR` / `MASTER_PORT` 正確且無衝突
- [ ] 所有檔案寫入都有 `is_main_process` 守衛
- [ ] `wait_for_everyone()` 放在任何「建目錄 / 建快取 → 其他 rank 讀取」的邊界上
- [ ] 有效 batch size（`batch × accum × num_gpus`）與單節點實驗相同
- [ ] `TMPDIR` 指向節點本機磁碟或 shared NFS，視需求而定
- [ ] 輸出路徑在 shared filesystem 上，所有節點都看得到
