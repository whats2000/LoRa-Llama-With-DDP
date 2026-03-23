# 使用 LLaMA 3.2 與 LoRA 進行醫學問答

以 LoRA 微調 `meta-llama/Llama-3.2-1B-Instruct`，在 **PathoQA** 資料集上進行醫學選擇題問答。

---

## 環境設定

### 方案 A — 標準 `venv`（通用方式）

```bash
# 1. 建立並啟用虛擬環境
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. 安裝相依套件
pip install -r requirements.txt
```

### 方案 B — `uv`（較快，建議使用）

```bash
# 1. 若尚未安裝 uv，請先安裝
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 建立虛擬環境並安裝所有相依套件
uv sync

# 3. 啟用（選擇性 — uv run 會自動使用）
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

> **新增或變更相依套件後**，請重新匯出鎖定檔：
> ```bash
> uv sync && uv export --no-hashes -o requirements.txt
> ```

---

## 設定檔

所有超參數、提示詞模板與檔案路徑皆集中於 **`configs/`** 目錄中，原始碼內不含寫死的字串。

每個實驗設定檔**僅覆寫與 `configs/base.yaml` 不同的部分**；兩者在執行時會進行深層合併。

| 檔案 | 實驗 |
|------|------|
| `configs/base.yaml` | 共用設定（模型、LoRA、訓練、提示詞文字） |
| `configs/zero_shot.yaml` | 零樣本提示 |
| `configs/few_shot.yaml` | 少樣本提示（4 個範例） |
| `configs/cot.yaml` | 思維鏈提示 |

`base.yaml` 主要區段：

| 區段 | 用途 |
|------|------|
| `paths` | 資料集、檢查點與輸出 CSV 路徑 |
| `data` | 訓練／驗證分割比例與隨機種子 |
| `model` | HuggingFace 模型 ID 與量化開關 |
| `lora` | LoRA rank、alpha、目標模組、dropout |
| `training` | 訓練週期數、批次大小、學習率、梯度累積 |
| `prompting` | 策略與所有提示詞文字模板 |

---

## 執行

```bash
# 執行單一實驗（零樣本、少樣本或思維鏈）
python main.py --config configs/zero_shot.yaml
python main.py --config configs/few_shot.yaml
python main.py --config configs/cot.yaml

# 使用 uv（無需手動啟用虛擬環境）
uv run main.py --config configs/zero_shot.yaml
uv run main.py --config configs/few_shot.yaml
uv run main.py --config configs/cot.yaml
```

> [!TIP]
> 使用 2 節點 × 8 GPU（共 16 張）時的有效批次大小：所有策略皆為 **768**。
> zero_shot／few_shot：每 GPU 48（受限於 512 token 的 VRAM）× 梯度累積 1 × 16 = 768。CoT：每 GPU 24 × 梯度累積 2 × 16 = 768（較小的每 GPU 批次以容納 1024 token 的上下文長度）。
> 可編輯 `configs/base.yaml` 中的 `training.batch_size` 與 `training.grad_accumulation_steps` 以適配您的硬體。

---

## 使用 Slurm 重現實驗

`scripts/run_experience_slurm.sh` 提供了現成的 Slurm 作業腳本。
該腳本使用 `accelerate launch` 在單一節點的 8 張 GPU 上依序執行所有三個實驗（零樣本 → 少樣本 → 思維鏈）。

```bash
# 實驗在 nano4.nchc.org.tw 上執行，但腳本可適用於任何具備類似資源的 Slurm 叢集。
sbatch --account=<使用者或專案ID> scripts/run_experience_slurm.sh

# 驗證訓練後模型，對驗證集進行評估，並將結果輸出至 JSONL 檔案。
sbatch --account=<使用者或專案ID> scripts/run_infer_validation_slurm.sh
```

將 `<使用者或專案ID>` 替換為您的叢集帳號名稱（例如您的使用者 ID 或計算專案 ID）。

腳本預設值：
- 分區：`dev`
- 每節點 8 張 GPU（`--gres=gpu:8`）
- 64 顆 CPU，使用所有可用記憶體
- 時間限制：2 小時
- 日誌輸出至 `logs/slurm_<JOBID>.out` / `.err`

若您的叢集使用不同的分區名稱或資源限制，請調整腳本內的 `#SBATCH` 指令。

---

## 專案結構

```
HW1_{student_id}/
├── dataset/
│   ├── dataset.csv          # 9,000 筆已標註的 PathoQA 範例
│   └── benchmark.csv        # 900 筆未標註的 Kaggle 測試題目
├── saved_models/
│   └── checkpoint/
│       ├── cot/             # 最佳思維鏈檢查點（驗證準確率最高的週期）
│       ├── few_shot/        # 最佳少樣本檢查點
│       └── zero_shot/       # 最佳零樣本檢查點
├── main.py
├── README.md
├── requirements.txt
│
│   （額外的專案檔案 — 壓縮檔中非必需，但存在於儲存庫中）
├── configs/
│   ├── base.yaml
│   ├── zero_shot.yaml
│   ├── few_shot.yaml
│   └── cot.yaml
├── pyproject.toml
├── uv.lock
└── src/
    ├── data.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

---

## 超參數（預設值）

| 參數 | 值 |
|------|-----|
| 模型 | `meta-llama/Llama-3.2-1B-Instruct` |
| 量化 | 無（bfloat16） |
| LoRA rank (`r`) | 64 |
| LoRA alpha | 128 |
| 目標模組 | 所有線性層 |
| 訓練週期數 | 10 |
| 批次大小（`training.batch_size`） | 每 GPU 48；**在單張消費級 GPU 上請降至 4**（總有效批次大小：768） |
| 梯度累積（`training.grad_accumulation_steps`） | 2；**若批次大小降至 4，請增加至 96**（維持有效批次大小：768） |
| 學習率 | 1e-4 |
| 提示策略 | CoT |
