# 口頭報告講稿 — LoRA-Llama PathoQA（約 8 分鐘）

> 對應 `report.ipynb` 各段，建議邊播 notebook 邊講。⭐ = 一定要講到的亮點。
> 時間重點放在**發現一、發現二**；方法部分快速帶過。

---

## 0. 開場（~45s）— 標題
- 用 **LoRA** 微調 **Llama-3.2-1B-Instruct** 解醫學病理選擇題（PathoQA），4 選 1。
- 限定 1B 小模型 → 我們研究的是「**方法**」，不是換更大的模型。
- 評估用 Kaggle 測試集（public == private，分數可信）。
- 一句話結論：**0.7700 → 0.7988**，靠三件事：LoRA rank、訓練配方、跨策略 ensemble。
- 而且得到兩個方法論發現，待會講——這是報告的亮點。

## 1. 程式架構（~40s）
- 整個專案 **config 驅動**：同一份 code、換 config 就是不同實驗，沒有寫死超參數。
- 一次實驗 = `run.sbatch` 用 accelerate 啟動 `main.py`，吃 `base.yaml` + 該 cell 的 `config.yaml`。
- 流程：main.py 合併 config → model.py 包 LoRA → data.py 組 prompt（答案字母當訓練目標）→ train.py 只訓 LoRA 參數、每 epoch 存最佳 → evaluate.py 產生提交檔。
- 每個 ablation 是 `experiments/` 下一個凍結的 cell（自己的 config + run script），結果都記在 `NOTES.md`。

## 2. 資料（~30s）
- 9000 題訓練、900 題測試。
- **這張圖**：答案分布偏斜（A 32% → D 18%）。我們檢查過——每個答案的 recall 平均（0.71~0.78），代表模型只是學到真實分布，**不是 bias**，不用去「修正」。（這是常見的錯誤方向，我們事先排除）

## 3. 五個方法（~1:40，每個約 20s）
- **A. LoRA**：在 7 個線性層加低秩 adapter，關鍵旋鈕 = rank `r`。
- **B. 訓練配方**：有效批次 = 每卡批次 × 梯度累積 × 卡數。預設 768 太大 → 只有約 110 步 → **訓練不足**；批次調小 = 更多步。
- **C. DoRA**：把更新拆成「大小 + 方向」，一個 flag `use_dora=True`。
- **D. Prompt 策略**：zero-shot vs few-shot（few-shot 在題目前放 k 個範例）。
- **E. Ensemble**：用 option log-likelihood 算出 A~D 的機率，多個 adapter 平均，不用再訓練。
- （每個方法旁都有連到實際 code / config 的連結，老師要看可以點。）

## 4. ⭐ 發現一：validation 會「排錯名次」（~1:00）
- **最重要的圖之一**。左圖：val 準確率在 r=32 最高（倒 U），但真實測試一路漲到 r=256。
- 右圖：兩者在高 rank **反相關**。val 覺得「最過擬合、最差」的 r=256，測試上其實**最好**。
- 教訓：**不能信 validation**，每個決定都用真實測試集判斷——這也是我們提交很多次的原因。
- （延伸：val_loss 在高 rank 爆高、看起來過擬合，但那不代表測試會差。）

## 5. ⭐ 發現二：ensemble 要的是「正交」多樣性（~1:20）
- 熱圖 = 不同 adapter 的「預測一致率」，越低越多樣。
- **關鍵對比**：
  - DoRA vs zero-shot 一致率 ~0.89（很像）→ 加進 ensemble 反而**變差**（0.7888→0.7855）。
  - few-shot vs zero-shot ~0.85（較不一樣）→ 加進去**變好**（0.7811→0.7888），即使 few-shot 單獨較弱。
- 結論：ensemble 的增益來自「**決策方式不同**」（不同策略 / 不同 shot 數），不是「更強但很像」的模型（更高 rank、DoRA、加權都沒用）。約 5 個成員就飽和。

## 6. 結論（~40s）
- 最佳 pipeline（**0.7988**）：5 個 adapter 的 option-LL ensemble（均勻平均），全部 r=256 / 有效批次 192 — `{zs, zs, fs-2, fs-4, fs-8}`。
- 三個帶走的教訓：(1) val 會排錯名次 → 看真實測試；(2) ensemble 要正交多樣性；(3) 訓練配方跟容量一樣重要。
- 距 leaderboard 第一（0.8088）還差約 0.01，我們試過的 LoRA / 訓練 / ensemble 槓桿都到頂了。

## 7. Checkpoints（~15s）
- 28 個 adapter 太大（~17GB）塞不進作業上傳 → 放 HF：`whats2000/lora-llama-pathoqa-checkpoints`，notebook 有下載 + 載入範例。

---

## 可能被問到（Q&A 準備）
- **Q：為什麼不用 CoT？** A：1B 模型的 CoT 太弱（proxy 0.59），而且常常沒寫出乾淨的「Answer: X」，抽取不可靠，所以放棄。
- **Q：為什麼 ensemble 用「機率平均」不用投票？** A：試過——幾何平均、加權（偏重 zero-shot）都比均勻算術平均差，因為加權會壓掉 few-shot 的多樣性貢獻。
- **Q：為什麼有效批次小反而好？** A：原本 768 只有 ~110 步，訓練不足；但太小（96）又會過擬合，所以 192 是甜蜜點。
- **Q：DoRA 不是比 LoRA 好嗎？** A：單獨在 r=256 確實略好（0.7811 > 0.7766），但其他 rank 不穩，且因為跟 zero-shot 太像，放進 ensemble 反而扣分。
