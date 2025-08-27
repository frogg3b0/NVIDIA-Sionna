# Part 1: Physical Layer Abstraction
## 前言: Physical Layer Abstraction 的觀念
#### 為甚麼我們需要 physical layer abstraction?
* 在大型網路模擬中，對每個發射機和接收機的完整物理層處理進行建模計算成本高昂
* Physical Layer Abstraction（或稱為 link-to-system mapping），是一種透過高效計算的方式**準確預測鏈路性能**，從而及時運行模擬的方法
#### Physical layer abstraction model 包含兩部分
* Link Quality Model: 依據發射器、干擾者位置、衰減（fading）等資訊估算每個 subcarrier 的通道品質
* Link Performance Model: 用這些 SINR + 調變/編碼預測錯誤率（PER 或 TBLER）

***

## 本章內文介紹
在這份 notebook 中，你將學會如何跳過模擬繁重的物理層處理流程
* 意思是我們不需要逐位元模擬發射、傳輸、接收、解碼等物理層流程
* 以最低計算成本**預測錯誤率**、**估計通道**，維持高準確性

我們會透過以下方式來抽象化每位用戶的  PHY computations 
* 針對每個使用者的每條 stream 對應的 SINR 做彙總（aggregation），合併為一個單一的 等效 SINR
* 使用 Sionna PHY 的前向糾錯 (FEC) 模組中**預先計算的表**，將有效 SINR 值對應到傳輸區塊誤碼率 (TBLER)
* 此方法能夠快速計算每個用戶的**傳輸速率**和 **HARQ feedback**，**使系統級模擬能夠擴展到數十個基地台和數百個用戶**  
<img width="2534" height="545" alt="image" src="https://github.com/user-attachments/assets/cdb81c13-52ce-4be8-8d45-075b19606067" />

***

## Import
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    if gpu_num!="":
        print(f'\nUsing GPU {gpu_num}\n')
    else:
        print('\nUsing CPU\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# Import Sionna
try:
    import sionna.sys
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
```

```python
# Additional external libraries
import numpy as np
import matplotlib.pyplot as plt

# Sionna components
from sionna.phy.utils import log2, insert_dims, db_to_lin
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.mimo import StreamManagement
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation
from sionna.phy import config

# Internal computational precision
sionna.phy.config.precision = 'single'  # 'single' or 'double'

# Set random seed for reproducibility
sionna.phy.config.seed = 45
```

***

## Instantiate a PHYAbstraction object
當你建立 [`PHYAbstraction()`](https://nvlabs.github.io/sionna/sys/api/abstraction.html#sionna.sys.PHYAbstraction) 物件時，它會自動載入「每個 MCS index 對應的 Effective SINR → BLER 映射表」
* 每一組 MCS（調變 + 編碼率）都有一張表，這張表的橫軸是 effective SINR，縱軸是 BLER
* 就可以透過這張表，告訴你在某個 SINR 下，這種 MCS 模式成功傳輸的機率是多少

舉例來說： 假設 MCS index = 15 是「16QAM + 1/2」，那它的映射表可能是這樣：  
```python
Effective SINR (dB)  | 	BLER
-2.0	               |  0.99
0.0	                 |  0.85
2.0	                 |  0.40
4.0	                 |  0.05
6.0	                 |  0.001
```
有關 MCS 的更多信息，請參閱 5G NR [Transport Block](https://nvlabs.github.io/sionna/phy/api/nr.html#sionna.phy.nr.TBConfig)

程式範例:  
```python
phy_abs = PHYAbstraction()   # 建立一個 PHYAbstraction 物件，載入所有 預先計算好的 MCS → BLER 映射表

mcs_category = 0      # 指定 MCS 類型: 0 for uplink, 1 for downlink
mcs_table_index = 1   # Table index: [1,2] for UL, [1,2,3,4] for DL
mcs_index = 15        # 指定一個具體的 MCS index（例如 index 15 可能代表 16QAM + code rate 1/2）

# Plot a BLER table
_ = phy_abs.plot(plot_subset={
    'category':
    {mcs_category: {'index':
                    {mcs_table_index:
                     {'MCS': [mcs_index, mcs_index+2]}}}}},  # MCS index
    show=True)
```
這段程式碼會載入 `PHYAbstraction` 的資料表
* 然後選出你指定的 uplink MCS 組合
* 畫出對應的 BLER vs. SINR 曲線圖
<img width="578" height="455" alt="image" src="https://github.com/user-attachments/assets/227be785-92e6-40d0-be83-0084f25f41f3" />
<img width="578" height="455" alt="image" src="https://github.com/user-attachments/assets/d0196519-8f5a-4b64-9bc0-d0d1170a2959" />

***

## Retrieve BLER values from interpolated tables
當你建立 `PHYAbstraction` class 時:  
* 系統會將預先計算好的 BLER 表格，根據 SINR 與 Code Block Size（CBS） 兩個變數，進行插值處理
* 所以之後查表（例如 SNR = 9.3dB、CBS = 82）也能給出準確的 BLER，而不會侷限於原始資料點
* 讓你能查到任意 SINR 與 CBS 組合的 BLER

下面的程式碼片段詳細示範了這個操作  
```python
# 設定查詢的 SINR 與 Code Block Size
snr_eff_db = 9
cb_size = 80

# 把這些數值轉換成插值表格的對應索引
snr_db_idx = phy_abs.get_idx_from_grid(snr_eff_db, 'snr')
cbs_idx = phy_abs.get_idx_from_grid(cb_size, 'cbs')

# 從插值表中查出 BLER 值
bler = phy_abs.bler_table_interp[mcs_category,
                                 mcs_table_index - 1,
                                 mcs_index,
                                 cbs_idx,
                                 snr_db_idx]

print(f'The BLER for MCS {mcs_index} at SNR {snr_eff_db} dB and CB size {cb_size} is {bler.numpy():.4f}')
```
整體流程說白話就是
1. 我要查 SNR = 9 dB、CBS = 80 bits 時的錯誤率（BLER）
2. 我先把這兩個浮點數轉成索引
3. 然後去內插表格裡查出錯誤率
4. 印出來

通常情況下，您不應直接呼叫此方法做內插，而應該使用 PHYAbstraction 類別提供的「call 方法」，直接把 SINR、Code Block Size、MCS 資訊丟進去，讓系統自動內插查表  

***

## Generate a new BLER table
除了Sionna內建的BLER查表，我們也可以自己產生一組BLER查表資料
* 你要測試某些 非預設 MCS 或 CBS 值
* 你使用了自訂或不同的 decoder 架構
* 想模擬出專屬系統的 BLER 表格，而不是用 Sionna 內建資料

程式範例:  
```python
# 定義模擬新 BLER table 的 MCS index
sim_set = {'category': {
    1:                         # 0 for PUSCH, 1 for PDSCH
    {'index': {
        2: {'MCS': [16]}       # index 2/ MCS 16 的 BLER 曲線   
    }}}}

# 執行新模擬時的 SINR 和程式碼區塊大小
sinr_dbs = np.linspace(5, 25, 25)    # 我要模擬從 5dB~25dB ， 25 個不同的 SINR 點
cb_sizes = [24, 200, 3000]           # 每個點都對應 3 種 CBS（短、中、長編碼）
# 這會總共生成 25 × 3 個 BLER 值

# Compute new BLER tables
new_table = phy_abs.new_bler_table(
    sinr_dbs,
    cb_sizes,
    sim_set,
    max_mc_iter=15,  # max n. Monte-Carlo iterations per SNR points
    batch_size=10,
    verbose=True)

# Plot the new BLER table
phy_abs.plot(plot_subset=sim_set,
             show=True);

```
系統幫你用 Monte Carlo 模擬生成：  
* downlink (PDSCH) 類別（category=1）
* MCS table 2 中的 MCS index 16
* 針對 3 種 Code Block Size：24、200、3000
* 針對 25 個 SNR（EbNo）點，範圍從 5 到 25 dB

<img width="578" height="455" alt="image" src="https://github.com/user-attachments/assets/6b7eecb0-b5d7-4ca5-9be5-4b297c09cc68" />

當你執行 `phy_abs.new_bler_table(...)` 時，模擬結果會自動覆蓋到 `phy_abs.bler_table` 的對應位置
另外，我們可以透過下面的 code，手動觀察、繪圖、分析這份結果
```python
new_table["category"][1]["index"][2]["MCS"][16]["CBS"][200]
```

輸出: 25 個 SNR 點(從5dB~25dB)的對應 BLER) 
```python
{'BLER': [1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  0.9933333396911621,
  0.846666693687439,
  0.5799999833106995,
  0.1666666716337204,
  0.019999999552965164,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0]}
```

*** 

## Bypass physical layer computations
我們接著要繞過(bypass)物理曾模擬，但依然可以拿到幾個重要的指標    
也就是說，當 `PHYAbstraction` 物件被實例化（建立）之後，我們就可以使用它來產出下列資訊:  
* Number of succesfully decoded bits: 成功傳送的實際資料量 bit
* Hybrid automatic repeat request (HARQ) value: 判斷使用者是否成功接收封包:
    * 若傳輸區塊成功接收則回傳 ACK（HARQ=1），否則回傳 NACK（HARQ=0），若缺少資訊則回傳 -1
* Effective SINR value: 可用來衡量使用者通道品質
* Block error rate (BLER): 代表一個 code block 解碼失敗的機率
* Transport block error rate (TBLER): 代表一個傳輸區塊中至少有一個 code block 解碼錯誤的機率

### Step 1. 設置模擬的「背景必要參數」  
```python
mcs_table_index = 1                           # [1;4] for downlink and [1;2] for uplink
direction = 'uplink'                          # 'downlink' or 'uplink'
mcs_category = int(direction == 'downlink')   # 判斷目前是否為 downlink， 因此這邊會輸出 0 (False)
num_ut = 80                                   # 80 個使用者 (UEs)

# Time/frequency resource grid
num_ofdm_symbols = 14                         # 每個 slot 裡 OFDM 符號的數量  
num_subcarriers = 1024                        # 每個 OFDM 符號內 subcarrier 數量 = 1024 
num_streams_per_ut = 1                        # 每個 UE 只傳 1 個 stream

# BLER target value
bler_target = .1                              # 這是設定模擬的目標 區塊錯誤率（BLER） ≤ 10%
```
* `mcs_table_index`:設定使用哪一張 MCS 表格， MCS 表格代表 modulation + coding rate 的組合（例如 QPSK + 1/2、64-QAM + 5/6 等）
    * 根據 3GPP TS 38.214 規範: uplink 支援 index 1 和 2，downlink 支援 index 1~4
* `direction = 'uplink'`: 你自己決定目前模擬的傳輸方向

### Step 2. 為每個 Stream 產生線性 SINR
* 目前這份教學採用簡化版流程，先用亂數產生每個 stream 的 SINR
* 之後的進階教學會介紹如何用**隨機通道模型**或**ray tracing**，建立更貼近真實環境的 SINR 分佈，而不是隨機亂數

```python
# Generate random SINR values across UT
# Target shape:
# […, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
sinr_db = config.tf_rng.uniform([1,1,num_ut,1], minval=-5, maxval=30)

# Generate random SINR values across UT, subcarriers, and OFDM symbols
sinr_db = sinr_db + config.tf_rng.normal(
    [num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], mean=0, stddev=2)
sinr = db_to_lin(sinr_db)
```
* `config.tf_rng.uniform([1,1,num_ut,1], minval=-5, maxval=30)`: 呼叫 TensorFlow 的亂數產生器，產出「均勻分布（uniform distribution）」的亂數
    * `[1,1,num_ut,1]`: 產出的 shape  `[num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]`
    * `minval=-5, maxval=30`: 指定亂數範圍：每個 SINR dB 值會落在 −5𝑑𝐵 ~ 30𝑑𝐵 之間
* `config.tf_rng.normal()`: 呼叫 TensorFlow 的亂數產生器，這次是常態分布亂數
    * `mean=0, stddev=2`: 擾動的平均值為 0，標準差為 2 dB
* ` db_to_lin(sinr_db)`: 將`sinr_db`由**dB**轉為**線性單位**

### Step 3. 為每位使用者 UE 選擇一個最適的 MCS
* 這段程式碼的目的，是要 在繞過實體層 (bypass PHY) 的模擬設定下，自動為每位 UE 選擇一個最適的**調變與編碼**
* 這個選擇會根據前面我們產生的 SINR

```python
illa = InnerLoopLinkAdaptation(phy_abs, bler_target)
mcs_index = illa(sinr=sinr,
                 mcs_table_index=mcs_table_index)
print("MCS indices: ",mcs_index.numpy())
```
1. 用 `phy_abs` 和 `bler_target` 兩個參數，建立一個 `InnerLoopLinkAdaptation` 類別的物件，命名為 `illa`
2. 什麼是 InnerLoopLinkAdaptation？
   * `InnerLoopLinkAdaptation` 是 Sionna 提供的模組，用來在繞過物理層的模擬。
   * 根據目前觀察到的 SINR，自動對每個 UE 決定最適的 MCS index
   * `phy_abs`: 這是先前建立的 `PHYAbstraction` 模組物件。負責根據 SINR 查詢對應的 BLER 值
   * `bler_target`: 使用者指定的 目標 BLER。 表示我們希望系統挑選出「最高的 MCS index」，但前提是 錯誤率不超過`bler_target`
3. `mcs_index = illa(sinr=sinr, mcs_table_index=mcs_table_index)`: 根據每個使用者的 SINR 值，查詢指定的 `mcs_table_index` 表，並回傳每個使用者的最佳 MCS

輸出: 
```python
MCS indices:  [14 18 10  5 27 23  7  5 13 27 25 27 24 23 12 27  5 27 19  2 16 17 27 27
 21 27 22 25  3 15 25 27 21  7 27  2 27  7 23 22  9 15 27 27 27 27 20 27
 27  7 21 16 16 17  5 27 25 12 27 23 27 16 27 27  3 12 27 27 27 17  4 27
 27 12 14 27  5 27 19 21]
```
* 輸出的 mcs_index 是一個陣列，長度為 num_ut = 80，即每個 UE 各自對應的 MCS 選擇

#### Step 4. 模擬物理層傳輸的表現
1. 前面已經用 `InnerLoopLinkAdaptation` 針對每位 user 計算出適合的 MCS index
2. 接下來，就是套用這些 MCS 與 SINR，模擬物理層傳輸的表現
3. `phy_abs(...)` 會根據 MCS 對應的 modulation/coding、SINR、資源大小，估計錯誤率與是否解碼成功

```python
# [batch_size, num_ut]
num_decoded_bits, harq_feedback, sinr_eff, bler, tbler = \
    phy_abs(mcs_index,
            sinr=sinr,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category)

sinr_eff = sinr_eff.numpy()                                            # 將張量 sinr_eff 轉換為 numpy array
assert np.all(num_decoded_bits.numpy()[harq_feedback.numpy()==0]==0)   # 驗證 HARQ 回報為 0 的情況下，num_decoded_bits 必須為 0
print("HARQ feedback: ", harq_feedback.numpy())                        # 列出 HARQ 結果
```
* 這段程式碼是 呼叫 `PHYAbstraction` 模組 `phy_abs(...)`，傳入每個使用者的 MCS index 以及對應的 per-stream SINR，來模擬實際物理層傳輸的結果
* 主要目的是獲得以下五個輸出
    1. `num_decoded_bits`： 每位使用者成功解碼的資料位元數
    2. `harq_feedback`： 每位使用者的 HARQ 回報（1 成功, 0 失敗）
    3. `sinr_eff`： 根據 MCS 與接收到的 SINR 計算出的 等效 SINR
    4. `bler`： Block Error Rate（每個 code block 的錯誤率）
    5. `tbler`： Transport Block Error Rate（整個傳輸區塊的錯誤率)
* `phy_abs(...)`: 模擬物理層傳輸行為，不實際傳送訊號，而是根據 SINR 與 MCS 表查表/內插，快速預測 BLER、解碼成功與否等結果

***

## Effective SINR
* 理想情況: 在 AWGN channel，每一個 symbol 所經歷的 SINR 是一樣的、不會變動，因此我們可以直接用一個 SINR 值對應出 BLER
* 實際情況: 在 Fading channel，每一個 symbol 所經歷的 SINR 是不同的，導致整個 codeword 是在一個「變動的 SINR 場景下」傳送的
* 所以，我們會將真實通道中的「變動 SINR 分布」壓縮為一個等效數值：Effective SINR

```python
ind_sort = np.argsort(sinr_eff)                    # 將使用者按 sinr_eff 排序，為了後續繪圖時有整齊漸進的順序    

sinr_t = np.transpose(sinr.numpy(), [2, 0, 1, 3])  # 維度交換
sinr_t = np.reshape(sinr_t, [num_ut, -1])          # 將每位 user 的 SINR tensor 攤平成一維陣列
sinr_t = sinr_t[ind_sort, :]                       # 根據先前的 ind_sort，將 sinr_t 中的 user 順序由 低 SINR 到高 SINR 排列


fig, ax = plt.subplots()
for ut in range(num_ut):
    label = 'Per-stream' if ut==0 else None
    ax.scatter([ut]*sinr_t.shape[1], 10*np.log10(sinr_t[ut, :]), c='y', alpha=.3, edgecolors='k', label=label)
ax.plot(10*np.log10(sinr_eff[ind_sort]), '-k', linewidth=3, label='Effective')
ax.plot(10*np.log10(sinr_t.mean(axis=1)), '--r', linewidth=2, label='Arithmetic mean')
ax.set_xlabel('User')
ax.set_ylabel('SINR [dB]')
ax.set_title('Per-stream vs. effective SINR')
ax.grid()
ax.legend(framealpha=1)
plt.show()
```
<img width="574" height="455" alt="image" src="https://github.com/user-attachments/assets/ace68579-edf6-4f9c-868a-2e67d30622f5" />

* 當 channel 很好時（右邊的 user），per-stream SINR 有些很高，但 effective SINR 會因為公式中使用指數平均而拉低
* 當 channel 很差時（左邊 user），per-stream SINR 本來就低，這時候 Effective SINR ≈ 平均 SINR

***

## Achieved vs. Shannon spectral efficiency
比較每個使用者**實際達成的頻譜效率**與**理論上在該 SINR 下可達到的 Shannon capacity bound**  

```python
# [num_ofdm_sym, num_subcarriers, num_ut]
is_re_allocated = np.sum(sinr.numpy(), axis=-1) > 0         # 判斷哪些 RE 被使用 
num_allocated_re = np.sum(is_re_allocated, axis=(-2, -3))   # 計算每個 user 被分配了多少 RE
se_achieved = num_decoded_bits / num_allocated_re           # 計算實際頻譜效率
se_shannon = log2(tf.cast(1, sinr_eff.dtype) + sinr_eff)    # Shannon 頻譜效率 
```
* `is_re_allocated = np.sum(sinr.numpy(), axis=-1) > 0 `: 判斷哪些 RE 被使用
    * `sinr` 是個 4 維 Tensor
    * `sinr.numpy()`：把 Tensor 轉成 numpy array，好操作
    * `np.sum(..., axis=-1)`：在最後一個維度（即 stream）上相加
    * `>0`: 代表 這個 RE 有被使用
    * 所以 `is_re_allocated` 現在 shape 是 `[num_ut, num_ofdm_symbols, num_subcarriers]`
* `num_allocated_re = np.sum(is_re_allocated, axis=(-2, -3))`: 計算每個 user 被分配了多少 RE
    * `axis=(-2, -3)`： 針對 OFDM symbols 與 subcarriers 做 sum
    * 輸出為 [num_ut]：每位 user 在整張頻時網格上用了幾格 RE
* `se_achieved = num_decoded_bits / num_allocated_re`: 計算實際頻譜效率
    * `num_decoded_bits`: 每個使用者成功 decode 的 bits 數
    * `num_allocated_re`: 每個使用者分配到的 RE 數

```python
def get_cdf(values):
    """
    Computes the Cumulative Distribution Function (CDF) of the input vector
    """
    values = np.array(values).flatten()
    n = len(values)
    sorted_val = np.sort(values)
    cumulative_prob = np.arange(1, n+1) / n
    return sorted_val, cumulative_prob

fig, ax = plt.subplots(figsize=(5,4))
ind_sort = np.argsort(se_shannon.numpy())
ax.plot(se_achieved.numpy()[ind_sort], label='Achieved')
ax.plot(se_shannon.numpy()[ind_sort], label='Shannon bound')
ax.set_title('Spectral Efficiency')
ax.set_xlabel('User')
ax.set_ylabel('SE [bits/s/Hz]')
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
```
<img width="490" height="390" alt="image" src="https://github.com/user-attachments/assets/779ea3fe-3f07-41ed-a8b8-3e8bdc76af44" />  

* 可以看到，實際達成的頻譜效率明顯低於Shannon capacity
    * 當 SINR 很差時，為了保證封包錯誤率低於某個門檻（例如 10%），系統會選擇比較保守的 MCS，也就是低速率導致頻譜效率被犧牲
    * 當 SINR 很高時，因為實務中系統的 MCS 模式是有限的，當 SINR 再高，頻譜效率也不會再上升，因為已經用到最高階的 modulation
* 當傳輸區塊（Transport Block）無法正確解碼時，會觀察到頻譜效率為零，這種情況的發生機率就是 TBLER

***

## Conclusion
1. `PHYAbstraction` class 透過計算等效 AWGN SINR（ or effective SINR），繞過了實際的實體層處理，並根據預先計算的對應表將此 SINR 映射為對應的 BLER
2. 在這份 Notebook 中，SINR 的生成與 MCS 的選擇流程是經過簡化的，因此許多模擬細節（例如多使用者通道估計、實際 SINR 擬合模型、動態 MCS 決策）都被省略或簡化了
3. 若你想更深入理解這些主題，可以參考以下的 Notebook
    1. [SINR computation from OFDM channel matrices](https://nvlabs.github.io/sionna/sys/tutorials/HexagonalGrid.html)
    2. [Link adaptation for MCS selection](https://nvlabs.github.io/sionna/sys/tutorials/LinkAdaptation.html)
