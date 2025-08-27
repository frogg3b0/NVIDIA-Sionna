# Part 2: Link Adaptation
* Link Adaptation 是 Layer 2 的核心功能之一
* 它透過「 動態調整傳輸參數 」以「匹配當前通道條件」來「優化單一無線鏈路的性能」
    * 在實際系統中，通道狀況會因為 fading、多徑、遮蔽而隨著時間變動
    * 因此我們需要根據估測到的 CSI 選擇合適的 MCS ，同時讓錯誤率不超過預設的Block Error Rate (BLER)，這就是 Link Adaptation 的任務
## Link Adaptation 的目標:
* 最大化實際吞吐量，同時維持區塊錯誤率（BLER）在可接受範圍內
    * 通常，這個問題會被簡化為「 維持 BLER 接近某個預先設定的目標值 」（稱為 BLER target，例如 10%）
    * 這個「目標值」是根據系統需求預先設計的
* 本教學中，我們將介紹 Sionna SYS 提供的兩種先進的 Link Adaptation 技術
    1. Inner-loop link adaptation (ILLA): 根據目前通道估計，在預設的 MCS 對照表（MCS → BLER）中，挑出最高階但 BLER ≤ 目標值 的那組 MCS
    2. Outer-loop link adaptation (OLLA): 根據 HARQ 回報結果來補償通道估計中非理想的情況

* 我們會利用由 Sionna RT 所產生的 ray-traced 通道樣本來進行上述的 LA 模擬
    *  代表你的通道樣本不是理想隨機模型，而是來自真實物理場景下的 ray tracing 結果

***

## Imports

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
    import sionna.rt
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
import matplotlib.pyplot as plt
import numpy as np

# Sionna components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, \
    RadioMapSolver, PathSolver, subcarrier_frequencies, Camera
from sionna.phy import config
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import dbm_to_watt, lin_to_db, log2, db_to_lin
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation
from sionna.phy.nr.utils import decode_mcs_index

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Internal computational precision
sionna.phy.config.precision = 'single'  # 'single' or 'double'

# Toggle to False to use the preview widget
# instead of rendering for scene visualization
no_preview = True
```

***

## Simulation parameter
這段程式碼是在「設定整體模擬實驗所需的參數」，這些參數將在 Sionna SYS 結合 Sionna RT 的 Link adaptation 模擬中被用到
* 我們假設通訊發生在 基地台和用戶終端之間的「下行鏈路」

```python
num_slots = 500                           # 在 5G NR 系統中，每個 slot 通常包含固定數量的 OFDM symbols
bler_target = .1                          # Block Error Rate (BLER) 的目標為 10%

# Time/frequency resource grid
carrier_frequency = 3.5
num_subcarriers = 1024
subcarrier_spacing = 30e3
num_ofdm_symbols = 13

# MCS table index
mcs_table_index = 1                       # 設定使用哪一張 MCS 表格，MCS表例如:（QPSK,1/2)(16QAM,3/4) 

num_bs = 1                                # 設定使用哪一張 MCS 表格，MCS表例如:（QPSK,1/2)(16QAM,3/4) 
num_bs_ant = 1                            # 只有一個 base station 
num_ut_ant = 1                            # base station天線數量 
num_streams_per_bs = num_ut_ant           # Number of streams per base station

# Base station transmit power
bs_power_dbm = 20                         # 基地台總發射功率為 20 dBm
bs_power_watt = dbm_to_watt(bs_power_dbm) # 將 dBm 單位轉為瓦特 

# Noise power per subcarrier
temperature = 294  # [K]
no = BOLTZMANN_CONSTANT * temperature * subcarrier_spacing  # 每條子載波的噪聲能量為 𝑁0
```

```python
# 發射功率均勻分佈在子載波和 stream 上
tx_power = np.ones(
    shape=[1, num_bs, num_streams_per_bs, num_ofdm_symbols, num_subcarriers])
tx_power *= bs_power_watt / num_streams_per_bs / num_subcarriers

# (Trivial) stream management: 1 user and 1 base station
rx_tx_association = np.ones([1, num_bs])
stream_management = StreamManagement(rx_tx_association, num_streams_per_bs)

# OFDM resource grid
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                             fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing,
                             num_tx=num_bs,
                             num_streams_per_tx=num_streams_per_bs)

# Subcarrier frequencies
frequencies = subcarrier_frequencies(num_subcarriers=num_subcarriers,
                                     subcarrier_spacing=subcarrier_spacing)
```
* `np.ones(...)`: 建立一個全為 1 的 Numpy 張量
