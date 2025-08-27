# Sionna-PHY
## Introduction to Sionna PHY
### 此套件提供一個可微分的鏈路層模擬器（link-level simulator）  
* link-level 模擬器: 是指針對發送端與接收端之間的點對點通訊鏈路進行模擬
* 可微分: 表示這個模擬器的各個模組都支援 TensorFlow 的自動微分，這樣才能串接 AI 模型進行訓練，以及實現端對端學習

***

### 此外，Sionna PHY還無縫整合多個通訊系統元件，例如:  
* 前向錯誤更正（FEC）
* 符合 5G NR 規範的 encoder/decoder
* 多輸入多輸出（MIMO）系統
* 正交分頻多工（OFDM）
* 符合 3GPP 規範的無線通訊通道模型

***

### 給初學者
* Part 0: "Hellow world!"
* Part 1: Getting Started with Sionna
* Part 2: Differentiable Communication Systems
* Part 3: Advanced Link-level Simulations
* Part 4: Toward Learned Receivers
* Part 5: Basic MIMO Simulations
* Part 6: Pulse-shaping Basics
* Part 7: Optical Channel with Lumped Amplification

### 給進階開發者
進階使用者可參閱後續內容，以深入理解 Sionna PHY 的內部運作方式，並學習如何擴展與自訂物理層演算法。
* 5G Channel Coding and Rate-Matching: Polar vs. LDPC Codes
* 5G NR PUSCH Tutorial
* Bit-Interleaved Coded Modulation (BICM)
* MIMO OFDM Transmissions over the CDL Channel Model
* Neural Receiver for OFDM SIMO Systems
* Realistic Multiuser MIMO OFDM Simulations
* OFDM MIMO Channel Estimation and Detection
* Introduction to Iterative Detection and Decoding
* End-to-end Learning with Autoencoders
* Weighted Belief Propagation Decoding
* Channel Models from Datasets
* Using the DeepMIMO Dataset with Sionna
* Link-level simulations with Sionna RT

***

### Sionna PYH 實作範例
#### 範例1: QAM symbols over an AWGN channel (Part 1)
<img width="738" height="790" alt="image" src="https://github.com/user-attachments/assets/8f9e3728-8de8-4f05-9df6-c0041e036aca" />
 

***

#### 範例2: 使用 5G Polor/LDCP 的 encoder/decoder (Part 1)
<img width="617" height="173" alt="image" src="https://github.com/user-attachments/assets/12769ebb-3014-40da-9cc6-bc2950a5b2df" />
   
<img width="1367" height="900" alt="image" src="https://github.com/user-attachments/assets/461863ee-5671-4ee6-9856-57a6de1e166d" />


***

### 範例3: OFDM Resource Grid (Part 3)
<img width="568" height="265" alt="image" src="https://github.com/user-attachments/assets/e259685e-02dc-47c5-ae97-c6da3850b313" />
<img width="605" height="455" alt="image" src="https://github.com/user-attachments/assets/7640e154-2577-4c9c-8458-0b63076beca4" />

***

### 範例4: 使用 3GPP model (Part 3)
<img width="737" height="485" alt="image" src="https://github.com/user-attachments/assets/11f616fa-3e5c-4ade-a7a1-77486e243110" />
<img width="1368" height="900" alt="image" src="https://github.com/user-attachments/assets/9c4c83d7-0079-42e4-a6d9-a615b8a02a7e" />

***

### 範例5: Implemention of an Advanced Neural Receiver (Part 4)
<img width="1782" height="522" alt="image" src="https://github.com/user-attachments/assets/4b7f7834-a7ff-4269-a259-831156d5ae98" />
<img width="739" height="658" alt="image" src="https://github.com/user-attachments/assets/58f9dbf0-dd88-47be-a4c1-49a016c9c42c" />
<img width="1368" height="900" alt="image" src="https://github.com/user-attachments/assets/03861bc9-ba06-49ef-9c99-7343c48995e3" />



