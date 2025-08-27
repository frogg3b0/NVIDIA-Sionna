# System Level (SYS)
* 這個模組支援可微分的系統層級模擬，這在深度學習或神經網路共同設計通訊系統時非常關鍵，因為它允許使用者透過反向傳播調整參數
* 支援多小區網路（multi-cell networks），也就是可模擬一個基地台不只一座的情境，模擬多個基地台之間的干擾與合作行為
![Uploading image.png…]()


## Introduction
* SYS 模組是基於物理層抽象模型所建構
* Sionna SYS 不僅涵蓋物理層，還進一步模擬 Layer 2 (資料連結層) 的核心功能，例如:
    * link adaption for adaptive modulation and Coding scheme selection : 根據通道狀況選擇最適合的 調變與編碼方式
        * Modulation and Coding scheme selection (MCS): 根據 當下的無線通道品質，動態選擇最合適的調變方式 (Modulation) 與編碼率（Coding scheme)
    * Downlink and uplink power control: 根據用戶通道品質，調整基地台或用戶的發射功率
    * User scheduling: 根據用戶需求等因素決定哪些 user 在當前時槽分配資源，或被分配多少子載波
* 在 Sionna SYS 系統層模擬中，你可以將多個基地台放置成六邊形蜂巢狀網格的形式 
