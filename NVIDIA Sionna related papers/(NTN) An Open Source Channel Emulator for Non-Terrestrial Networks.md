# An Open Source Channel Emulator for Non-Terrestrial Networks
## Abstract
非地面網路（Non-Terrestrial Networks, NTN）是實現 6G 中「無所不在的連接性」這一目標的關鍵技術之一。然而，由於在 NTN 中**取得真實世界資料的成本昂貴**，因此需要能夠模擬多種 NTN 場景的準確通道模型，以進行通訊技術的開發與測試  

本研究中，我們實作了由 3GPP 所提供的多種 NTN 通道模型，並將其封裝於一個開源框架中
* 此框架可整合進現有的 Python 框架 Sionna™ 中，使研究者能透過鏈路層（link-level）模擬來探討 NTNs
* 透過開源此框架，我們讓使用者能根據特定應用情境加以調整，而無需自行實作底層複雜的數學模型
* 此框架以 Python 實作，作為現有 Sionna™ 框架的擴充，而 Sionna™ 已內建大量符合 5G 標準的通訊模組

***

## Introduction
### 非地面網路（NTNs）是實現「無所不在覆蓋（ubiquitous coverage）」目標的關鍵技術之一。  5412
* Non-Terrestrial Networks (NTNs)：包含衛星（LEO/MEO/GEO）、高空平台（HAPs）、無人機等非地面通信節點。
* ubiquitous coverage：代表無論城市、鄉村、海上、山區甚至空中等，任何地方都能連網

### NTNs 面臨的挑戰
* 同一衛星波束下，使用者之間存在顯著不同的通道條件
* 距離遙遠造成的長時間延遲
* 非地面元件高速移動導致的顯著 Doppler 效應
因此，在模擬 NTN 通道時，必須使用完整的通道模型；若僅依賴簡化模型，將無法充分捕捉其所面臨的全部複雜性

### 開發上述技術方案需要大量數據
* 來自 NTN 的真實世界資料仍然有限且代價昂貴
* 例如，衛星發射與連線測試的時間與頻譜資源昂貴，無法像地面 5G 那樣收集大規模通道量測數據

### 解決方法: Sioona 
* 我們的開源框架，透過使用支援 3GPP TR38.811 標準中的通道模型來進行模擬
    * 3GPP TR38.811: 該標準所包含的模型是根據真實世界量測資料所建立
    * 研究人員與開發者能在複雜的非地面條件下測試
* 本文提出一個用於「生成 NTN 通道模型」的全新開源框架，此框架可整合進「鏈路層模擬」，特別是現有的 Sionna™ 軟體框架中
* 我們所選擇實作的模型皆基於**真實世界量測資料**，因我們期望這些模型能最貼近實際情境

### 補充說明: 何為 Sionna
* Sionna 是一個以 Python 開發、專為通訊研究設計的工具集
    * 支援 5G/6G PHY 層元件模擬，例如 OFDM、LDPC 編碼、beamforming、MIMO detection 等
* Sionna 利用 GPU 加速，並採用模組化、開源的設計，從而實現彈性且高效能的模擬
* 透過開源此框架，讓所有人都能使用這些高階通道模型，而無需自行實作其複雜的底層數學模型
    * 使用者不需深入 3GPP 的數學公式，即可快速呼叫模擬器生成通道資料


### 在本論文發表時，該框架可於 GitHub 上取得，名稱為 [OpenNTN](https://github.com/ant-uni-bremen/OpenNTN)

***

## II. CAPABILITIES AND APPLICATIONS OF THE FRAMEWORK
### 在本節中，我們將列出使用者在使用此框架時可直接調整的主要參數  

#### 自定義通道模型
我們的框架允許使用者定義多種可調整的通道模型，這些模型是根據 3GPP TR38.811 標準建立的  
* 代表這些模型不是固定參數，而是可根據情境進行自定義（如高度、頻率、角度等）
* 3GPP TR38.811：此為 NTN 相關的官方技術報告，定義了 LEO/MEO/GEO 等場景的物理通道模型
  
該標準所提供的模型描述了具有頻率選擇性（frequency selectivity）的多徑通道  
一般來說，這些通道被分為三種情境:  
1. 高密度都會區（dense urban）
2. 都會區（urban）
3. 郊區（suburban）
此外，像是 Line-of-Sight, LOS 機率等參數，也會隨不同情境而改變

#### 自定義傳輸相關參數
選定模擬情境後，使用者還可以進一步調整傳輸相關參數，這些參數包含：  
* 傳輸方向（上行或下行）
* 載波頻率: 必須位於 S 波段（本標準定義為 1.9GHz 至 4GHz）或 Ka 波段（定義為 19GHz 至 40GHz）範圍內。
* 仰角（使用者與衛星與地平線之間的角度): 必須在 10 度到 90 度之間
* 衛星高度: 高度需設定於 600 公里至 36,000 公里之間
* 使用者端與衛星的天線型態: 衛星與使用者端（UTs）可使用單天線或陣列天線
* 使用者端數量: 使用者端的數量可以自由設定，但模擬中衛星的數量始終為 1

#### 模型功能
* 所定義的模型可用來產生通道脈衝響應（CIR），包含對每一組發射與接收天線之間的所有路徑，每一條路徑包含：
    * path coefficient：複數增益（振幅與相位）
    * path delay：訊號經由該路徑的時間延遲
* 產生出的 CIR 可直接使用，或進一步處理為預先定義之 OFDM 系統的通道矩陣
* 藉由使用陣列天線與多個使用者端，此框架支援 MU-MIMO 模擬
* 框架亦提供對 Doppler 效應的控制
    *  使用者可選擇開啟或關閉，視是否需在模擬中考慮由衛星運動所導致的 Doppler 位移而定
    *  但在某些情況下（例如已知衛星精確位置與軌道時）可假設該效應是可預測並被補償的
*  框架的模組化設計使得使用者能進一步自訂模擬情境，例如修改使用者移動行為、速度與地理環境等參數
*  此框架可與其現有的 5G 元件相容，例如 LDPC 編碼、通道估計與 OFDM 等模組
*  框架的 GPU 加速特性能確保模擬效率，使其能夠處理大規模模擬研究

***

## III. NON-TERRESTRIAL CHANNEL MODEL
### 模型定義
* NTN 通道模型背後的數學模型定義於 3GPP TR38.811 標準中，該標準是對地面通道模擬標準（3GPP TR38.901）的延伸
* 此外，模型的定義還參考了其他文件，例如由國際電信聯盟（ITU）所提供的環境損耗描述
* 以及 3GPP TR38.821 中進一步的參數與場景描述
由於整體建模過程非常複雜，且散見於多份文檔中，本節將對我們所實作的流程做一個總結，幫助讀者掌握足夠的知識，以便自訂框架並充分利用其開源特性

***

### 實作的模型具備 frequency-selective multipath model  
該框架的輸入是一個由使用者定義的場景與其參數，輸出則是所有發射與接收天線對之間的通道路徑集
* input: 場景類型（urban/suburban）、衛星高度、頻率、仰角、天線配置等
* output: CIR (各條路徑的 path coefficient/ path delay)

***

### 建模流程
#### 整個建模流程分為三個部分：
#### 這三個流程又可細分為16個步驟
<img width="512" height="212" alt="image" src="https://github.com/user-attachments/assets/f766c294-4422-4161-91f9-cb5ce8179f21" />

* 藍色方框: 這些步驟是 NTN（非地面網路）專有的；或是與3GPP TR38.901 中定義的地面通道模型產生有顯著差異
* 紅色方框: 這些步驟和地面模擬中的做法幾乎一樣

***

### A. Topology and Large Scale Parameters
#### 1) Satellite and Users Layout
* 會產生衛星與使用者的佈局配置，也就是建立 3D 幾何場景，這個配置包括每個使用者端（UT）與衛星在全域坐標系中的位置與方向
    * 每個使用者與衛星的空間座標（x, y, z）會建立在地球為參考系的全域座標系中
* 像是「使用者之間的距離」，會根據所選場景隨機抽樣:
    * 在「高密度都會區」（dense urban）中，使用者間距離較近
    * 在「郊區」（suburban）中，距離則較遠
* UT 與衛星間的距離依賴於「仰角」與「衛星高度」，而 UT 的方向為隨機設定，而衛星則始終朝向地球表面正下方

<img width="281" height="246" alt="image" src="https://github.com/user-attachments/assets/abc4a8ac-abb4-409a-a953-c9d7a92a3aff" />

***

#### 2) Antenna and Topology
所有終端和衛星都會分配一個天線。天線可以是單一天線，也可以是陣列天線。 

##### 可用的 Satellite Antenna 如下:  
1. Reflector antenna with a circular aperture

<img width="512" height="124" alt="image" src="https://github.com/user-attachments/assets/a0434508-333b-4c6f-bd13-38fe951b6043" />

這是一個以「圓形反射面天線」為基礎的 Airy Pattern，依據 TR38.811 定義
* θ： 與主波束方向（boresight）之間的夾角，0 度表示正中央。
* 𝑎： 天線孔徑（aperture）半徑。
* 𝑘=2𝜋𝑓/𝑐 ：波數，f 是載波頻率、c 是光速。
* 𝐽1： 第一類第一階貝索函數（Bessel function）

2. Uniform rectangular panel array with dual linear polarization antenna pattern

##### 可用的 UT Antennas 如下: 
1. Quasi isotropic antennas for handheld UTs (全向性手持式終端裝置)
2. Dual linear polarized patterns (雙線性極化)    
3. very small aperture terminal (VSAT) antennas

***

#### 3) Propagation Condition
每個使用者端會被指定為「 LOS」或「 NLOS」狀態  
* 每個狀態（LOS 或 NLOS）的機率，是根據場景與仰角 𝜃 決定的 [(參考 3GPP 38.811 version 15.4.0 6.6.1-1)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3234)

<img width="614" height="269" alt="image" src="https://github.com/user-attachments/assets/0493cf42-87b6-47ff-9f3e-cb438bf85134" />


* 與地面模擬不同，衛星模擬中不考慮室內通訊情況

***

#### 4) Atmospheric, Rain & Cloud, Scintillation
由於訊號在衛星與地球表面之間傳播，會產生多種損耗，這些損耗在地面通訊中並不存在  
由於這些損耗的計算涉及多個步驟，這裡僅列出名稱  
* PL_g：氣體損耗（或對流層損耗），僅在 Ka 波段考慮，對 S 波段為 0
* PL_s：閃爍損耗或電離層損耗，僅在 S 波段考慮，對 Ka 波段為 0
* 雨衰與雲衰已整合進其他損耗中，未單獨列出

***

#### 5) Calculate Losses
我們定義總損耗為 PL_total = PL_b ​+ PL_g + PL_s​ + PL_e​
* PL_e: 建築穿透損耗，因為模擬中不考慮室內 UT，因此此項始終為 0
* PL_b: 基本路徑損耗，FSPL (d, fc) + SF + CL(θ,fc​)
    *  FSPL (d, fc): free space pathloss = 32.45+20 log10 (fc)+20 log10(d)
    *  SF: The shadow fading， SF ∼ N(0, σ2)
    *  CL: The clutter loss，依據仰角 𝜃 的常數，若為 LOS 情況則為 0
* PL_g：氣體損耗（或對流層損耗），僅在 Ka 波段考慮，對 S 波段為 0
* PL_s：閃爍損耗或電離層損耗，僅在 S 波段考慮，對 Ka 波段為 0

***
#### 6) Correlated LSPs (SF, DS, AS, K)
對每個使用者，我們從高斯分布中抽樣下列參數：  
* 延遲展開度（DS）
* 到達仰角（ZOA）
* 發射仰角（ZOD）
* Rician factor 𝐾

這些參數的分布是[參考3GPP 38.811 version 15.4.0 6.7.2)](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3234)

***  

### B. Small Scale Parameter
* 本部分的方法是基於 [3GPP 38.901 version 18.0.0  Chapter 7.5](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173)
* 參數定義則基於 [3GPP 38.811 version 15.4.0](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3234)

***
* 在每對發送端與接收端之間:
    * 產生 𝑚 個 cluster
        * 每個 cluster 中包含 𝑛 條 ray
    * 所有在同一時刻抵達接收端的 ray，會被歸為一個 path

<img width="347" height="354" alt="image" src="https://github.com/user-attachments/assets/24442a4e-9ee5-4bcc-9aa5-9819a0739656" />

#### 1) Cluster Delays:
對每對發射器與接收器產生 𝑚 個 cluster，每個cluster會產生一個基礎延遲
* 在這個 cluster 內又會再產生 n 條ray，會在此delay基礎上加上微小變動 (delay spread)
* 延遲值會根據使用者的 delay spread 與場景特定比例因子進行縮放
若為 LOS 狀況，延遲還會根據 Rician factor 𝐾 再調整

#### 2) Cluster Powers:
每個 cluster 都會被指派一個 power（功率），這個功率由以下三個因素決定：  
1. Delay Spread： DS 越大，表示路徑越分散
2. delay distribution factor
3. shadowing term: 描述大尺度 shadow fading ，從高斯分佈隨機抽樣
接著，所有 cluster 的功率會做正規化，讓總和等於 1  

#### 3) Arrival & Departure Angles:
每條 ray（在每個 cluster 中的成員）都會被分配一對角度：  
* AoD：從 Tx 發射出去的方向
* AoA：到達 Rx 的方向

#### 4) Random Coupling of Rays:
為了讓模擬出來的通道具有更真實的隨機性，模型會將每個 cluster 中的 ray 順序隨機打亂  

#### 5) Calculate Cross Polarization:
Cross Polarization: 「發射」與「接收」的波在正交極化下，所遭受的能量轉換與損耗  
* 每條 ray 都會被賦予一個「極化交叉比」（Cross Polarization Ratio, XPR）
* 這個 XPR 值從 高斯分佈抽樣

***

### C. Channel Coefficient Generation
* 本部分的方法是基於 [3GPP 38.901 version 18.0.0  Chapter 7.5](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173)  
* 使用非地面特定參數，並添加了由於法拉第旋轉和時變多普勒頻移引起的相位旋轉，產生復值通道增益 ，同樣參考 [3GPP 38.811 version 15.4.0](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3234)

***

這段落為 NTN 通道模型的最終階段，是將前面已生成的多徑資訊（cluster/ray、角度、功率、delay）進一步轉換為複數形式的通道增益值  
整體可拆分為 5 個步驟:  
1. Random Initial Phases: 對於每條 ray ，初始相位取自 (−π, π) 上的均勻分佈
2. Faraday Rotation Calculation: 「地球磁場」和「電離介質」會為射線引入旋轉，稱為**法拉第旋轉**(基於載波頻率計算)
3. Time-varying Doppler Shift: 在 NTN 情況下，會額外計算「時變衛星多普勒頻移」，並將 DL, UE 頻移相加
4. Channel Coefficients: 利用所有計算結果，計算每條 ray 的通道係數
<img width="438" height="328" alt="image" src="https://github.com/user-attachments/assets/ad117b3e-f81e-481b-b475-f6a718bbc401" />
5. Apply Losses and Shadowing: 最後，將產生的「通道係數值」乘以 「PL_total」，以施加 loss/ shadowing (此操作不考慮傳輸功率，傳輸功率目前始終被視為 1)

***

## IV. USER INTERFACE
### 本節我們簡要介紹:  
* 該模擬框架的使用者介面
* 設定方式、輸入參數與輸出結果

### 如同第三章所述，生成通道需經歷四個步驟
1. 天線設定: Tx/Rx 元件數、陣列形狀、波束寬度、方向圖
2. 場景建構: propagation model（LoS/NLoS）、shadow fading 模型、延遲展開（DS）、角度展開（AS）
3. 拓撲定義: 定義用戶數、位置、高度、速度，以及衛星高度
4. 通道係數模型建立: 計算 CIR 中的每條 ray（功率、角度、多普勒、相位等）
Fig. 4 展示了使用此框架的各個步驟與所需的參數類型

<img width="604" height="350" alt="image" src="https://github.com/user-attachments/assets/6c7762b6-6d82-40b7-9af7-ca638eacdefe" />

* 通道模型產生後，會導出通道衝激響應（CIR），可用來產生所需的通道
* Sionna™ 框架已內建函式，可將 CIR 轉為 OFDM 時域通道或頻域通道，兩者都需搭配 `resource grid` 使用 (可參考Sionna PYH)

***

## V. CALIBRATION AND TESTING
為確保我們的實作正確性，我們進行了三步驟的完整測試流程：  
1. 根據 3GPP 標準進行校準
2. 執行 傳統單元測試（unit tests）
3. 與 其他模擬模型進行效能驗證與比較

***

### A. Calibration using the 3GPP TR38.821 standard
我們依據 3GPP TR38.821 標準 所提供的校準指南來驗證我們的模型  
* 該標準定義了多種測試場景，包括不同的 衛星高度、載波頻率 與 仰角，並提供對應的 benchmark 供比對
* 共定義了 30 種不同的測試情境，並提供來自真實世界量測的參考數據
* 在每個情境中，標準提供了對應的 link budget ，這些值也會在我們的模擬中產出，以便進行比對
* 利用這些值以及預先定義的參數（例如天線增益和發射功率），我們可以計算出 SNR

作者使用自己的模擬框架計算不同 NTN 場景下的 自由空間損耗（FSPL）、大氣損耗（Atmospheric Loss）、閃爍損耗（Scintillation Loss），進而計算 SNR   
* 並與 3GPP 標準值對照，發現誤差非常小（最大僅 0.6 dB），因此認為模擬框架是準確的

***

### B. Unit Testing of Framework Components
第二步是使用傳統的 單元測試（unit test） 來確保所有模組的功能正確性:  
* 參數產生器（parameter generation）
* 都卜勒效應處理（Doppler effect handling）
* 天線配置（antenna configurations）
執行這些測試可確保每個模組都能如預期運作，並在程式碼更新過程中維持其正確性。

*** 

### C. Comparison with Other Existing Models
除了依據 3GPP 標準與傳統單元測試驗證我們的模型外，我們也進行了與其他現有模擬框架的比較  
* 由於我們框架中的模型基於機率分布（例如距離、路徑數、都卜勒偏移等），是以隨機抽樣方式產生，因此無法進行完全的「解析驗證（analytical verification）」
* 此外，其他模擬框架的重點不同，可能使用不同的建模假設，因此結果之間產生 一定程度的誤差或不一致 是可以預期的

我們用來比較的對象包括以下幾種模擬框架與資料集：  
* MATLAB 實作版本：依據 3GPP TR 38.811 標準【14】
* C++ ns-3 模擬器：也依據 TR 38.811 所建構【13】
* Gigayasa 平台：支援 3GPP CDL（Cluster Delay Line）與 TDL（Tapped Delay Line）模型【15】
* 盧森堡大學提供的資料集：包含 NTN 場景下的通道係數【16】

***

## VI. OUTLOOK AND CONCLUSION
### Conclusion
1. 在本研究中，我們基於 3GPP TR38.811 標準，提出了一個用於模擬 NTN 的開源框架
2. 我們的框架允許使用者模擬各種場景和環境，包括密集城區、城區和郊區場景​​
3. 透過全面的驗證，我們確保了框架的準確性和穩健性
    1. 根據 3GPP 標準進行校準
    2. 單元測試
    3. 性能比較

***

### Outlook
未來，我們計劃從多個方向擴展該框架的功能:  
1. 一旦完整的參數集可用，我們計劃整合 3GPP TR38.811 標準的分析模型
   * 這將使不同仰角的模擬更加精確，消除目前對非標準角度縮放參數的依賴
2. 隨著 3GPP 和其他組織發布新標準，我們將定期更新框架，以跟上最新發展
3. 除了目前基於 3GPP 的模型外，我們還計劃整合來自國際電信聯盟 (ITU) 等組織的其他模型，以涵蓋更多方面，例如衛星星座和高級大氣條件
4. 從靜態拓樸（drop-based）擴展到動態拓樸模擬
   * 將加入「使用者或衛星移動」的情境，模擬拓樸隨時間變化的動態過程 
