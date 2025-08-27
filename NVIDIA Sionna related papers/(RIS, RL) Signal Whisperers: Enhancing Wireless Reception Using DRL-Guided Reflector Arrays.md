# Signal Whisperers: Enhancing Wireless Reception Using DRL-Guided Reflector Arrays

## III. SYSTEM MODEL
### B. Assumptions 
#### 具體功能與特性
我們的模擬利用了 NVIDIA Sionna  deterministic propagation engine  
該引擎整合了詳細的三維幾何結構、材料特定的電磁特性以及多次反射和散射效應，以近似真實世界的訊號行為
* 整合 3D 幾何建模： 模擬真實建築物、牆面、障礙物的三維結構
* 材料特定的電磁特性: 使用 ITU 建議值為不同材質（如混凝土、金屬）設定真實的反射係數與穿透係數
* 多次反射和散射效應: 追蹤每一道訊號在多次反射下的衰減與方向變化

#### 目的與動機
這種 Ray tracing model 特別適合用於毫米波頻段，原因是:  
* 毫米波的反射與繞射特性與光學波段相似
* 實證研究表明， Ray tracing  與測量結果相比，可實現低於 3 dB 的 root mean square errors
* 並且通常可以透過對主要多路徑分量 (MPC) 進行建模來捕捉 90% 以上的接收功率
* 雖然確定性光線追蹤無法捕捉所有隨機小尺度現象，但這些未建模的路徑通常對標準室內和城市環境中的總接收功率貢獻微乎其微

在我們的設定中，我們最多考慮 8 次反射，因為額外的反射對接收功率的影響可以忽略不計

#### NVIDIA Sionna 功能
* dynamic channel modeling with **time-varying multipath delays**: 模擬 CIR 中每個 path 的延遲隨著時間的變化
* dynamic channel modeling with **Doppler shifts**

#### Sionna RT v.s. 傳統模擬
##### 傳統的RIS建模方法: 
1. 大多使用 Rayleigh/Rician fading 模型（即統計分布）來描述每一個單元格的反射行為。
2. 但這些計算效率高的方法，常常過度簡化實際環境中的幾何

##### Sionna RT:  
1. 具備高幾何解析度，能準確模擬:反射角度、毫米波下材質(水泥or金屬or玻璃)的電磁特性
2. 不同材料具有特定的反射係數，每次反射都會根據這些係數衰減訊號功率
3. 考慮了繞射效應

#### 實驗模擬假設
1. 裝置位置預設
    * 所有設備位置在模擬一開始就設定好
    * 實務上可透過像是 超寬頻（UWB） 技術完成定位
2. 反射器旋轉角度假設
    * 雖然每個 tile 有角度上下限，但這裡假設 每個 tile 可連續旋轉
3. UE 提供 CSI feedback
    * 假設用戶裝置（UE）能將其估計的通道資訊（CSI）回傳給反射器
    * 這項假設是合理的，因為：
        * UE 在接收時會自動進行 CSI 偵測
        * UE 也會傳送其自身位置，所以傳送 CSI 的額外成本很低

*** 

### C. Simulation Environment
* compute engine: 使用 NVIDIA Sionna 進行 propagation calculations
* 3D 模型場景建立: 利用 Blender 的建模工具，將牆壁、地板、天花板和反射器等結構元素與預先定義的方向和元素角度配置結合
* 模型中的材質: 依據 [Sionna 定義的材質](https://nvlabs.github.io/sionna/rt/api/radio_materials.html)列表進行標記  
    * 這些材料會極大地影響反射和透射係數，進而影響使用者的訊號強度
    * 為確保模擬的準確性和可靠性，我們遵循[國際電信聯盟 (ITU) 制定的所有材料相對介電常數和電導率](https://www.itu.int/rec/R-REC-P.2040/en)

以下是本研究的工作流程  
* 這個流程可以無縫地將模型在 Blender（建模引擎）與 Sionna（計算引擎）之間傳遞
    * Blender： 負責建立場景中的 3D 幾何與反射器模型
    * Sionna： 接收這些模型，進行電磁波傳播模擬（ray tracing）
  
<img width="5771" height="2855" alt="image" src="https://github.com/user-attachments/assets/99909b21-8a52-4f06-800b-1218df77737d" />  
   
#### 流程(左半部的Environment):  
1. 在 Blender 裡建立預設場景：空間結構（牆壁、地板、反射器、AP、UE 等），並設定反射器的位置與 tile 初始角度
2. 進入 Sionna RT ，並根據這個靜態場景與反射器角度，執行 一次完整的 ray tracing 模擬
根據 ray tracing 模擬的結果，產生 Reward 和 Observation

#### 流程(右半部的Agent):  
* 給定某個觀察值 (Observation)，應該怎麼設定 tile 的角度（Action）
    * 輸入： Observation 來自模擬場景的輸出
    * 輸出： Action（調整反射器 tile 的角度）
    * 回饋： Path Gain 作為 Reward，幫助 agent 評估動作好壞

#### 流程(右半部的Replay Buffer):  
* 將每一筆（Observation, Action, Reward, Next Observation）紀錄進緩衝區
* 用於 agent 的離線訓練與批次更新

***

## IV. REFLECTOR CONTROL WITH DEEP REINFORCEMENT LEARNING
### A. Overview of Reflector Control with Deep Reinforcement Learning
* DRL 是一種透過「智能體 (agent) 與環境互動」來學習最佳決策策略的方法

#### 1) Markov Decision Process (MDP):  
深度強化學習 (DRL) 框架本質上是一個馬可夫決策過程 (MDP)，這裡的 MDP 包含 5 個元素:  
* S (States)：反射器觀察到的環境狀態
* A (Actions)：可採取的動作，例如各 tile 的旋轉角度
* P (Transition probabilities)：從一個狀態轉移到下一個狀態的機率（可以是模擬或學習到的）
* R (Reward function)：給予某個動作的回饋（例如某用戶接收到的訊號強度）
* γ (Discount factor)：用來衡量**未來**回饋的價值，0~1之間

每個 time step 下:  
* agent 可以觀察到當前的狀態 $\{s^t} \in S\$
* 根據策略 π 做出行動 $\{a^t} \in A\$
* 根據行動得到一個 reward $\{r^t}\$
* 根據機率分布轉換到新狀態  $\{s^{t + 1}} \in S\$
我們的目標是**學習一個最優策略 π***，使長期獎勵最大化

***

#### State
在 DRL 框架下，狀態 (State) 是 agent 在某一時刻能觀察到的資訊。在這個反射器控制問題裡，狀態由三個部分構成：
1. CSI： 代表無線通道的即時特性。
2. 反射器所有 tile 的旋轉角度： 反映反射器當前的結構配置
3. 使用者位置： UE 的空間座標，幫助 agent 理解 CSI 與 tile 配置之間的幾何關係

#### Action
Action 最終只需要三個控制值（每列）：  
* Δϕ：該列的水平旋轉
* Δθ：該列中間 tile 的仰角
* ΔR：焦點距離（控制聚焦的位置）

#### Reward 
* 幫助 DRL agent 判斷「這次調整 RIS 角度到底好不好」
* 依據的是**使用者的平均路徑增益 (path gain)** → 也就是所有 UE 收到訊號的強度表現
* 計算 path gain: 用 Sionna radio solver 生成整個環境的「radio map」，就可以在圖上看到**每個位置對應一個 path gain 值**
* Reward 的組成部分:
    *  平均 path gain: 取所有使用者的 path gain 平均，這是主要指標
    *  difference: 觀察「這一步比上一步有沒有更好？」如果 path gain 比前一時刻更高 → 加分；反之 → 減分
    *  penalty: 當 agent 執行的操作，**導致旋轉運動超出系統角度限制**時，會受到懲罰

***

#### 2) Policy: π($\{a^t}\$| $\{s^t} \$)
Policy 就是 agent 的行為準則，決定**在某個狀態下**該**採取動作的機率分布**  
* 所有可能動作的機率總和 = 1

#### 3) State-Action Value Function Qπ($\{s^t} \$\{a^t}\$)
代表**如果在狀態 st 採取動作 at**，並且之後**遵循策略 π**，下來能期待獲得多少總回報  
* 不只看當下的 reward，還會把未來的 reward **乘上一個介於0和1之間的折扣因子 γ** 後，再加總起來

#### 4) Bellman Equation:
一個狀態動作的價值 = 立即獎勵 + 未來所有可能狀態的加權價值

#### 5) Experience Definition
一個完整的互動紀錄包含：(𝑠𝑡,𝑎𝑡,𝑟𝑡+1,𝑠𝑡+1) → 當前狀態、採取的動作、得到的獎勵、下一個狀態  
* 用途：此公式使 agent 能夠從歷史互動中學習，並相應地更新其策略

#### 6) Relay Buffer
一個「記憶資料庫」，存放 agent 的互動經驗  
* 如果直接用時間序列的資料學習，樣本之間高度相關 → 學習不穩定
* Replay Buffer 會隨機抽取樣本來訓練 → 打破相關性、避免發散

***

### B. Soft Actor-Critic Algorithms
此段落和Sionna無關，故快速帶過觀念

#### 為什麼選 SAC？
* SAC 是 off-policy：比 on-policy效率高，因為它能重複利用過去資料
* 引入 entropy最大化 → 這讓 policy 不會太快收斂到次佳解，保有探索能力
* 比 DDPG / TD3 更穩定：能處理更複雜的情境與多用戶障礙環境

#### SAC 的核心思想
* Policy π：是一個隨機策略，不是單一動作，而是動作分布
* 目標是最大化「期望回報 + 策略熵」
* 這樣 agent 會學到「既要好，也要敢嘗試新角度」

#### Actor-Critic 架構
* Actor (policy network)：負責輸出動作（如何旋轉 RIS）
* Critic (Q network)：負責評估動作的價值（這樣做會不會增加 path gain）
* SAC 交替訓練 Actor 和 Critic，確保策略和價值評估一起進步

*** 

### C. Neural Network Design
此段落和Sionna無關，故快速帶過觀念

#### Neural Network 設計的四個核心部分
1. Embedding Position (位置嵌入)
    * 把 UE 的座標位置（低維度數字）映射到更高維度的特徵空間
    * 技術來自 NeRF (Neural Radiance Fields)，用 sin/cos 週期函數轉換座標
    * 在這個研究中 → UE 位置經過這種編碼，可以讓 agent 更快學到「位置 → 反射角度 → path gain」的關聯
2. Feed-forward Block (FF Block)
    * Actor 與 Critic 都用 前饋塊 (FF Block) + 殘差連接 (residual connection)
3. Critic Network (評估網路)
    * 輸入狀態：UE 位置 + RIS 角度 + CSI
    * 輸入動作（RIS 調整的向量） → 經過自己的 FC block
    * 最後輸出一個 Q-value（評估這個 state-action pair 的期望回報）
4. Actor Network (策略網路)
    * 輸出不是單一動作，而是動作分布的兩個參數
    * Actor 的策略是 隨機性 (stochastic policy)，不是只給一個確定角度，而是動作分布

***

## V. RESULTS AND DISCUSSION
* 本節評估反射器在NLOS條件下，是否能增強多用戶的訊號接收
* 採用 DRL（深度強化學習） 來最佳化 RIS 的波束配置與 path gain

###　A. Experimental Setup
實驗在一個L形走廊中進行，以分析存在障礙物時的無線訊號傳播，如圖所示:  
<img width="462" height="782" alt="image" src="https://github.com/user-attachments/assets/49211286-d784-4ac0-9ef2-2a2dc00cfc1c" />  

系統配置:  
* AP
    * 位置: [1,0]
    * 高度: 2.5 m
    * 多天線配置
    * 頻率: 28GHz (毫米波)
* 障礙物 (obstacle)
    * 1 個實心橢圓形，代表人體
    * 高度：1.8 m
    * 位置：[−6.0, −3.5]
* RIS
    * 位置：[2.1, −3.9]
    * 由 72 個六角形 tile 組成
* 用戶配置:  
    * 三個 UE，單天線
    * 區域範圍：固定，或是在範圍 x ∈ [−11, −8], y ∈ [−4.25, −2.75] 均勻分布
* 3D 模型的材質設計
    * 牆壁：plasterboard（石膏板）
    * 天花板：board material（板材）
    * 地板：concrete（混凝土）
    * 障礙物：wood（木材）
    * 參數依據：ITU Recommendation （同 Table I 的介電常數與導電率數據）
* 反射器初始狀態
    * 每個 tile 的角度（θ, ϕ）在每次實驗 episode 開始時，隨機分佈於 [−π/4, π/4]
<img width="344" height="153" alt="image" src="https://github.com/user-attachments/assets/29646ecd-c2f5-4f38-87dd-44c1e9d51909" />

*** 

### B. Hyperparameters
這段沒有提到 Sionna，所以只整理重點觀念  

#### 演算法選擇與網路結構
* 使用 SAC 演算法（Soft Actor-Critic）
* Actor & Critic 網路：各 2 層全連接層 (FC)，每層 256 neurons
* 啟用函數：GELU（平滑、有效的非線性）
* 優化器：AdamW（自適應學習率 + 權重衰減，增強泛化能力）

#### 學習率與訓練參數
* Critic 學習率：4.0×10^−4
* Actor 學習率：1.8×10^−4
* 折扣因子 𝛾=0.985
* 經驗回放：buffer 容量 3×10^4筆 transition
* 每次更新 batch size = 256 → 確保梯度估計穩定

#### Entropy Regularization
* 熵係數 𝛼：自動調整
* 目標熵 = −𝑑𝑖𝑚(𝐴)
* 保證 agent 在學習過程中持續有足夠的探索行為

#### Target Network 更新
* 使用平滑更新 (soft update)係數 ，𝜏=0.005

#### 訓練初始化
* 訓練開始前，先執行 1001 次互動步驟：填充 replay buffer，收集多樣化的經驗樣本、同時計算 CSI 與角度的統計量 → 用來正規化輸入特徵
* 好處：加快後續學習並提升穩定性

#### 計算需求
* 單次實驗迭代需要約 24 小時
* 原因：每個樣本模擬耗時約 4 秒（NVIDIA 3090 GPU）
* 加速方法：可透過 平行運算縮短訓練時間

*** 

* 為了模擬真實場景，反射器的初始配置在每次實驗開始時都會在實際物理約束範圍內隨機化
* 使用者位置在訓練和評估階段均保持不變

* Baseline 比較: 三種情境下的 平均 Path Gain（所有 UE）：
    1. 沒有反射器 → 約 −105.1 dB
    2. 大金屬平板（固定反射器） → 約 −87.2 dB
    3. 可控 RIS + DRL → 約 −75 dB
可見 DRL 控制的 RIS 明顯優於傳統固定反射面
<img width="183" height="351" alt="image" src="https://github.com/user-attachments/assets/47e3715a-0aba-405c-a67b-403e76a3d457" />
<img width="183" height="351" alt="image" src="https://github.com/user-attachments/assets/ad6a8e47-e303-41f4-be65-7cd1c89015c9" />
<img width="288" height="351" alt="image" src="https://github.com/user-attachments/assets/2f3133e4-9a64-416a-a834-fed81756619f" />

***

### E. Tracking of Mobile Users
實驗目的:  
* 測試 RIS + DRL 系統在使用者移動 (mobility) 情境下的表現
* 與先前靜態實驗不同，這裡不會重新初始化，而是要求 DRL 動態調整 RIS 配置以追蹤移動中的 UE

實驗設置:  
* 環境：L 型走廊
* AP 與 RIS → 固定位置。
* 3 個 UE 沿預設軌跡移動，每 10 個 simulation steps 移動 0.25 m
* RIS tile 初始角度 → 常態分布

適應能力:  
* DRL agent 成功調整 RIS，使訊號覆蓋範圍在**新 UE 位置**仍保持高品質
<img width="534" height="437" alt="image" src="https://github.com/user-attachments/assets/8ebd9883-30b5-4ef7-8fa9-3e8b8b00c465" />

性能數據:  
* 平均 Path Gain：從 −88 dB → −75 dB，提升約 13 dB。
* 標準差區間：不同初始 RIS 配置會造成差異，但收斂後趨勢一致。
* 在約 第 3 個 episode (step 30) 後，系統能穩定維持性能，即使 UE 持續移動
<img width="613" height="506" alt="image" src="https://github.com/user-attachments/assets/b8a5ef44-a0eb-4012-85c1-04ebd93065d2" />

***

### F. Ablation Study for Network Size
實驗目的:  
* 檢驗 Actor / Critic 網路規模（neuron 數量） 對 DRL 性能的影響
* 比較兩種網路大小：128 neurons vs. 256 neurons

固定 UE 位置場景:  
* 256-node 網路表現比 128-node 差
* 原因：網路太大 → 學習更複雜，但資料有限 → 易 overfitting 或收斂困難
<img width="600" height="472" alt="image" src="https://github.com/user-attachments/assets/fa05d322-e3f9-4f3f-b694-3e76c997dc0a" />

隨機 UE 位置場景:  
* 同樣趨勢：128-node 網路顯著優於 256-node
* 小網路在資料有限的情況下表現更好
<img width="600" height="480" alt="image" src="https://github.com/user-attachments/assets/c351aed9-0177-4ad2-8800-dcbea4223424" />

結論:  
* 較小網路 (128 neurons) → 適合這類 RIS 控制的 DRL 問題
* 較大網路 (256 neurons) → 雖有更強表達力，但在樣本不足時反而失效

*** 

### G. Performance Trade-off for Applying Deep Learning
計算負擔 (Computational Overhead)  
* 使用 DRL 控制 RIS → 相較靜態或啟發式方法需要額外運算。
* 但可透過 模型量化 (quantization) 與 網路精簡大幅降低
* 一旦 DRL 訓練完成，推論開銷很小，可在嵌入式硬體上即時執行

延遲 (Latency)  
* 影響最大的計算開銷來自「重新最佳化 / 更新配置」，而非持續推論

能耗 (Energy Consumption)  
* 量化模型 → 減少運算能耗
* 系統設計 → 主要能耗來自偶爾更新，非持續運算
* RIS 本身為 被動元件，不需持續消耗電能
* 僅在使用者位置或環境改變時進行少量調整
* 結果：在邊緣運算/電池供電情境下仍能運作高效，且能實現 ~12 dB path gain 改善，不增加明顯能源或運營負擔

### H. Path to Practicality: Future Work
這裡作者提出未來可能的研究方向，目的是讓 DRL 控制的 RIS 系統更貼近真實部署  

#### 擴展環境場景
* 目前限制：研究主要在 L 型走廊 進行模擬與測試
* 未來計畫：將框架擴展至更多不同類型的場景
* 目的：驗證 DRL 方法在各種複雜環境下仍然能自適應

#### 多智能體強化學習 (MARL)
* 現有方法：只考慮單一 RIS，且優化目標是「所有使用者的平均 path gain」
* 問題：這種方法忽略了不同使用者之間的需求衝突
* 改進：採用 MARL，讓每個 RIS 都有一個 agent，負責不同使用者或使用者群組，可以針對 不同的 QoS (服務品質需求) 做更細緻的最佳化。在高密度環境下，能解決 公平性 (fairness) 與 資源分配衝突。
* 效果：系統效能提升的同時，也能確保不同使用者的需求都被兼顧

#### On-device DRL（嵌入式強化學習）
* 願景：將 DRL 直接整合到 RIS 裡，使系統能夠在實際部署時，邊運行邊學習
* RIS 能從 即時量測數據 學習，不再完全依賴模擬
* RIS 能夠隨著時間自我優化，針對不同的電磁環境與使用者行為不斷調整
* 意義：縮短「模擬 → 實際」的落差，達到真正的自適應

***

## VI. CONCLUSION AND FUTURE WORK
### 本研究的主要貢獻:
* 提出並驗證了 基於 DRL 的 RIS 自調整設計
* 在有遮蔽的環境中，系統大幅提升多使用者的平均 path gain
* 與 baseline（無 RIS、固定金屬板）相比，DRL 控制 RIS 明顯更優

### 學術與應用意義
* 本研究展示了 智慧無線環境 (Smart Radio Environments, SRE) 的實現潛力，為未來的 動態自適應無線通訊系統 提供了基礎

### 未來展望
* 持續優化 DRL 演算法，使學習更快、更穩定
* 探索更多複雜場景（如大型會議室、高密度用戶、更多障礙物）
* 進一步提升 RIS 系統的適應性與通訊效能，向實際部署推進
