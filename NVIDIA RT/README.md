# Introduction to Sionna RT
### 為什麼需要 ray tracing 
* 它可以模擬真實環境下、某個特定位置的無線通道，比傳統隨機模型更精確

### Sionna RT 是什麼
* 一個 開源、GPU 加速、可微分的 ray tracing 引擎，專門用來做無線電波傳播建模
* 建立在 Mitsuba 3（光線渲染系統） 之上
* 使用 Dr.Jit（可微分 JIT 編譯器）來自動計算梯度，方便用於優化與 AI 訓練

### Sionna RT 可以拿來做甚麼
* 能計算與物體材質、天線模式、散射特性、方位與位置相關的梯度 → 這對 天線設計、波束成形、最佳化反射面配置 等研究非常重要。
* 可結合 Blender + OpenStreetMap 快速建立 3D 場景，再透過 Sionna RT 的 API 與 3D Viewer 進行無線場景模擬。
