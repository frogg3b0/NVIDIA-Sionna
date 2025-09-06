# Introduction to Sionna RT

## Imports
```python
# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

# Other imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

no_preview = True # Toggle to False to use the preview widget

# Import relevant components from Sionna RT
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
```

***

## Loading and Visualizing Scenes
* Sionna RT 可以載入外部場景檔案（Mitsuba 的 XML 檔案格式），也可以載入內建場景
* 在本例中，我們載入了一個「德國慕尼黑聖母教堂周圍區域」的範例場景

### 1. 載入場景
```python
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile
```
* `load_scene()`： Sionna RT 的函式，用來載入場景
* 可以載入:
    * 「外部 Mitsuba XML 場景檔（自建或從 Blender 匯出）」
    * 「內建場景」，像這裡用的是德國慕尼黑的聖母教堂附近區域

 ### 2. 預覽場景
 開啟一個 互動式 3D 預覽視窗（只能在 Jupyter Notebook 用）  
 
 ```python
if not no_preview:
    scene.preview();
```

* `if not no_preview`: 如果 no_preview 是 False，就執行 `scene.preview()`
    * no_preview = True → 跳過 preview（適合在純伺服器環境）。
    * no_preview = False → 開啟互動視窗（適合在本地 Jupyter Notebook）。
* `scene.preview()`: 開啟一個互動式 3D 預覽視窗，讓你直接在 Jupyter Notebook 裡操作場景，在正式渲染或 ray tracing 計算前，先調整好攝影機角度。

### 3. 渲染高品質影像
先用 preview 選好視角 → 再高品質 render  

```python
if not no_preview:
    scene.render(camera="preview", num_samples=512);
```

* `render()`： 把場景渲染成高品質影像。
* `camera="preview"`： 使用剛剛 preview 的視角
* `num_samples=512`： 每像素採樣 512 次 → 更清晰、更真實，但耗時更久

### 4. 渲染並輸出成檔案
```python
if not no_preview:
    scene.render_to_file(camera="preview",
                         filename="scene.png",
                         resolution=[650,500]);
```
* `render_to_file()`： 渲染並存成圖片檔（PNG、JPEG...）
* `resolution=[650,500]`： 設定輸出的解析度（寬 650 × 高 500）

### 5. 建立並定義camara的位置以及觀看方向

```python
my_cam = Camera(position=[-250,250,150], look_at=[-15,30,28])
scene.render(camera=my_cam, resolution=[650, 500], num_samples=512); 
```

* `my_cam =`: 建立了一個名為 `my_cam` 的 Camera 物件
    * 攝影機所在的點是 [-250, 250, 150]
    * 它觀看的目標點是 [-15, 30, 28]
* `scene.render()`: 這行把整個場景 scene 用你剛剛設定的 `my_cam` 來渲染出圖
<img width="761" height="590" alt="image" src="https://github.com/user-attachments/assets/b0bd0c63-117c-46a0-83d2-f273449e7841" />

***

## Inspecting SceneObjects and Editing of Scenespp
這段主要是在介紹如何用 Sionna RT 去檢視與修改場景中的物件（SceneObjects），包括它們的：  
* 位置與方向（position, orientation, scaling, look_at）
* 移動速度（velocity） → Doppler shift 模擬
* 材質（radio_material） → 頻率相關性、對無線波的影響

***

### 1. 觀察場景中有哪些物件
每個 場景 (Scene) 都由多個 SceneObjects 組成，比如建築物、地面、樹木等  
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
scene.objects
```

```python
{'building_1': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbcce0>,
 'building_6': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbd400>,
 'building_5': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbc9e0>,
 'building_4': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbdc40>,
 'building_3': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbdbe0>,
 'building_2': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbddf0>,
 'floor': <sionna.rt.scene_object.SceneObject at 0x7f0fbadbde80>}
```

* `merge_shapes=False`：表示不要把所有物件合併成單一物件，而是保持各自獨立（這樣才能逐個編輯）
* `scene.objects`: 會列出所有物件的名稱與對應的 SceneObject 實例

### 2. 檢查物體的位置、方向、縮放、速度
```python
print("Position (x,y,z) [m]: ", floor.position)
print("Orientation (alpha, beta, gamma) [rad]: ", floor.orientation)
print("Scaling: ", floor.scaling)
print("Velocity (x,y,z) [m/s]: ", floor.velocity)
```

```python
Position (x,y,z) [m]:  [[-0.769669, 0.238537, -0.0307941]]
Orientation (alpha, beta, gamma) [rad]:  [[0, 0, 0]]
Scaling:  [1]
Velocity (x,y,z) [m/s]:  [[0, 0, 0]]
```

### 3. RadioMaterial 無線電波材質
```python
floor.radio_material

```

```python
ITURadioMaterial type=concrete
                 eta_r=5.240
                 sigma=0.123
                 thickness=0.100
                 scattering_coefficient=0.000
                 xpd_coefficient=0.000
```
這會顯示物件的無線電材質屬性：  
* `type`:	材質類型，如 concrete、glass 等
* `eta_r`:	相對介電常數（real part of permittivity）
* `sigma`:	電導率（會影響穿透損耗）
* `thickness`:	厚度
* `scattering_coefficient`:	散射係數
* `xpd_coefficient`:	交叉極化衰減係數（XPD）

***

## Ray tracing of Propagation Paths
這段的目的是說明如何在載入的 Sionna RT 場景中：  
* 放置發射器與接收器（Tx / Rx）
* 設定它們的天線陣列（Antenna Arrays）
    * 使用`scene.tx_array`定義 tx_array
    * 使用`scene.rx_array`定義 rx_array  
* 呼叫 `PathSolver` 進行 Ray tracing ，求得多徑傳播路徑（Propagation Paths）

***
### Part 1 — 場景與天線設定
#### 載入場景
```python
scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
```

#### 設定 Tx 天線陣列
使用`scene.tx_array`定義 tx_array  

```python
scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V"
)
```

#### 設定 Rx 天線陣列

```python
scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="dipole",
    polarization="cross"
)

```

***

### Part 2 — 放置 Transmitter / Receiver

```python
tx = Transmitter(name="tx", position=[8.5, 21, 27], display_radius=2)
scene.add(tx)

rx = Receiver(name="rx", position=[45, 90, 1.5], display_radius=2)
scene.add(rx)

tx.look_at(rx) # 讓 Tx 的方向朝向 Rx → 這會旋轉 Tx，使它發射主波束對準接收端

```
* `display_radius`: 用於視覺化的大小設定，純外觀設置與物理性質無關
* `name`: 必須是唯一的識別碼

***

### Part 3 — 建立一個可重複使用的 PathSolver 

```python
p_solver = PathSolver()
```
* 建立一個 路徑求解器實例，並把它命名成`p_solver`(它可以被重複使用)

***

### Part 4 — 呼叫該 PathSolver，把要求解的環境輸入進去
```python
paths = p_solver(
    scene=scene,
    max_depth=5,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=True,
    synthetic_array=False,
    seed=41
)

```
* `scene`:	要分析的場景物件
* `max_depth`:	射線可經過的最大交互次數（e.g. 5 表示最多 5 次反射/折射）
* `los`:	是否考慮 Line-of-Sight
* `specular_reflection`:	是否考慮鏡面反射
* `diffuse_reflection`:	是否考慮漫射（需隨機取樣方向）
* `refraction`:	是否考慮穿透與折射
* `synthetic_array`:	是否使用陣列中心的近似模型（若為 False → 模擬所有 Tx-Rx 天線對）
* `seed`:	隨機種子（確保漫反射等過程可重現）

最終的輸出`path`，包含發射器和接收器之間的所有路徑
* 這些路徑是deterministic
* 因為`diffuse_reflection`是隨機抽樣方向，因此可固定某個`seed`確保每次模擬可重現

***

### Part 5 — 把剛剛場景輸出的path可視化出來
```python
if no_preview:
    scene.render(camera=my_cam, paths=paths, clip_at=20);
else:
    scene.preview(paths=paths, clip_at=20);
```

* `no_preview = True`： 輸出靜態渲染圖像
    * 使用 `my_cam` 指定的 camera 視角
    * `paths=paths`： 把 ray tracing 得到的 propagation paths 畫出來（射線路徑可視化）
    * `clip_at=20`： 表示只畫 20 公尺以內 的 propagation paths
    
* `no_preview = False`： 使用互動式 3D 預覽視窗

<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/a3b61e7b-517e-452b-ba8a-a406e21ac549" />

***

## From Paths to Channel Impulse and Frequency Responses
### `路徑名稱.cir()`、`路徑名稱.cfr()`、`路徑名稱.tap()`
這段是在講：如何將 ray tracing 得到的多徑 `paths`，轉換成可以用於通訊系統模擬的 baseband 通道響應資訊（CIR / CFR）  

你已經從 PathSolver 得到`paths`，接下來的目標是：  
* `paths.cir`: 模擬 channel impulse response（連續時間 baseband）
* `paths.taps`	: 離散化後的 CIR（可對應 OFDM tap）
* `paths.cfr`: 頻域通道響應（例如 OFDM subcarrier 頻率響應）

### 補充說明
如果前面用PathSolver 求得的路徑變數命名為 `ppaatthh`

```python
ppaatthh = p_solver(scene=scene, ...)
```

那麼你後續要使用 通道相關的計算函數，就直接呼叫:  
* `ppaatthh.cir()`
* `ppaatthh.cfr()`
* `ppaatthh.taps()`

其他名稱同理

***
## `路徑名稱.cir`範例
### 1. 將 CIR 的值賦予到 a, tau

```python
a, tau = paths.cir(normalize_delays=True, out_type="numpy")

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
print("Shape of a: ", a.shape)

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
print("Shape of tau: ", tau.shape)
```

```python
Shape of a:  (1, 2, 1, 1, 20, 1)
Shape of tau:  (1, 2, 1, 1, 20)
```
* `paths.cir(...)`: 將 ray tracing 的結果轉換為 基頻等效的 CIR
    * `normalize_delays=True`: 讓最早一條 path 的 delay = 0
    * `out_type="numpy"`: 輸出格式指定為 Numpy
* 輸出：
    * `a`： `每個 path 的複數振幅（包含衰減、相位、極化等）
    * `tau`： 每個 path 的時延（以秒為單位）

### 2. 針對某個(tx_ant,rx_ant)的組合，把他們每一條path 對應的 a, tau 印出來
```python
t = tau[0,0,0,0,:]/1e-9 # Scale to ns
a_abs = np.abs(a)[0,0,0,0,:,0]
a_max = np.max(a_abs)

# And plot the CIR
plt.figure()
plt.title("Channel impulse response")
plt.stem(t, a_abs)
plt.xlabel(r"$\tau$ [ns]")
plt.ylabel(r"$|a|$");
```

* `tau[0,0,0,0,:]`: 擷取第 1 個 Rx antenna 與第 1 個 Tx antenna 的所有 path delay
* `np.abs(a)[0,0,0,0,:,0]`: 擷取第 1 個 Rx antenna 與第 1 個 Tx antenna的第 1 個 time step的所有 path magnitude
*  `a_max = np.max(a_abs)`: 計算最大振福

<img width="615" height="457" alt="image" src="https://github.com/user-attachments/assets/e31daafc-d36a-49d3-bc90-575a4536dcf7" />  

***
## `路徑名稱.cfr()`範例
### 1. 設定 OFDM 系統參數

```python
# OFDM system parameters
num_subcarriers = 1024
subcarrier_spacing=30e3

# Compute frequencies of subcarriers relative to the carrier frequency
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
```
* `subcarrier_frequencies(num_subcarriers, subcarrier_spacing)`: 根據「subcarrier數量」、「subcarrier spacing」，來產生每個子載波的頻率
    * 輸出為 shape: `(1024,)` 的實數 array
    * 結果會對稱分佈在 `[-15.36 MHz, ..., 0, ..., +15.36 MHz]`
 
### 2. 計算 Channel Frequency Response（CFR）

```python
h_freq = paths.cfr(
    frequencies=frequencies,
    normalize=True,
    normalize_delays=True,
    out_type="numpy"
)

print("Shape of h_freq: ", h_freq.shape)
# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
# Output: (1, 2, 1, 1, 1, 1024)

```
根據先前 ray tracing 產生的路徑 `path`，以及上面設定的 OFDM 參數，求對應的 CFR

### 3. 畫 CFR 幅度圖
```python
# Plot absolute value
plt.figure()
plt.plot(np.abs(h_freq)[0,0,0,0,0,:]);
plt.xlabel("Subcarrier index");
plt.ylabel(r"|$h_\text{freq}$|");
plt.title("Channel frequency response");
```
<img width="572" height="455" alt="image" src="https://github.com/user-attachments/assets/12e5ec48-5911-4155-8caa-912d2c6b21f2" />

***
## `路徑名稱.tap()`範例
### 為什麼需要 channel taps？

通道 taps 是從連續時間 CIR 轉為離散時間域 impulse response $h[\ell]$:  

$$
\sum_k a_k \cdot \delta(t - \tau_k)
$$

這表示「每條 path 都貢獻一個以 $tau_{k}$ 為中心的 sinc 函數」:  

$$
h[\ell] = \sum_k a_k \cdot \text{sinc}\left( \frac{\ell - \tau_k / T_s}{1} \right)
$$


因為 sinc 函數的時間響應是無限長的，所以在計算 taps 時，**必須選擇一段有限的區間去截斷 sinc，加總僅保留感興趣的部分。

---

### 1. 計算 channel taps

```python
taps = paths.taps(
    bandwidth=100e6,       # 低通濾波器頻寬 100 MHz
    l_min=-6,              # tap index 起始點
    l_max=100,             # tap index 結束點
    sampling_frequency=None,  # 預設為 Nyquist rate = 1 / bandwidth
    normalize=True,        # 能量正規化
    normalize_delays=True, # 最早 path delay = 0
    out_type="numpy"
)

# Output shape
# [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_taps]
print(taps.shape)  # (1, 2, 1, 1, 1, 107)
```

### 2. 畫 CIR tap 圖

```python
# 畫圖
plt.figure()
plt.stem(np.arange(-6, 101), np.abs(taps)[0,0,0,0,0]);
plt.xlabel(r"Tap index $\ell$");
plt.ylabel(r"|$h[\ell]|$");
plt.title("Discrete channel taps");
```

<img width="571" height="459" alt="image" src="https://github.com/user-attachments/assets/a8d47912-904c-471c-82bb-a039bfa48d62" />

***

## Radio Map

### 什麼是 Radio Map?
Radio Map = 空間平面 (𝑥,𝑦) 上每個點的「接收品質量測值」，常用於：  
* 基地台選址
* Beam coverage 分析
* RIS 調整
* 環境感知（Radio Environment Map）

More information about radio maps can be found in the detailed Tutorial on [Radio Maps](https://nvlabs.github.io/sionna/rt/tutorials/Radio-Maps.html)
### 如何產生 Radio Map? 使用 `RadioMapSolver`

### 1. 建立一個 `RadioMapSolver` 實例
```python
rm_solver = RadioMapSolver()
```

### 2. 呼叫求解器，並輸入要解的地圖

```python
rm = rm_solver(
    scene=scene,
    max_depth=5,
    cell_size=[1, 1],         # 每個像素的解析度（單位：公尺）
    samples_per_tx=10**6      # 每個發射器要模擬的射線數量（控制精度與時間）
)

```

### 3. 顯示 Radio Map（靜態圖 or 互動視窗）

```python
if no_preview:
    scene.render(camera=my_cam, radio_map=rm)
else:
    scene.preview(radio_map=rm)
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/44df0d9b-40ee-4ef1-9330-3d033846a2c6" />
