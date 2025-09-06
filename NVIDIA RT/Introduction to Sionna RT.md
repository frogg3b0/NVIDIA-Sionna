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
