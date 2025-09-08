# Tutorial on Mobility 

## 本篇目的
目的是說明如何模擬發射器、接收器或場景物件在移動時，通道會產生什麼變化  
* 你會學會如何透過修改場景物件的 `position` 與 `orientation` 來「移動物件」
* 你會了解每個物件的 `velocity` 屬性如何影響 Doppler shift
* 你會學會如何利用 `Paths.doppler` 屬性計算每條 propagation path 的 Doppler shift

***
## Background Information
### 你可以透過兩種方式模擬「物件移動對 CIR 的影響」
1. 移動幾何位置並每步重新 Ray Tracing
2. 固定幾何並根據 Doppler 模擬時間演化 (物體移動的距離很短，幾何位置幾乎沒變)

為了計算特定路徑的 Doppler shift（如下圖所示）:  
* 這條 path 是從一個 Tx 發出，途中經過 「n 個散射點」（例如建築牆面、車子、玻璃）最終到達一個 Rx
* 在這條 path 中，圖示中「每個點都可能屬於一個有速度的物體」，例如:
    * 一個會動的車子反射面
    * 一個移動中的發射器
    * 一個移動中的接收器
* 因此電磁波經過這些點後，可能會產生反射/折射/散射，進而又有一個新的傳播方向 $k_n$
* **上述的這些資訊，會用來計算這整條路徑的Doppler shift 𝑓Δ**

<img width="700" height="257" alt="image" src="https://github.com/user-attachments/assets/cc3d405c-2669-4fb8-a084-0b2bf5c5b21d" />

***

## GPU Configuration and Imports

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      RadioMapSolver, PathSolver
from sionna.rt.utils import r_hat, subcarrier_frequencies
```

***

## Controlling Position and Orientation of Scene Objects
場景中的每個物件都具有「位置」和「方向」屬性，這些屬性可以檢查和修改。為了演示這一點，我們加載一個包含簡單街道峽谷和幾輛汽車的場景。  

### 1. 載入一個包含簡單街道和幾輛汽車的場景
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                   merge_shapes=False)
cam = Camera(position=[50,0,130], look_at=[10,0,0])

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();
```
* `load_scene()`: 載入一個內建的場景模型
* `merge_shapes=False`: 不要把場景內的物件合併成單一幾何結構，方能控制每台車的位置
* `cam = Camera(position=[50,0,130], look_at=[10,0,0])`: `cam`的位置[50,0,130]，朝向[10,0,0] 
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/780e2479-ec65-49fb-a204-670578f6632f" />

### 2. 條列出場景中所有的物件
```python
scene.objects
```

```python
{'building_1': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd7370>,
 'building_2': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd41f0>,
 'building_3': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd45b0>,
 'building_4': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd6020>,
 'building_5': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd7010>,
 'building_6': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd43d0>,
 'car_1': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd6ef0>,
 'car_2': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd4f40>,
 'car_3': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd5120>,
 'car_4': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd4e80>,
 'car_5': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd5c30>,
 'car_6': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd62c0>,
 'car_7': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd6410>,
 'car_8': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd7c10>,
 'floor': <sionna.rt.scene_object.SceneObject at 0x7fd8dfcd7be0>}
```
* 從輸出的結果可以知道，場景內有六個建築、8輛車、地板

### 3. 選取單一車輛並檢查其位置與朝向

```python
car_2 = scene.get("car_2")
print("Position: ", car_2.position.numpy()[:,0])
print("Orientation: ", car_2.orientation.numpy()[:,0])
```

```python
Position:  [25.         5.5999994  0.7500001]
Orientation:  [0. 0. 0.]
```

* `car_2.position`: 拿到這台車的「位置」，是 TensorFlow 的 tensor，形狀為 (3, 1)
* `.numpy()`: 把 TensorFlow tensor 轉成 NumPy
* `[:, 0]`取出所有列（3 個軸），第 0 欄

### 4. 設定該車輛的位置與朝向
```python
car_2.position += [0, 10, 0]
car_2.orientation = [np.pi/2, 0, 0]

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();
```

* `car_2.position += [0, 10, 0]`: 把 car_2 沿著 y 軸「往上」或「往北」移動 10 公尺
* `car_2.orientation = [np.pi/2, 0, 0]`:  把 car_2 繞 z 軸 轉 90 度

<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/458bcb3f-1b6f-4133-a205-7d9242227825" />   

***

### 接續前段: 當一台車上有發射天線時，觀察其在街道場景中行駛時的信號覆蓋效果變化

### 1. 載入場景與設定 `Camera()`
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars,
                   merge_shapes=False)
cam =  Camera(position=[50,0,130], look_at=[10,0,0])
```

### 2. 設定 TX 與該陣列
```python
scene.add(Transmitter("tx", position=[22.7, 5.6, 0.75], orientation=[np.pi,0,0]))
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
scene.rx_array = scene.tx_array
```
* 在 `car_2` 的車頭位置（約 22.7 公尺 x 軸）加上一個發射器
* `orientation=[π, 0, 0]`： 代表方向轉 180°（朝負 x 軸方向，向後發射）
* 設定使用 3GPP tr38901 垂直極化（V）的 1x1 平面天線陣列 作為 TX 和 RX

### 3. 建立 RadioMapSolver
```python
rm_solver = RadioMapSolver()
```
* `RadioMapSolver` 是 Sionna RT 提供的工具，用來計算：
    * 每個位置的 路徑增益（Path Gain）
    * RSS（接收信號強度）
    * SINR
    * 並輸出 2D 覆蓋圖（可視化）

### 4. 設定移動參數：每次移動 10 公尺，共模擬 3 次
```python
displacement_vec = [10, 0, 0]     # 每次向 x 軸方向移動 10 公尺
num_displacements = 2            # 共移動 2 次 + 初始位置 = 3 次繪圖
```

### 5. 模擬循環：每一輪都畫一張覆蓋圖，然後更新車輛位置

```python
for _ in range(num_displacements+1):

    # 每一輪計算 Radio Map
    rm = rm_solver(scene=scene,
                   samples_per_tx=20**6,
                   refraction=True,
                   max_depth=10,
                   center=[0,0,0.5],
                   orientation=[0,0,0],
                   size=[186,121],
                   cell_size=[2,2])

    # 渲染 radio map  
    scene.render(camera=cam, radio_map=rm,
                 num_samples=512, rm_show_color_bar=True,
                 rm_vmax=-40, rm_vmin=-150)

    # 發射器往 -x 方向移動 10 公尺
    scene.get("tx").position -= displacement_vec

    # 移動前 5 台車（car_1 ~ car_5）往 -x 方向移動 10 公尺
    for j in range(1,6):
        scene.get(f"car_{j}").position -= displacement_vec

    # 移動後 3 台車（car_6 ~ car_8）往 +x 方向移動 10 公尺
    for j in range(6,9):
        scene.get(f"car_{j}").position += displacement_vec
```
<img width="822" height="590" alt="image" src="https://github.com/user-attachments/assets/ea947b24-5f69-4d01-aa87-79d9a6c67ee9" />  
<img width="822" height="590" alt="image" src="https://github.com/user-attachments/assets/08dab5d3-2db8-4336-9428-831eefd48829" />  
<img width="822" height="590" alt="image" src="https://github.com/user-attachments/assets/d1bf95ed-9d51-4044-8c31-ba0e042d030b" />

***

## Time Evolution of Channels Via Doppler Shift
* 這一段會示範如何透過設定物體的速度向量 `velocity` ，來模擬短時間尺度下的通道演變，特別是  Doppler Shift 對通道的影響  
* 為什麼要用 Velocity 模擬短時間通道變化
    * 前一節提到，我們可以透過 `.position += ...` 移動物體，但這種「逐格移動」方式不適合模擬高速移動或連續時間下的通道變化
    * 因此改用 velocity 向量 來設定「移動速度」，由 Sionna 自動計算 Doppler shift

### 1. 載入簡單反射場景：這個場景只包含一個簡單的金屬反射面
```python
scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
```

### 2. 設定反射器的速度向量
```python
print("Velocity vector: ", scene.get("reflector").velocity.numpy()[:,0])

scene.get("reflector").velocity = [0, 0, -20]
print("Velocity vector after update: ", scene.get("reflector").velocity.numpy()[:,0])
```

```python
Velocity vector:  [0. 0. 0.]
Velocity vector after update:  [  0.   0. -20.]
```
* 初始速度為 [0, 0, 0]：靜止不動
* 更新後速度為 [0, 0, -20]：沿著 z 軸方向以 -20 m/s 向下移動

### 3. 建立發射器與接收器
```python
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = scene.tx_array
scene.add(Transmitter("tx", [-25, 0.1, 50]))
scene.add(Receiver("rx",    [ 25, 0.1, 50]))
```
* 發射器位在 `[-25, 0.1, 50]`
* 接收器位在 `[ 25, 0.1, 50]`

### 4. 使用 PathSolver 計算傳播路徑
```python
p_solver = PathSolver()
paths = p_solver(scene=scene, max_depth=1)
```
* `max_depth=1`： 只計算一次反射路徑（即 TX → reflector → RX）
* 這時候 Sionna 會根據反射板的位置、移動速度，自動導入 Doppler Shift 到每條路徑中，並將結果存在 `paths`

### 5. 渲染結果：畫出 TX、RX、反射器與光路
```python

if no_preview:
    cam = Camera(position=[0, 100, 50], look_at=[0,0,30])
    scene.render(camera=cam, paths=paths);
else:
    scene.preview(paths=paths)
```

### 6. 印出每一條路徑的互動類型與 Doppler
```python
print("Path interaction (0=LoS, 1=Specular reflection): ", paths.interactions.numpy())
print("Doppler shifts (Hz): ", paths.doppler.numpy())
```

```python
Path interaction (0=LoS, 1=Specular reflection):  [[[[0 1]]]]
Doppler shifts (Hz):  [[[   0.      -417.68835]]]
```
* `[[[[0 1]]]]`: 表示這條路徑有兩段 interaction
    * `0`： 第一段是 Line-of-Sight (LoS)，代表 TX → RX
    * `1`: 第二段是 Specular Reflection，代表TX → Reflector → RX (經過一次鏡面反射)
* `[[[   0.      -417.68835]]]`:  對應每段 interaction 的 Doppler shift
    * `0` : LoS 且 TX/RX 是靜止的，所以都卜勒為 0 Hz
    * `-417.68835`: 經過反射器的路線，由於反射器 向下以 -20 m/s 速度移動，產生了都卜勒偏移

***
### Example: Delay-Doppler Spectrum
這一段的目的是: 透過前面計算的 Doppler shift 資訊，建立一個時間變化的通道響應`h(t, τ)`，並進一步將它轉換為一個 延遲-都卜勒頻譜（Delay-Doppler Spectrum）

### 1. 設定 OFDM System parameter
```python
num_ofdm_symbols = 1024
num_subcarriers = 1024
subcarrier_spacing = 30e3

ofdm_symbol_duration = 1/subcarrier_spacing
delay_resolution = ofdm_symbol_duration/num_subcarriers
doppler_resolution = subcarrier_spacing/num_ofdm_symbols

print("Delay   resolution (ns): ", int(delay_resolution/1e-9))
print("Doppler resolution (Hz): ", int(doppler_resolution))
```

```python
Delay   resolution (ns):  32
Doppler resolution (Hz):  29
```
* 什麼是 `Delay resolution`:
    * 因為 IFFT 後的 delay bin 數有限，當兩條路徑的延遲差異落在同一 bin 時系統無法區分它們，此時看起來像是同一條 path
    * 因此我們計算的結果是每一個 delay bin 可分辨的 時間差最小單位為 32 ns，代表你只能分辨兩個 path 的傳播距離差是否超過 9.6 公尺
* 什麼是 `Doppler resolution`:
    * 原因基本上和上面相同
    * 因此我們計算的結果是每個 Doppler bin 可分辨的頻移為 29 Hz，表示你可以分辨物體速度是否差異超過約 1.24 公尺/秒

### 2. 設定 TX 速度、使用 `PathSolver` 計算場景路徑、 使用 `路徑名稱.cfr` 計算該 path 對應的 channel frequency response

```python
tx_velocity = [30, 0, 0]
scene.get("tx").velocity = tx_velocity

# Recompute the paths
paths = p_solver(scene=scene, max_depth=1, refraction=False)

# Compute channel frequency response with time evolution
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)

h = paths.cfr(frequencies=frequencies,
              sampling_frequency=1/ofdm_symbol_duration,
              num_time_steps=num_ofdm_symbols,
              normalize_delays=False, normalize=True, out_type="numpy")
```

### 3. 開始計算 Delay-Doppler spectrum

```python
h = np.squeeze(h)
h = np.fft.fftshift(h, axes=1)
h_delay = np.fft.ifft(h, axis=1, norm="ortho")
h_delay_doppler = np.fft.fft(h_delay, axis=0, norm="ortho")
h_delay_doppler = np.fft.fftshift(h_delay_doppler, axes=0)
```
* `np.squeeze(h)`: 去除維度=1的維度
    * h 經過 `path.cfr` 之後的 shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
    * 進一步透過 `np.squeeze(h)`之後的 shape: [num_time_steps, num_subcarriers]
* `np.fft.fftshift(h, axes=1)`: 針對 `axes=1`，把分量 0 移到中心
* `np.fft.ifft(h, axis=1, norm="ortho")`: 針對`h`的 `axes=1`(subcarrier)做 IFFT ，得到 delay domain 的 CIR
* `np.fft.fft(h_delay, axis=0, norm="ortho")`: 針對`h_delay`的 `axes=0`(不同 OFDM symbol)做 FFT，得到 Doppler domain資訊
* 輸出為一張 Delay-Doppler map，其中：
    * x 軸是 delay
    * y 軸是 Doppler shift

### 4. 建立 Delay-Doppler 頻譜的座標網格
```python
doppler_bins = np.arange(-num_ofdm_symbols/2*doppler_resolution,
                          num_ofdm_symbols/2*doppler_resolution,
                          doppler_resolution)

delay_bins = np.arange(0,
                       num_subcarriers*delay_resolution,
                       delay_resolution) / 1e-9

x, y = np.meshgrid(delay_bins, doppler_bins)
```
* `doppler_bins`: 每隔一個 `doppler_resolution`創建一個 doppler bin ，一共有 1024 個bin
* `delay_bins`: 每隔一個 `delay_resolution/1e-9`創建一個 delay bin ，一共有 1024 個bin
* `np.meshgrid(delay_bins, doppler_bins)`: 建立 2D 網格座標，橫軸為 delay/縱軸為 doppler

```python
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(111, projection='3d')
```
* `plt.figure(figsize=(8, 10))`: 建立一個新的 figure，設定圖形的尺寸為 8 x 10 英吋
* `fig.add_subplot(111, projection='3d')`: 新增一個 3D 的子圖到這個 figure，`111` 的意思是：在 1x1 的網格中放第 1 個圖

```python
offset = 20
x_start = int(num_subcarriers/2)-offset
x_end = int(num_subcarriers/2)+offset
y_start = 0
y_end = offset

x_grid = x[x_start:x_end,y_start:y_end]
y_grid = y[x_start:x_end,y_start:y_end]
z_grid = np.abs(h_delay_doppler[x_start:x_end,y_start:y_end])
```

* x 軸： 從中心的 num_subcarriers 往正負各取 20 個 bin
* y 軸： 從 Doppler 的 0Hz 開始往上取 20 個 bin

### 5. 繪製 3D Delay-Doppler 圖形
```python
surf = ax.plot_surface(x_grid,
                       y_grid,
                       z_grid,
                       cmap='viridis', edgecolor='none')
```
* 將 `x_grid`, `y_grid`, `z_grid` 三個 2D 陣列當成 `x, y, z` 三維座標，畫出一個「顏色漸層的 3D 表面圖」
*  `cmap='viridis'`: 指定顏色漸層的樣式，`viridis` 是一種從紫到黃的 colormap，適合表示強度等級
*  `edgecolor='none'`: 不畫格線邊框，讓圖更平滑乾淨
```python
ax.set_xlabel('Delay (ns)')               # 設定 X 軸名稱 為 Delay (ns)
ax.set_ylabel('Doppler (Hz)')             # 設定 Y 軸名稱 為 Doppler (Hz)
ax.set_zlabel('Magnitude');               # 設定 Z 軸名稱 為 Magnitude
ax.zaxis.labelpad=2                       # 調整 Z 軸標籤與 Z 軸之間的距離 
ax.view_init(elev=53, azim=-32)           # 設定整個 3D 圖的 視角 (從哪個角度去看這張3D圖)
ax.set_title("Delay-Doppler Spectrum");   # 設定整張圖的標題為 Delay-Doppler Spectrum 
```

<img width="665" height="684" alt="image" src="https://github.com/user-attachments/assets/3f7ad33d-66eb-4204-ae4d-d3473e36820e" />

### 6. 驗證圖上的 Peak 是否與 ray tracing 模擬出的 delay & Doppler shift 一致
先看 Delay-Doppler Spectrum:  
* 第一個 peak： 160 ns、350 Hz Doppler
* 第二個 peak： 370 ns、-260 Hz Doppler
這代表圖上有兩個 dominant 路徑：一條 LoS，一條反射

#### 使用 `paths` 驗證理論值（Ground Truth）
```python
print("Delay - LoS Path (ns) :", paths.tau[0,0,0]/1e-9)
print("Doppler - LoS Path (Hz) :", paths.doppler[0,0,0])

print("Delay - Reflected Path (ns) :", paths.tau[0,0,1].numpy()/1e-9)
print("Doppler - Reflected Path (Hz) :", paths.doppler[0,0,1])
```

```python
Delay - LoS Path (ns) : 166.782
Doppler - LoS Path (Hz) : 350.242
Delay - Reflected Path (ns) : 372.93597188181593
Doppler - Reflected Path (Hz) : -261.056
```
兩者非常吻合，因此可確認： Delay-Doppler Spectrum 成功反映出路徑的實際物理參數

***

## Comparison of Doppler- vs Position-based Time Evolution
待編輯
