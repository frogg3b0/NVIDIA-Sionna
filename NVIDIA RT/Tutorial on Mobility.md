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

### 3. 選取單一車輛並檢查其位置與方向

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
