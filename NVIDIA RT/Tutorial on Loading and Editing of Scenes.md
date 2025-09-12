# Tutorial on Loading and Editing of Scenes
本筆記會說明如何使用 Sionna RT 模組載入與編輯場景，你將會學到以下幾點：  
* 使用 `load_scene()` 函數載入場景，並學會選擇是否要合併場景中的物件
* 學會如何新增與刪除場景中的物件
* 學會如何在場景中平移、旋轉與縮放物件

***

## Imports
```python
%matplotlib inline
import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

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
                      PathSolver, ITURadioMaterial, SceneObject
```

***

## Loading Scenes and Merging Objects
使用 Sionna RT 載入場景會透過 `load_scene()` 函數完成  
* 這個函數預設會合併具有相似屬性（例如相同 radio material）的物件
* 這麼做是因為減少場景中物件的數量可以顯著加快 ray tracing 的速度

#### 關閉合併選項（merge_shapes=False）
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=False) # Disable merging of objects
```

#### 列印場景中所有物件與其材質
```python
for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}'
```

```python
building_1     itu_glass
building_6     itu_wood
building_5     itu_glass
building_4     itu_marble
building_3     itu_marble
building_2     itu_brick
floor          itu_concrete
```

#### 重新載入場景並啟用物件合併
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True) # Enable merging of objects (default)

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')
```

```python
floor          itu_concrete
building_2     itu_brick
no-name-1      itu_marble
building_6     itu_wood
no-name-2      itu_glass
```
我們可以看到，共享相同無線電材質的物體已被合併  

#### 排除特定物體不合併
```python
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True, # Enable merging of objects
                   merge_shapes_exclude_regex=r'building_[0-2]$') 

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')
```

```python
building_1     itu_glass
building_2     itu_brick
floor          itu_concrete
no-name-3      itu_marble
building_5     itu_glass
building_6     itu_wood
```
* `merge_shapes_exclude_regex=r'building_[0-2]$'`: `building_0`, `building_1`, `building_2` 不進行合併

***

## Editing Scenes

### 1. 載入 Etoile 場景並指定攝影機位置
```python
scene = load_scene(sionna.rt.scene.etoile) # Objects are merged by default

cam = Camera(position=[-360,145,400], look_at=[-115,33,1.5])
if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/b85739ae-3d02-434f-a154-4b1391773dd0" />

### 2. 在場景中添加物件
可參考這篇 [Scence Object 網站](https://nvlabs.github.io/sionna/rt/api/scene_object.html)說明  

### 2-1. 根據 ITU 規範建立金屬車輛的材質
```python
num_cars = 10
car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.8, 0.1, 0.1))
```

### 2-2. 使用 `SceneObject` 建立多個車輛物件並加入場景
```python
cars = [SceneObject(fname=sionna.rt.scene.low_poly_car, # Simple mesh of a car
                    name=f"car-{i}",
                    radio_material=car_material)
        for i in range(num_cars)]
scene.edit(add=cars)

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();
```

* `SceneObject` 說明

```python
SceneObject(
    fname=模型檔案路徑 (str),           # 必須是 .obj 或內建模型名稱
    name=此物件在場景中的名稱 (str),     # 可用於編輯、刪除、辨識
    radio_material=ITU 材質物件        # 對應電磁反射特性
)
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/6ae4fb68-43d9-4057-bd88-88ba574e133c" />  

### 2-3. 將先前加入場景的 10 輛紅色金屬車 排成一個圓圈
```python

c = mi.Point3f(-127, 37, 1.5)           # 圓心座標
r = 100                                 # 圓半徑  
thetas = dr.linspace(mi.Float, 0., dr.two_pi, num_cars, endpoint=False)  # 圓周上的角度
# Cars positions
cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.)*r    # 將 10 輛車平均分布在[0, 2π) 範圍

# Orientations
# Compute points the car "look-at" to set their orientation
d = dr.normalize(cars_positions - c)
# Tangent vector to the circle at the car position
look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
look_at_points = cars_positions + look_at_dirs

# Set the cars positions and orientations
for i in range(num_cars):
    cars[i].position = mi.Point3f(cars_positions.x[i], cars_positions.y[i], cars_positions.z[i])
    cars[i].look_at(mi.Point3f(look_at_points.x[i], look_at_points.y[i], look_at_points.z[i]))

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/1942c43e-6bea-413a-b7c5-8fbfdc4f15d5" />  

### 補充說明:
* 縮放物件: `cars[0].scaling = 2.0`
* 刪除物件: `scene.edit(remove=[cars[-1]])`

***

## Path Computation with the Edited Scene
這段程式碼展示了如何在 編輯後的 Sionna RT 場景中建立與計算傳播路徑（Paths），包括加入發射器、接收器、設定天線陣列，並呼叫 PathSolver 計算多徑路徑  

### 1. 加入發射器 Transmitter
```python
scene.remove("tx")
scene.add(Transmitter("tx", position=[-36.59, -65.02, 25.], display_radius=2))
```

### 2. 在每台車上放置接收器 Receiver
```python
for i in range(num_cars):
    scene.remove(f"rx-{i}")
    scene.add(Receiver(f"rx-{i}", position=[cars_positions.x[i],
                                            cars_positions.y[i],
                                            cars_positions.z[i] + 3],
                      display_radius=2))
```

### 3. 設定 TX/RX 天線陣列
```python
scene.tx_array = PlanarArray(num_cols=1,
                             num_rows=1,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array
```

### 4. 使用 `PathSolver()` 計算傳播路徑並可視化出來
```python
p_solver = PathSolver()
paths = p_solver(scene, max_depth=5)

if no_preview:
    scene.render(camera=cam, paths=paths);
else:
    scene.preview(paths=paths);
```
<img width="766" height="590" alt="image" src="https://github.com/user-attachments/assets/b1029f93-616d-484b-b919-babe0681deb2" />


