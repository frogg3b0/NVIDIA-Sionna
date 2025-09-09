# 使用 Blender 創建 xml 檔
## Sionna 3D 地圖介紹
在Sionna中使用的地圖需為 `Mitsuba` 格式，以下流程為教學如何使用 Blender + OpenStreetMap 來輸出成 `Mitsuba` 格式的檔案
* Blender: 3D圖形製作軟體
* OpenStreetMap: 開源的地圖資訊(包含建築物的資訊)
* Mitsuba: 適合用於 Sionna ray tracing 的環境

以下教學皆參考 [github/s87315teve](https://github.com/s87315teve/sionna_introduction/tree/main/3dmap_tutorial) 的文章，並根據自身情況作些許微調

***
## 環境架設
### 1. 安裝 [Blender](https://www.blender.org/)
建議安裝 Blender4.2.9，可至 [blender-release](https://download.blender.org/release/) 下載，最新的可能會有相容性問題

***

### 2. 安裝Blender-OSM插件
* 進入 [Blosm for Blender: Google 3D cities, OpenStreetMap, terrain](https://prochitecture.gumroad.com/l/blender-osm)
    * 並在圖上金額的位置輸入想要donate的金額 (可以輸入0)
    * 並點擊 "I want this!" 進入下一步

<img width="1092" height="363" alt="image" src="https://github.com/user-attachments/assets/4dccd01f-0eb2-4533-a063-c2192a218972" />

* 下載後會是一個 `blosm.zip` 的檔案，不用解壓縮
* 打開Blender -> Edit -> Preferences -> Add-ons -> 點選右上像V的圖案，點選 "Install from Disk" 後選擇 `blosm.zip`
<img width="1178" height="884" alt="image" src="https://github.com/user-attachments/assets/d70d0656-df0f-4855-b022-882998378145" />

* 安裝完Blosm後，在Preferences紅色箭頭的地方輸入未來OSM地圖儲存的路徑
<img width="1728" height="1275" alt="image" src="https://github.com/user-attachments/assets/151326fc-971c-4763-b2c2-6f8db978a044" />

***

### 3. 安裝 [Blender-Mitsuba](https://github.com/mitsuba-renderer/mitsuba-blender)插件
* 在 [Mitsuba-blender-release](https://github.com/mitsuba-renderer/mitsuba-blender/releases) 下載v0.4.0的 `mitsuba-blender.zip`，同樣不用解壓縮
* 按照和 Step 2 相同的步驟安裝
* 安裝完後可能會缺少部分套件，請依照下圖的位置點選 "Install dependencies"
<img width="1504" height="1249" alt="image" src="https://github.com/user-attachments/assets/6b3d1cf8-dbe9-4e6d-ac27-a203916e5ed8" />

#### 錯誤回報及解決方式
某些情況，可能會看到 "Failed to load Mitsuba package"，而相應的解決方式如下:  
1. 找到你存放 blender 的資料夾
2. 對 blender 點右鍵 -> 內容 ，並在路徑後面添加“ --python-use-system-env” (雙引號不用加)
<img width="404" height="501" alt="image" src="https://github.com/user-attachments/assets/2b94ae37-4e93-4fac-9452-a958c882c49d" />

* 之後在 Blender 內，到File -> Export就能看到Blender已經支援輸出Mitsuba格式了
<img width="527" height="636" alt="image" src="https://github.com/user-attachments/assets/bc951941-f433-4c0d-9b4f-0583e233a14d" />

***

## 如何使用 Blender 創建 `.xml`檔
此部分的教學包含: 地圖匯入 Blender -> 輸出成 Mitsuba  
更詳細的內容可參考這份影片教學: [Sionna RT: Scene Creation with Blender using OpenStreetMap](https://www.youtube.com/watch?v=7xHLDxUaQ7c)
### 1. 匯入 "OpenStreetMap 地圖" 至 Blender
* 在 Blender 內，打開 Blosm 插件，並點選 select
<img width="891" height="487" alt="image" src="https://github.com/user-attachments/assets/0d0819f6-4c40-451a-8099-15f4cd132cb7" />

* 之後會進到地圖頁面，把地圖移動到自己想要的範圍 (e.g. 交大工程四館)
* 然後按下Show selection rectangle
* 確認範圍後按下Copy，電腦就會複製指定範圍的GPS座標
<img width="1318" height="973" alt="image" src="https://github.com/user-attachments/assets/fea2ede8-ea82-4146-9054-4deb31f6526c" />


* 回到Blender，按照下圖1~3的順序操作
    1. 按下paste貼上GPS資訊
    2. 確認資料源是 "OpenStreetMap"
    3. 按下 "import" 方能導入地圖資訊
<img width="1919" height="1053" alt="image" src="https://github.com/user-attachments/assets/2c49081d-f996-493b-8c84-8a93d5d6d73d" />


* 導入地圖後如下所示 (此份教學以交大光復校區為例子)
<img width="1919" height="1053" alt="image" src="https://github.com/user-attachments/assets/6b6a88c2-fd44-473a-a605-351dd94f0873" />


### 2. 使用教學
1. 匯入OpenStreetMap地圖
    * 因為 OpenStreetMap 的物件可能不包含地面資訊，所以需要自己生成地面物件
    * 可以根據[這份影片](https://www.youtube.com/watch?v=7xHLDxUaQ7c)參考其他設定，像是根據 [NVIDIA Sionna官網](https://nvlabs.github.io/sionna/rt/api/radio_materials.html) 設定建築物材質等

1-1. 在場景內加入地平面
<img width="744" height="422" alt="image" src="https://github.com/user-attachments/assets/ea485f81-9747-48ff-9244-ce393f35cc64" />  
* 此時加入的地板大小為2x2的正方形，後續還須對它進行縮放  

1-2. 縮放地板大小，以覆蓋整張地圖
<img width="1983" height="972" alt="image" src="https://github.com/user-attachments/assets/e7430b5f-62b5-4f04-80c5-acde1b4320bf" />
* 在右側部分調整(x,y,z)縮放大小

    * 設定完後點選 File -> Export -> Mitsuba -> Mitsuba Export
2. 看見資料夾有 <檔名>.xml 和 meshes 就代表輸出完成
    * 之後使用需要讓xml檔和meshes資料夾在同一個路徑，路徑如下
   
folder  
├── <檔名>.xml  
└── meshes  
　 ├── object1.ply  
　 ├── object2.ply  
　 ├── object3.ply  
　 └── ⋮   
  
 ## 如何匯入到 Sionna 做使用
現在我們已經有一份 `.xml`檔(地圖檔)，以及數份的`.ply`檔案，接下來會示範如何將此份地圖匯入到 Sionna 做使用

```python
scene_path="my_scene/nycu_campus/nycu_campus.xml"
scene = load_scene(
    filename="/home/wicoms5090/sionna_env/NYCU ED/NYCU ED.xml",  
    merge_shapes=True                           
)

print(f"size of the scene: {scene.size}")
scene.preview()
```



