# Scenes
`sionna.rt.load_scene(filename=None, merge_shapes=True, merge_shapes_exclude_regex=None)`
* `filename=在Ubuntu上的路徑`: 路徑內需要有.xml檔以及其他的meshs檔
* `merge_shapes=True`: 如果設定為 True，則「相同無線電材質的物件」將合併
* `merge_shapes_exclude_regex=None`: 指定某些物件不參與合併，前提是 `merge_shapes=True` 才會用到這個參數

***
### 補充說明: 如何從 Windows 上，把.xml檔案匯入到ubuntu後，再進一步上傳到jupyter

#### 1. 準備你的場景資料夾（在 Windows 上）
假設你有以下結構，確認所有用到的 mesh 都跟 xml 放在一起:  

```python
C:\Users\frogg\Desktop\XML path\NYCU ED\
├── scene.xml
├── building.obj
├── ground.obj
└── ...（其他 mesh 檔案）
```

#### 2. 將資料夾上傳到 Ubuntu Server
` `scp -r "C`:\Users\frogg\Desktop\XML path\NYCU ED" wicoms5090@<你的Ubuntu IP>:/home/wicoms5090/sionna_scenes/`
* `scp -r`：遞迴上傳整個資料夾
* `"..."`：引號避免空格出錯
* `wicoms5090@<你的Ubuntu IP>`：請改成你 Ubuntu server 的帳號與 IP
* `/home/wicoms5090/sionna_scenes/`：Ubuntu 上的目標目錄

#### 3. 在 Python 中載入場景
* 假設你上傳到的是 `/home/wicoms5090/sionna_scenes/NYCU ED/scene.xml`
* 那麼你在 Sionna 中這樣寫即可：

```python
scene = load_scene(
    filename="/home/wicoms5090/sionna_scenes/NYCU ED/scene.xml",
    merge_shapes=True
)
```
***
