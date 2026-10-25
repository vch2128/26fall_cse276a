# HW2 新檔案說明

## 1. 檔案要放哪裡（temp/ 下的路徑 = repo 裡的路徑）

| temp 裡的檔案 | 搬到 | 用途 |
|---|---|---|
| `robot_control/robot_control/velocity_control.py` | `robot_control/robot_control/`（取代 HW1 版，HW1 在 git 歷史裡） | 從 HW1 改的：一樣用 `drive()` / `turn()` 和轉向表，多了 `look()` + EKF，變成閉迴路 |
| `robot_control/robot_control/landmark_detector.py` | `robot_control/robot_control/` | YOLO 偵測物件 → 每個 landmark 的距離 + 方位角 → `/landmarks` |
| `robot_control/config/hw2_waypoints.txt` | `robot_control/config/` | (0,0,0)→(1,0,0)→(1,2,π)→(0,0,0) |
| `robot_control/config/landmarks.txt` | `robot_control/config/` | landmark 地圖：類別、世界座標、高度 |
| `robot_control/launch/hw2_launch.py` | `robot_control/launch/` | 一次開 camera + Foxglove + motor_control + detector |
| `tools/plot_trajectory.py` | `tools/` | 筆電上跑：畫軌跡圖、算總距離、每個 waypoint 誤差 |
| `report_hw2.md` | 交作業的資料夾 | 報告草稿，TODO 要填實測數字 |

舊檔案要自己改（我沒動）：
- `robot_control/setup.py`：`console_scripts` 加 `'landmark_detector = robot_control.landmark_detector:main',`；
  launch 那行改成 `('share/' + package_name + '/launch', glob('launch/*.py')),`
  （velocity_control 名字沒變，不用加）

## 2. 跟 HW1 差在哪

```
camera ──image──▶ landmark_detector ──/landmarks (class, r, b)──▶ velocity_control ──[L,R]──▶ motor_control ──▶ 車
                                                                    EKF: predict = HW1 運動學模型
                                                                         update  = 停下來時看到的 landmark
```

HW1：轉 → 開 → 轉，位置全靠運動學模型猜。
HW2 每個 waypoint：
1. `look()`：停 1 秒，用看到的 landmark 修正 (x, y, heading)；離目標 < 5 cm 就算到
2. 轉向目標（一樣查 CCW/CW 表）。目標很近又在後面 → 不轉 180°，直接倒車
3. 往前開，**一次最多 0.5 m**，然後回到 1. 再看一次
4. 轉到 θ → `look()` → 角度還差 > 4° 就再轉一點

- **只用停下來時拍的影像**（`look()` 期間），所以沒有模糊和 YOLO 延遲的問題。
- **EKF**：predict 就是 HW1 的 `x += d cos θ, y += d sin θ, θ += 轉的角度`（分散到整段動作，方便每 0.5 s 記一次位置）；
  update 用 landmark 距離 + 方位角。對不上的偵測（誤判）會被 Mahalanobis gate 丟掉；連續被丟太多次代表是估計錯了，就放大不確定度重新定位。
- 看不到 landmark → 只有 predict，跟 HW1 一樣。

## 3. 要做的事（照順序）

1. **安裝 YOLO**（車上一次）：`pip install ultralytics`（Ubuntu 24.04 可能要加 `--break-system-packages` 或用 venv）。第一次跑會自動下載 `yolov8n.pt`。
2. **選 landmark 物件**：要是 COCO 類別，每種類別放一個最簡單。建議夠大的：`chair`、`teddy bear`、`backpack`、`potted plant`、`bottle`、`tv`。改 `landmarks.txt` 的類別名稱。
3. **擺 landmark**，量世界座標填進 `landmarks.txt`（原點 = 起點，x = 起始朝向，y = 左邊）。預設擺法讓每一段路都看得到至少一個：
   - (2, 0)：第 1 段往 +x、最後回到 (0,0,0) 時
   - (1, 3)：第 2 段往 +y 時
   - (-1, 2)：在 (1,2,π) 時
   - (-0.5, -1)：第 3 段從 (1,2) 開回 (0,0) 時
   場地不夠可以縮近，但離 waypoint 最好至少 0.7 m（太近會被畫面切到）。
4. **量 `CAM_X`**（`landmark_detector.py` 最上面）：相機到兩輪中間往前的距離。
5. **校正物體高度**：物體放相機正前方 1.0 m，跑 `ros2 run robot_control landmark_detector`，看印出的 `r=`。
   新高度 = 舊高度 × 1.0 / r。每個物體做一次。再左右移，確認 `b` 正負對（左邊 = 正）。
   ⚠️ `robot_vision/config/camera_parameter.yaml` 的 p1 = 1.21 看起來不正常（一般 |p1| < 0.01）。如果物體在畫面邊邊時 `b` 明顯錯，加 `-p use_distortion:=false` 比較，或重新校正相機。
6. **檢查 HW1 的校正**：`SECONDS_PER_METER` 和 `CCW_TABLE` / `CW_TABLE` 沿用 HW1。
   因為 `look()` 之後的修正大多是**小角度轉彎（5–20°）**，表裡最小只有 45°，小角度是外插的。
   建議用 `calibrate` 量 10° 和 20° 要幾秒，加進兩個表，例如 `CCW_TABLE = [(10, ?), (20, ?), (45, 0.68), ...]`。
7. **測試**：先 `test_line.txt`，再 `hw2_waypoints.txt`（預設）。
   ```bash
   ros2 launch robot_control hw2_launch.py              # 終端機 1
   ros2 run robot_control velocity_control              # 終端機 2，車擺在 (0,0,0)
   ros2 run robot_control velocity_control test_line.txt
   ```
   Foxglove 連 `ws://<robot-ip>:8765` 看 `/landmarks/debug/compressed`（綠框 = 有用，紅框 = 被切到不用）。
   `ros2 topic hz /landmarks` 最好 ≥ 3 Hz，太慢加 `-p imgsz:=416` 或 `320`。
8. **正式跑 + 錄影**：地上每個 waypoint 貼小標記。每到一個 waypoint 會停 2 秒，**量車子真實位置跟角度**記到 `truth.csv`（每行 `x,y,theta_deg`）。
9. **出報告數字**：
   ```bash
   scp 'ubuntu@<robot-ip>:~/ros2_ws/logs/hw2_*' logs/
   python3 tools/plot_trajectory.py logs/hw2_<時間> --truth truth.csv --unit 0.5 --out trajectory.png
   ```
   印出總移動距離（平移 + 旋轉）、每個 waypoint 的估計值與誤差，存軌跡圖（每 0.5 s 一點）。
10. 填 `report_hw2.md` 的 TODO、影片上傳 YouTube（unlisted）、壓縮 `hw2_<名字>` 交。

## 4. 可以調的參數（`velocity_control.py` 最上面）

| 參數 | 說明 |
|---|---|
| `SEGMENT` | 一次最多開多遠就停下來看。小 → 準但慢、比較不順 |
| `LOOK_SECONDS` | 每次停多久收偵測。長 → 平均更多張，比較準 |
| `POS_TOL`, `ANG_TOL_DEG`, `MAX_TRIES` | 多準才算到、最多修幾次 |
| `SIG_R_REL`, `SIG_B` | 量測雜訊：距離（框高度）比方位角不準很多 |
| `SIG_TRANS`, `SIG_ROT` | 運動學模型多不可信 |

`landmark_detector` 的 `conf`、`imgsz`、`model` 是 ROS 參數（`-p imgsz:=416`）。
