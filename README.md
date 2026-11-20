# 低重力漂浮目标视觉抓取

Isaac Sim 6.1 里的零重力抓取仿真。固定基座的 UR5e 携带 Robotiq 2F-85，追逐一颗自由漂浮的纹理小球。物理场景重力为 `0 m/s²`。

`--auto-grasp` 的正式闭环只使用相机观测：

1. 两台外部相机构成双目，用绿色目标分割得到像素，再把像素反投影成射线并三角化；
2. 六维常速度卡尔曼滤波估计位置和速度，短时丢检时按当前速度外推；
3. 腕部单目相机提供视线，与滤波位置融合；
4. Lula 逆解把夹爪中心送到预测拦截点，经过靠近、速度同步、闭合和撤回；
5. 目标靠近图像边缘时，用归一化图像误差把拦截点拉回视场中心。

仿真真值只写入 `[oracle-metrics]` 和 `outputs/episode_metrics.jsonl`，不进入闭合决策。`--oracle-baseline` 是读取真值的上限对照。

`space_grasp_sim.py` 里还保留了 `VisualGraspController`。它在预抓取之后用腕部归一化图像误差做图像平面对准，当前 `main()` 没有实例化它。默认自动抓取走 `StereoKalmanGraspController`。

## 场景

- UR5e 固定在世界原点，夹爪为 Robotiq 2F-85。`tool0` 到指垫中点约 `0.129 m`。
- PhysX 重力大小为 0，求解器为 TGS，开启 CCD。小球线阻尼和角阻尼为 0。
- 小球默认半径 `0.045 m`，基准质量 `0.18 kg`，质量按半径三次方缩放。碰撞体是完整球体；表面深色和橙色斑点只参与渲染。
- 小球从机械臂下方的发射口出发，默认以约 `0.035 m/s` 穿过抓取区。速度设为 0 时，在可达工作区内生成静止目标。
- 小球恢复系数 `0.78`，面板 `0.72`。面板材质的恢复系数组合模式为 `max`，低速弹跳阈值 `0.005 m/s`。
- 越过工作区边界默认 `0.12 m` 后，目标失效并关闭碰撞。手动演示模式下，失效后默认等待 1 秒再生成下一颗。自动抓取和真值对照不会自动换目标。
- 外部双目为 640×480，焦距 20 mm，分别放在 `(1.20, -0.42, 0.88)` 和 `(1.20, 0.42, 0.88)`，看向抓取区中心。
- 腕部相机焦距 8 mm。每个渲染帧根据实测关节和 Lula 正运动学更新位姿，避免 GPU/Fabric 下 USD 子节点不同步。
- 物理默认 120 Hz，界面和 RGB 默认 15 Hz，追逐控制 30 Hz。默认关闭 DLSS，避免双目边缘不一致。

## 控制状态

正式通路的状态顺序：

`SEARCH_STEREO` → `COARSE_APPROACH` → `WRIST_TRACK_APPROACH` → `VELOCITY_SYNCHRONIZATION` → `CLOSE` → `RETRACT_AND_TRANSPORT` → `VERIFY` → `DONE`

粗接近停在球前方约 `0.12 m`。细接近把指垫送到预测拦截点。相对速度和位置误差同时满足后闭合。接触确认或小球进入指间后撤回约 `0.06 m` 并检查是否夹住。失败会张开夹爪并重试，超过 `configs/controller.yaml` 里的次数或超时后中止。按 `B` 可以整场景重置。

日志前缀：`[vision]`、`[estimate]`、`[control]`、`[stereo-kf]`、`[oracle-metrics]`。真值对照使用 `[cheats]`。

## 目录

```text
space_grasp_sim.py          场景、感知、卡尔曼滤波和抓取状态机
space_grasp/config.py       合并 configs/*.yaml
space_grasp/control/        视场安全边距和回中偏移
space_grasp/estimation/     观测和估计消息；估计来源不能是 oracle
space_grasp/evaluation/     真值误差和 episode 记录
space_grasp/logging/        限频文本日志
configs/                    相机、估计器、控制器和评测阈值
scripts/                    Windows 上的 Isaac Sim 安装和启动脚本
tests/                      不启动 Isaac 的单元测试
```

## 环境

仿真界面使用 Windows 上的 Isaac Sim 6.1。NVIDIA 6.1 工作站支持矩阵不包含 WSL，因此不要把 Isaac Sim 装进当前 WSL。下面的安装和启动脚本是 PowerShell，因为它们调用的是 Windows 原生 Python 3.12 环境。

本仓库开发时使用的环境：

- Windows 11，Python 3.12 虚拟环境，默认路径 `E:\isaacsim\venv-6.1`
- `isaacsim[all,extscache]==6.1.0.0`
- `torch==2.11.0`（CUDA 13.0 轮子）
- 场景资产来自 Isaac Sim 资源根目录，首次构建场景需要能访问该资源

UR5e 与 Robotiq 模型、相机和物理都由 Isaac Sim 提供，不随本仓库分发。

## 安装

在 PowerShell 中执行。安装脚本默认把环境和 pip 缓存放在 `E:\isaacsim`。首次安装前需要自行阅读并接受 NVIDIA Omniverse EULA；确认后再加 `-AcceptEula`。

```powershell
cd E:\MscapeTech\space_grasp_sim
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\install_isaacsim_windows.ps1 -CompatibilityOnly
E:\isaacsim\venv-6.1\Scripts\isaacsim.exe isaacsim.exp.compatibility_check
.\scripts\install_isaacsim_windows.ps1
```

脚本内部的核心安装命令是：

```powershell
E:\isaacsim\venv-6.1\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
E:\isaacsim\venv-6.1\Scripts\python.exe -m pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
```

## 运行

打开界面并一直运行到关闭窗口：

```powershell
cd E:\MscapeTech\space_grasp_sim
.\scripts\run_windows.ps1
```

视觉闭环抓取。示例把球改小并给一个较低的穿越速度：

```powershell
.\scripts\run_windows.ps1 -AutoGrasp -BallRadius 0.012 -BallSpeed 0.05
```

真值上限对照。正式实验不要加这个开关：

```powershell
.\scripts\run_windows.ps1 -AutoGrasp -OracleBaseline -BallRadius 0.012 -BallSpeed 0.05
```

静止小球、指定种子，并在共同视野内采样可达位置：

```powershell
.\scripts\run_windows.ps1 -BallRadius 0.03 -BallSpeed 0 -Seed 7
```

运行固定时长、无窗口保存三路 RGB，或只生成一个目标：

```powershell
.\scripts\run_windows.ps1 -Duration 20 -Seed 42
.\scripts\run_windows.ps1 -Headless -Duration 10 -CaptureDir E:\isaacsim-data\episode-0001
.\scripts\run_windows.ps1 -SinglePass -DespawnMargin 0.15 -RespawnDelay 2
```

每次运行会导出 `outputs/space_grasp_scene.usda`。打开 RGB 保存后，三台相机的每个渲染帧都会写盘。

阈值放在 `configs/`：

| 文件 | 作用 |
| --- | --- |
| `cameras.yaml` | 分辨率、感知频率、双目残差上限 |
| `estimator.yaml` | 常速度模型、测量噪声、丢失后的外推时间 |
| `controller.yaml` | 粗细接近阈值、闭合相对速度、视场边距、重试次数 |
| `evaluation.yaml` | 真值日志间隔和 `episode_metrics.jsonl` 文件名 |

`controller.ibvs_gain` 目前没有接到默认状态机。视场回中使用 `visibility.recenter_gain`。

## 键盘

先点击 Isaac Sim 主视口，再使用键盘。自动抓取和真值对照会关闭键盘遥操作。

| 键 | 作用 |
| --- | --- |
| `W` / `S` | 末端世界坐标 `+X` / `-X` |
| `A` / `D` | 末端世界坐标 `+Y` / `-Y` |
| `Z` / `X` | 末端世界坐标 `+Z` / `-Z` |
| `I` / `K` | 夹爪局部 Roll |
| `J` / `L` | 夹爪局部 Pitch |
| `U` / `O` | 夹爪局部 Yaw |
| `V` / `C` | 打开 / 闭合夹爪 |
| `T` | 机械臂回到初始关节 |
| `B` | 重置机械臂、夹爪、小球和抓取状态 |
| `Space` | 停止持续移动 |

默认平移 `0.12 m/s`，旋转 `45°/s`。逆解失败时该次命令被丢弃，终端打印 IK 警告。

## 测试

单元测试不启动 Isaac Sim，只检查三角化、估计消息和 `space_grasp/control` 不调用真值接口：

```powershell
cd E:\MscapeTech\space_grasp_sim
python -m unittest discover -s tests -t .
```

## 仿真假设

这里的 0 g 用来验证自由漂浮目标的视觉抓取，不是完整轨道动力学。基座固定，因此不模拟机械臂动作对航天器的动量耦合，也不包含姿态控制、轨道摄动和真空热环境。物理后端是 PhysX。

## 参考

- [Isaac Sim 6.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)
- [Isaac Sim 6.1 Python 安装](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html)
- [Isaac Sim 6.1 工作站安装](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_workstation.html)
- [Manipulator Robot Assets（UR5e 与 Robotiq）](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/assets/usd_assets_robots_manipulator.html)
- [Isaac Sim 6.1 Camera Sensors](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_camera.html)
