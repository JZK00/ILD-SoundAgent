# VELCRO Respiratory Sound Agent

VELCRO是一个面向科研验证的呼吸音推理与模型证据定位工具。系统使用OPERA-CT音频表征和患者级多位点模型，输出患者级纤维化相关概率、位点贡献以及时间段级模型证据。

> 本项目不是临床诊断工具。时间定位结果表示音频遮挡对模型预测的影响，不等同于经医生确认的病理异常声段。

## 功能

- 读取WAV、M4A和MP3音频；
- 提取并缓存768维OPERA-CT特征；
- 使用3个初始化模型进行患者级集成推理；
- 计算位点级有符号遮挡贡献；
- 定位对患者级预测有影响的时间段；
- 绘制位点贡献、Log-Mel频谱和时间证据曲线。

当前不包含交互式界面或自动报告生成。

## 项目结构

```text
resp_agent/
  audio_io.py          Audio decoding and validation
  opera_encoder.py     OPERA-CT feature extraction and cache
  patient_model.py     Deployment checkpoint loading
  inference.py         Patient-level inference and site attribution
  localization.py      Time-interval evidence localization
  visualization.py     Static evidence visualization
  schemas.py           Output schemas and terminology constraints
configs/
  localization.yaml    Localization parameters
scripts/
  export_task1_visualizations.py
tests/
```

## 环境

安装Python依赖：

```bash
pip install -r requirements.txt
```

另外需要：

- 可用的OPERA代码与OPERA-CT checkpoint；
- `OPERA_ROOT`环境变量指向OPERA项目根目录；
- 部署模型文件位于`checkpoints/deployment/`；
- 读取M4A或MP3时，系统`PATH`中需要有FFmpeg。

Windows PowerShell示例：

```powershell
$env:OPERA_ROOT = "D:\OPERA"
```

## Python接口

```python
from resp_agent.inference import predict_patient
from resp_agent.localization import localize_suspected_abnormal_segments
from resp_agent.patient_model import load_deployment_model

deployment = load_deployment_model("site_self_attention")
site_audio = {
    "Site 1": "path/to/site1.wav",
    "Site 4": "path/to/site4.wav",
}

prediction = predict_patient(deployment, site_audio)
localization = localize_suspected_abnormal_segments(deployment, site_audio)
```

## 导出可视化

六个位点文件遵循当前VELCRO数据命名方式时，可运行：

```powershell
python scripts/export_task1_visualizations.py `
  --audio-dir "path/to/six-site-recordings" `
  --patient-id 001 `
  --theme paper
```

默认输出到`task1_visualization_outputs/patient001/`。该目录已被`.gitignore`排除。

## 定位方法

定位模块采用2秒窗口和0.5秒步长。每个窗口执行3次确定性低电平噪声遮挡，并分别保留3个患者模型初始化的概率变化。窗口证据根据幅度、prominence、峰值与中位数比值、方向一致性和跨初始化稳定性归类为：

- `focal`：存在满足预设判据的局部证据；
- `diffuse`：证据分布较广，没有明确局部峰；
- `minimal`：整体影响低于工程阈值；
- `unstable`：方向或峰值位置稳定性不足。

所有判据均配置在`configs/localization.yaml`。这些参数尚未经过带时间段临床标注的数据验证，不应解释为医学诊断标准。

## 模型与性能声明

部署checkpoint由全部55名患者重训产生，并使用3个初始化组成集成模型。该checkpoint本身没有独立held-out评估，因此不能用其训练队列输出作为泛化性能证据。论文性能结果应引用预先定义的nested cross-validation OOF统计。

示例患者001属于部署模型训练队列。相关输出只用于验证推理与可视化链路，不属于独立测试结果。

## 数据与模型文件

仓库默认不跟踪：

- 原始音频；
- OPERA特征缓存；
- 部署checkpoint；
- 患者级导出图片；
- 本地运行日志。

公开数据、模型或患者衍生图片前，应确认数据授权、去标识化要求及第三方模型许可。

## 测试

```bash
python tests/test_localization.py
python tests/test_visualization.py
```

部分测试需要OPERA checkpoint、部署模型和本地音频数据。纯单元测试与真实数据集成测试后续可进一步拆分为独立测试标记。
