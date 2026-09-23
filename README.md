# Healthcare 6.0 · ClinRisk

基于 Flask、PyTorch 和 scikit-learn 的健康评估 Web 项目，提供骨折影像分析、心脏病风险预测、账户管理和记录查询。界面支持中英文切换。

## 功能

- **骨折影像分析**：上传影像，使用多任务 Faster R-CNN 检测并绘制边界框，结合 ResNet50 进行骨折分类，支持 10 种骨折类别。
- **心脏病风险预测**：根据年龄、血压、胆固醇等输入，使用逻辑回归或感知机模型进行预测。
- **账户与记录**：注册、登录、退出，以及评估记录查询和条件筛选。
- **统计展示**：展示记录中的年龄、性别和骨折类别分布。

本项目为学习与演示用途，预测结果不能替代临床诊断。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| Web 后端 | Flask、Flask-CORS、Jinja2 |
| 影像模型 | PyTorch 2.6.0、torchvision 0.21.0 |
| 心脏预测 | scikit-learn 1.6.1 |
| 数据处理与图表 | pandas、NumPy、Matplotlib、Seaborn |
| 数据库 | SQLite |
| 前端 | Bootstrap、HTML、CSS、JavaScript |

已在 macOS Apple Silicon、Python 3.12 环境下启动验证。本地启动脚本使用 `fcntl`，适用于 macOS / Linux；Windows 可使用 WSL。当前代码在 CUDA 可用时使用 CUDA，否则使用 CPU，未启用 Apple MPS。

## 快速开始

### 1. 获取源码并安装依赖

```bash
git clone https://github.com/andyyyyaa/healthcare_6.0.git
cd healthcare_6.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. 下载模型

模型作为 [models-v1 Release 附件](https://github.com/andyyyyaa/healthcare_6.0/releases/tag/models-v1) 提供，不存入 Git 历史。下载以下四个文件并放在项目根目录：

| 文件 | 用途 |
| --- | --- |
| `best_multitask_frcnn_verbose3.pt` | 骨折检测与全局分类 |
| `model.pth` | ResNet50 骨折类型分类 |
| `heart_model_lr.pkl` | 心脏病逻辑回归模型 |
| `heart_model_p.pkl` | 心脏病感知机模型 |

也可使用已登录且拥有仓库访问权限的 GitHub CLI：

```bash
gh release download models-v1 --repo andyyyyaa/healthcare_6.0 \
  --pattern '*.pt' --pattern '*.pth' --pattern '*.pkl' --pattern 'SHA256SUMS' --dir .
shasum -a 256 -c SHA256SUMS
```

仅加载可信来源的模型文件。保持 scikit-learn 1.6.1 与已保存心脏模型的版本一致。首次启动时，torchvision 还会下载官方 Faster R-CNN 基础权重，并缓存到 `.local/torch`，因此需要网络连接。

### 3. 初始化数据库

```bash
python init_database.py
```

该命令只创建缺失的数据表，可以重复执行，不删除已有记录。仓库不包含原有账户或诊断数据，新环境需要自行注册账户。

旧脚本 `create_database.py` 会删除并重建数据表；已有数据时不要运行它。

### 4. 启动服务

```bash
python start_local.py
```

等待模型加载，浏览器打开 **http://127.0.0.1:5050**。按 `Control+C` 停止服务。

macOS 用户完成上述安装、模型下载和数据库初始化后，也可以双击 `启动本地服务.command` 启动，双击 `停止本地服务.command` 停止。

修改端口：

```bash
PORT=5051 python start_local.py
```

后台运行并保存日志：

```bash
mkdir -p .local
nohup .venv/bin/python -u start_local.py > .local/server.log 2>&1 &
```

## 项目结构

```text
healthcare_6.0/
├── app.py                    # Flask 路由与模型推理
├── start_local.py            # 本地启动、会话密钥和缓存配置
├── start_server.py           # 服务器启动入口
├── init_database.py          # 无损创建缺失的数据表
├── create_database.py        # 旧版数据库重建脚本（会清空数据）
├── train_model.py            # 心脏预测模型训练脚本
├── infer-fracture.py         # 独立骨折推理脚本
├── templates/                # Jinja2 页面模板
├── static/                   # 样式、脚本、公共图片与前端依赖
├── requirements.txt          # Python 依赖版本
├── LOCAL_SETUP.md            # 本地运行说明
└── DEPLOYMENT.md             # 服务器部署参考
```

## 数据与配置

- `user_db.sqlite3` 在本地保存账户和评估记录，不提交到 GitHub。
- `.local/secret_key` 由本地启动脚本自动生成并持续复用；也可通过 `SECRET_KEY` 环境变量设置。
- 虚拟环境、日志、缓存、生成的标注图片和统计图均由 `.gitignore` 排除。
- 本地测试影像与 `static/heart.csv` 未上传；启动和使用现有模型不需要训练数据。运行 `train_model.py` 时，需自行提供具有 `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target` 列的 CSV。
- 当前账户代码使用明文密码，并有输出登录信息的调试日志；此版本适合本地演示。对外部署前需完善密码哈希、日志脱敏、访问控制和 HTTPS 配置。[部署说明](DEPLOYMENT.md) 是配置参考，不代表已完成生产环境加固。

## 常见问题

**提示找不到模型文件**：确认四个 Release 附件位于 `app.py` 同级目录，且文件名未更改。

**提示 `no such table`**：在虚拟环境中运行 `python init_database.py`。

**端口被占用**：macOS 的 5000 端口可能由系统使用；本地默认使用 5050，也可通过 `PORT` 更改。

**首次启动较慢或基础权重下载失败**：首次启动需要下载并加载 torchvision 权重；检查网络后重试，缓存会保存在 `.local/torch`。

**出现 `unexpected keys: ['cls_loss_fn.pos_weight']`**：本地启动时会输出该检查点额外字段提示，目前已验证模型能够加载并启动服务；这不构成模型准确率或临床有效性的验证。
