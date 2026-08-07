# CCCS-Cost Calculator（Streamlit 版）

这是根据现有 CCCS-Cost Calculator 页面迁移的 Streamlit 原型，保留两种评估模式、30 个疾病节点、ICD-10 批量识别、演示风险计算、结果分层和主要影响因素展示。

> 重要：当前计算逻辑仅复现原网页中的交互演示算法，未接入论文冻结模型，不可用于临床诊疗或费用管理决策。

## 本地运行

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装依赖并启动：

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器一般会自动打开 `http://localhost:8501`。

## 部署到 Streamlit Community Cloud

1. 将本文件夹上传至 GitHub 仓库。
2. 在 Streamlit Community Cloud 新建应用。
3. 选择仓库、分支和入口文件 `app.py`。
4. 部署后获得 `*.streamlit.app` 网址。

## 文件说明

- `app.py`：Streamlit 页面和交互。
- `model.py`：疾病数据、ICD-10 规则和演示计算逻辑。
- `style.css`：按原网页视觉定制的界面样式。
- `requirements.txt`：部署依赖。

