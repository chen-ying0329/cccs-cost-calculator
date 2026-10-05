# CCCS–Cost（HONAM-M3冻结模型接入版）
- 单次COPD住院费用评估及CSV/XLSX批量预测；
- 中文主页和English version英文入口。
- Assessment of single COPD hospitalization costs and batch prediction via CSV/XLSX;
- Chinese homepage and English version entry.
## 本地运行

```powershell
pip install -r requirements.txt
streamlit run app.py
```
模型采用2011—2017年资料开发，并在2018—2020年新患者队列中进行单中心时间验证。超出研究中心或资料时段属于外推。该工具仅供研究复核，不用于临床诊疗、医保支付或资源配置决策。

The model was developed using data from 2011 to 2017, and single-center temporal validation was performed in a new patient cohort from 2018 to 2020. Application beyond the study center or data period constitutes extrapolation. This tool is for research review only and shall not be used for clinical diagnosis and treatment, medical insurance payment, or resource allocation decision-making.
