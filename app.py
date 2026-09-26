from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from model import (
    CLASSIFICATION_THRESHOLD,
    DISEASES,
    HIGH_COST_AMOUNT,
    MODEL_VERSION,
    SERVICES,
    TRAINING_TIME_MAX,
    calculate_cci_from_icd,
    calculate_risk,
    get_ensemble,
    predict_batch,
    recognize_icd_codes,
)


ROOT = Path(__file__).parent
st.set_page_config(
    page_title="CCCS–Cost｜COPD医疗费用预测",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


DEFAULTS = {
    "page_mode": "single",
    "age": 79,
    "bmi": 26.0,
    "insurance": "城乡居民医保/新农合",
    "admission_route": "门诊",
    "department": "呼吸专科(PCCM)",
    "admission_date": date(2020, 9, 1),
    "stay_days": 12,
    "admission_count": 1,
    "history_outpatient": 0,
    "history_inpatient": 1,
    "diastolic_bp": 77.0,
    "respiratory_rate": 20.0,
    "temperature": 36.5,
    "pulse": 82.0,
    "smoking": 1,
    "cci": 2,
    "icd_text": "",
    "search": "",
    "show_result": False,
    "cci_categories": [],
}


def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    for disease in DISEASES:
        st.session_state.setdefault(f"disease_{disease.id}", False)
    for service in SERVICES:
        st.session_state.setdefault(f"service_{service.id}", 0)


def invalidate_result() -> None:
    st.session_state.show_result = False


def clear_all() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, list) else value
    for disease in DISEASES:
        st.session_state[f"disease_{disease.id}"] = False
    for service in SERVICES:
        st.session_state[f"service_{service.id}"] = 0


def selected_disease_ids() -> list[str]:
    return [d.id for d in DISEASES if st.session_state.get(f"disease_{d.id}")]


def recognize_codes() -> None:
    for disease_id in recognize_icd_codes(st.session_state.icd_text):
        st.session_state[f"disease_{disease_id}"] = True
    cci, categories = calculate_cci_from_icd(st.session_state.icd_text)
    st.session_state.cci = cci
    st.session_state.cci_categories = categories
    st.session_state.show_result = False


def heading(step: int, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="card-heading"><span class="step-number">{step}</span>'
        f'<div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="正在加载HONAM-M3模型数据……")
def warm_model():
    return get_ensemble()


def result_html(result: dict, selected_count: int) -> str:
    risk_class = "high" if result["positive"] else ("medium" if result["probability"] >= 20 else "low")
    threshold_text = (
        f"达到预先固定的{CLASSIFICATION_THRESHOLD:.2f}模型分类阈值"
        if result["positive"]
        else f"未达到预先固定的{CLASSIFICATION_THRESHOLD:.2f}模型分类阈值"
    )
    rotation = max(0, min(180, result["probability"] * 1.8)) - 90
    impacts = []
    for item in result["impacts"]:
        width = min(100, abs(float(item["value"])) * 45)
        text = "推高" if item["direction"] == "up" else "降低"
        impacts.append(
            f'<div class="impact-row"><span>{escape(item["label"])}</span><div><i class="{item["direction"]}" '
            f'style="width:{width:.1f}%"></i></div><b class="{item["direction"]}">{text}</b></div>'
        )
    return f"""
    <div class="result-head"><div><div class="eyebrow">模型计算结果</div><h2>高费用概率</h2></div></div>
    <div class="result-grid">
      <article class="risk-panel">
        <div class="panel-label">高费用住院概率</div>
        <div class="gauge-wrap"><div class="gauge"><div class="gauge-mask"></div>
          <div class="needle" style="transform:rotate({rotation:.1f}deg)"></div><div class="needle-pin"></div></div>
          <div class="gauge-value">{result['probability']:.1f}%</div><div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>
        <div class="risk-pill {risk_class}">{result['risk']}</div><p class="risk-note">{threshold_text}</p>
      </article>
      <div class="summary-panels">
        <div class="metric-row">
          <article class="metric-card"><span>CCI</span><strong>{result['cci']:.0f}</strong><small>Quan ICD-10</small></article>
          <article class="metric-card"><span>CCCS</span><strong>{result['cccs']:.2f}</strong><small>冻结SPCA评分</small></article>
          <article class="metric-card"><span>共病节点</span><strong>{selected_count}</strong><small>30节点网络</small></article>
          <article class="metric-card"><span>高费用界值</span><strong>¥{HIGH_COST_AMOUNT:,.0f}</strong><small>开发集P80</small></article>
        </div>
        <article class="explain-panel"><div class="panel-label">HONAM单变量主效应贡献（不含二阶交互）</div>{''.join(impacts)}</article>
      </div>
    </div>
    <div class="result-disclaimer"><strong>研究用途</strong><span>该结果来自内部验证模型，不是临床诊断或费用支付规则。页面的0.50是依据现有数据预先固定的分类评价阈值，不代表任一临床场景的最佳决策阈值。</span></div>
    """


initialize_state()

with st.sidebar:
    st.markdown('<div class="eyebrow">评估方式</div><div class="sidebar-title">选择数据入口</div>', unsafe_allow_html=True)
    st.radio(
        "数据入口",
        ["single", "batch"],
        format_func=lambda x: "01　住院费用评估" if x == "single" else "02　批量数据预测",
        key="page_mode",
        label_visibility="collapsed",
        on_change=invalidate_result,
    )
    st.markdown(
        f"""
        <div class="sidebar-rule"></div><div class="eyebrow">模型状态</div>
        <div class="status-line"><span>计算逻辑</span><b class="status-live">正式冻结</b></div>
        <div class="status-line"><span>结局金额界值</span><b>¥{HIGH_COST_AMOUNT:,.2f}</b></div>
        <div class="status-line"><span>分类阈值</span><b>{CLASSIFICATION_THRESHOLD:.2f}</b></div>
        <div class="status-line"><span>模型版本</span><b>HONAM-M3</b></div>
        <div class="notice">ⓘ　五种子HONAM集成；CCCS使用30共病节点、45条共病稳定边、标准化参数与SparsePCA载荷。</div>
        """,
        unsafe_allow_html=True,
    )

with st.container(key="topbar"):
    brand_col, pill_col, clear_col = st.columns([8, 1.8, 0.85], vertical_alignment="center")
    with brand_col:
        st.markdown('<div class="top-brand"><div class="brand-mark">C</div><div><div class="brand-name">CCCS–Cost</div><div class="brand-sub">COPD医疗费用预测工具</div></div></div>', unsafe_allow_html=True)
    with pill_col:
        st.markdown('<span class="demo-pill">真实数据研究版</span>', unsafe_allow_html=True)
    with clear_col:
        st.button("清空", on_click=clear_all, use_container_width=True)

if st.session_state.page_mode == "batch":
    st.markdown('<div class="workspace-head"><div class="eyebrow">BATCH INFERENCE</div><h1>批量数据预测</h1><p>可直接读取现有“预测变量矩阵.xlsx”的工作表，适用于多名患者或多次住院记录的Excel/CSV，系统逐行计算每条记录的高费用概率，不需要在网页上逐个录入。</p></div>', unsafe_allow_html=True)
    required = list(warm_model().feature_names)
    with st.container(border=True):
        st.markdown("**正式模型所需20个字段**")
        st.code("、".join(required), language=None, wrap_lines=True)
        st.caption("矩阵中的史_吸烟史、史_化疗、查_微生物培养、查_病理检查等会自动映射为字段。")
        upload = st.file_uploader("上传CSV或XLSX", type=["csv", "xlsx"])
        if upload is not None:
            try:
                if upload.name.lower().endswith(".csv"):
                    source = pd.read_csv(upload)
                else:
                    sheets = pd.read_excel(upload, sheet_name=None)
                    sheet_name = st.selectbox("选择工作表", list(sheets), index=list(sheets).index("特征_标签") if "特征_标签" in sheets else 0)
                    source = sheets[sheet_name]
                st.success(f"已读取 {len(source):,} 行、{len(source.columns)} 列")
                st.dataframe(source.head(10), use_container_width=True)
                if st.button("运行HONAM-M3模型批量预测", type="primary"):
                    with st.spinner("正在计算五种子集成概率……"):
                        output = predict_batch(source)
                    st.session_state["batch_output"] = output
            except Exception as exc:
                st.error(f"文件读取或字段检查失败：{exc}")
        if "batch_output" in st.session_state:
            output = st.session_state["batch_output"]
            c1, c2, c3 = st.columns(3)
            c1.metric("预测记录数", f"{len(output):,}")
            c2.metric("平均预测概率", f"{output['HONAM_M3_高费用概率'].mean():.1%}")
            c3.metric("0.50模型阳性", f"{output['HONAM_M3_0.5分类'].mean():.1%}")
            st.dataframe(output.head(100), use_container_width=True)
            data = output.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载批量预测结果CSV", data, "HONAM_M3_batch_predictions.csv", "text/csv")
else:
    st.markdown('<div class="workspace-head"><div class="eyebrow">FORMAL HONAM-M3</div><h1>医疗费用预测</h1><p>录入正式M3变量并选择30个冻结共病节点，系统自动计算CCCS与HONAM概率。</p></div>', unsafe_allow_html=True)

    with st.container(key="basic_card"):
        heading(1, "基本信息与住院时点", "字段定义与正式模型一致")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.number_input("年龄", 18, 110, key="age", on_change=invalidate_result)
        with c2: st.number_input("BMI", 8.0, 60.0, step=0.1, key="bmi", on_change=invalidate_result)
        with c3: st.selectbox("医保类型", ["公费/医疗救助", "其他支付方式", "其他社会保险", "城乡居民医保/新农合", "城镇职工医保", "自费"], key="insurance", on_change=invalidate_result)
        with c4: st.selectbox("入院途径", ["其他", "外院转入", "急诊", "门诊"], key="admission_route", on_change=invalidate_result)
        with c5: st.selectbox("入院科室", ["其他内科系统", "呼吸专科(PCCM)", "外科系统", "心血管内科", "急诊", "未知", "老年与综合医学", "重症监护(ICU/CCU)"], key="department", on_change=invalidate_result)
        d1, d2, d3 = st.columns(3)
        with d1: st.date_input("入院日期", key="admission_date", on_change=invalidate_result)
        with d2: st.number_input("住院天数", 1, 365, key="stay_days", on_change=invalidate_result)
        with d3: st.number_input("本次为第几次住院（住院次数）", 1, 99, key="admission_count", on_change=invalidate_result)
        time_trend = 12 * (st.session_state.admission_date.year - 2011) + (st.session_state.admission_date.month - 10)
        if time_trend < 0 or time_trend > TRAINING_TIME_MAX:
            st.warning(f"该日期对应TimeTrend_month={time_trend}，超出训练数据范围0–{TRAINING_TIME_MAX}（2011-10至2020-09）；属于时间外外推，概率需谨慎解释。")

    with st.container(key="dynamic_card"):
        heading(2, "生命体征、既往利用与二值变量", "未知值建议按原始病历核实；页面当前要求完整录入")
        v1, v2, v3, v4 = st.columns(4)
        with v1: st.number_input("舒张压（mmHg）", 20.0, 180.0, key="diastolic_bp", on_change=invalidate_result)
        with v2: st.number_input("呼吸频率（次/分）", 5.0, 80.0, key="respiratory_rate", on_change=invalidate_result)
        with v3: st.number_input("体温（℃）", 30.0, 45.0, step=0.1, key="temperature", on_change=invalidate_result)
        with v4: st.number_input("脉搏（次/分）", 20.0, 240.0, key="pulse", on_change=invalidate_result)
        h1, h2, h3, h4 = st.columns(4)
        with h1: st.number_input("历史总门诊次数", 0, 999, key="history_outpatient", on_change=invalidate_result)
        with h2: st.number_input("历史总住院次数", 0, 999, key="history_inpatient", on_change=invalidate_result)
        with h3: st.selectbox("吸烟史", [1, 0], format_func=lambda x: "是" if x == 1 else "否", key="smoking", on_change=invalidate_result)
        with h4: st.number_input("CCI", 0, 30, key="cci", on_change=invalidate_result, help="可由下方ICD-10编码按Quan算法自动计算，也可核实后手动修改。")
        service_cols = st.columns(3)
        for index, service in enumerate(SERVICES):
            with service_cols[index]:
                st.selectbox(service.label, [0, 1], format_func=lambda x: "否" if x == 0 else "是", key=f"service_{service.id}", on_change=invalidate_result)

    with st.container(key="diagnosis_card"):
        heading(3, "共病诊断与CCCS", "冻结的30个疾病节点；ICD识别结果需人工核对")
        st.text_area("ICD-10编码", key="icd_text", placeholder="例如：I50.9, E11.9, N18.3", help="用逗号、空格、分号或换行分隔。")
        if st.button("识别节点并计算CCI", on_click=recognize_codes):
            pass
        if st.session_state.cci_categories:
            st.caption("CCI识别类别：" + "、".join(st.session_state.cci_categories))
        st.text_input("搜索疾病名称、编码或系统分类", key="search", label_visibility="collapsed")
        query = st.session_state.search.strip().lower()
        filtered = [d for d in DISEASES if not query or query in f"{d.name}{d.code}{d.group}".lower()]
        columns = st.columns(3)
        for index, disease in enumerate(filtered):
            with columns[index % 3]:
                st.checkbox(f"**{disease.name}**  \n{disease.group}　`{disease.code}`", key=f"disease_{disease.id}", on_change=invalidate_result)
        st.markdown(f'<div class="selection-count">已选 {len(selected_disease_ids())} / 30 个冻结节点</div>', unsafe_allow_html=True)

    with st.container(key="calculate_bar"):
        left, right = st.columns([4, 1.25], vertical_alignment="center")
        with left:
            st.markdown('<div class="calc-copy"><span>不会把患者输入写入项目数据文件</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("计算HONAM-M3模型　→", type="primary", use_container_width=True):
                st.session_state.show_result = True

    if st.session_state.show_result:
        binary = {service.feature_name: st.session_state[f"service_{service.id}"] for service in SERVICES}
        features = {
            "年龄": st.session_state.age,
            "BMI": st.session_state.bmi,
            "医保类型": st.session_state.insurance,
            "住院次数": st.session_state.admission_count,
            "吸烟史": st.session_state.smoking,
            "是否化疗": binary["是否化疗"],
            "历史总门诊次数": st.session_state.history_outpatient,
            "历史总住院次数": st.session_state.history_inpatient,
            "舒张压": st.session_state.diastolic_bp,
            "呼吸频率": st.session_state.respiratory_rate,
            "体温": st.session_state.temperature,
            "脉搏": st.session_state.pulse,
            "入院科室": st.session_state.department,
            "入院途径": st.session_state.admission_route,
            "是否做过微生物培养": binary["是否做过微生物培养"],
            "是否做过病理检查": binary["是否做过病理检查"],
            "TimeTrend_month": time_trend,
            "住院天数": st.session_state.stay_days,
            "CCI": st.session_state.cci,
        }
        with st.spinner("正在运行五种子HONAM集成……"):
            result = calculate_risk(selected_disease_ids=selected_disease_ids(), **features)
        with st.container(key="results_card"):
            st.markdown(result_html(result, len(selected_disease_ids())), unsafe_allow_html=True)
            with st.expander("查看模型与网络明细"):
                network_names = ["节点数", "加权边强度", "加权网络密度", "加权全局效率", "跨模块连接比例", "桥接核心暴露"]
                st.dataframe(pd.DataFrame({"网络特征": network_names, "数值": result["network_features"]}), hide_index=True, use_container_width=True)
                st.write("五个种子概率（%）：", ", ".join(f"{x:.1f}" for x in result["seed_probabilities"]))
                st.caption(f"模型版本：{MODEL_VERSION}")
            selected_names = "、".join(d.name for d in DISEASES if d.id in selected_disease_ids()) or "无"
            text = "\n".join([
                "CCCS–Cost｜HONAM-M3研究结果",
                f"高费用定义：2020年价格标准化总费用 > {HIGH_COST_AMOUNT:,.2f}元",
                f"预测概率：{result['probability']:.1f}%",
                f"0.50模型分类：{'阳性' if result['positive'] else '阴性'}",
                f"CCI：{result['cci']:.0f}", f"CCCS：{result['cccs']:.2f}",
                f"共病节点：{selected_names}", f"模型：{MODEL_VERSION}",
                "说明：仅供辅助研究，不用于临床诊疗、支付或资源配置决策。",
            ])
            st.download_button("下载本次结果文本", text, "HONAM_M3_result.txt", "text/plain")

st.markdown('<div class="page-footer"><b>CCCS–Cost Calculator</b><span>Frozen research model · Internal validation only</span></div>', unsafe_allow_html=True)
