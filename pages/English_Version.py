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


ROOT = Path(__file__).parent.parent
st.set_page_config(
    page_title="CCCS–Cost | COPD cost prediction",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


FEATURE_EN = {
    "年龄": "Age", "BMI": "BMI", "性别": "Sex", "医保类型": "Insurance type",
    "住院次数": "Admission number", "吸烟史": "Smoking history", "是否化疗": "Prior chemotherapy",
    "历史总门诊次数": "Prior outpatient visits", "历史总住院次数": "Prior hospitalizations",
    "舒张压": "Diastolic blood pressure", "呼吸频率": "Respiratory rate", "体温": "Temperature",
    "脉搏": "Pulse", "入院科室": "Admission department", "入院途径": "Admission route",
    "是否做过微生物培养": "Microbiological culture", "是否做过物理检查": "Physical examination",
    "是否做过病理检查": "Pathology examination", "住院天数": "Length of stay", "CCI": "CCI", "CCCS": "CCCS",
    "TimeTrend_month": "Time trend"
}
RISK_EN = {"高风险": "High risk", "中风险": "Intermediate risk", "低风险": "Low risk"}
SEX_EN = {"男": "Male", "女": "Female"}
INSURANCE_EN = {"公费/医疗救助": "Public funding / medical assistance", "其他支付方式": "Other payment",
    "其他社会保险": "Other social insurance", "城乡居民医保/新农合": "Resident insurance / NCMS",
    "城镇职工医保": "Urban employee insurance", "自费": "Self-pay"}
ROUTE_EN = {"其他": "Other", "外院转入": "Transfer from another hospital", "急诊": "Emergency", "门诊": "Outpatient referral"}
DEPT_EN = {"其他内科系统": "Other medical department", "呼吸专科(PCCM)": "Pulmonary and Critical Care Medicine",
    "外科系统": "Surgical department", "心血管内科": "Cardiology", "急诊": "Emergency department",
    "未知": "Unknown", "老年与综合医学": "Geriatrics / general medicine", "重症监护(ICU/CCU)": "Intensive care (ICU/CCU)"}
SERVICE_EN = {"chemotherapy": "Prior chemotherapy", "microbiology": "Microbiological culture during this admission",
    "physical_exam": "Physical examination during this admission", "pathology": "Pathology examination during this admission"}
GROUP_EN = {"循环系统疾病": "Cardiovascular", "呼吸系统疾病": "Respiratory", "内分泌代谢疾病": "Endocrine/metabolic",
    "泌尿生殖系统疾病": "Genitourinary", "消化系统疾病": "Digestive", "血液系统疾病": "Hematologic",
    "肌肉骨骼疾病": "Musculoskeletal", "肿瘤": "Malignancy", "风湿免疫疾病": "Rheumatic/immune", "神经精神疾病": "Neuropsychiatric"}
DISEASE_EN = {
    "hypertension": "Hypertension", "coronary": "Coronary heart disease", "heart_failure": "Heart failure",
    "arrhythmia": "Arrhythmia", "valvular": "Valvular heart disease", "cerebrovascular": "Cerebrovascular disease",
    "peripheral_artery": "Peripheral arterial disease / atherosclerosis", "vte": "Venous thromboembolism",
    "pneumonia": "Pneumonia / pulmonary infection", "asthma": "Asthma", "bronchiectasis": "Bronchiectasis",
    "ild": "Interstitial lung disease / pneumoconiosis", "pleural": "Pleural disease", "sleep_apnea": "Sleep apnea",
    "upper_airway": "Upper-airway disease", "diabetes": "Diabetes", "dyslipidemia": "Dyslipidemia",
    "thyroid": "Thyroid disease", "malnutrition": "Malnutrition / hypoproteinemia", "ckd": "Chronic kidney disease / renal failure",
    "bph": "Benign prostatic hyperplasia", "liver": "Chronic liver disease / cirrhosis", "fatty_liver": "Fatty liver",
    "gastritis_ulcer": "Gastritis / peptic ulcer", "gallbladder": "Cholelithiasis / gallbladder disease",
    "anemia": "Anemia", "osteoporosis": "Osteoporosis", "malignancy": "Malignancy",
    "rheumatic": "Rheumatic / connective-tissue disease", "neuropsych": "Mental / neurocognitive disorder"
}

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


@st.cache_resource(show_spinner="Loading the frozen HONAM-M3 model...")
def warm_model():
    return get_ensemble()


def result_html(result: dict, selected_count: int) -> str:
    risk_class = "high" if result["positive"] else ("medium" if result["probability"] >= 20 else "low")
    threshold_text = (
        f"At or above the prespecified {CLASSIFICATION_THRESHOLD:.2f} classification threshold"
        if result["positive"]
        else f"Below the prespecified {CLASSIFICATION_THRESHOLD:.2f} classification threshold"
    )
    rotation = max(0, min(180, result["probability"] * 1.8)) - 90
    impacts = []
    for item in result["impacts"]:
        width = min(100, abs(float(item["value"])) * 45)
        text = "Higher" if item["direction"] == "up" else "Lower"
        impacts.append(
            f'<div class="impact-row"><span>{escape(FEATURE_EN.get(item["label"], item["label"]))}</span><div><i class="{item["direction"]}" '
            f'style="width:{width:.1f}%"></i></div><b class="{item["direction"]}">{text}</b></div>'
        )
    return f"""
    <div class="result-head"><div><div class="eyebrow">MODEL RESULT</div><h2>High-cost probability</h2></div></div>
    <div class="result-grid">
      <article class="risk-panel">
        <div class="panel-label">Probability of high-cost hospitalization</div>
        <div class="gauge-wrap"><div class="gauge"><div class="gauge-mask"></div>
          <div class="needle" style="transform:rotate({rotation:.1f}deg)"></div><div class="needle-pin"></div></div>
          <div class="gauge-value">{result['probability']:.1f}%</div><div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>
        <div class="risk-pill {risk_class}">{RISK_EN.get(result['risk'], result['risk'])}</div><p class="risk-note">{threshold_text}</p>
      </article>
      <div class="summary-panels">
        <div class="metric-row">
          <article class="metric-card"><span>CCI</span><strong>{result['cci']:.0f}</strong><small>Quan ICD-10</small></article>
          <article class="metric-card"><span>CCCS</span><strong>{result['cccs']:.2f}</strong><small>Frozen ordinary PCA score</small></article>
          <article class="metric-card"><span>Comorbidity nodes</span><strong>{selected_count}</strong><small>30-node network</small></article>
          <article class="metric-card"><span>High-cost cutoff</span><strong>¥{HIGH_COST_AMOUNT:,.0f}</strong><small>Development-set P80</small></article>
        </div>
        <article class="explain-panel"><div class="panel-label">HONAM main-effect contributions (excluding pairwise interactions)</div>{''.join(impacts)}</article>
      </div>
    </div>
    <div class="result-disclaimer"><strong>Research use only</strong><span>This result is generated by a single-center temporally validated model and is not a clinical diagnosis or payment rule. The 0.50 cutoff is a prespecified evaluation threshold, not an optimal decision threshold for every setting.</span></div>
    """


initialize_state()

with st.sidebar:
    st.markdown(
        '<a class="language-link" href="/" target="_self">↩️&nbsp;&nbsp;中文版 / Chinese version</a>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">ASSESSMENT MODE</div><div class="sidebar-title">Select an input mode</div>', unsafe_allow_html=True)
    st.radio(
        "Input mode",
        ["single", "batch"],
        format_func=lambda x: "01  Individual prediction" if x == "single" else "02  Batch prediction",
        key="page_mode",
        label_visibility="collapsed",
        on_change=invalidate_result,
    )
    st.markdown(
        f"""
        <div class="sidebar-rule"></div><div class="eyebrow">MODEL STATUS</div>
        <div class="status-line"><span>Inference pipeline</span><b class="status-live">Frozen</b></div>
        <div class="status-line"><span>High-cost cutoff</span><b>¥{HIGH_COST_AMOUNT:,.2f}</b></div>
        <div class="status-line"><span>Classification threshold</span><b>{CLASSIFICATION_THRESHOLD:.2f}</b></div>
        <div class="status-line"><span>Model version</span><b>HONAM-M3</b></div>
        <div class="notice">ⓘ Five-seed HONAM ensemble. CCCS uses 30 comorbidity nodes, 33 stable positive edges, maximum-edge scaling, Leiden modules, and ordinary PCA loadings.</div>
        """,
        unsafe_allow_html=True,
    )

with st.container(key="topbar"):
    brand_col, pill_col, clear_col = st.columns([8, 1.8, 0.85], vertical_alignment="center")
    with brand_col:
        st.markdown('<div class="top-brand"><div class="brand-mark">C</div><div><div class="brand-name">CCCS–Cost</div><div class="brand-sub">COPD Medical Cost Prediction Tool</div></div></div>', unsafe_allow_html=True)
    with pill_col:
        st.markdown('<span class="demo-pill">Research version</span>', unsafe_allow_html=True)
    with clear_col:
        st.button("Reset", on_click=clear_all, use_container_width=True)

if st.session_state.page_mode == "batch":
    st.markdown('<div class="workspace-head"><div class="eyebrow">BATCH INFERENCE</div><h1>Batch prediction</h1><p>Upload a CSV or Excel feature matrix for multiple patients or hospital episodes. The model calculates one high-cost probability per row.</p></div>', unsafe_allow_html=True)
    required = list(warm_model().feature_names)
    with st.container(border=True):
        st.markdown("**Twenty-one fields required by the frozen model**")
        st.code("、".join(required), language=None, wrap_lines=True)
        st.caption("Recognized aliases in the source matrix are mapped automatically to the frozen model fields.")
        upload = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
        if upload is not None:
            try:
                if upload.name.lower().endswith(".csv"):
                    source = pd.read_csv(upload)
                else:
                    sheets = pd.read_excel(upload, sheet_name=None)
                    sheet_name = st.selectbox("Select a worksheet", list(sheets), index=list(sheets).index("特征_标签") if "特征_标签" in sheets else 0)
                    source = sheets[sheet_name]
                st.success(f"Loaded {len(source):,} rows and {len(source.columns)} columns")
                st.dataframe(source.head(10), use_container_width=True)
                if st.button("Run HONAM-M3 batch prediction", type="primary"):
                    with st.spinner("Calculating five-seed ensemble probabilities..."):
                        output = predict_batch(source)
                    st.session_state["batch_output"] = output
            except Exception as exc:
                st.error(f"File reading or field validation failed: {exc}")
        if "batch_output" in st.session_state:
            output = st.session_state["batch_output"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted records", f"{len(output):,}")
            c2.metric("Mean predicted probability", f"{output['HONAM_M3_高费用概率'].mean():.1%}")
            c3.metric("Positive at 0.50", f"{output['HONAM_M3_0.5分类'].mean():.1%}")
            st.dataframe(output.head(100), use_container_width=True)
            data = output.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download batch predictions (CSV)", data, "HONAM_M3_batch_predictions.csv", "text/csv")
else:
    st.markdown('<div class="workspace-head"><div class="eyebrow">FORMAL HONAM-M3</div><h1>Medical Cost Prediction</h1><p>Enter the frozen M3 predictors and select comorbidity nodes. The system calculates CCCS and the HONAM-M3 probability.</p></div>', unsafe_allow_html=True)

    with st.container(key="basic_card"):
        heading(1, "Patient and hospitalization information", "Fields are aligned with the frozen research model")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.number_input("Age", 18, 110, key="age", on_change=invalidate_result)
        with c2: st.number_input("BMI", 8.0, 60.0, step=0.1, key="bmi", on_change=invalidate_result)
        with c3: st.selectbox("Sex", ["男", "女"], format_func=lambda x: SEX_EN[x], key="sex", on_change=invalidate_result)
        with c4: st.selectbox("Insurance type", list(INSURANCE_EN), format_func=lambda x: INSURANCE_EN[x], key="insurance", on_change=invalidate_result)
        with c5: st.selectbox("Admission route", list(ROUTE_EN), format_func=lambda x: ROUTE_EN[x], key="admission_route", on_change=invalidate_result)
        with c6: st.selectbox("Admission department", list(DEPT_EN), format_func=lambda x: DEPT_EN[x], key="department", on_change=invalidate_result)
        d1, d2, d3 = st.columns(3)
        with d1: st.date_input("Admission date", key="admission_date", on_change=invalidate_result)
        with d2: st.number_input("Length of stay", 1, 365, key="stay_days", on_change=invalidate_result)
        with d3: st.number_input("Current admission number", 1, 99, key="admission_count", on_change=invalidate_result)
        time_trend = 12 * (st.session_state.admission_date.year - 2011) + (st.session_state.admission_date.month - 10)
        if time_trend < 0 or time_trend > TRAINING_TIME_MAX:
            st.warning("This date is outside the study period (October 2011 to September 2020); interpret temporal extrapolation cautiously.")

    with st.container(key="dynamic_card"):
        heading(2, "Vital signs, prior utilization, and binary predictors", "Verify unknown values against the source record; complete entry is currently required")
        v1, v2, v3, v4 = st.columns(4)
        with v1: st.number_input("Diastolic blood pressure (mmHg)", 20.0, 180.0, key="diastolic_bp", on_change=invalidate_result)
        with v2: st.number_input("Respiratory rate (/min)", 5.0, 80.0, key="respiratory_rate", on_change=invalidate_result)
        with v3: st.number_input("Temperature (°C)", 30.0, 45.0, step=0.1, key="temperature", on_change=invalidate_result)
        with v4: st.number_input("Pulse (/min)", 20.0, 240.0, key="pulse", on_change=invalidate_result)
        h1, h2, h3, h4 = st.columns(4)
        with h1: st.number_input("Prior outpatient visits", 0, 999, key="history_outpatient", on_change=invalidate_result)
        with h2: st.number_input("Prior hospitalizations", 0, 999, key="history_inpatient", on_change=invalidate_result)
        with h3: st.selectbox("Smoking history", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", key="smoking", on_change=invalidate_result)
        with h4: st.number_input("CCI", 0, 30, key="cci", on_change=invalidate_result, help="Can be calculated from ICD-10 codes using the Quan algorithm or entered manually after verification.")
        service_cols = st.columns(4)
        for index, service in enumerate(SERVICES):
            with service_cols[index]:
                st.selectbox(SERVICE_EN.get(service.id, service.label), [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=f"service_{service.id}", on_change=invalidate_result)

    with st.container(key="diagnosis_card"):
        heading(3, "Comorbidity diagnoses and CCCS", "Thirty frozen disease nodes; verify ICD-derived selections manually")
        st.text_area("ICD-10 codes", key="icd_text", placeholder="e.g., I50.9, E11.9, N18.3", help="Separate codes with commas, spaces, semicolons, or line breaks.")
        if st.button("Recognize nodes and calculate CCI", on_click=recognize_codes):
            pass
        if st.session_state.cci_categories:
            st.caption("Recognized CCI categories: " + ", ".join(st.session_state.cci_categories))
        st.text_input("Search by disease name, code, or system", key="search", label_visibility="collapsed")
        query = st.session_state.search.strip().lower()
        filtered = [d for d in DISEASES if not query or query in f"{d.name}{d.code}{d.group}{DISEASE_EN.get(d.id, '')}".lower()]
        columns = st.columns(3)
        for index, disease in enumerate(filtered):
            with columns[index % 3]:
                st.checkbox(f"**{DISEASE_EN.get(disease.id, disease.name)}**  \n{GROUP_EN.get(disease.group, disease.group)}  `{disease.code}`", key=f"disease_{disease.id}", on_change=invalidate_result)
        st.markdown(f'<div class="selection-count">Selected {len(selected_disease_ids())} / 30 frozen nodes</div>', unsafe_allow_html=True)

    with st.container(key="calculate_bar"):
        left, right = st.columns([4, 1.25], vertical_alignment="center")
        with left:
            st.markdown('<div class="calc-copy"><span>Patient inputs are not written to the project data files</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("Calculate HONAM-M3 probability →", type="primary", use_container_width=True):
                st.session_state.show_result = True

    if st.session_state.show_result:
        binary = {service.feature_name: st.session_state[f"service_{service.id}"] for service in SERVICES}
        features = {
            "Age": st.session_state.age,
            "BMI": st.session_state.bmi,
            "Sex": st.session_state.sex,
            "Insurance type": st.session_state.insurance,
            "住院次数": st.session_state.admission_count,
            "Smoking history": st.session_state.smoking,
            "是否化疗": binary["是否化疗"],
            "Prior outpatient visits": st.session_state.history_outpatient,
            "Prior hospitalizations": st.session_state.history_inpatient,
            "舒张压": st.session_state.diastolic_bp,
            "呼吸频率": st.session_state.respiratory_rate,
            "体温": st.session_state.temperature,
            "脉搏": st.session_state.pulse,
            "Admission department": st.session_state.department,
            "Admission route": st.session_state.admission_route,
            "是否做过微生物培养": binary["是否做过微生物培养"],
            "是否做过物理检查": binary["是否做过物理检查"],
            "是否做过病理检查": binary["是否做过病理检查"],
            "Length of stay": st.session_state.stay_days,
            "CCI": st.session_state.cci,
        }
        with st.spinner("Running the five-seed HONAM ensemble..."):
            result = calculate_risk(selected_disease_ids=selected_disease_ids(), **features)
        with st.container(key="results_card"):
            st.markdown(result_html(result, len(selected_disease_ids())), unsafe_allow_html=True)
            with st.expander("View model and network details"):
                network_names = ["Node count", "Weighted edge strength", "Weighted density", "Weighted global efficiency", "Cross-module connection ratio", "Bridge-core exposure"]
                st.dataframe(pd.DataFrame({"Network feature": network_names, "Value": result["network_features"]}), hide_index=True, use_container_width=True)
                st.write("Five seed-specific probabilities (%):", ", ".join(f"{x:.1f}" for x in result["seed_probabilities"]))
                st.caption(f"Model version: {MODEL_VERSION}")
            selected_names = ", ".join(DISEASE_EN.get(d.id, d.name) for d in DISEASES if d.id in selected_disease_ids()) or "None"
            text = "\n".join([
                "CCCS–Cost | HONAM-M3 research result",
                f"High-cost definition: 2020 price-standardized total cost > CNY {HIGH_COST_AMOUNT:,.2f}",
                f"Predicted probability: {result['probability']:.1f}%",
                f"Classification at 0.50: {'Positive' if result['positive'] else 'Negative'}",
                f"CCI：{result['cci']:.0f}", f"CCCS：{result['cccs']:.2f}",
                f"Comorbidity nodes: {selected_names}", f"Model: {MODEL_VERSION}",
                "Note: For research support only; not for clinical diagnosis, payment, or resource-allocation decisions.",
            ])
            st.download_button("Download this result (text)", text, "HONAM_M3_result.txt", "text/plain")

st.markdown('<div class="page-footer"><b>CCCS–Cost Calculator</b><span>Frozen research model · Temporal validation</span></div>', unsafe_allow_html=True)
