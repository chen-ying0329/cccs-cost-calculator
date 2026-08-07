from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from model import DISEASES, SERVICES, calculate_risk, recognize_icd_codes


ROOT = Path(__file__).parent

st.set_page_config(
    page_title="CCCS-Cost Calculator｜COPD高费用住院风险计算器",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(ROOT / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


DEFAULTS = {
    "mode": "admission",
    "age": 68,
    "sex": "男",
    "bmi": 22.4,
    "insurance": "城镇职工医保",
    "admission_route": "急诊",
    "department": "呼吸与危重症医学科",
    "stay_days": 5,
    "previous_admissions": 1,
    "search": "",
    "icd_text": "",
    "show_icd": False,
    "show_result": False,
}


def initialize_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    for disease in DISEASES:
        st.session_state.setdefault(f"disease_{disease.id}", False)
    for service in SERVICES:
        st.session_state.setdefault(f"service_{service.id}", False)


def invalidate_result() -> None:
    st.session_state.show_result = False


def clear_all() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for disease in DISEASES:
        st.session_state[f"disease_{disease.id}"] = False
    for service in SERVICES:
        st.session_state[f"service_{service.id}"] = False


def selected_disease_ids() -> list[str]:
    return [d.id for d in DISEASES if st.session_state.get(f"disease_{d.id}")]


def selected_service_ids() -> list[str]:
    return [s.id for s in SERVICES if st.session_state.get(f"service_{s.id}")]


def recognize_codes() -> None:
    for disease_id in recognize_icd_codes(st.session_state.icd_text):
        st.session_state[f"disease_{disease_id}"] = True
    st.session_state.show_result = False


def heading(step: int, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="card-heading">
          <span class="step-number">{step}</span>
          <div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_html(result: dict, selected_count: int) -> str:
    risk_class = {"低风险": "low", "中风险": "medium", "高风险": "high"}[result["risk"]]
    threshold_text = "超过30%的演示预警阈值" if result["probability"] >= 30 else "未超过30%的演示预警阈值"
    rotation = max(0, min(180, result["probability"] * 1.8)) - 90
    impacts = []
    for item in result["impacts"]:
        direction = item["direction"]
        label = escape(str(item["label"]))
        width = min(100, abs(float(item["value"])) * 74)
        text = "推高" if direction == "up" else "降低"
        impacts.append(
            f'<div class="impact-row"><span>{label}</span><div><i class="{direction}" '
            f'style="width:{width:.1f}%"></i></div><b class="{direction}">{text}</b></div>'
        )
    impacts_html = "".join(impacts) or '<div class="empty-impact">录入更多信息后可查看个体影响因素</div>'

    return f"""
    <div class="result-head">
      <div><div class="eyebrow">演示计算结果</div><h2>个体风险概览</h2></div>
    </div>
    <div class="result-grid">
      <article class="risk-panel">
        <div class="panel-label">高费用住院概率</div>
        <div class="gauge-wrap" aria-label="演示风险概率 {result['probability']:.1f}%">
          <div class="gauge">
            <div class="gauge-mask"></div>
            <div class="needle" style="transform:rotate({rotation:.1f}deg)"></div>
            <div class="needle-pin"></div>
          </div>
          <div class="gauge-value">{result['probability']:.1f}%</div>
          <div class="gauge-scale"><span>低</span><span>中</span><span>高</span></div>
        </div>
        <div class="risk-pill {risk_class}">{result['risk']}</div>
        <p class="risk-note">{threshold_text}</p>
      </article>
      <div class="summary-panels">
        <div class="metric-row">
          <article class="metric-card"><span>CCI</span><strong>{result['cci']}</strong><small>演示计分</small></article>
          <article class="metric-card"><span>CCCS</span><strong>{result['cccs']:.2f}</strong><small>演示算法</small></article>
          <article class="metric-card"><span>共病节点</span><strong>{selected_count}</strong><small>已识别</small></article>
          <article class="metric-card"><span>数据完整度</span><strong>{result['completeness']}%</strong><small>{'信息完整' if result['completeness'] == 100 else '仍可补充'}</small></article>
        </div>
        <article class="explain-panel">
          <div class="panel-label">主要影响因素</div>
          {impacts_html}
        </article>
      </div>
    </div>
    <div class="result-disclaimer"><strong>重要说明</strong><span>此页面为交互原型，风险概率、CCI、CCCS与影响因素均未接入论文冻结模型。正式发布前需替换为经过内部及外部验证的模型和预处理参数。</span></div>
    """


initialize_state()

with st.sidebar:
    st.markdown('<div class="eyebrow">评估设置</div><div class="sidebar-title">选择评估时点</div>', unsafe_allow_html=True)
    st.radio(
        "评估模式",
        ["admission", "dynamic"],
        format_func=lambda x: "01　入院快速评估" if x == "admission" else "02　住院动态评估",
        key="mode",
        label_visibility="collapsed",
        on_change=invalidate_result,
    )
    st.caption("基本信息与共病诊断" if st.session_state.mode == "admission" else "增加服务利用与住院进程")
    st.markdown(
        """
        <div class="sidebar-rule"></div>
        <div class="eyebrow">模型状态</div>
        <div class="status-line"><span>计算逻辑</span><b class="status-demo">演示</b></div>
        <div class="status-line"><span>冻结阈值</span><b>待接入</b></div>
        <div class="status-line"><span>模型版本</span><b>Prototype 1.0</b></div>
        <div class="notice">ⓘ　真实发布版将读取冻结的30个疾病节点、网络边权、标准化参数与模型文件。当前概率仅用于体验页面。</div>
        """,
        unsafe_allow_html=True,
    )

with st.container(key="topbar"):
    brand_col, pill_col, clear_col = st.columns([8, 1.6, 0.85], vertical_alignment="center")
    with brand_col:
        st.markdown(
            """
            <div class="top-brand"><div class="brand-mark">C</div><div>
              <div class="brand-name">CCCS–Cost</div>
              <div class="brand-sub">COPD高费用住院风险计算器</div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )
    with pill_col:
        st.markdown('<span class="demo-pill">界面演示版</span>', unsafe_allow_html=True)
    with clear_col:
        st.button("清空", on_click=clear_all, use_container_width=True)

title = "入院快速风险评估" if st.session_state.mode == "admission" else "住院期间动态风险评估"
subtitle = (
    "填写基本信息并录入共病诊断，后台自动生成CCI与CCCS。"
    if st.session_state.mode == "admission"
    else "在入院信息基础上增加住院日、检查与治疗变量，动态更新风险。"
)
head_left, head_right = st.columns([8, 1], vertical_alignment="center")
with head_left:
    st.markdown(
        f'<div class="workspace-head"><div class="eyebrow">CCCS-COST CALCULATOR</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown('<span class="step-badge">共 3 步</span>', unsafe_allow_html=True)

with st.container(key="basic_card"):
    heading(1, "患者基本信息", "带 * 的项目为必填项")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.number_input("年龄 *", 18, 110, key="age", on_change=invalidate_result)
    with c2:
        st.selectbox("性别 *", ["男", "女"], key="sex", on_change=invalidate_result)
    with c3:
        st.number_input("BMI *", 8.0, 60.0, step=0.1, key="bmi", on_change=invalidate_result)
    with c4:
        st.selectbox(
            "医保类型 *",
            ["城镇职工医保", "城乡居民医保/新农合", "自费", "公费/医疗救助", "其他社会保险", "其他/商业保险", "未知/缺失"],
            key="insurance",
            on_change=invalidate_result,
        )
    with c5:
        st.selectbox("入院途径 *", ["急诊", "门诊", "转院", "其他"], key="admission_route", on_change=invalidate_result)

with st.container(key="diagnosis_card"):
    heading(2, "共病诊断", "选择疾病名称，或批量粘贴ICD-10编码")
    selected_before = selected_disease_ids()
    tool_left, tool_right = st.columns([5, 1])
    with tool_left:
        st.text_input("搜索疾病名称、编码或系统分类", key="search", label_visibility="collapsed")
    with tool_right:
        if st.button("粘贴 ICD-10", use_container_width=True):
            st.session_state.show_icd = not st.session_state.show_icd

    if st.session_state.show_icd:
        with st.container(border=True):
            st.text_area(
                "批量录入诊断编码",
                key="icd_text",
                placeholder="例如：J96.00, I50.9, E11.9\n可用逗号、空格或换行分隔",
            )
            icd_button, icd_note = st.columns([1, 4], vertical_alignment="center")
            with icd_button:
                st.button("识别并加入", type="primary", on_click=recognize_codes, use_container_width=True)
            with icd_note:
                st.caption("演示版支持常见ICD-10前缀匹配")

    if selected_before:
        tags = "".join(
            f'<span class="selected-tag">{escape(d.name)}</span>'
            for d in DISEASES
            if d.id in selected_before
        )
        st.markdown(f'<div class="selected-tags">{tags}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="selection-count">已选 {len(selected_before)} 项</div>', unsafe_allow_html=True)

    query = st.session_state.search.strip().lower()
    filtered = [
        d for d in DISEASES
        if not query or query in f"{d.name}{d.code}{d.group}".lower()
    ]
    with st.container(key="diagnosis_list"):
        columns = st.columns(3)
        for index, disease in enumerate(filtered):
            with columns[index % 3]:
                st.checkbox(
                    f"**{disease.name}**  \n{disease.group}　`{disease.code}`",
                    key=f"disease_{disease.id}",
                    on_change=invalidate_result,
                )

if st.session_state.mode == "dynamic":
    with st.container(key="dynamic_card"):
        heading(3, "住院动态信息", "住院进程、检查与治疗变量")
        with st.expander("展开填写动态信息", expanded=False):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.selectbox(
                    "入院科室",
                    ["呼吸与危重症医学科", "急诊科", "重症医学科", "心内科", "其他"],
                    key="department",
                    on_change=invalidate_result,
                )
            with d2:
                st.number_input("当前住院日", 1, 365, key="stay_days", on_change=invalidate_result)
            with d3:
                st.number_input("既往住院次数", 0, 99, key="previous_admissions", on_change=invalidate_result)
            st.markdown("**已完成的检查或治疗**")
            service_cols = st.columns(3)
            for index, service in enumerate(SERVICES):
                with service_cols[index % 3]:
                    st.checkbox(service.label, key=f"service_{service.id}", on_change=invalidate_result)

with st.container(key="calculate_bar"):
    calc_text, calc_button = st.columns([4, 1.25], vertical_alignment="center")
    with calc_text:
        st.markdown(
            '<div class="calc-copy"><strong>信息仅保存在当前浏览器页面</strong><span>请勿将演示结果用于诊疗或费用管理决策</span></div>',
            unsafe_allow_html=True,
        )
    with calc_button:
        if st.button("计算高费用风险　→", type="primary", use_container_width=True):
            st.session_state.show_result = True

selected_ids = selected_disease_ids()
if st.session_state.show_result:
    result = calculate_risk(
        mode=st.session_state.mode,
        age=st.session_state.age,
        sex=st.session_state.sex,
        bmi=st.session_state.bmi,
        insurance=st.session_state.insurance,
        admission_route=st.session_state.admission_route,
        selected_disease_ids=selected_ids,
        selected_service_ids=selected_service_ids(),
        current_stay_days=st.session_state.stay_days,
        previous_admissions=st.session_state.previous_admissions,
    )
    with st.container(key="results_card"):
        st.markdown(result_html(result, len(selected_ids)), unsafe_allow_html=True)
        selected_names = "、".join(d.name for d in DISEASES if d.id in selected_ids) or "无"
        result_text = "\n".join(
            [
                "CCCS-Cost Calculator｜演示结果",
                f"评估模式：{'入院快速评估' if st.session_state.mode == 'admission' else '住院动态评估'}",
                f"高费用住院概率：{result['probability']:.1f}%（演示值）",
                f"风险分层：{result['risk']}",
                f"CCI：{result['cci']}",
                f"CCCS：{result['cccs']:.2f}（演示算法）",
                f"已识别共病：{selected_names}",
                "说明：当前页面仅用于界面与交互演示，尚未接入论文冻结模型，不可用于临床决策。",
            ]
        )
        with st.expander("复制或下载结果"):
            st.code(result_text, language=None, wrap_lines=True)
            st.download_button("下载结果文本", result_text, file_name="cccs_cost_result.txt", mime="text/plain")

st.markdown(
    '<div class="page-footer"><b>CCCS–Cost Calculator</b><span>Research prototype · Not for clinical use</span></div>',
    unsafe_allow_html=True,
)

