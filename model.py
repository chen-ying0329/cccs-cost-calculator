"""CCCS-Cost prototype data and risk calculation.

This module intentionally reproduces the demonstration logic from the existing
web prototype. It is not a validated clinical model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


@dataclass(frozen=True)
class Disease:
    id: str
    name: str
    code: str
    group: str
    cci: int
    impact: float


@dataclass(frozen=True)
class Service:
    id: str
    label: str
    impact: float


DISEASES = [
    Disease("pneumonia", "肺炎", "J12–J18", "呼吸系统", 0, 0.80),
    Disease("resp_failure", "呼吸衰竭", "J96", "呼吸系统", 0, 1.45),
    Disease("bronchiectasis", "支气管扩张", "J47", "呼吸系统", 1, 0.65),
    Disease("emphysema", "肺气肿", "J43", "呼吸系统", 1, 0.50),
    Disease("pulm_heart", "肺源性心脏病", "I27", "心血管", 0, 0.95),
    Disease("hypertension", "高血压", "I10–I15", "心血管", 0, 0.35),
    Disease("coronary", "冠心病", "I20–I25", "心血管", 1, 0.58),
    Disease("heart_failure", "心力衰竭", "I50", "心血管", 1, 1.10),
    Disease("arrhythmia", "心律失常", "I47–I49", "心血管", 0, 0.48),
    Disease("valvular", "心脏瓣膜病", "I34–I39", "心血管", 0, 0.48),
    Disease("stroke", "脑卒中", "I60–I69", "神经系统", 1, 0.72),
    Disease("dementia", "痴呆", "F00–F03", "神经系统", 1, 0.62),
    Disease("paralysis", "偏瘫或截瘫", "G81–G82", "神经系统", 2, 0.90),
    Disease("diabetes", "糖尿病", "E10–E14", "代谢系统", 1, 0.54),
    Disease("diabetes_comp", "糖尿病伴并发症", "E10–E14†", "代谢系统", 2, 0.78),
    Disease("dyslipidemia", "血脂异常", "E78", "代谢系统", 0, 0.22),
    Disease("hyperuricemia", "高尿酸血症", "E79", "代谢系统", 0, 0.20),
    Disease("obesity", "肥胖", "E66", "代谢系统", 0, 0.30),
    Disease("ckd", "慢性肾脏病", "N18", "肾脏系统", 2, 1.00),
    Disease("liver_mild", "轻度肝病", "K70–K77", "消化系统", 1, 0.55),
    Disease("liver_severe", "中重度肝病", "K72/K76.6", "消化系统", 3, 0.92),
    Disease("ulcer", "消化性溃疡", "K25–K28", "消化系统", 1, 0.34),
    Disease("gerd", "胃食管反流病", "K21", "消化系统", 0, 0.20),
    Disease("tumor", "恶性肿瘤", "C00–C75", "肿瘤", 2, 1.12),
    Disease("metastasis", "转移性实体瘤", "C76–C80", "肿瘤", 6, 1.62),
    Disease("anemia", "贫血", "D50–D64", "血液系统", 0, 0.42),
    Disease("osteoporosis", "骨质疏松", "M80–M82", "其他", 0, 0.20),
    Disease("osa", "阻塞性睡眠呼吸暂停", "G47.3", "呼吸系统", 0, 0.35),
    Disease("depression", "抑郁或焦虑障碍", "F32–F41", "精神心理", 0, 0.26),
    Disease("malnutrition", "营养不良", "E40–E46", "代谢系统", 0, 0.76),
]

DISEASE_BY_ID = {d.id: d for d in DISEASES}

SERVICES = [
    Service("blood_gas", "血气分析", 0.32),
    Service("ct", "CT检查", 0.25),
    Service("mri", "MRI检查", 0.42),
    Service("ultrasound", "超声检查", 0.18),
    Service("xray", "X线检查", 0.14),
    Service("operation", "手术或操作", 0.62),
    Service("chemotherapy", "既往化疗", 0.58),
    Service("targeted", "既往靶向治疗", 0.55),
    Service("endocrine", "既往内分泌治疗", 0.32),
]

SERVICE_BY_ID = {s.id: s for s in SERVICES}

_EDGE_PAIRS = {
    pair
    for pair in [
        ("resp_failure", "pneumonia"),
        ("heart_failure", "pulm_heart"),
        ("coronary", "heart_failure"),
        ("diabetes", "ckd"),
        ("diabetes_comp", "ckd"),
        ("hypertension", "stroke"),
        ("hypertension", "coronary"),
        ("malnutrition", "resp_failure"),
        ("anemia", "ckd"),
        ("arrhythmia", "heart_failure"),
        ("emphysema", "resp_failure"),
        ("osa", "obesity"),
        ("tumor", "malnutrition"),
        ("metastasis", "malnutrition"),
    ]
}

ICD_RULES = [
    (r"\bJ1[2-8](?:\.\w+)?\b", "pneumonia"),
    (r"\bJ96(?:\.\w+)?\b", "resp_failure"),
    (r"\bJ47(?:\.\w+)?\b", "bronchiectasis"),
    (r"\bJ43(?:\.\w+)?\b", "emphysema"),
    (r"\bI27(?:\.\w+)?\b", "pulm_heart"),
    (r"\bI1[0-5](?:\.\w+)?\b", "hypertension"),
    (r"\bI2[0-5](?:\.\w+)?\b", "coronary"),
    (r"\bI50(?:\.\w+)?\b", "heart_failure"),
    (r"\bI4[7-9](?:\.\w+)?\b", "arrhythmia"),
    (r"\bI6\d(?:\.\w+)?\b", "stroke"),
    (r"\bF0[0-3](?:\.\w+)?\b", "dementia"),
    (r"\bG8[12](?:\.\w+)?\b", "paralysis"),
    (r"\bE1[0-4](?:\.\w+)?\b", "diabetes"),
    (r"\bE78(?:\.\w+)?\b", "dyslipidemia"),
    (r"\bE79(?:\.\w+)?\b", "hyperuricemia"),
    (r"\bE66(?:\.\w+)?\b", "obesity"),
    (r"\bN18(?:\.\w+)?\b", "ckd"),
    (r"\bK2[5-8](?:\.\w+)?\b", "ulcer"),
    (r"\bK21(?:\.\w+)?\b", "gerd"),
    (r"\bC[0-7]\d(?:\.\w+)?\b", "tumor"),
    (r"\bC(?:7[6-9]|80)(?:\.\w+)?\b", "metastasis"),
    (r"\bD[5-6]\d(?:\.\w+)?\b", "anemia"),
    (r"\bM8[0-2](?:\.\w+)?\b", "osteoporosis"),
    (r"\bG47\.3\w*\b", "osa"),
    (r"\bF(?:3[2-9]|4[0-1])(?:\.\w+)?\b", "depression"),
    (r"\bE4[0-6](?:\.\w+)?\b", "malnutrition"),
]


def recognize_icd_codes(text: str) -> list[str]:
    """Return disease IDs recognized by the prototype's ICD-10 rules."""

    upper = text.upper()
    return [disease_id for pattern, disease_id in ICD_RULES if re.search(pattern, upper)]


def calculate_risk(
    *,
    mode: str,
    age: float,
    sex: str,
    bmi: float,
    insurance: str,
    admission_route: str,
    selected_disease_ids: Iterable[str],
    selected_service_ids: Iterable[str] = (),
    current_stay_days: float = 0,
    previous_admissions: float = 0,
) -> dict:
    """Calculate the same demonstration values used by the source prototype."""

    disease_ids = list(dict.fromkeys(selected_disease_ids))
    service_ids = list(dict.fromkeys(selected_service_ids))
    selected = [DISEASE_BY_ID[x] for x in disease_ids if x in DISEASE_BY_ID]
    selected_id_set = {d.id for d in selected}

    cci = sum(d.cci for d in selected)
    edge_count = sum(
        1
        for a in selected_id_set
        for b in selected_id_set
        if a < b and tuple(sorted((a, b))) in _EDGE_PAIRS
    )
    possible_edges = len(selected) * (len(selected) - 1) / 2 if len(selected) > 1 else 0
    density = edge_count / possible_edges if possible_edges else 0
    cccs = (
        (len(selected) - 3.4) / 2.2 + edge_count * 0.34 + density * 0.9
        if selected
        else -1.55
    )

    disease_impact = sum(d.impact for d in selected)
    service_impact = sum(
        SERVICE_BY_ID[x].impact for x in service_ids if x in SERVICE_BY_ID
    )

    logit = (
        -4.45
        + max(0, age - 55) * 0.022
        + max(0, 19 - bmi) * 0.075
        + disease_impact * 0.19
        + cci * 0.11
        + cccs * 0.16
        + (0.32 if admission_route == "急诊" else 0)
        + (0.19 if insurance == "自费" else 0)
    )

    if mode == "dynamic":
        logit += (
            max(0, current_stay_days - 3) * 0.095
            + min(previous_admissions, 5) * 0.08
            + service_impact * 0.38
        )

    probability = min(92.0, max(2.3, 100 / (1 + math.exp(-logit))))
    risk = "高风险" if probability >= 30 else "中风险" if probability >= 15 else "低风险"
    completeness_fields = [age, sex, bmi, insurance, admission_route]
    completeness = round((sum(bool(x) for x in completeness_fields) + bool(selected)) / 6 * 100)

    factors = [
        {
            "label": d.name,
            "value": d.impact * 0.19 + d.cci * 0.11,
            "direction": "up",
        }
        for d in selected
    ]
    if age >= 65:
        factors.append({"label": "年龄", "value": max(0.12, (age - 55) * 0.022), "direction": "up"})
    if 0 < bmi < 19:
        factors.append({"label": "较低BMI", "value": (19 - bmi) * 0.075, "direction": "up"})
    if mode == "dynamic" and current_stay_days > 3:
        factors.append({"label": "当前住院日", "value": (current_stay_days - 3) * 0.095, "direction": "up"})
    if admission_route == "急诊":
        factors.append({"label": "急诊入院", "value": 0.32, "direction": "up"})
    if 19 <= bmi <= 25:
        factors.append({"label": "BMI处于常见范围", "value": -0.16, "direction": "down"})
    factors = sorted(factors, key=lambda x: abs(x["value"]), reverse=True)[:6]

    return {
        "cci": cci,
        "edge_count": edge_count,
        "density": density,
        "cccs": cccs,
        "probability": probability,
        "risk": risk,
        "completeness": completeness,
        "impacts": factors,
    }
