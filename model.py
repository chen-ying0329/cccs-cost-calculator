"""Frozen CCCS scorer and five-seed HONAM-M3 inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from math import comb
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from formal_model import FrozenHONAMEnsemble


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
HIGH_COST_AMOUNT = 32998.753036000024
CLASSIFICATION_THRESHOLD = 0.50
MODEL_VERSION = "HONAM-M3 formal 5-seed ensemble / 2026-07-31"
TRAINING_TIME_MIN = 0
TRAINING_TIME_MAX = 107


@dataclass(frozen=True)
class Disease:
    id: str
    name: str
    code: str
    group: str


@dataclass(frozen=True)
class Service:
    id: str
    label: str
    feature_name: str


_NODE_ROWS = [
    ("hypertension", "高血压", "I10–I15", "循环系统疾病"),
    ("coronary", "冠心病（缺血性心脏病）", "I20–I25", "循环系统疾病"),
    ("heart_failure", "心力衰竭（含心功能不全）", "I50等", "循环系统疾病"),
    ("arrhythmia", "心律失常", "I47–I49", "循环系统疾病"),
    ("valvular", "心脏瓣膜病", "I34–I39", "循环系统疾病"),
    ("cerebrovascular", "脑血管病", "I60–I69等", "循环系统疾病"),
    ("peripheral_artery", "外周动脉疾病及动脉粥样硬化", "I70–I79等", "循环系统疾病"),
    ("vte", "静脉血栓栓塞症（含肺栓塞）", "I26/I80/I82", "循环系统疾病"),
    ("pneumonia", "肺炎及肺部感染", "J12–J18等", "呼吸系统疾病"),
    ("asthma", "支气管哮喘", "J45–J46", "呼吸系统疾病"),
    ("bronchiectasis", "支气管扩张", "J47", "呼吸系统疾病"),
    ("ild", "间质性肺病及尘肺", "J60–J70/J84", "呼吸系统疾病"),
    ("pleural", "胸膜疾病（胸腔积液/气胸/胸膜肥厚）", "J90–J94", "呼吸系统疾病"),
    ("sleep_apnea", "睡眠呼吸暂停（重叠综合征）", "G47.3", "呼吸系统疾病"),
    ("upper_airway", "上气道疾病（慢性鼻窦炎/鼻炎）", "J30–J32", "呼吸系统疾病"),
    ("diabetes", "糖尿病", "E10–E14", "内分泌代谢疾病"),
    ("dyslipidemia", "血脂异常", "E78", "内分泌代谢疾病"),
    ("thyroid", "甲状腺疾病", "E00–E07", "内分泌代谢疾病"),
    ("malnutrition", "营养不良及低蛋白血症", "E40–E46等", "内分泌代谢疾病"),
    ("ckd", "慢性肾脏病及肾衰竭", "N18–N19", "泌尿生殖系统疾病"),
    ("bph", "良性前列腺增生", "N40", "泌尿生殖系统疾病"),
    ("liver", "慢性肝病及肝硬化", "K70–K77", "消化系统疾病"),
    ("fatty_liver", "脂肪肝", "K76.0", "消化系统疾病"),
    ("gastritis_ulcer", "胃炎及消化性溃疡", "K25–K29", "消化系统疾病"),
    ("gallbladder", "胆石症及胆囊疾病", "K80–K82", "消化系统疾病"),
    ("anemia", "贫血", "D50–D64", "血液系统疾病"),
    ("osteoporosis", "骨质疏松", "M80–M82", "肌肉骨骼疾病"),
    ("malignancy", "恶性肿瘤（实体及血液系统）", "C00–C97", "肿瘤"),
    ("rheumatic", "风湿性及结缔组织病", "M05–M35", "风湿免疫疾病"),
    ("neuropsych", "精神与神经认知障碍", "F00–F99/G30等", "神经精神疾病"),
]
DISEASES = [Disease(*row) for row in _NODE_ROWS]
DISEASE_BY_ID = {d.id: d for d in DISEASES}

SERVICES = [
    Service("chemotherapy", "既往化疗", "是否化疗"),
    Service("microbiology", "本次住院做过微生物培养", "是否做过微生物培养"),
    Service("pathology", "本次住院做过病理检查", "是否做过病理检查"),
]

ICD_RULES = [
    (r"^I1[0-5]", "hypertension"), (r"^I2[0-5]", "coronary"),
    (r"^(I50|I099|I110|I130|I132|I255|I42|I43)", "heart_failure"),
    (r"^I4[7-9]", "arrhythmia"), (r"^I3[4-9]", "valvular"),
    (r"^(G45|G46|H340|I6[0-9])", "cerebrovascular"),
    (r"^(I7[0-9]|K55|Z95[89])", "peripheral_artery"),
    (r"^(I26|I80|I82)", "vte"), (r"^(J1[2-8]|J20|J21|J22)", "pneumonia"),
    (r"^J4[56]", "asthma"), (r"^J47", "bronchiectasis"),
    (r"^(J6[0-9]|J70|J84)", "ild"), (r"^J9[0-4]", "pleural"),
    (r"^G473", "sleep_apnea"), (r"^J3[0-2]", "upper_airway"),
    (r"^E1[0-4]", "diabetes"), (r"^E78", "dyslipidemia"),
    (r"^E0[0-7]", "thyroid"), (r"^(E4[0-6]|E8809)", "malnutrition"),
    (r"^N1[89]", "ckd"), (r"^N40", "bph"),
    (r"^(B18|K7[0-7]|I85|Z944)", "liver"), (r"^K760", "fatty_liver"),
    (r"^K2[5-9]", "gastritis_ulcer"), (r"^K8[0-2]", "gallbladder"),
    (r"^D[5-6][0-9]", "anemia"), (r"^M8[0-2]", "osteoporosis"),
    (r"^C[0-9][0-9]", "malignancy"), (r"^(M0[5-6]|M3[1-5])", "rheumatic"),
    (r"^(F[0-9][0-9]|G30|G311)", "neuropsych"),
]

CCI_PREFIX = {
    "心肌梗死": ["I21", "I22", "I252"],
    "充血性心力衰竭": ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
    "外周血管疾病": ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"],
    "脑血管疾病": ["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    "痴呆": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "慢性肺疾病": ["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"],
    "风湿性疾病": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    "消化性溃疡": ["K25", "K26", "K27", "K28"],
    "轻度肝病": ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
    "无并发症糖尿病": ["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E120", "E121", "E126", "E128", "E129", "E130", "E131", "E136", "E138", "E139", "E140", "E141", "E146", "E148", "E149"],
    "糖尿病伴并发症": ["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E122", "E123", "E124", "E125", "E127", "E132", "E133", "E134", "E135", "E137", "E142", "E143", "E144", "E145", "E147"],
    "偏瘫截瘫": ["G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831", "G832", "G833", "G834", "G839"],
    "中重度肾病": ["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054", "N055", "N056", "N057", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"],
    "中重度肝病": ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    "转移性实体瘤": ["C77", "C78", "C79", "C80"],
    "艾滋病HIV": ["B20", "B21", "B22", "B24"],
}
CCI_WEIGHTS = {"心肌梗死": 1, "充血性心力衰竭": 1, "外周血管疾病": 1, "脑血管疾病": 1, "痴呆": 1, "慢性肺疾病": 1, "风湿性疾病": 1, "消化性溃疡": 1, "轻度肝病": 1, "无并发症糖尿病": 1, "偏瘫截瘫": 2, "中重度肾病": 2, "糖尿病伴并发症": 2, "恶性肿瘤": 2, "中重度肝病": 3, "转移性实体瘤": 6, "艾滋病HIV": 6}


def normalize_icd_codes(text: str) -> list[str]:
    return sorted({token.upper().replace(".", "") for token in re.split(r"[;；,，、\s]+", text or "") if token.strip()})


def recognize_icd_codes(text: str) -> list[str]:
    found = set()
    for code in normalize_icd_codes(text):
        for pattern, disease_id in ICD_RULES:
            if re.search(pattern, code):
                found.add(disease_id)
    return [d.id for d in DISEASES if d.id in found]


def calculate_cci_from_icd(text: str) -> tuple[int, list[str]]:
    codes = normalize_icd_codes(text)
    flags = {name: False for name in CCI_WEIGHTS}
    for code in codes:
        for category, prefixes in CCI_PREFIX.items():
            if any(code.startswith(prefix) for prefix in prefixes):
                flags[category] = True
        if code.startswith("C") and len(code) >= 3 and code[1:3].isdigit():
            n = int(code[1:3])
            if n <= 26 or 30 <= n <= 34 or 37 <= n <= 41 or n == 43 or 45 <= n <= 58 or 60 <= n <= 76 or 81 <= n <= 85 or n == 88 or 90 <= n <= 97:
                flags["恶性肿瘤"] = True
    if flags["糖尿病伴并发症"]:
        flags["无并发症糖尿病"] = False
    if flags["中重度肝病"]:
        flags["轻度肝病"] = False
    if flags["转移性实体瘤"]:
        flags["恶性肿瘤"] = False
    categories = [name for name, present in flags.items() if present]
    return int(sum(CCI_WEIGHTS[name] for name in categories)), categories


@lru_cache(maxsize=1)
def load_cccs_params() -> dict:
    return json.loads((ASSETS / "CCCS_frozen_params.json").read_text(encoding="utf-8"))


def cccs_score(node_names: Iterable[str]) -> tuple[float, float, np.ndarray]:
    params = load_cccs_params()
    node_to_index = {name: index for index, name in enumerate(params["node_order"])}
    present = sorted({node_to_index[name] for name in node_names if name in node_to_index})
    present_set = set(present)
    k = len(present)
    weights = {tuple(sorted((int(i), int(j)))): float(params["weight_norm"][f"{i}_{j}"]) for i, j in params["global_edges"]}
    all_edges = [(i, j, w) for (i, j), w in weights.items() if i in present_set and j in present_set]
    positive_edges = [(i, j, w) for i, j, w in all_edges if w > 0]
    strength = float(sum(w for _, _, w in positive_edges))
    density = strength / comb(k, 2) if k >= 2 else 0.0
    if k >= 2:
        local = {node: pos for pos, node in enumerate(present)}
        distances = np.full((k, k), np.inf)
        np.fill_diagonal(distances, 0.0)
        for i, j, weight in positive_edges:
            a, b = local[i], local[j]
            distances[a, b] = distances[b, a] = min(distances[a, b], 1.0 / weight)
        for mid in range(k):
            distances = np.minimum(distances, distances[:, [mid]] + distances[[mid], :])
        inverse = np.zeros_like(distances)
        mask = np.isfinite(distances) & (distances > 0)
        inverse[mask] = 1.0 / distances[mask]
        efficiency = float(inverse.sum() / (k * (k - 1)))
    else:
        efficiency = 0.0
    modules = {int(i): int(v) for i, v in params["module"].items()}
    cross = sum(1 for i, j, _ in all_edges if modules.get(i) != modules.get(j))
    cross_ratio = float(cross / len(all_edges)) if all_edges else 0.0
    bridge_scores = {int(i): float(v) for i, v in params["bc_score"].items()}
    bridge = float(np.mean([bridge_scores[i] for i in present])) if present else 0.0
    features = np.array([k, strength, density, efficiency, cross_ratio, bridge], dtype=float)
    standardized = (features - np.asarray(params["scaler_mean"])) / np.asarray(params["scaler_scale"])
    component = np.asarray(params["spca_components"]).reshape(-1)
    # sklearn SparsePCA.transform projects by ridge regression.  The original
    # training run used ridge_alpha=0.01, so a simple dot product is not exact.
    projection_denominator = float(np.dot(component, component) + 0.01)
    raw = float(
        np.dot(standardized - np.asarray(params["spca_mean"]), component)
        / projection_denominator
        * float(params["cccs_sign"])
    )
    scaled = float(np.clip((raw - float(params["scale_lo"])) / (float(params["scale_hi"]) - float(params["scale_lo"])) * 100.0, 0.0, 100.0))
    return raw, scaled, features


@lru_cache(maxsize=1)
def get_ensemble() -> FrozenHONAMEnsemble:
    return FrozenHONAMEnsemble(ASSETS)


def selected_node_names(selected_disease_ids: Iterable[str]) -> list[str]:
    return [DISEASE_BY_ID[x].name for x in dict.fromkeys(selected_disease_ids) if x in DISEASE_BY_ID]


def calculate_risk(*, selected_disease_ids: Iterable[str], **features) -> dict:
    names = selected_node_names(selected_disease_ids)
    cccs_raw, cccs, network = cccs_score(names)
    row = dict(features)
    row["CCCS"] = cccs
    frame = pd.DataFrame([row], columns=get_ensemble().feature_names)
    probability, seeds = get_ensemble().predict_proba(frame)
    p = float(probability[0])
    risk = "模型阳性" if p >= CLASSIFICATION_THRESHOLD else ("概率较高" if p >= 0.20 else "概率较低")
    return {
        "probability": p * 100.0,
        "risk": risk,
        "positive": p >= CLASSIFICATION_THRESHOLD,
        "threshold": CLASSIFICATION_THRESHOLD,
        "cci": float(row["CCI"]),
        "cccs": cccs,
        "cccs_raw": cccs_raw,
        "network_features": network,
        "seed_probabilities": (seeds[0] * 100.0).tolist(),
        "impacts": get_ensemble().local_main_effects(frame),
        "model_version": MODEL_VERSION,
    }


def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    model = get_ensemble()
    frame = frame.copy()
    aliases = {
        "史_吸烟史": "吸烟史",
        "史_化疗": "是否化疗",
        "查_微生物培养": "是否做过微生物培养",
        "查_病理检查": "是否做过病理检查",
    }
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    for column in ["吸烟史", "是否化疗", "是否做过微生物培养", "是否做过病理检查"]:
        if column in frame.columns:
            frame[column] = frame[column].replace(
                {"是": 1, "否": 0, "未知": np.nan, "有": 1, "无": 0, True: 1, False: 0}
            ).infer_objects(copy=False)
    required = list(model.feature_names)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"批量文件缺少字段：{', '.join(missing)}")
    probability, seeds = model.predict_proba(frame[required])
    out = frame.copy()
    out["HONAM_M3_高费用概率"] = probability
    out["HONAM_M3_0.5分类"] = (probability >= CLASSIFICATION_THRESHOLD).astype(int)
    for seed in range(5):
        out[f"seed_{seed}_概率"] = seeds[:, seed]
    return out
