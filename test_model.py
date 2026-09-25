import numpy as np

from model import calculate_cci_from_icd, calculate_risk, cccs_score, recognize_icd_codes


def test_frozen_cccs_reproduces_saved_scores():
    cases = [
        (["心力衰竭（含心功能不全）", "糖尿病"], 12.179416),
        (["高血压", "冠心病（缺血性心脏病）", "心律失常", "肺炎及肺部感染"], 50.978172),
        (["恶性肿瘤（实体及血液系统）"], 0.387398),
    ]
    for nodes, expected in cases:
        assert np.isclose(cccs_score(nodes)[1], expected, atol=1e-5)


def test_icd_recognition_and_quan_cci():
    text = "I50.9, E11.9, N18.3"
    assert {"heart_failure", "diabetes", "ckd"} <= set(recognize_icd_codes(text))
    score, categories = calculate_cci_from_icd(text)
    assert score == 4
    assert set(categories) == {"充血性心力衰竭", "无并发症糖尿病", "中重度肾病"}


def test_ep000002_matches_locked_ensemble_prediction():
    result = calculate_risk(
        selected_disease_ids=["hypertension", "coronary", "arrhythmia", "pneumonia"],
        年龄=82, BMI=20.0, 医保类型="城乡居民医保/新农合", 住院次数=1,
        吸烟史=0, 是否化疗=0, 历史总门诊次数=0, 历史总住院次数=2,
        舒张压=60, 呼吸频率=18, 体温=36.5, 脉搏=78,
        入院科室="心血管内科", 入院途径="急诊",
        是否做过微生物培养=1, 是否做过病理检查=0,
        TimeTrend_month=40, 住院天数=10, CCI=1,
    )
    assert np.isclose(result["cccs"], 50.9781720996728, atol=1e-6)
    assert np.isclose(result["probability"] / 100.0, 0.264500, atol=1e-6)
