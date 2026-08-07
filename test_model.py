from model import calculate_risk, recognize_icd_codes


def test_default_empty_case_matches_prototype_floor():
    result = calculate_risk(
        mode="admission",
        age=68,
        sex="男",
        bmi=22.4,
        insurance="城镇职工医保",
        admission_route="急诊",
        selected_disease_ids=[],
    )
    assert result["probability"] >= 2.3
    assert result["risk"] == "低风险"
    assert result["cci"] == 0
    assert result["cccs"] == -1.55


def test_icd_recognition():
    ids = recognize_icd_codes("J96.00, I50.9, E11.9, N18.3")
    assert {"resp_failure", "heart_failure", "diabetes", "ckd"} <= set(ids)


def test_dynamic_mode_adds_service_and_stay_effects():
    common = dict(
        age=74,
        sex="男",
        bmi=18.2,
        insurance="自费",
        admission_route="急诊",
        selected_disease_ids=["resp_failure", "heart_failure", "ckd"],
    )
    admission = calculate_risk(mode="admission", **common)
    dynamic = calculate_risk(
        mode="dynamic",
        selected_service_ids=["blood_gas", "operation"],
        current_stay_days=12,
        previous_admissions=4,
        **common,
    )
    assert dynamic["probability"] > admission["probability"]


def test_reference_case_matches_existing_web_prototype():
    result = calculate_risk(
        mode="admission",
        age=68,
        sex="男",
        bmi=22.4,
        insurance="城镇职工医保",
        admission_route="急诊",
        selected_disease_ids=[
            "pneumonia",
            "resp_failure",
            "bronchiectasis",
            "heart_failure",
            "ckd",
        ],
    )
    assert round(result["probability"], 1) == 8.8
    assert result["cci"] == 4
    assert round(result["cccs"], 2) == 0.73

