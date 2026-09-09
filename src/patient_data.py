"""Patient metadata and checks for cleaned-corpus/annotation consistency."""

import hashlib

from clean_cases import extract_age, extract_gender


def new_patient(text: str) -> dict:
    age, age_method = extract_age(text)
    gender, gender_method = extract_gender(text)
    return {"case_id": "pasted-" + hashlib.sha256(text.encode()).hexdigest()[:16],
            "article_id": "", "age": age, "gender": gender, "age_method": age_method,
            "gender_method": gender_method, "age_upstream": "", "gender_upstream": "",
            "source_case_ids": "", "data_source": "new_text"}


def validate_annotations(cases: list[dict], rows: list[dict], source: str) -> None:
    """Fail explicitly on old raw-case IDs or spans in a different text version."""
    by_id = {case["case_id"]: case for case in cases}
    foreign = sorted({r["case_id"] for r in rows} - by_id.keys())
    if foreign:
        raise ValueError(f"{source} contains IDs outside the selected corpus: {', '.join(foreign[:5])}. "
                         "Regenerate both extraction CSVs with --cases data/interim/cases_clean.csv.")
    for row in rows:
        case = by_id[row["case_id"]]
        text = case["case_text"]
        start, end = int(row["start"]), int(row["end"])
        if not 0 <= start < end <= len(text) or text[start:end] != row["surface_text"]:
            raise ValueError(f"{source}: offsets do not match cleaned text for {row['case_id']}. "
                             "Regenerate the extraction CSVs from the selected corpus.")
