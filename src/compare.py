import json

from .config import LEAD_H, REPORTS, tag

XGB = "xgboost ensemble (calibrated)"
CAL = "calibration (test, 8 equal-size bins)"


def main():
    runs = {lead: json.loads((REPORTS / f"metrics{tag(lead)}.json").read_text()) for lead in (0, LEAD_H)}
    print(f"{'':34}{'at arrival':>12}{f'{LEAD_H} h ahead':>12}")
    rows = [("ROC-AUC", lambda m: m[XGB]["roc_auc"]), ("PR-AUC", lambda m: m[XGB]["pr_auc"]),
            ("Brier (lower is better)", lambda m: m[XGB]["brier"]),
            ("calibration gap (lower is better)", lambda m: m[CAL]["mean |gap|"]),
            ("rule ROC-AUC", lambda m: m["rule: backlog > 1 day"]["roc_auc"]),
            ("logistic ROC-AUC", lambda m: m["logistic regression"]["roc_auc"])]
    rows += [(f"accuracy when {c}", lambda m, c=c: m["accuracy by confidence"].get(c)) for c in ("High", "Medium", "Low")]
    rows += [(f"share {c}", lambda m, c=c: m["confidence mix on test"].get(c)) for c in ("High", "Medium", "Low")]
    for name, get in rows:
        print(f"{name:34}" + "".join(f"{str(get(runs[lead])):>12}" for lead in (0, LEAD_H)))


if __name__ == "__main__":
    main()
