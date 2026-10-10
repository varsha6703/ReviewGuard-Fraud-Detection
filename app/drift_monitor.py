import json, os, logging
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO)
REFERENCE_PATH = "artifacts/reference_scores.json"

def check_drift(live_scores):
    if not os.path.exists(REFERENCE_PATH):
        return {"status":"reference_missing","message":"Train model first to create reference scores."}
    with open(REFERENCE_PATH) as f:
        reference=json.load(f)
    if len(reference)<2 or len(live_scores)<2:
        return {"status":"insufficient_data","reference_count":len(reference),"live_count":len(live_scores)}
    stat,pvalue=ks_2samp(reference, live_scores)
    drift=bool(pvalue < 0.05)
    result={"status":"drift_detected" if drift else "stable","ks_statistic":float(stat),
            "p_value":float(pvalue),"threshold":0.05,"reference_count":len(reference),
            "live_count":len(live_scores)}
    if drift:
        logging.warning("MOCK DRIFT ALERT: KS p-value %.6f < 0.05", pvalue)
    return result

if __name__ == "__main__":
    print(check_drift([]))
