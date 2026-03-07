# Churn Sentinel — Pipeline Run Report
Generated: 2026-03-07 22:06:56
Mode: TEST

## Summary

| Metric | Value |
|--------|-------|
| Total users processed | 10 |
| Flagged (above threshold) | 3 |
| Below threshold (no action) | 7 |
| Emails generated | 3 |
| Discount eligible | 3 |
| Avg churn probability | 42.0% |
| Pipeline duration | 3.6s |

## Risk Type Breakdown

| Risk Type | Count | % of Flagged |
|-----------|-------|-------------|
| Disengagement | 3 | 100.0% |
| Support Issue | 0 | 0.0% |

## Flagged Users

| User ID | Churn Prob | Risk Type | Discount | Subject |
|---------|-----------|-----------|----------|---------|
| USR_0001 | 90.8% | disengagement | 10% | Checking in to see how we can better sup |
| USR_0005 | 78.6% | disengagement | 5% | Checking In on Your Experience with Our  |
| USR_0006 | 69.8% | disengagement | 5% | Checking in and a Special Offer to Help  |

## Output Files
- Email log : `outputs/email_log.json`
- This report : `reports/pipeline_run.md`

---
*Churn Sentinel v1.0 — XGBoost + Multi-Agent AI*
