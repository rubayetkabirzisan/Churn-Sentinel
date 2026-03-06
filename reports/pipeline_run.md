# Churn Sentinel — Pipeline Run Report
Generated: 2026-03-07 01:30:42
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
| Pipeline duration | 5.3s |

## Risk Type Breakdown

| Risk Type | Count | % of Flagged |
|-----------|-------|-------------|
| Disengagement | 3 | 100.0% |
| Support Issue | 0 | 0.0% |

## Flagged Users

| User ID | Churn Prob | Risk Type | Discount | Subject |
|---------|-----------|-----------|----------|---------|
| USR_0001 | 90.8% | disengagement | 10% | Checking in on Your Journey with Our Pla |
| USR_0005 | 78.6% | disengagement | 5% | Checking in on your success with us |
| USR_0006 | 69.8% | disengagement | 5% | Checking in to See How You're Doing with |

## Output Files
- Email log : `outputs/email_log.json`
- This report : `reports/pipeline_run.md`

---
*Churn Sentinel v1.0 — XGBoost + Multi-Agent AI*
