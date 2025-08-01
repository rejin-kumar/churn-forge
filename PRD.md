# 📄 PRD: Synthetic Churn Dataset Generator

## Project Name
Customer Churn Dataset Generator (Pattern-Based)

---

## Overview
Build an internal tool/script that generates a synthetic dataset simulating customer behavior in a hosting/domain service business. The generated dataset must include churn labels (`customer_status: active/inactive`) that are **driven by specific, configurable behavioral patterns**, not randomly assigned.

### Use Cases:
- Prototyping churn prediction models
- Simulating user behavior for ML workflows
- Testing dashboards or analytics pipelines
- Creating sandbox datasets for training/experimentation

---

## User
Internal data scientist, ML engineer, or analyst using Cursor. They should be able to:
- Modify churn logic
- Control the number of generated records
- Export the dataset
- Understand and trace which patterns drive churn in the data

---

## Inputs
| Parameter         | Type   | Default | Description                              |
|------------------|--------|---------|------------------------------------------|
| `n_customers`     | int    | 100    | Number of synthetic customer records     |
| `random_seed`     | int    | None    | Optional. Sets deterministic output      |

---

## Output
- A `.csv` file, e.g. `churn_dataset_realistic.csv`
- 100+ rows of synthetic customer data
- Includes 15+ columns reflecting behavior
- Target column: `customer_status` with values `'active'` or `'inactive'`

---

## Schema (Columns)

| Column Name                   | Type     | Description |
|------------------------------|----------|-------------|
| `customer_id`                | string   | UUID |
| `customer_name`              | string   | Generated with Faker |
| `customer_email`             | string   | Generated with Faker |
| `support_tickets`            | int      | 0–10 |
| `avg_resolution_time`        | float    | in hours |
| `critical_tickets_sla_breach`| int      | 0–4 |
| `product_renewals`           | int      | 0–5 |
| `tenure_months`              | int      | 1–48 |
| `monthly_spend`              | float    | USD 5–150 |
| `last_login`                 | date     | Derived from tenure |
| `total_products`             | int      | 1–10 |
| `products_transferred_out`   | int      | 0–3 |
| `avg_load_time`              | float    | in seconds |
| `downtime_minutes`           | int      | 0–300 |
| `product_usage_percent`      | float    | 0–100% |
| `customer_status`            | string   | `'active'` or `'inactive'` |

---

## Churn Logic

Use a rule-based scoring function to determine if a customer should churn.

### Score Calculation

| Feature                       | Condition                        | Score Impact |
|------------------------------|----------------------------------|--------------|
| Support tickets              | `> 4`                            | +1 |
| Avg. resolution time         | `> 48 hours`                     | +1 |
| Critical tickets SLA breach | `> 1`                            | +1 |
| Product renewals             | `< 2`                            | +1 |
| Tenure                       | `< 6 months`                     | +1 |
| Monthly spend                | `< $20`                          | +1 |
| Days since last login        | `> 45 days`                      | +1 |
| Products transferred out     | `> 1`                            | +1 |
| Avg. load time               | `> 5 seconds`                    | +1 |
| Downtime                     | `> 120 minutes`                  | +1 |
| Product usage                | `< 30%`                          | +1 |

### Churn Assignment

- If total score ≥ 5 → `customer_status = 'inactive'`
- Else → `'active'`
- Add ~10% noise by randomly flipping the status to simulate human unpredictability

---

## Functional Requirements
- Must generate data as a single Python script or notebook
- Columns must match schema above
- Churn logic must be centralized and easy to modify
- Save final output as CSV
- Print preview (first 5 rows) to console after generation

---

## Non-Functional Requirements
- Fast runtime: < 2 seconds for 100 rows
- Deterministic output with seed
- Only `pandas`, `numpy`, and `faker` dependencies

---

## Future Considerations
- CLI support (e.g., `python generate.py --rows 5000`)
- Add user segments or plan types
- Output in multiple formats (JSON, DB insert, etc.)
- Include behavior drift (churn logic changing over time)

