# 📄 PRD: Synthetic Churn Dataset Generator

## Project Name
Hosting & Domain Service Churn Dataset Generator

---

## Overview
Build an internal tool/script that generates a synthetic dataset simulating customer behavior in a hosting and domain service business. The generated dataset includes comprehensive product usage, support interactions, financial metrics, and engagement data. Churn prediction features are **driven by realistic behavioral patterns** commonly observed in the hosting industry.

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
| `n_customers`     | int    | 1000   | Number of synthetic customer records     |
| `random_seed`     | int    | None    | Optional. Sets deterministic output      |
| `reference_date`  | date   | "2024-12-01" | Reference date for data generation |

---

## Output
- A `.csv` file, e.g. `synthetic_churn_dataset.csv`
- 1000+ rows of synthetic customer data
- Includes 50+ columns reflecting hosting/domain service behavior
- Dataset focused on behavioral features for churn prediction (target column removed)

---

## Schema (Columns)

| Column Name                   | Type     | Description |
|------------------------------|----------|-------------|
| `FIRST_OF_MONTH`             | date     | First day of the month for the data record |
| `ACCOUNT_ID`                 | int      | DWH unique identifier for the account |
| `PERSON_ORG_ID`              | int      | Person/Organization identifier |
| `PP_SERVICE_CREATED_DATE`    | date     | Service creation date of the primary service instance |
| `PP_SERVICE_DELETION_DATE`   | date     | Service deletion date of the primary service instance (nullable) |
| `PP_EXPIRATION_DATE`         | date     | Expiration date of the primary service instance |
| `PP_PREMIUM_FLAG`            | string   | Indicates if the domain is premium (Y/N) |
| `PP_BUNDLE_FLAG`             | string   | Indicates if the service is part of a bundle (Y/N) |
| `PP_DISCOUNT_AMT`            | float    | Final amount paid after all discounts |
| `PP_NET_AMT`                 | float    | Net amount (difference between gross and net) |
| `TERM_IN_MONTHS`             | int      | Term for which the product is billed |
| `RENEWAL_COUNT`              | int      | Total count of renewals and acquisitions |
| `ALL_PRODUCT_CNT`            | int      | Count of all service transactions for primary service instance |
| `ALL_PRODUCT_NET_AMT`        | float    | Sum of net amount for all service transactions |
| `ECOMMERC_PRODUCT_CNT`       | int      | Count of ecommerce products |
| `ECOMMERC_NET_AMT`           | float    | Net amount for ecommerce products |
| `ADV_HOSTING_PRODUCT_CNT`    | int      | Count of advanced hosting products |
| `ADV_HOSTING_NET_AMT`        | float    | Net amount for advanced hosting products |
| `HOSTING_PRODUCT_CNT`        | int      | Count of shared hosting products |
| `HOSTING_NET_AMT`            | float    | Net amount for shared hosting products |
| `DIY_WEBSIT_PRODUCT_CNT`     | int      | Count of DIY website products |
| `DIY_WEBSIT_NET_AMT`         | float    | Net amount for DIY website products |
| `PRO_SERVICES_PRODUCT_CNT`   | int      | Count of professional services products |
| `PRO_SERVICES_NET_AMT`       | float    | Net amount for professional services products |
| `HOSTING_ADD_ONS_PRODUCT_CNT`| int      | Count of hosting add-ons/marketing products |
| `HOSTING_ADD_ONS_NET_AMT`    | float    | Net amount for hosting add-ons/marketing products |
| `EMAIL_PRODUCTIVITY_PRODUCT_CNT` | int  | Count of email and productivity products |
| `EMAIL_PRODUCTIVITY_NET_AMT` | float    | Net amount for email and productivity products |
| `SECURITY_PRODUCT_CNT`       | int      | Count of security and backup products |
| `SECURITY_NET_AMT`           | float    | Net amount for security and backup products |
| `PREMIUM_DOMAIN_PRODUCT_CNT` | int      | Count of premium domain products |
| `PREMIUM_DOMAIN_NET_AMT`     | float    | Net amount for premium domain products |
| `DOMAIN_VAS_PRODUCT_CNT`     | int      | Count of domain VAS products |
| `DOMAIN_VAS_NET_AMT`         | float    | Net amount for domain VAS products |
| `DOMAIN_PRODUCT_CNT`         | int      | Count of domain products |
| `STATE`                      | string   | Customer state/status |
| `HAS_ECOMMERCE`              | string   | Has ecommerce product (Y/N) |
| `HAS_WORDPRESS`              | string   | Has WordPress product (Y/N) |
| `NPS_PROMOTER_COUNT`         | int      | Net Promoter Score - Promoter count |
| `NPS_DETRACTOR_COUNT`        | int      | Net Promoter Score - Detractor count |
| `TOTAL_CONTACTS`             | int      | Total support contacts |
| `WORDPRESS_CONTACTS`         | int      | WordPress support contacts |
| `DOMAIN_CONTACTS`            | int      | Domain support contacts |
| `EMAIL_CONTACTS`             | int      | Email support contacts |
| `CPANEL_CONTACTS`            | int      | cPanel support contacts |
| `ACCOUNT_CONTACTS`           | int      | Account support contacts |
| `BILLING_CONTACTS`           | int      | Billing support contacts |
| `RETENTION_CONTACTS`         | int      | Retention support contacts |
| `SALES_CONTACTS`             | int      | Sales support contacts |
| `SSL_CONTACTS`               | int      | SSL support contacts |
| `TOS_CONTACTS`               | int      | Terms of Service support contacts |
| `SUCCESS_LOGIN`              | int      | Successful login count |
| `TOTAL_LOGIN`                | int      | Total login attempts |


---

## Churn Logic

Use a rule-based scoring function to determine if a customer should churn, based on realistic patterns observed in hosting/domain service businesses.

### Score Calculation

| Feature                       | Condition                        | Score Impact |
|------------------------------|----------------------------------|--------------|
| **Service Age**              | Days since `PP_SERVICE_CREATED_DATE` < 90 | +2 |
| **Service Expiring Soon**    | Days to `PP_EXPIRATION_DATE` < 30 | +3 |
| **Low Product Diversity**    | `ALL_PRODUCT_CNT` < 2            | +1 |
| **Low Spend**                | `ALL_PRODUCT_NET_AMT` < $50      | +2 |
| **No Renewals**              | `RENEWAL_COUNT` = 0              | +2 |
| **High Support Contact**     | `TOTAL_CONTACTS` > 5             | +1 |
| **Billing Issues**           | `BILLING_CONTACTS` > 2           | +2 |
| **Retention Calls**          | `RETENTION_CONTACTS` > 0         | +1 |
| **Low Login Activity**       | `SUCCESS_LOGIN` = 0 OR `TOTAL_LOGIN` < 5 | +2 |
| **NPS Detractor**            | `NPS_DETRACTOR_COUNT` > 0        | +1 |
| **Premium Domain Issues**    | `PP_PREMIUM_FLAG` = 'Y' AND `PREMIUM_DOMAIN_PRODUCT_CNT` = 0 | +1 |
| **Short Term Contract**      | `TERM_IN_MONTHS` <= 3            | +1 |
| **High Discount Dependency** | `PP_DISCOUNT_AMT` / `PP_NET_AMT` > 0.5 | +1 |
| **No Core Products**         | `HOSTING_PRODUCT_CNT` = 0 AND `DOMAIN_PRODUCT_CNT` = 0 | +2 |
| **Technical Support Issues** | `CPANEL_CONTACTS` + `EMAIL_CONTACTS` > 3 | +1 |

### Churn Assignment

- Churn scoring logic implemented (target column removed from dataset)
- Else → `'active'`
- Add ~10% noise by randomly flipping the status to simulate real-world unpredictability

### Additional Logic

**Automatic Active Status (Override):**
- `NPS_PROMOTER_COUNT` > 0 AND `ALL_PRODUCT_NET_AMT` > $100 → Force `'active'`
- `RENEWAL_COUNT` > 3 AND `ALL_PRODUCT_CNT` > 5 → Force `'active'`

**Automatic Inactive Status (Override):**
- `PP_SERVICE_DELETION_DATE` is not null → Force `'inactive'`
- `BILLING_CONTACTS` > 5 → Force `'inactive'`

---

## Functional Requirements
- Must generate data as a single Python script or notebook
- Columns must match the updated schema above (50+ columns)
- Churn logic must be centralized and easily modifiable
- Generate realistic date ranges for service lifecycle
- Create correlated product counts and revenue amounts
- Support contact data should follow realistic patterns
- Login activity should correlate with engagement
- Save final output as CSV with proper data types
- Print summary statistics and churn distribution after generation

---

## Non-Functional Requirements
- Fast runtime: < 10 seconds for 1000 rows
- Deterministic output with seed
- Dependencies: `pandas`, `numpy`, `faker`, and `datetime`
- Memory efficient for large datasets (up to 100K+ rows)

---

## Future Considerations
- CLI support (e.g., `python generate.py --rows 5000`)
- Add user segments or plan types
- Output in multiple formats (JSON, DB insert, etc.)
- Include behavior drift (churn logic changing over time)

