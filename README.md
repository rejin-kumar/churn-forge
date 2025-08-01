# 📊 Customer Churn Dataset Generator

A synthetic dataset generator for customer churn prediction training data. Creates realistic customer behavior data for hosting/domain service businesses with pattern-based churn labels.

## ✨ Features

- **Pattern-based churn logic** (not random assignment)
- **Realistic data distributions** using appropriate statistical models
- **Fast generation** (<2 seconds for 100 records)
- **Configurable parameters** for easy customization
- **Reproducible results** with optional random seed
- **CLI support** for batch generation

## 📋 Generated Schema

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | UUID |
| `customer_name` | string | Generated with Faker |
| `customer_email` | string | Generated with Faker |
| `support_tickets` | int | 0–10 |
| `avg_resolution_time` | float | Hours |
| `critical_tickets_sla_breach` | int | 0–4 |
| `product_renewals` | int | 0–5 |
| `tenure_months` | int | 1–48 |
| `monthly_spend` | float | USD 5–150 |
| `last_login` | date | YYYY-MM-DD |
| `total_products` | int | 1–10 |
| `products_transferred_out` | int | 0–3 |
| `avg_load_time` | float | Seconds |
| `downtime_minutes` | int | 0–300 |
| `product_usage_percent` | float | 0–100% |
| **`customer_status`** | string | **`'active'` or `'inactive'`** |

## 🚀 Quick Start

### Option 1: Python Script

```bash
# Install dependencies
pip install pandas numpy faker

# Generate 100 records with default settings
python3 generate_churn_dataset.py

# Generate custom dataset
python3 generate_churn_dataset.py --rows 1000 --seed 42 --output my_churn_data.csv
```

### Option 2: Jupyter Notebook

1. Open `churn_dataset_generator.ipynb`
2. Modify configuration in the second cell if needed
3. Run all cells to generate your dataset

### CLI Options

```bash
python3 generate_churn_dataset.py [OPTIONS]

Options:
  --rows, -n     Number of records to generate (default: 100)
  --seed, -s     Random seed for reproducible results
  --output, -o   Output filename (default: churn_dataset_realistic.csv)
```

## 🎯 Churn Logic

The generator uses a rule-based scoring system with 11 behavioral indicators:

| Condition | Score Impact |
|-----------|--------------|
| Support tickets > 4 | +1 |
| Avg resolution time > 48h | +1 |
| Critical SLA breaches > 1 | +1 |
| Product renewals < 2 | +1 |
| Tenure < 6 months | +1 |
| Monthly spend < $20 | +1 |
| Last login > 45 days ago | +1 |
| Products transferred out > 1 | +1 |
| Avg load time > 5s | +1 |
| Downtime > 120 minutes | +1 |
| Product usage < 30% | +1 |

**Churn Assignment:**
- Score ≥ 5 → `customer_status = 'inactive'`
- Score < 5 → `customer_status = 'active'`
- 10% random noise added for realism

## 🔧 Customization

### Modify Churn Logic

Edit the `calculate_churn_score()` function to change scoring rules:

```python
# Example: More aggressive churn threshold
base_status = ['inactive' if score >= 3 else 'active' for score in scores]

# Example: Add new scoring rule
('High support load', (df['support_tickets'] > 2) & (df['avg_resolution_time'] > 24))
```

### Adjust Data Distributions

Modify generation functions to change customer profiles:

```python
# Example: Higher spending customers
monthly_spend = np.random.lognormal(mean=4.0, sigma=0.6, size=n_customers)

# Example: Different tenure distribution
tenure_months = np.random.gamma(shape=3, scale=12, size=n_customers)
```

## 📊 Sample Output

```
Generating 100 customer records...

🎯 Churn Scoring Rules Applied:
   Support tickets > 4: 8 customers affected
   Avg resolution time > 48h: 23 customers affected
   Critical SLA breaches > 1: 12 customers affected
   Product renewals < 2: 31 customers affected
   Tenure < 6 months: 22 customers affected
   Monthly spend < $20: 28 customers affected
   Last login > 45 days ago: 19 customers affected
   Products transferred out > 1: 7 customers affected
   Avg load time > 5s: 15 customers affected
   Downtime > 120 min: 18 customers affected
   Product usage < 30%: 26 customers affected

📊 Churn Assignment Summary:
   Base rule assignment: 12 inactive / 100 total
   After 10% noise: 11 inactive / 100 total

✅ Dataset saved to: churn_dataset_realistic.csv
📊 Dataset shape: (100, 16)
📈 Churn rate: 11.0% (11 inactive / 100 total)
⚡ Generation time: 0.09 seconds
```

## 📁 Files

- **`generate_churn_dataset.py`** - Main Python script with CLI support
- **`churn_dataset_generator.ipynb`** - Interactive Jupyter notebook
- **`requirements.txt`** - Python dependencies
- **`churn_dataset_realistic.csv`** - Generated sample dataset

## 🎮 Use Cases

- Prototyping churn prediction models
- Testing ML pipelines and feature engineering
- Creating training datasets for experimentation
- Simulating user behavior analysis
- Dashboard and analytics testing

## 📝 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0
- faker >= 15.0.0

---

Generated synthetic data is perfect for ML model development, testing, and experimentation without privacy concerns! 🚀 