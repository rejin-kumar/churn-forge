# 🔧 Configuration Guide: Synthetic Dataset Generator

## Overview

This configuration system allows you to create synthetic datasets for any domain by defining your schema, generation rules, and business logic in JSON configuration files. No more hardcoded schemas!

## Quick Start

1. **Choose a starting point:**
   - Use `template_config.json` for a simple, generic dataset
   - Use `template_config.json` as a starting template for any domain
   - Create your own based on `config_schema.json`

2. **Customize your config:**
   - Define your columns in the `schema.columns` section
   - Set up relationships between columns
   - Configure your target variable logic
   - Adjust generation parameters

3. **Generate your dataset:**
   ```bash
   python generate_dataset.py --config my_config.json
   ```

---

## Configuration Structure

### 🔧 Generation Parameters
```json
"generation_params": {
  "n_records": 1000,           // Number of records to generate
  "random_seed": 42,           // For reproducible results
  "reference_date": "2024-12-01", // Base date for calculations
  "output_filename": "data.csv"    // Output file name
}
```

### 📊 Schema Definition

Each column in your dataset is defined with:

```json
{
  "name": "column_name",
  "type": "int|float|string|date|boolean",
  "description": "What this column represents",
  "generation_method": "random|calculated|lookup|correlated|sequential",
  "required": true,
  "nullable": false,
  "generation_params": {
    // Method-specific parameters
  }
}
```

#### Generation Methods

| Method | Description | Example Use Case |
|--------|-------------|------------------|
| `random` | Generate random values within constraints | Age, income, ratings |
| `sequential` | Auto-incrementing values | IDs, account numbers |
| `calculated` | Derive from other columns using formulas | Total spend = monthly_spend × months |
| `correlated` | Generate values correlated with other columns | Satisfaction score ↔ Spend amount |
| `lookup` | Pick from predefined choices with probabilities | Categories, segments, status |

#### Generation Parameters by Type

**Integer/Float:**
```json
"generation_params": {
  "min": 1,
  "max": 100,
  "distribution": "normal|uniform|exponential|beta"
}
```

**String:**
```json
"generation_params": {
  "choices": ["A", "B", "C"],
  "probabilities": [0.5, 0.3, 0.2],  // Optional
  "pattern": "regex_pattern"          // Optional
}
```

**Date:**
```json
"generation_params": {
  "date_range": {
    "start": "2020-01-01",
    "end": "2024-12-01"
  },
  "distribution": "uniform"
}
```

**Calculated:**
```json
"generation_params": {
  "formula": "column1 * column2 + random(10, 50)"
}
```

**Correlated:**
```json
"generation_params": {
  "correlation_target": "other_column",
  "correlation_strength": 0.7,
  "formula": "if other_column > 100 then random(80, 100) else random(20, 60)"
}
```

### 🔗 Relationships

Define how columns relate to each other:

```json
"relationships": [
  {
    "type": "correlation",
    "columns": ["spend", "satisfaction"],
    "rule": "positive correlation",
    "strength": 0.8
  },
  {
    "type": "constraint", 
    "columns": ["total_logins", "successful_logins"],
    "rule": "total_logins >= successful_logins",
    "strength": 1.0
  },
  {
    "type": "dependency",
    "columns": ["end_date", "start_date", "duration"],
    "rule": "end_date = start_date + duration",
    "strength": 1.0
  }
]
```

### 🎯 Target Logic

Configure your target variable (churn, conversion, etc.):

```json
"target_logic": {
  "target_column": "risk_score",
  "scoring_rules": [
    {
      "name": "low_engagement",
      "condition": "login_count < 5",
      "score_impact": 2,
      "description": "User rarely logs in"
    }
  ],
  "override_rules": [
    {
      "type": "force_negative",
      "condition": "premium_user == true and satisfaction > 8",
      "description": "Happy premium users are safe"
    }
  ],
  "noise_percentage": 0.1,
  "threshold": {
    "high_risk": 6,
    "medium_risk": 3, 
    "low_risk": 0
  }
}
```

#### Scoring Rules

Each rule adds points to a customer's risk score:

```json
{
  "name": "descriptive_name",
  "condition": "column_name operator value",
  "score_impact": 1,  // Points added if condition is true
  "description": "Human readable explanation"
}
```

**Supported Operators:**
- `<`, `>`, `<=`, `>=`, `==`, `!=`
- `is null`, `is not null`
- `in [list]`, `not in [list]`
- Logical: `and`, `or`

**Examples:**
```json
"condition": "age < 25"
"condition": "status == 'inactive'"
"condition": "last_login is null"
"condition": "spend < 50 and support_tickets > 3"
"condition": "segment in ['basic', 'trial']"
```

#### Override Rules

Force specific outcomes regardless of score:

```json
{
  "type": "force_positive|force_negative",
  "condition": "complex_condition",
  "description": "Why this override exists"
}
```

### 📈 Data Quality

Add realistic imperfections:

```json
"data_quality": {
  "missing_data_percentage": 0.05,  // 5% missing values
  "outlier_percentage": 0.02,       // 2% outliers
  "duplicate_percentage": 0.01      // 1% duplicates
}
```

---

## Formula Language

Use simple expressions in calculated fields:

### Built-in Functions
- `random(min, max)` - Random number in range
- `choice(['A', 'B', 'C'])` - Random choice from list
- `days_since(date_column)` - Days between date and reference_date
- `days_until(date_column)` - Days from reference_date to date
- `months_since(date_column)` - Months between dates
- `if condition then value1 else value2` - Conditional logic

### Examples
```json
"formula": "monthly_spend * months_since(signup_date)"
"formula": "if premium_flag == 'Y' then random(100, 500) else random(10, 50)"
"formula": "base_price + (addon_count * random(10, 30))"
```

---

## Example Use Cases

### 🛒 E-commerce Dataset
```json
{
  "name": "purchase_amount",
  "type": "float", 
  "generation_method": "correlated",
  "generation_params": {
    "correlation_target": "customer_tier",
    "formula": "if customer_tier == 'gold' then random(100, 500) else random(20, 100)"
  }
}
```

### 🏦 Financial Services
```json
{
  "name": "credit_score",
  "type": "int",
  "generation_method": "correlated", 
  "generation_params": {
    "correlation_target": "default_risk",
    "formula": "if default_risk == 'high' then random(300, 600) else random(650, 850)"
  }
}
```

### 📱 SaaS Product
```json
{
  "name": "feature_usage_count",
  "type": "int",
  "generation_method": "correlated",
  "generation_params": {
    "correlation_target": "plan_type",
    "formula": "if plan_type == 'enterprise' then random(50, 200) else random(5, 30)"
  }
}
```

---

## Best Practices

### 🎯 Schema Design
1. **Start simple** - Begin with core columns, add complexity gradually
2. **Use realistic distributions** - `exponential` for counts, `normal` for ratings
3. **Define relationships early** - Plan dependencies before implementation
4. **Add correlation carefully** - Too much correlation reduces data realism

### 🔧 Generation Methods
1. **Sequential IDs** - Always use for primary keys
2. **Calculated totals** - Derive sums/aggregations from component parts
3. **Correlated behavior** - Link satisfaction scores to spending patterns
4. **Realistic probabilities** - Use actual business knowledge for choice distributions

### 🎯 Target Logic
1. **Domain expertise** - Base scoring rules on real business insights  
2. **Balanced scoring** - Avoid rules that apply to too many/few records
3. **Test thresholds** - Validate that your risk buckets have reasonable distributions
4. **Add noise** - 5-15% noise makes predictions more realistic

### 🚀 Performance
1. **Batch relationships** - Process correlations efficiently
2. **Limit calculated fields** - Too many formulas slow generation
3. **Use appropriate distributions** - Some are faster than others
4. **Test with small samples** - Validate logic before generating large datasets

---

## Common Patterns

### Customer Lifecycle
```json
"columns": [
  {"name": "signup_date", "generation_method": "random"},
  {"name": "first_purchase_date", "generation_method": "calculated", 
   "generation_params": {"formula": "signup_date + random(1, 30)"}},
  {"name": "days_active", "generation_method": "calculated",
   "generation_params": {"formula": "days_since(signup_date)"}}
]
```

### Product Portfolio
```json
"columns": [
  {"name": "product_count", "generation_method": "random"},
  {"name": "total_value", "generation_method": "calculated",
   "generation_params": {"formula": "product_count * random(25, 100)"}}
]
```

### Support Interactions
```json
"columns": [
  {"name": "total_tickets", "generation_method": "random"},
  {"name": "resolved_tickets", "generation_method": "calculated",
   "generation_params": {"formula": "total_tickets * random(0.7, 0.95)"}}
]
```

---

## Troubleshooting

### Common Issues

**"Column not found" errors:**
- Ensure calculated columns reference existing columns
- Check column name spelling and case sensitivity

**Unrealistic correlations:**
- Verify correlation_strength values (0.0 to 1.0)
- Test with small datasets first

**Slow generation:**
- Reduce number of calculated columns
- Simplify complex formulas
- Use simpler distributions for large datasets

**Poor target variable distribution:**
- Adjust scoring rule weights
- Modify threshold values
- Add more diverse scoring conditions

---

Ready to create your own dataset? Start with `template_config.json` and customize it for your domain! 