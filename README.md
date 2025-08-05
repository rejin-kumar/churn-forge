# 🚀 ChurnForge: Universal Synthetic Dataset Generator

A powerful, **configuration-driven** synthetic dataset generator for any domain. Create realistic customer behavior data with sophisticated business logic, temporal patterns, and bulk generation capabilities - all through simple JSON configuration files.

## ✨ Key Features

### 🎯 **Domain-Agnostic Configuration System**
- **Any Business Domain**: E-commerce, SaaS, Finance, Hosting, Retail, etc.
- **Flexible Schema**: Define unlimited columns with custom types and generation methods
- **Business Logic Engine**: Configurable scoring rules, correlations, and relationships
- **No Code Required**: Everything defined through JSON configuration

### 🔧 **Advanced Generation Capabilities**
- **Multiple Generation Methods**: Random, calculated, correlated, sequential, lookup
- **Statistical Distributions**: Normal, exponential, uniform, beta
- **Realistic Relationships**: Cross-column correlations and dependencies
- **Data Quality Control**: Missing values, outliers, duplicates

### ⚡ **Production-Ready Performance**
- **Fast Generation**: <2 seconds for 1,000 records
- **Batch Processing**: Generate millions of records across multiple months
- **Temporal Evolution**: Customer lifecycle simulation with churn patterns
- **CLI Support**: Full command-line interface with parameter overrides

## 🚀 Quick Start

### 1. **Single Dataset Generation**

```bash
# Use any pre-built configuration
python3 config_dataset_generator.py --config template_config.json --rows 1000

# Use simple template for any domain
python3 config_dataset_generator.py --config template_config.json --rows 5000 --seed 42
```

### 2. **Temporal Customer Lifecycle**

```bash
# Generate 12 months of evolving customer data
python3 config_temporal_generator.py --config temporal_config.json --months 12

# Custom temporal generation
python3 config_temporal_generator.py --config template_config.json --initial-customers 1000 --monthly-new 50
```

### 3. **Batch Monthly Generation**

```bash
# Generate multiple monthly datasets
python3 config_batch_generator.py --config batch_config.json --rows 100000 --num-months 6

# Production-scale generation
python3 config_batch_generator.py --config template_config.json --rows 1000000 --num-months 13
```

## 📊 Pre-Built Configurations

| Configuration | Domain | Features | Use Case |
|---------------|--------|----------|----------|
| `template_config.json` | Generic | Simple schema, basic churn logic | Quick start, any domain |
| `template_config.json` | Generic | Simple template for any domain | Universal template |
| `temporal_config.json` | Customer Lifecycle | Behavioral evolution, lifecycle stages | Customer journey analysis |
| `batch_config.json` | Batch Processing | Optimized for large datasets | Production data generation |

## 🔧 Creating Your Own Configuration

### Simple Example (E-commerce)
```json
{
  "dataset_config": {
    "name": "E-commerce Customer Dataset",
    "schema": {
      "columns": [
        {
          "name": "customer_id",
          "type": "int",
          "generation_method": "sequential"
        },
        {
          "name": "purchase_amount",
          "type": "float",
          "generation_method": "random",
          "generation_params": {
            "min": 10.0,
            "max": 500.0,
            "distribution": "exponential"
          }
        },
        {
          "name": "customer_segment",
          "type": "string", 
          "generation_method": "random",
          "generation_params": {
            "choices": ["premium", "standard", "basic"],
            "probabilities": [0.2, 0.5, 0.3]
          }
        }
      ]
    },
    "target_logic": {
      "scoring_rules": [
        {
          "name": "low_spend",
          "condition": "purchase_amount < 50",
          "score_impact": 2,
          "description": "Low spending indicates churn risk"
        }
      ]
    }
  }
}
```

📖 **Complete Guide**: See `CONFIG_GUIDE.md` for comprehensive documentation.

## 🎯 Advanced Features

### **1. Business Logic Engine**
```json
"target_logic": {
  "scoring_rules": [
    {"condition": "monthly_spend < 25", "score_impact": 2},
    {"condition": "support_tickets > 5", "score_impact": 1},
    {"condition": "satisfaction_score < 5", "score_impact": 3}
  ],
  "override_rules": [
    {
      "type": "force_negative", 
      "condition": "customer_segment == 'premium' and satisfaction_score > 8"
    }
  ]
}
```

### **2. Column Relationships**
```json
"relationships": [
  {
    "type": "correlation",
    "columns": ["spending", "satisfaction"],
    "rule": "positive correlation",
    "strength": 0.8
  }
]
```

### **3. Temporal Customer Evolution**
```json
"behavioral_evolution": {
  "login_activity": {
    "degradation_rate": 0.15,
    "improvement_rate": 0.05
  },
  "spending_patterns": {
    "churn_reduction_rate": 0.25,
    "loyalty_increase_rate": 0.1
  }
}
```

## 📊 Sample Output

```
📊 Dataset Summary: E-commerce Customer Dataset
============================================================
Records: 1,000
Features: 15
Memory usage: 0.12 MB

🎯 Target Variable: churn_risk_score
count    1000.000000
mean        2.340000
std         1.920000
min         0.000000
max         8.000000

📈 Risk Distribution:
High Risk: 120 (12.0%)
Medium Risk: 380 (38.0%) 
Low Risk: 500 (50.0%)

✅ Generation complete! File: e_commerce_dataset.csv
⚡ Generation time: 1.24 seconds
```

## 🎮 Use Cases

### **ML & AI Training**
- Churn prediction model development
- Customer segmentation analysis
- Feature engineering experimentation
- A/B testing simulation

### **Business Intelligence** 
- Dashboard and analytics testing
- Customer journey mapping
- Retention strategy modeling
- Revenue forecasting

### **Data Engineering**
- Pipeline testing and validation
- ETL process development
- Data quality assessment
- Performance benchmarking

## 📁 File Structure

### **Core Generators**
- **`config_dataset_generator.py`** - Single dataset generation from config
- **`config_temporal_generator.py`** - Temporal customer lifecycle simulation
- **`config_batch_generator.py`** - Batch monthly dataset generation

### **Configuration Files**
- **`template_config.json`** - Simple template for any domain
- **`temporal_config.json`** - Customer lifecycle configuration
- **`batch_config.json`** - Batch processing configuration
- **`config_schema.json`** - Master configuration schema

### **Documentation**
- **`CONFIG_GUIDE.md`** - Complete configuration guide with examples
- **`TEMPORAL_README.md`** - Temporal generation documentation
- **`BATCH_README.md`** - Batch processing documentation

### **Legacy (Backward Compatibility)**
- **`template_config.json`** - Generic template for any domain
- **`PRD.md`** - Original product requirements document

## 🔗 CLI Options

### **Single Dataset**
```bash
python3 config_dataset_generator.py --config CONFIG_FILE [OPTIONS]

Options:
  --config, -c     Configuration file path (required)
  --rows, -n       Number of records (overrides config)
  --seed, -s       Random seed (overrides config)
  --output, -o     Output filename (overrides config)
```

### **Temporal Generation**
```bash
python3 config_temporal_generator.py --config CONFIG_FILE [OPTIONS]

Options:
  --config, -c          Configuration file path
  --months, -m          Number of months to generate
  --initial-customers   Starting customer count
  --monthly-new         New customers per month
  --output-dir, -o      Output directory
```

### **Batch Generation**
```bash
python3 config_batch_generator.py --config CONFIG_FILE [OPTIONS]

Options:
  --config, -c       Configuration file path (required)
  --rows, -r         Rows per dataset
  --num-months, -n   Number of months
  --start-month      Starting month (YYYY-MM)
  --verify, -v       Verify generated datasets
  --summary          Generate summary report
```

## 📝 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0  
- faker >= 15.0.0
- python-dateutil >= 2.8.0

## 🚀 Migration from Legacy System

### **Single Dataset Generation**
```bash
# New way (configurable)
python3 config_dataset_generator.py --config template_config.json --rows 1000
```

### **Temporal Generation**
```bash
# New way (configurable)  
python3 config_temporal_generator.py --config temporal_config.json
```

### **Batch Generation**
```bash
# New way (configurable)
python3 config_batch_generator.py --config batch_config.json --rows 1000000
```

---

## 🌟 Why ChurnForge?

✅ **Domain-Agnostic**: Works for any business domain  
✅ **No Coding Required**: Everything configured through JSON  
✅ **Production-Ready**: Handles millions of records efficiently  
✅ **Realistic Data**: Sophisticated business logic and relationships  
✅ **ML-Optimized**: Perfect for training and testing AI models  
✅ **Open Source**: MIT licensed, free for commercial use  

Transform your data science workflow with realistic, configurable synthetic datasets! 🚀 