# 🔄 Batch Churn Dataset Generator

This system generates synthetic churn datasets for **multiple months** with **configurable record counts** per dataset, supporting any business domain through JSON configuration files.

## ✨ NEW: Configuration-Based System

🚀 **Now supports any domain through JSON configuration!** The batch generator has been upgraded to use the flexible configuration system:

### **New Config-Based Usage**
```bash
# Generate batch datasets for any domain
python3 config_batch_generator.py --config batch_config.json --rows 100000 --num-months 6

# Use hosting domain configuration for batch generation
python3 config_batch_generator.py --config template_config.json --rows 1000000 --num-months 13

# Create custom domain-specific batch datasets
python3 config_batch_generator.py --config my_ecommerce_config.json --rows 500000 --verify
```

### **Advanced Batch Configuration**
The new system supports sophisticated batch generation through configuration:

```json
{
  "dataset_config": {
    "name": "E-commerce Batch Dataset",
    "generation_params": {
      "n_records": 1000000,
      "random_seed": 42,
      "output_filename": "ecommerce_batch.csv"
    },
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
            "max": 1000.0,
            "distribution": "exponential"
          }
        }
      ]
    },
    "target_logic": {
      "scoring_rules": [
        {
          "condition": "purchase_amount < 50",
          "score_impact": 2
        }
      ]
    }
  }
}
```

### **Benefits of Config-Based System**
✅ **Domain-Agnostic**: Works for any business (e-commerce, SaaS, finance, etc.)  
✅ **No Coding Required**: Everything configured through JSON  
✅ **Flexible Schema**: Define unlimited columns with custom generation rules  
✅ **Business Logic**: Configurable scoring rules and relationships  
✅ **Verification Built-In**: Automatic quality checks and reporting  

📖 **Complete Configuration Guide**: See `CONFIG_GUIDE.md` for details on creating custom batch configurations.

---

## 📁 Files

| File | Description | Status |
|------|-------------|--------|
| `config_batch_generator.py` | **NEW** - Config-based batch generation script | ✅ Recommended |
| `batch_config.json` | **NEW** - Batch configuration template | ✅ Recommended |
| `batch_usage_example.py` | Usage examples and documentation | ✅ Updated |
| `template_config.json` | Generic template for any domain | 📝 Template |
| `PRD.md` | Product requirements and schema | 📖 Reference |

## 🚀 Quick Start

### **Recommended: Configuration-Based Usage**
```bash
# Generate batch datasets with any configuration
python3 config_batch_generator.py --config template_config.json --rows 100000 --num-months 3

# Production-scale batch generation
python3 config_batch_generator.py --config template_config.json --rows 1000000 --num-months 13 --verify

# Custom configuration with summary report
python3 config_batch_generator.py --config batch_config.json --rows 500000 --num-months 6 --summary
```

### **Alternative: Using Different Configurations**
```bash
# Generate with hosting domain configuration
python3 config_batch_generator.py --config template_config.json --rows 500000 --num-months 6

# Generate with temporal lifecycle patterns
python3 config_batch_generator.py --config temporal_config.json --rows 100000 --num-months 12

# Generate with custom configuration
python3 config_batch_generator.py --config my_custom_config.json --rows 250000 --verify
```

## 📊 What Gets Generated

### **Config-Based Output (New)**
- **Configurable months** (any period you specify)
- **Configurable rows** per dataset
- **Unlimited columns** (defined in your config)
- **Custom filename patterns** based on dataset name
- **Domain-specific business logic** and relationships
- **Automatic verification** and quality reports

### **Sample Config-Based Output Files**
```
batch_datasets/
├── my_ecommerce_dataset_202407.csv   # July 2024
├── my_ecommerce_dataset_202408.csv   # August 2024
├── my_ecommerce_dataset_202409.csv   # September 2024
├── batch_summary_report.csv          # Analytics summary
└── batch_generation.log              # Generation logs
```



## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--rows, -r` | Rows per dataset | 1,000,000 |
| `--seed, -s` | Random seed | 42 |
| `--output-dir, -o` | Output directory | `datasets` |
| `--verify, -v` | Force verification | Auto |
| `--no-verify` | Skip verification | - |

## 📋 Features

### 🎯 Realistic Data Generation
- **Behavioral patterns** drive churn scoring
- **Correlated metrics** (spend vs. products, contacts vs. issues)
- **Realistic date ranges** and service lifecycles
- **Industry-specific** hosting/domain business logic

### 🔄 Batch Processing
- **Monthly datasets** with proper reference dates
- **Unique seeds** per month for variation
- **Progress tracking** and detailed logging
- **Error handling** (continues if one month fails)
- **Automatic verification** of generated files

### 📈 Monitoring
- **Real-time progress** updates
- **Generation timing** and performance metrics
- **File size** and row count verification
- **Summary statistics** after completion

## 🕐 Performance

### Estimated Times
- **Per dataset**: ~2-3 minutes (1M rows)
- **Total time**: ~20-30 minutes (13 datasets)
- **Memory usage**: ~2-4 GB peak

### Optimization Tips
- Use SSD storage for faster I/O
- Close other applications during generation
- Monitor disk space (need ~4-5 GB free)

## 🔍 Verification

The system automatically verifies each generated dataset:

✅ **Row Count**: Confirms expected number of rows  
✅ **Column Count**: Validates schema completeness  
✅ **Data Quality**: Checks for null values  
✅ **File Size**: Estimates reasonable file sizes  

## 🛠 Programmatic Usage

```python
from batch_generate_datasets import BatchDatasetGenerator

# Create generator
generator = BatchDatasetGenerator(
    rows_per_month=1000000,
    random_seed=42,
    output_dir="datasets"
)

# Generate all datasets
files = generator.generate_all_datasets()

# Verify results
generator.verify_datasets(files)
```

## 📄 Dataset Schema

Each dataset follows the schema defined in `PRD.md`:

### Key Columns
- **Customer Info**: `ACCOUNT_ID`, `PERSON_ORG_ID`, `STATE`
- **Service Details**: `PP_SERVICE_CREATED_DATE`, `PP_EXPIRATION_DATE`
- **Financial**: `PP_NET_AMT`, `ALL_PRODUCT_NET_AMT`, product counts
- **Support**: `TOTAL_CONTACTS`, `BILLING_CONTACTS`, contact types
- **Engagement**: `SUCCESS_LOGIN`, `TOTAL_LOGIN`, `NPS_*_COUNT`

### Churn Logic
The data includes realistic churn patterns based on:
- Service age and expiration timing
- Product diversity and spending
- Support contact patterns  
- Login activity and engagement
- Billing issues and retention calls

## 🎨 Example Output

Run `python batch_usage_example.py` to see:
- Usage examples and commands
- Expected file structure
- Dataset schema details
- Performance estimates

## ⚡ Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas>=1.3.0`
- `numpy>=1.20.0`
- `faker>=15.0.0`
- `python-dateutil>=2.8.0`

## 🔧 Troubleshooting

### Common Issues
- **Memory errors**: Reduce `--rows` parameter
- **Disk space**: Ensure 4-5 GB free space
- **Import errors**: Check dependencies installation
- **Permission errors**: Ensure write access to output directory

### Log Files
Check `datasets/batch_generation.log` for detailed execution logs.

---

## 🎯 Ready to Generate?

```bash
# Start the batch generation
python batch_generate_datasets.py

# Watch the progress in real-time
tail -f datasets/batch_generation.log
```

**⚠️ Note**: Generation takes 20-30 minutes for 13M total rows. Plan accordingly! 