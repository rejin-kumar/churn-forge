# 🔄 Batch Churn Dataset Generator

This system generates synthetic churn datasets for **13 months** (July 2024 to July 2025), with each dataset containing **1 million customer records**.

## 📁 Files

| File | Description |
|------|-------------|
| `batch_generate_datasets.py` | Main batch generation script |
| `batch_usage_example.py` | Usage examples and documentation |
| `generate_churn_dataset.py` | Core dataset generator (existing) |
| `PRD.md` | Product requirements and schema |

## 🚀 Quick Start

### Basic Usage
```bash
# Generate all 13 datasets with default settings
python batch_generate_datasets.py
```

### Custom Usage
```bash
# Custom row count and output directory
python batch_generate_datasets.py --rows 500000 --output-dir my_datasets

# With specific seed for reproducibility
python batch_generate_datasets.py --seed 123 --verify
```

## 📊 What Gets Generated

### Datasets
- **13 monthly datasets** (July 2024 → July 2025)
- **1,000,000 rows** per dataset (configurable)
- **50+ columns** per dataset (follows PRD schema)
- **~230 MB** per CSV file
- **~3 GB** total data

### Output Files
```
datasets/
├── churn_dataset_202407.csv  # July 2024
├── churn_dataset_202408.csv  # August 2024
├── churn_dataset_202409.csv  # September 2024
├── churn_dataset_202410.csv  # October 2024
├── churn_dataset_202411.csv  # November 2024
├── churn_dataset_202412.csv  # December 2024
├── churn_dataset_202501.csv  # January 2025
├── churn_dataset_202502.csv  # February 2025
├── churn_dataset_202503.csv  # March 2025
├── churn_dataset_202504.csv  # April 2025
├── churn_dataset_202505.csv  # May 2025
├── churn_dataset_202506.csv  # June 2025
├── churn_dataset_202507.csv  # July 2025
└── batch_generation.log      # Generation log
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