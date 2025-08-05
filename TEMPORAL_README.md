# 🕰️ Temporal Churn Dataset Generator

The **Temporal Churn Dataset Generator** creates realistic customer churn datasets that evolve over time, simulating a true customer lifecycle with behavioral patterns, churn events, and new customer acquisition.

## ✨ NEW: Configuration-Based System

🚀 **Now supports any domain through JSON configuration!** The temporal generator has been upgraded to use the flexible configuration system:

### **New Config-Based Usage**
```bash
# Use any configuration for temporal generation
python3 config_temporal_generator.py --config temporal_config.json --months 12

# Use hosting domain config with temporal evolution
python3 config_temporal_generator.py --config template_config.json --initial-customers 1000 --monthly-new 50

# Create your own domain-specific temporal config
python3 config_temporal_generator.py --config my_saas_temporal_config.json
```

### **Advanced Temporal Configuration**
The new system supports sophisticated temporal behavior through configuration:

```json
{
  "temporal_config": {
    "lifecycle_behavior": {
      "monthly_churn_rate": 0.025,
      "seasonal_effects": {
        "high_churn_months": [1, 6, 12],
        "low_churn_months": [3, 9]
      }
    },
    "behavioral_evolution": {
      "login_activity": {
        "degradation_rate": 0.15,
        "improvement_rate": 0.05
      },
      "spending_patterns": {
        "churn_reduction_rate": 0.25,
        "loyalty_increase_rate": 0.1
      }
    },
    "customer_segments": {
      "high_value": {"churn_resistance": 0.7},
      "at_risk": {"early_warning_months": 3},
      "loyal": {"churn_rate_reduction": 0.6}
    }
  }
}
```

📖 **Complete Configuration Guide**: See `CONFIG_GUIDE.md` for details on creating custom temporal configurations.

---

## 🎯 Key Features

### 🔄 Customer Lifecycle Management
- **Customer Evolution**: Existing customers' behavior changes month-to-month
- **Realistic Churn**: 2-3% monthly churn rate with observable patterns
- **New Acquisitions**: Fresh customers added each month
- **Behavioral Patterns**: Degrading metrics before churn (login activity, support contacts, product usage)

### 📊 Realistic Churn Behavior
- **Pre-churn Signals**: Decreasing logins, increasing support contacts, billing issues
- **Gradual Degradation**: Products and spending reduce over time for at-risk customers
- **Outlier Behaviors**: ~5% of customers have unique patterns (high spenders, support-heavy, etc.)
- **Retention Patterns**: Some customers show improvement and stay longer

## 🚀 Quick Start

### **Recommended: Configuration-Based Usage**
```bash
# Generate 12 months of temporal data with any domain
python3 config_temporal_generator.py --config temporal_config.json --months 12

# Custom parameters with configuration override
python3 config_temporal_generator.py --config template_config.json --initial-customers 500 --monthly-new 25 --months 6
```

### **Alternative: CLI with Different Configurations**
```bash
# Generate using hosting domain configuration
python3 config_temporal_generator.py --config template_config.json --months 6

# Generate using batch-optimized configuration
python3 config_temporal_generator.py --config batch_config.json --initial-customers 500 --monthly-new 25
```

### Programmatic Usage (Config-Based)
```python
from config_temporal_generator import ConfigTemporalGenerator

# Create generator with any configuration
generator = ConfigTemporalGenerator('temporal_config.json')

# Override settings if needed
generator.initial_customers = 1000
generator.monthly_new_customers = 50
generator.num_months = 12

# Generate datasets
files = generator.generate_all_datasets()
```

### Programmatic Usage (Alternative Config)
```python
from config_temporal_generator import ConfigTemporalGenerator

# Create generator with different configuration
generator = ConfigTemporalGenerator('template_config.json')

# Override settings for specific needs
generator.initial_customers = 1000
generator.monthly_new_customers = 50
generator.num_months = 12
generator.output_dir = "hosting_temporal_data"

# Generate datasets
files = generator.generate_all_datasets()
```

## 📋 Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--initial-customers` | Starting customer count | 1000 |
| `--monthly-new` | New customers per month | 50 |
| `--months` | Number of months to generate | 12 |
| `--seed` | Random seed for reproducibility | 42 |
| `--output-dir` | Output directory | `temporal_datasets` |

## 🔍 What Makes It Temporal?

### Traditional vs Temporal Generation

**Traditional (Static)**:
- Each month = independent 1M customers
- No customer persistence across months
- No behavioral evolution
- No realistic churn patterns

**Temporal (Dynamic)**:
- Customers tracked across multiple months
- Behavior evolves based on trends and risk factors
- Realistic churn with observable warning signs
- New customers replace churned ones

### Customer Evolution Examples

#### 📉 Customer Approaching Churn
```
Month 1: 50 logins, 2 support contacts, $120 spend
Month 2: 35 logins, 4 support contacts, $110 spend  
Month 3: 18 logins, 8 support contacts, $85 spend   ← Churn signals
Month 4: [CHURNED] - Customer disappears from dataset
```

#### 📈 Healthy Customer Evolution
```
Month 1: 25 logins, 1 support contact, $80 spend
Month 2: 28 logins, 0 support contacts, $95 spend
Month 3: 32 logins, 1 support contact, $110 spend   ← Growing customer
Month 4: 30 logins, 0 support contacts, $105 spend
```

## 📊 Generated Output Structure

### File Naming Convention
```
temporal_datasets/
├── 202407.csv  # July 2024 (1000 customers)
├── 202408.csv  # August 2024 (~1045 customers)
├── 202409.csv  # September 2024 (~1088 customers)
└── temporal_generation.log   # Detailed generation log
```

### Monthly Evolution Metrics
- **Total Customers**: Previous month + new - churned
- **Churn Rate**: Typically 2-3% monthly
- **New Customers**: Configurable (default: 50/month)
- **Customer Overlap**: ~97% month-to-month retention

## 🎭 Behavioral Patterns

### 1. Churn Risk Factors
The system tracks multiple risk indicators:

| Risk Factor | Pattern | Impact |
|-------------|---------|---------|
| **Login Decline** | 30%+ decrease in successful logins | High |
| **Support Spike** | Billing contacts > 2, retention calls | Very High |
| **Product Reduction** | Gradual decrease in service count | Medium |
| **Spend Decline** | Consistent decrease in monthly spend | Medium |
| **Service Age** | New customers (< 90 days) more likely to churn | Low |

### 2. Outlier Customer Types (~5% of base)

#### 🏆 High Spenders
- 5-20x normal spending amounts
- 10-25 products
- Lower churn risk
- Stable behavior over time

#### 📞 Support Heavy
- 15-30 total support contacts/month
- 3-8 billing contacts
- Higher churn risk
- May improve or churn quickly

#### 🔐 Login Addicts  
- 50-200 successful logins/month
- Very engaged users
- Very low churn risk
- Consistent behavior

#### 🛍️ Product Hoarders
- 15-35 total products
- Multiple service types
- Low churn risk
- Tend to add more products over time

### 3. Temporal Trends

#### Customer Behavior Evolution
```python
# Each customer has evolving trends:
login_trend: -1 (decreasing) to +1 (increasing)
support_trend: -0.5 to +1 (support contact changes)
spend_trend: -0.5 to +0.5 (spending changes)
product_trend: -0.3 to +0.3 (product count changes)
```

#### Churn Probability Calculation
```python
base_churn_rate = 2.5%  # Monthly base rate
risk_multiplier = 1 + (risk_trend × months_active × 0.1)

# Behavioral multipliers:
if login_trend < -0.5: risk_multiplier *= 1.5
if support_trend > 0.5: risk_multiplier *= 1.3
if spend_trend < -0.3: risk_multiplier *= 1.4
```

## 📈 Analysis Capabilities

### Run Analysis Examples
```bash
# See temporal patterns in action
python temporal_usage_example.py
```

### Key Metrics to Track
1. **Monthly Churn Rate**: Should stay around 2-3%
2. **Customer Lifecycle Length**: Average survival time
3. **Churn Signal Detection**: Behavioral changes before churn
4. **Customer Segment Analysis**: Different behavior patterns
5. **Retention Effectiveness**: Customers who recover from risk

### Sample Analysis Output
```
Month    Customers  New  Churned  Churn%  AvgLogin  AvgSupport  AvgSpend
202407   100        10   0        0.0     25.5      2.1         $85.30
202408   108        10   2        1.9     24.8      2.3         $87.50
202409   116        10   2        1.9     26.1      2.0         $89.20
202410   124        10   2        1.7     25.9      2.4         $88.10
```

## 🔬 Use Cases for AI Training

### 1. Churn Prediction Models
- **Multi-month features**: Customer behavior across time windows
- **Trend detection**: Rate of change in key metrics
- **Early warning**: Predict churn 1-3 months in advance
- **Feature engineering**: Time-series derived features

### 2. Customer Segmentation
- **Behavioral patterns**: Group customers by evolution patterns
- **Risk scoring**: Real-time churn probability
- **Lifecycle stages**: New, growing, stable, at-risk, churning

### 3. Intervention Modeling
- **Retention campaigns**: Which customers to target
- **Timing optimization**: When to intervene
- **Campaign effectiveness**: Simulate intervention outcomes

## ⚡ Performance & Scale

### Generation Speed
- **Small Dataset** (100 customers, 6 months): ~5 seconds
- **Medium Dataset** (1K customers, 12 months): ~30 seconds  
- **Large Dataset** (10K customers, 24 months): ~5 minutes

### Memory Usage
- **Customer State Tracking**: ~1KB per customer per month
- **Data Generation**: ~2-4GB peak for large datasets
- **Output Files**: ~200-300KB per 1000 customers per month

### Scalability Notes
- Linear scaling with customer count
- Memory usage grows with tracked months
- Can generate 100K+ customers with sufficient RAM

## 🛠️ Technical Implementation

### Customer State Tracking
Each customer maintains:
- **Behavioral trends** (login, support, spending patterns)
- **Churn risk evolution** over time
- **Data consistency** across months
- **Realistic state transitions**

### Churn Logic
1. **Risk Calculation**: Based on current behavior + trends
2. **Probability Weighting**: Historical patterns + current state
3. **Realistic Timing**: Customers don't churn immediately
4. **Signal Generation**: Observable patterns before churn

### Data Consistency
- **ID Persistence**: Customer IDs remain consistent
- **Temporal Logic**: Dates, service ages, renewal counts evolve properly
- **Relationship Preservation**: Related metrics stay correlated
- **Business Rules**: Realistic hosting/domain service patterns

## 🎯 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Basic Example**:
   ```bash
   python temporal_usage_example.py
   ```

3. **Generate Production Data**:
   ```bash
   python temporal_batch_generator.py --initial-customers 5000 --months 24
   ```

4. **Analyze Results**:
   ```python
   import pandas as pd
   
       # Load and analyze temporal patterns
    df_jul = pd.read_csv('temporal_datasets/202407.csv')
    df_aug = pd.read_csv('temporal_datasets/202408.csv')
   
   # Find churned customers
   jul_customers = set(df_jul['ACCOUNT_ID'])
   aug_customers = set(df_aug['ACCOUNT_ID'])
   churned = jul_customers - aug_customers
   ```

## 🔮 Advanced Features

### Custom Churn Logic
```python
# Override churn probability calculation
class CustomTemporalGenerator(TemporalBatchGenerator):
    def _calculate_churn_probability(self, customer):
        # Custom business logic here
        return custom_probability
```

### Seasonal Patterns
```python
# Add seasonal churn variations
seasonal_multiplier = 1.2 if month in ['11', '12'] else 1.0
churn_prob *= seasonal_multiplier
```

### Industry-Specific Patterns
- **Hosting/Domain**: Service expiration patterns
- **SaaS**: Usage-based churn signals  
- **E-commerce**: Purchase frequency patterns
- **Subscription**: Billing cycle influences

---

## 🎉 Ready to Generate Temporal Data?

```bash
# Start generating realistic temporal churn data
python temporal_batch_generator.py --initial-customers 1000 --months 12

# Watch progress in real-time
tail -f temporal_datasets/temporal_generation.log
```

The temporal generator creates datasets perfect for training AI models that understand customer behavior evolution and can predict churn with realistic lead times! 