#!/usr/bin/env python3
"""
Config-Based Temporal Dataset Generator - Usage Examples

This script demonstrates how to use the new configuration-based temporal generator
for realistic customer lifecycle datasets across any business domain.
"""

import os
import pandas as pd
from config_temporal_generator import ConfigTemporalGenerator


def example_basic_usage():
    """Basic usage example with template configuration."""
    print("🚀 EXAMPLE 1: Basic Temporal Generation with Template Config")
    print("=" * 60)
    print("Generating customer lifecycle data using template_config.json...")
    
    # Create generator with template configuration
    # Note: This will use template_config.json as base for column definitions
    generator = ConfigTemporalGenerator('template_config.json')
    
    # Override some settings for demo
    generator.initial_customers = 100      # Start with 100 customers  
    generator.monthly_new_customers = 10   # Add 10 new customers each month
    generator.num_months = 6               # Generate 6 months
    generator.output_dir = "demo_temporal"
    
    # Generate temporal datasets
    files = generator.generate_all_datasets()
    
    print(f"\n✅ Generated {len(files)} monthly datasets:")
    for file in files:
        print(f"   📄 {file}")
    
    # Show customer evolution
    analyze_customer_evolution(files)


def example_domain_specific():
    """Example using hosting configuration for domain-specific temporal generation."""
    print("\n🏢 EXAMPLE 2: Domain-Specific Temporal Generation")
    print("=" * 60)
    print("Generating hosting/domain service customer lifecycle...")
    
    # Create generator with hosting domain configuration
    generator = ConfigTemporalGenerator('template_config.json')
    
    # Custom settings for hosting domain
    generator.initial_customers = 500
    generator.monthly_new_customers = 25
    generator.num_months = 12
    generator.output_dir = "hosting_temporal"
    
    print(f"\n📊 Configuration:")
    print(f"   • Domain: {generator.temporal_config['name']}")
    print(f"   • Initial customers: {generator.initial_customers}")
    print(f"   • Monthly new customers: {generator.monthly_new_customers}")
    print(f"   • Number of months: {generator.num_months}")
    print(f"   • Base configuration: {generator.temporal_config.get('base_config_path', 'template_config.json')}")
    
    # Note: Uncomment to actually generate (takes time)
    # files = generator.generate_all_datasets()
    print("\n💡 To run: Uncomment the generation line above")


def example_custom_temporal_config():
    """Example showing custom temporal configuration."""
    print("\n⚙️ EXAMPLE 3: Custom Temporal Configuration")
    print("=" * 60)
    
    print("📝 Creating custom temporal configuration for SaaS business:")
    
    custom_config = {
        "temporal_config": {
            "name": "SaaS Customer Lifecycle",
            "description": "SaaS subscription lifecycle with realistic patterns",
            "base_config_path": "template_config.json",
            "temporal_params": {
                "initial_customers": 1000,
                "monthly_new_customers": 100,
                "num_months": 12,
                "output_dir": "saas_temporal"
            },
            "lifecycle_behavior": {
                "monthly_attrition_rate": 0.035,  # 3.5% monthly churn for SaaS
                "new_customer_protection_months": 3,  # 3-month honeymoon period
                "seasonal_effects": {
                    "enabled": True,
                    "high_attrition_months": [1, 12],  # New Year and end of year
                    "low_attrition_months": [6, 7, 8]  # Summer stability
                }
            },
            "customer_segments": {
                "enterprise": {
                    "percentage": 0.1,
                    "retention_bonus": 0.8,
                    "spend_threshold": 1000
                },
                "startup": {
                    "percentage": 0.3,
                    "retention_bonus": 0.4,
                    "growth_potential": 1.5
                }
            }
        }
    }
    
    print("\n📋 Custom Configuration Features:")
    print("   • SaaS-specific churn rate (3.5% monthly)")
    print("   • New customer protection period (3 months)")
    print("   • Seasonal churn patterns")
    print("   • Enterprise vs Startup customer segments")
    print("   • Growth and retention modeling")
    
    print("\n💡 To use this configuration:")
    print("   1. Save the config to 'saas_temporal_config.json'")
    print("   2. Run: python3 config_temporal_generator.py --config saas_temporal_config.json")


def example_cli_usage():
    """Show command line usage examples."""
    print("\n💻 EXAMPLE 4: Command Line Usage")
    print("=" * 60)
    
    print("🚀 Quick start commands:")
    print("\n1️⃣ Generic temporal generation:")
    print("   python3 config_temporal_generator.py --config template_config.json --months 6")
    
    print("\n2️⃣ Domain-specific with overrides:")
    print("   python3 config_temporal_generator.py --config template_config.json \\")
    print("       --initial-customers 1000 --monthly-new 50 --months 12")
    
    print("\n3️⃣ Custom output directory:")
    print("   python3 config_temporal_generator.py --config template_config.json \\")
    print("       --months 3 --output-dir my_temporal_data")
    
    print("\n4️⃣ Production scale:")
    print("   python3 config_temporal_generator.py --config template_config.json \\")
    print("       --initial-customers 10000 --monthly-new 500 --months 24")


def analyze_customer_evolution(files):
    """Analyze customer evolution patterns from generated files."""
    if not files or not os.path.exists(files[0]):
        print("\n⚠️ No files to analyze (generation may have been skipped)")
        return
        
    print("\n📊 CUSTOMER EVOLUTION ANALYSIS")
    print("-" * 40)
    
    try:
        # Read first and last files
        first_df = pd.read_csv(files[0])
        last_df = pd.read_csv(files[-1])
        
        print(f"\n📈 Customer Growth:")
        print(f"   Month 1: {len(first_df):,} customers")
        print(f"   Month {len(files)}: {len(last_df):,} customers")
        print(f"   Net Growth: {len(last_df) - len(first_df):,} customers")
        
        # Analyze status distribution if available
        if 'CUSTOMER_STATUS' in last_df.columns:
            status_counts = last_df['CUSTOMER_STATUS'].value_counts()
            print(f"\n📊 Customer Status Distribution (Month {len(files)}):")
            for status, count in status_counts.items():
                percentage = (count / len(last_df)) * 100
                print(f"   • {status}: {count:,} ({percentage:.1f}%)")
        
        # Analyze risk scores if available
        risk_columns = [col for col in last_df.columns if 'risk' in col.lower() or 'score' in col.lower()]
        if risk_columns:
            risk_col = risk_columns[0]
            print(f"\n🎯 Risk Score Analysis ({risk_col}):")
            print(f"   • Mean: {last_df[risk_col].mean():.2f}")
            print(f"   • Std: {last_df[risk_col].std():.2f}")
            print(f"   • High Risk (>6): {(last_df[risk_col] > 6).sum():,} customers")
        
    except Exception as e:
        print(f"\n⚠️ Analysis error: {e}")


def show_configuration_options():
    """Show available configuration options."""
    print("\n⚙️ CONFIGURATION OPTIONS")
    print("=" * 60)
    
    print("\n📁 Available Configurations:")
    configs = [
        ("template_config.json", "Generic template", "Any domain"),
        ("template_config.json", "Generic template", "Simple schema, any domain"),
        ("batch_config.json", "Batch processing", "Optimized for large datasets"),
        ("temporal_config.json", "Customer lifecycle", "Advanced temporal patterns")
    ]
    
    for config, description, features in configs:
        print(f"   📄 {config}")
        print(f"      • {description}")
        print(f"      • {features}")
        print()
    
    print("🎯 Creating Custom Configurations:")
    print("   1. Start with template_config.json")
    print("   2. Add temporal_config section for lifecycle behavior")
    print("   3. Define customer segments and behavioral evolution")
    print("   4. Set seasonal patterns and risk prediction features")
    print("   5. See CONFIG_GUIDE.md for complete documentation")


def show_temporal_features():
    """Show temporal-specific features."""
    print("\n🕰️ TEMPORAL FEATURES")
    print("=" * 60)
    
    print("🔄 Customer Lifecycle Management:")
    print("   • Customer state tracking across months")
    print("   • Realistic attrition/departure patterns")
    print("   • New customer acquisition each month")
    print("   • Behavioral evolution over time")
    
    print("\n📊 Behavioral Patterns:")
    print("   • Login activity degradation before departure")
    print("   • Spending pattern changes")
    print("   • Support ticket trends")
    print("   • Satisfaction score evolution")
    
    print("\n🎯 Advanced Features:")
    print("   • Customer segmentation (high-value, at-risk, loyal)")
    print("   • Seasonal effects on behavior")
    print("   • Early warning signals (1-3 months ahead)")
    print("   • Intervention modeling potential")
    
    print("\n📈 ML-Ready Outputs:")
    print("   • Time-series features for prediction models")
    print("   • Customer journey tracking")
    print("   • Risk probability scores")
    print("   • Lifecycle stage classification")


def migration_from_legacy():
    """Show migration from legacy temporal system."""
    print("\n🔄 MIGRATION FROM LEGACY SYSTEM")
    print("=" * 60)
    
    print("🆕 All generation now uses the config-based system:")
    print("   from config_temporal_generator import ConfigTemporalGenerator")
    print("   generator = ConfigTemporalGenerator('temporal_config.json')")
    print("   files = generator.generate_all_datasets()")
    
    print("\n✅ New way (configurable):")
    print("   from config_temporal_generator import ConfigTemporalGenerator")
    print("   generator = ConfigTemporalGenerator('temporal_config.json')")
    print("   files = generator.generate_all_datasets()")
    
    print("\n🎉 Benefits of Migration:")
    print("   ✓ Any business domain supported")
    print("   ✓ Flexible customer lifecycle modeling")
    print("   ✓ Advanced behavioral evolution patterns")
    print("   ✓ Better seasonal and segment modeling")
    print("   ✓ No coding required for customization")
    print("   ✓ JSON-driven configuration")


if __name__ == "__main__":
    example_basic_usage()
    example_domain_specific()
    example_custom_temporal_config()
    example_cli_usage()
    show_configuration_options()
    show_temporal_features()
    migration_from_legacy()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to generate temporal datasets!")
    print("📖 See CONFIG_GUIDE.md for temporal configuration details")
    print("📊 See TEMPORAL_README.md for complete documentation")
    print("🔗 Visit README.md for more examples")
    print("=" * 60) 