#!/usr/bin/env python3
"""
Usage Examples for Config-Based Batch Dataset Generator

This file demonstrates how to use the new configuration-based batch dataset generator
for any domain with flexible JSON configuration.
"""

from config_batch_generator import ConfigBatchGenerator


def example_usage():
    """Show different ways to use the config-based batch generator."""
    
    print("📋 CONFIG-BASED BATCH DATASET GENERATOR - USAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Using template config (generic)
    print("\n1️⃣  USING TEMPLATE CONFIG (Generic):")
    print("   Generates datasets for any domain using template_config.json")
    print("   Flexible schema and business rules")
    print("   Output: ./batch_datasets/ directory")
    print("\n   Command:")
    print("   python3 config_batch_generator.py --config template_config.json --rows 10000 --num-months 3")
    
    # Example 2: Using hosting config (domain-specific)
    print("\n2️⃣  USING HOSTING CONFIG (Domain-specific):")
    print("   Generates hosting/domain service datasets with 50+ columns")
    print("   Complex business logic and relationships")
    print("\n   Command:")
    print("   python3 config_batch_generator.py --config template_config.json --rows 100000 --num-months 6")
    
    # Example 3: Custom batch config
    print("\n3️⃣  USING BATCH CONFIG (Optimized for large datasets):")
    print("   Pre-configured for batch processing with verification")
    print("   Includes data quality checks and summary reports")
    print("\n   Command:")
    print("   python3 config_batch_generator.py --config batch_config.json --rows 1000000 --num-months 12 --verify")
    
    # Example 4: Custom output and verification
    print("\n4️⃣  CUSTOM OUTPUT & VERIFICATION:")
    print("   Custom directory, starting month, and full verification")
    print("\n   Command:")
    print("   python3 config_batch_generator.py --config template_config.json \\")
    print("       --rows 50000 --num-months 6 --start-month 2024-01 \\")
    print("       --output-dir my_datasets --verify --summary")
    
    print("\n" + "=" * 60)


def programmatic_example():
    """Show how to use the batch generator programmatically."""
    
    print("\n🐍 PROGRAMMATIC USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n💡 Example 1: Basic programmatic usage")
    print("""
from config_batch_generator import ConfigBatchGenerator

# Create batch generator with any configuration
generator = ConfigBatchGenerator(
    config_path='template_config.json',
    rows_per_month=10000,
    random_seed=42,
    output_dir='my_datasets'
)

# Generate 6 months of data
files = generator.generate_all_datasets(
    num_months=6,
    start_month='2024-01',
    verify=True
)

print(f"Generated {len(files)} files")
""")
    
    print("\n💡 Example 2: Domain-specific generation")
    print("""
# E-commerce example using hosting config as base
generator = ConfigBatchGenerator(
            config_path='template_config.json',
    rows_per_month=100000,
    output_dir='ecommerce_data'
)

# Generate large datasets with verification
files = generator.generate_all_datasets(
    num_months=12,
    verify=True,
    no_skip_verification=True
)

# Generate summary report
generator.generate_summary_report(files)
""")

    print("\n💡 Example 3: Custom configuration workflow")
    print("""
# 1. Create your custom config (JSON file)
# 2. Use it with the batch generator

generator = ConfigBatchGenerator(
    config_path='my_custom_domain_config.json',
    rows_per_month=50000
)

files = generator.generate_all_datasets(num_months=3)
""")


def configuration_examples():
    """Show configuration file examples."""
    
    print("\n⚙️  CONFIGURATION FILE EXAMPLES")
    print("=" * 60)
    
    print("\n📄 Available configurations:")
    print("   • template_config.json     - Generic template for any domain")
    print("   • template_config.json     - Generic template for any domain")
    print("   • batch_config.json        - Optimized for batch processing")
    print("   • temporal_config.json     - Customer lifecycle patterns")
    
    print("\n📖 Creating custom configurations:")
    print("   1. Copy template_config.json")
    print("   2. Modify schema, business rules, and target logic")
    print("   3. See CONFIG_GUIDE.md for complete documentation")
    
    print("\n🎯 Domain examples you can create:")
    print("   • E-commerce (products, orders, customer segments)")
    print("   • SaaS (subscriptions, usage, feature adoption)")
    print("   • Finance (transactions, risk scores, compliance)")
    print("   • Healthcare (patients, treatments, outcomes)")
    print("   • Education (students, courses, performance)")


def migration_guide():
    """Show how to migrate from legacy system."""
    
    print("\n🔄 MIGRATION FROM LEGACY SYSTEM")
    print("=" * 60)
    
    print("\n🆕 All generation now uses the config-based system:")
    print("   python3 config_batch_generator.py --config batch_config.json --rows 1000000")
    
    print("\n✅ New way (configurable):")
    print("   python3 config_batch_generator.py --config template_config.json --rows 1000000")
    
    print("\n🎉 Benefits of new system:")
    print("   ✓ Any business domain supported")
    print("   ✓ Unlimited customization through JSON")
    print("   ✓ No coding required")
    print("   ✓ Better data quality and relationships")
    print("   ✓ Built-in verification and reporting")


if __name__ == "__main__":
    example_usage()
    programmatic_example()
    configuration_examples()
    migration_guide()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to generate your datasets!")
    print("📖 See CONFIG_GUIDE.md for complete documentation")
    print("🔗 Visit README.md for more examples")
    print("=" * 60) 