#!/usr/bin/env python3
"""
Usage Examples for Batch Dataset Generator

This file demonstrates how to use the batch dataset generator
without actually running the generation (which takes time for 1M rows).
"""

from batch_generate_datasets import BatchDatasetGenerator


def example_usage():
    """Show different ways to use the batch generator."""
    
    print("📋 BATCH DATASET GENERATOR - USAGE EXAMPLES")
    print("=" * 50)
    
    # Example 1: Default usage (1M rows per month)
    print("\n1️⃣  DEFAULT USAGE:")
    print("   Generates 13 datasets (July 2024 - July 2025)")
    print("   Each dataset: 1,000,000 rows")
    print("   Output: ./datasets/ directory")
    print("\n   Command:")
    print("   python batch_generate_datasets.py")
    
    # Example 2: Custom parameters
    print("\n2️⃣  CUSTOM PARAMETERS:")
    print("   Custom row count, seed, and output directory")
    print("\n   Command:")
    print("   python batch_generate_datasets.py --rows 500000 --seed 123 --output-dir my_datasets")
    
    # Example 3: Programmatic usage
    print("\n3️⃣  PROGRAMMATIC USAGE:")
    print("""
   from batch_generate_datasets import BatchDatasetGenerator
   
   # Create generator
   generator = BatchDatasetGenerator(
       rows_per_month=1000000,  # 1M rows per dataset
       random_seed=42,          # For reproducible results
       output_dir="datasets"    # Output directory
   )
   
   # Generate all datasets
   files = generator.generate_all_datasets()
   
   # Verify datasets
   generator.verify_datasets(files)
   """)
    
    # Show expected outputs
    print("\n📁 EXPECTED OUTPUT FILES:")
    print("   The batch job will create these 13 files:")
    
    months = [
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"
    ]
    
    for month in months:
        year, month_num = month.split("-")
        filename = f"churn_dataset_{year}{month_num}.csv"
        print(f"   • {filename}")
    
    # Show estimated details
    print("\n📊 ESTIMATED DETAILS:")
    print("   • Total datasets: 13")
    print("   • Rows per dataset: 1,000,000")
    print("   • Total rows: 13,000,000")
    print("   • Estimated file size per dataset: ~230 MB")
    print("   • Estimated total size: ~3 GB")
    print("   • Estimated generation time: 20-30 minutes")
    
    # Command line options
    print("\n⚙️  COMMAND LINE OPTIONS:")
    options = [
        ("--rows, -r", "Number of rows per dataset (default: 1,000,000)"),
        ("--seed, -s", "Base random seed for reproducibility (default: 42)"),
        ("--output-dir, -o", "Output directory (default: 'datasets')"),
        ("--verify, -v", "Force verification of generated datasets"),
        ("--no-verify", "Skip dataset verification"),
        ("--help, -h", "Show help message")
    ]
    
    for option, description in options:
        print(f"   {option:<20} {description}")


def show_month_details():
    """Show details about each month that will be generated."""
    print("\n📅 MONTHLY DATASET DETAILS:")
    print("-" * 50)
    
    # Create a generator instance to get the month dates
    generator = BatchDatasetGenerator()
    month_dates = generator.get_month_dates()
    
    for i, (year, month, reference_date) in enumerate(month_dates, 1):
        filename = f"churn_dataset_{year}{month:02d}.csv"
        print(f"{i:2d}. {filename}")
        print(f"    Reference Date: {reference_date}")
        print(f"    Month: {year}-{month:02d}")
        print()


def show_dataset_schema():
    """Show the dataset schema that will be generated."""
    print("\n📋 DATASET SCHEMA:")
    print("=" * 30)
    print("Each dataset will contain these columns from the PRD:")
    print()
    
    # Key columns from the PRD
    key_columns = [
        "FIRST_OF_MONTH", "ACCOUNT_ID", "PERSON_ORG_ID",
        "PP_SERVICE_CREATED_DATE", "PP_SERVICE_DELETION_DATE", "PP_EXPIRATION_DATE",
        "PP_PREMIUM_FLAG", "PP_BUNDLE_FLAG", "PP_DISCOUNT_AMT", "PP_NET_AMT",
        "TERM_IN_MONTHS", "RENEWAL_COUNT", "ALL_PRODUCT_CNT", "ALL_PRODUCT_NET_AMT",
        "TOTAL_CONTACTS", "BILLING_CONTACTS", "SUCCESS_LOGIN", "TOTAL_LOGIN"
    ]
    
    print("Key columns (subset of 50+ total columns):")
    for col in key_columns:
        print(f"  • {col}")
    
    print("\n📈 Churn Logic Features:")
    print("  • Service age and expiration dates")
    print("  • Product diversity and spend amounts") 
    print("  • Support contact patterns")
    print("  • Login activity metrics")
    print("  • NPS scores and billing issues")


if __name__ == "__main__":
    example_usage()
    show_month_details()
    show_dataset_schema()
    
    print("\n" + "=" * 60)
    print("🚀 TO START GENERATION:")
    print("   python batch_generate_datasets.py")
    print("\n⚠️  NOTE: Generation will take 20-30 minutes for 13M total rows")
    print("=" * 60) 