#!/usr/bin/env python3
"""
Batch Dataset Generator for Churn Analysis

Generates synthetic churn datasets for 13 months (July 2024 to July 2025).
Each dataset contains 1 million customer records.
"""

import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from pathlib import Path

# Import the existing generator
from generate_churn_dataset import ChurnDatasetGenerator


class BatchDatasetGenerator:
    """Batch generator for multiple monthly datasets."""
    
    def __init__(self, rows_per_month=1000000, random_seed=42, output_dir="datasets"):
        """
        Initialize the batch generator.
        
        Args:
            rows_per_month (int): Number of rows per monthly dataset (default: 1M)
            random_seed (int): Base random seed for reproducible results
            output_dir (str): Directory to save generated datasets
        """
        self.rows_per_month = rows_per_month
        self.base_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'batch_generation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_month_dates(self):
        """
        Generate list of reference dates for July 2024 to July 2025 (13 months).
        
        Returns:
            list: List of (year, month, reference_date) tuples
        """
        start_date = datetime(2024, 7, 1)  # July 2024
        dates = []
        
        for i in range(13):  # 13 months
            current_date = start_date + relativedelta(months=i)
            dates.append((
                current_date.year,
                current_date.month,
                current_date.strftime("%Y-%m-%d")
            ))
            
        return dates
    
    def generate_monthly_dataset(self, year, month, reference_date, dataset_index):
        """
        Generate a single monthly dataset.
        
        Args:
            year (int): Year for the dataset
            month (int): Month for the dataset
            reference_date (str): Reference date for data generation
            dataset_index (int): Index for seed generation
            
        Returns:
            str: Path to the generated file
        """
        # Create unique seed for each month
        monthly_seed = self.base_seed + dataset_index if self.base_seed else None
        
        # Create filename
        filename = f"churn_dataset_{year}{month:02d}.csv"
        filepath = self.output_dir / filename
        
        self.logger.info(f"📊 Generating dataset for {year}-{month:02d}")
        self.logger.info(f"   • Reference Date: {reference_date}")
        self.logger.info(f"   • Rows: {self.rows_per_month:,}")
        self.logger.info(f"   • Seed: {monthly_seed}")
        self.logger.info(f"   • Output: {filepath}")
        
        start_time = datetime.now()
        
        try:
            # Generate dataset
            generator = ChurnDatasetGenerator(
                n_customers=self.rows_per_month,
                random_seed=monthly_seed,
                reference_date=reference_date
            )
            
            dataset = generator.generate_dataset()
            
            # Save dataset
            dataset.to_csv(filepath, index=False)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            
            self.logger.info(f"   ✅ Completed in {generation_time:.2f} seconds")
            self.logger.info(f"   💾 File size: {file_size_mb:.1f} MB")
            self.logger.info(f"   📋 Shape: {dataset.shape}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"   ❌ Failed to generate dataset: {str(e)}")
            raise
    
    def generate_all_datasets(self):
        """
        Generate all 13 monthly datasets.
        
        Returns:
            list: List of generated file paths
        """
        month_dates = self.get_month_dates()
        generated_files = []
        
        self.logger.info("🚀 Starting batch dataset generation")
        self.logger.info(f"📅 Generating datasets for {len(month_dates)} months")
        self.logger.info(f"📊 {self.rows_per_month:,} rows per dataset")
        self.logger.info(f"💾 Output directory: {self.output_dir.absolute()}")
        self.logger.info("-" * 60)
        
        total_start_time = datetime.now()
        
        for i, (year, month, reference_date) in enumerate(month_dates, 1):
            try:
                self.logger.info(f"\n[{i}/{len(month_dates)}] Processing {year}-{month:02d}")
                
                filepath = self.generate_monthly_dataset(year, month, reference_date, i)
                generated_files.append(filepath)
                
                # Progress update
                progress = (i / len(month_dates)) * 100
                self.logger.info(f"📈 Progress: {progress:.1f}% ({i}/{len(month_dates)} complete)")
                
            except Exception as e:
                self.logger.error(f"Failed to generate dataset for {year}-{month:02d}: {str(e)}")
                # Continue with next month even if one fails
                continue
        
        total_time = (datetime.now() - total_start_time).total_seconds()
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 BATCH GENERATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Generated: {len(generated_files)}/{len(month_dates)} datasets")
        self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"📁 Output directory: {self.output_dir.absolute()}")
        
        if generated_files:
            total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in generated_files)
            total_rows = len(generated_files) * self.rows_per_month
            self.logger.info(f"💾 Total data size: {total_size_mb:.1f} MB")
            self.logger.info(f"📋 Total rows generated: {total_rows:,}")
            
            self.logger.info("\n📁 Generated files:")
            for filepath in generated_files:
                filename = os.path.basename(filepath)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                self.logger.info(f"   • {filename} ({size_mb:.1f} MB)")
        
        return generated_files
    
    def verify_datasets(self, generated_files):
        """
        Verify the generated datasets for basic integrity.
        
        Args:
            generated_files (list): List of file paths to verify
        """
        import pandas as pd
        
        self.logger.info("\n🔍 DATASET VERIFICATION")
        self.logger.info("-" * 30)
        
        verification_results = {}
        
        for filepath in generated_files:
            try:
                filename = os.path.basename(filepath)
                self.logger.info(f"Verifying {filename}...")
                
                # Load dataset
                df = pd.read_csv(filepath)
                
                # Basic checks
                checks = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'has_nulls': df.isnull().sum().sum() > 0,
                    'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
                }
                
                verification_results[filename] = checks
                
                # Expected row count check
                if checks['row_count'] == self.rows_per_month:
                    self.logger.info(f"   ✅ Row count: {checks['row_count']:,}")
                else:
                    self.logger.warning(f"   ⚠️  Row count: {checks['row_count']:,} (expected: {self.rows_per_month:,})")
                
                self.logger.info(f"   📋 Columns: {checks['column_count']}")
                self.logger.info(f"   💾 Size: {checks['file_size_mb']:.1f} MB")
                
                if checks['has_nulls']:
                    self.logger.warning(f"   ⚠️  Contains null values")
                else:
                    self.logger.info(f"   ✅ No null values")
                    
            except Exception as e:
                self.logger.error(f"   ❌ Verification failed: {str(e)}")
                verification_results[filename] = {'error': str(e)}
        
        return verification_results


def main():
    """Main function to run the batch generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch generate monthly churn datasets')
    parser.add_argument('--rows', '-r', type=int, default=1000000,
                       help='Number of rows per dataset (default: 1,000,000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    parser.add_argument('--verify', '-v', action='store_true',
                       help='Verify generated datasets after creation')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip dataset verification')
    
    args = parser.parse_args()
    
    # Create batch generator
    batch_generator = BatchDatasetGenerator(
        rows_per_month=args.rows,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    try:
        # Generate all datasets
        generated_files = batch_generator.generate_all_datasets()
        
        # Verify datasets if requested
        if args.verify or (not args.no_verify and len(generated_files) > 0):
            batch_generator.verify_datasets(generated_files)
        
        return generated_files
        
    except KeyboardInterrupt:
        batch_generator.logger.info("\n⚠️  Generation interrupted by user")
        return []
    except Exception as e:
        batch_generator.logger.error(f"❌ Batch generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    generated_files = main() 