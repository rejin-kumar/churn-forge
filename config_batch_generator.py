#!/usr/bin/env python3
"""
Configuration-Based Batch Dataset Generator

Generates multiple monthly synthetic datasets using JSON configuration files.
Supports any domain by leveraging the flexible config system.
"""

import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from pathlib import Path
import argparse
import json
from typing import List, Dict, Any
from config_dataset_generator import ConfigDatasetGenerator


class ConfigBatchGenerator:
    """Batch generator for multiple monthly datasets using configuration files."""
    
    def __init__(self, config_path: str, rows_per_month: int = 1000000, 
                 random_seed: int = 42, output_dir: str = "batch_datasets"):
        """
        Initialize the batch generator.
        
        Args:
            config_path (str): Path to the JSON configuration file
            rows_per_month (int): Number of rows per monthly dataset
            random_seed (int): Base random seed for reproducible results
            output_dir (str): Directory to save generated datasets
        """
        self.config_path = config_path
        self.rows_per_month = rows_per_month
        self.base_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        self.dataset_name = self.config['dataset_config']['name']
        
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
        
        self.logger.info(f"🚀 Initialized Config Batch Generator")
        self.logger.info(f"   • Config: {config_path}")
        self.logger.info(f"   • Dataset: {self.dataset_name}")
        self.logger.info(f"   • Rows per month: {rows_per_month:,}")
        self.logger.info(f"   • Output directory: {output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get_month_dates(self, start_month: str = "2024-07", num_months: int = 13) -> List[tuple]:
        """
        Generate list of reference dates for batch generation.
        
        Args:
            start_month (str): Starting month in YYYY-MM format
            num_months (int): Number of months to generate
            
        Returns:
            List of tuples: (year, month, reference_date_string)
        """
        start_date = datetime.strptime(f"{start_month}-01", "%Y-%m-%d")
        dates = []
        
        for i in range(num_months):
            current_date = start_date + relativedelta(months=i)
            dates.append((
                current_date.year,
                current_date.month,
                current_date.strftime("%Y-%m-%d")
            ))
            
        return dates

    def generate_monthly_dataset(self, year: int, month: int, reference_date: str, 
                                dataset_index: int) -> str:
        """
        Generate a single monthly dataset using the configuration.
        
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
        
        # Create filename based on config name
        safe_name = "".join(c for c in self.dataset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_').lower()
        filename = f"{safe_name}_{year}{month:02d}.csv"
        filepath = self.output_dir / filename
        
        self.logger.info(f"📊 Generating dataset for {year}-{month:02d}")
        self.logger.info(f"   • Reference Date: {reference_date}")
        self.logger.info(f"   • Rows: {self.rows_per_month:,}")
        self.logger.info(f"   • Seed: {monthly_seed}")
        self.logger.info(f"   • Output: {filepath}")
        
        start_time = datetime.now()
        
        try:
            # Create generator with config
            generator = ConfigDatasetGenerator(self.config_path)
            
            # Override generation parameters
            generator.n_records = self.rows_per_month
            generator.random_seed = monthly_seed
            generator.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
            
            # Reset numpy seed if provided
            if monthly_seed is not None:
                import numpy as np
                np.random.seed(monthly_seed)
            
            # Generate dataset
            dataset = generator.generate_dataset()
            
            # Update the FIRST_OF_MONTH column to match the current month
            dataset['FIRST_OF_MONTH'] = datetime.strptime(f"{year}-{month:02d}-01", "%Y-%m-%d").date()
            
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

    def generate_all_datasets(self, start_month: str = "2024-07", num_months: int = 13) -> List[str]:
        """
        Generate all monthly datasets for the specified period.
        
        Args:
            start_month (str): Starting month in YYYY-MM format
            num_months (int): Number of months to generate
            
        Returns:
            List[str]: Paths to all generated files
        """
        self.logger.info(f"🎯 Starting batch generation")
        self.logger.info(f"   • Period: {start_month} to {num_months} months")
        self.logger.info(f"   • Total datasets: {num_months}")
        self.logger.info(f"   • Total estimated rows: {self.rows_per_month * num_months:,}")
        
        start_time = datetime.now()
        generated_files = []
        
        # Get month dates
        month_dates = self.get_month_dates(start_month, num_months)
        
        try:
            for dataset_index, (year, month, reference_date) in enumerate(month_dates):
                filepath = self.generate_monthly_dataset(year, month, reference_date, dataset_index)
                generated_files.append(filepath)
                
                # Progress update
                progress = ((dataset_index + 1) / len(month_dates)) * 100
                self.logger.info(f"🔄 Progress: {dataset_index + 1}/{len(month_dates)} ({progress:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"❌ Batch generation failed: {str(e)}")
            raise
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_hours = total_time / 3600
        
        self.logger.info(f"🎉 Batch generation completed!")
        self.logger.info(f"   • Total time: {total_hours:.2f} hours")
        self.logger.info(f"   • Files generated: {len(generated_files)}")
        self.logger.info(f"   • Average time per dataset: {total_time/len(generated_files):.2f} seconds")
        
        return generated_files

    def verify_datasets(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Verify the generated datasets for consistency and quality.
        
        Args:
            file_paths (List[str]): List of file paths to verify
            
        Returns:
            Dict[str, Any]: Verification results
        """
        self.logger.info("🔍 Starting dataset verification...")
        
        verification_results = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'total_rows': 0,
            'issues': []
        }
        
        for filepath in file_paths:
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                
                # Basic checks
                if len(df) == 0:
                    verification_results['issues'].append(f"{filepath}: Empty dataset")
                    verification_results['invalid_files'] += 1
                    continue
                
                if len(df) != self.rows_per_month:
                    verification_results['issues'].append(
                        f"{filepath}: Expected {self.rows_per_month:,} rows, got {len(df):,}"
                    )
                
                # Check for required columns based on config
                expected_columns = [col['name'] for col in self.config['dataset_config']['schema']['columns']]
                missing_columns = set(expected_columns) - set(df.columns)
                if missing_columns:
                    verification_results['issues'].append(
                        f"{filepath}: Missing columns: {missing_columns}"
                    )
                
                verification_results['valid_files'] += 1
                verification_results['total_rows'] += len(df)
                
                self.logger.info(f"   ✅ {os.path.basename(filepath)}: {len(df):,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                verification_results['issues'].append(f"{filepath}: Verification failed - {str(e)}")
                verification_results['invalid_files'] += 1
                self.logger.error(f"   ❌ {os.path.basename(filepath)}: {str(e)}")
        
        # Summary
        self.logger.info("📊 Verification Summary:")
        self.logger.info(f"   • Valid files: {verification_results['valid_files']}")
        self.logger.info(f"   • Invalid files: {verification_results['invalid_files']}")
        self.logger.info(f"   • Total rows: {verification_results['total_rows']:,}")
        
        if verification_results['issues']:
            self.logger.warning(f"   ⚠️  Issues found: {len(verification_results['issues'])}")
            for issue in verification_results['issues']:
                self.logger.warning(f"     - {issue}")
        else:
            self.logger.info("   🎉 All datasets passed verification!")
        
        return verification_results

    def generate_summary_report(self, file_paths: List[str]) -> str:
        """
        Generate a summary report of all generated datasets.
        
        Args:
            file_paths (List[str]): List of generated file paths
            
        Returns:
            str: Path to the summary report
        """
        self.logger.info("📈 Generating summary report...")
        
        import pandas as pd
        summary_data = []
        
        for filepath in file_paths:
            try:
                df = pd.read_csv(filepath)
                filename = os.path.basename(filepath)
                
                # Extract month from filename or data
                month_str = filename.split('_')[-1].replace('.csv', '')
                if 'FIRST_OF_MONTH' in df.columns:
                    first_date = pd.to_datetime(df['FIRST_OF_MONTH'].iloc[0])
                    month_str = first_date.strftime('%Y%m')
                
                summary = {
                    'month': month_str,
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
                }
                
                # Add dataset-specific metrics if target column exists
                target_logic = self.config['dataset_config'].get('target_logic', {})
                target_column = target_logic.get('target_column')
                
                if target_column and target_column in df.columns:
                    summary.update({
                        f'avg_{target_column}': df[target_column].mean(),
                        f'max_{target_column}': df[target_column].max(),
                        f'min_{target_column}': df[target_column].min()
                    })
                    
                    # Risk distribution if thresholds exist
                    thresholds = target_logic.get('threshold', {})
                    if thresholds:
                        high_threshold = thresholds.get('high_risk', 6)
                        medium_threshold = thresholds.get('medium_risk', 3)
                        
                        high_risk = (df[target_column] >= high_threshold).sum()
                        medium_risk = ((df[target_column] >= medium_threshold) & 
                                      (df[target_column] < high_threshold)).sum()
                        low_risk = (df[target_column] < medium_threshold).sum()
                        
                        summary.update({
                            'high_risk_count': high_risk,
                            'medium_risk_count': medium_risk,
                            'low_risk_count': low_risk,
                            'high_risk_pct': (high_risk / len(df)) * 100,
                            'medium_risk_pct': (medium_risk / len(df)) * 100,
                            'low_risk_pct': (low_risk / len(df)) * 100
                        })
                
                summary_data.append(summary)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {filepath}: {str(e)}")
        
        # Create summary DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'batch_summary_report.csv'
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"📋 Summary report saved: {summary_path}")
        
        # Print key statistics
        total_rows = summary_df['rows'].sum()
        total_size_gb = summary_df['file_size_mb'].sum() / 1024
        
        print(f"\n📊 Batch Generation Summary Report")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Total files: {len(summary_df)}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total size: {total_size_gb:.2f} GB")
        print(f"Average rows per file: {total_rows / len(summary_df):,.0f}")
        
        return str(summary_path)


def main():
    """Main function to run the config-based batch generation."""
    parser = argparse.ArgumentParser(description='Batch generate monthly datasets from configuration file')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to the JSON configuration file')
    parser.add_argument('--rows', '-r', type=int, default=1000000,
                       help='Number of rows per dataset (default: 1,000,000)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='batch_datasets',
                       help='Output directory (default: batch_datasets)')
    parser.add_argument('--start-month', type=str, default='2024-07',
                       help='Starting month in YYYY-MM format (default: 2024-07)')
    parser.add_argument('--num-months', '-n', type=int, default=13,
                       help='Number of months to generate (default: 13)')
    parser.add_argument('--verify', '-v', action='store_true',
                       help='Verify generated datasets after creation')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip dataset verification')
    parser.add_argument('--summary', action='store_true', default=True,
                       help='Generate summary report (default: True)')
    
    args = parser.parse_args()
    
    # Create batch generator
    batch_generator = ConfigBatchGenerator(
        config_path=args.config,
        rows_per_month=args.rows,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
    
    try:
        # Generate all datasets
        generated_files = batch_generator.generate_all_datasets(
            start_month=args.start_month,
            num_months=args.num_months
        )
        
        # Verify datasets if requested
        if args.verify or (not args.no_verify and len(generated_files) > 0):
            batch_generator.verify_datasets(generated_files)
        
        # Generate summary report
        if args.summary and generated_files:
            batch_generator.generate_summary_report(generated_files)
        
        return generated_files
        
    except KeyboardInterrupt:
        batch_generator.logger.info("\n⚠️  Generation interrupted by user")
        return []
    except Exception as e:
        batch_generator.logger.error(f"❌ Batch generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    generated_files = main() 