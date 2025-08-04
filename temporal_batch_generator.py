#!/usr/bin/env python3
"""
Temporal Churn Dataset Generator

Generates synthetic churn datasets across multiple months with realistic customer lifecycle:
- Customers can churn out (disappear from subsequent months)
- New customers are added each month
- Existing customers evolve their behavior over time
- Churned customers show behavioral patterns before disappearing
- Maintains steady 2-3% monthly churn rate
"""

import os
import sys
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import uuid
from contextlib import redirect_stdout
from io import StringIO

# Import the existing generator for base functionality
from generate_churn_dataset import ChurnDatasetGenerator


class CustomerState:
    """Tracks an individual customer's state across months."""
    
    def __init__(self, customer_id: int, creation_month: str, initial_data: Dict):
        self.customer_id = customer_id
        self.creation_month = creation_month
        self.is_active = True
        self.churn_month = None
        self.months_since_creation = 0
        self.churn_risk_trend = 0.1  # Starting risk level
        
        # Store customer data
        self.data = initial_data.copy()
        
        # Behavioral trend tracking
        self.login_trend = 0  # -1 decreasing, 0 stable, 1 increasing
        self.support_trend = 0  # How support contacts are trending
        self.spend_trend = 0  # How spending is trending
        self.product_trend = 0  # How product count is trending
        
    def evolve_behavior(self, month_delta: int) -> Dict:
        """Evolve customer behavior for the next month."""
        self.months_since_creation += month_delta
        
        # Calculate churn probability based on current risk and trends
        churn_prob = self._calculate_churn_probability()
        
        # Decide if customer churns this month
        if np.random.random() < churn_prob and self.months_since_creation > 2:
            self.is_active = False
            return self._generate_final_month_data()
        
        # Evolve the customer's data
        evolved_data = self._evolve_customer_data()
        self.data.update(evolved_data)
        
        return self.data.copy()
    
    def _calculate_churn_probability(self) -> float:
        """Calculate the probability this customer will churn this month."""
        base_prob = 0.025  # 2.5% base monthly churn rate
        
        # Risk increases over time if trends are negative
        risk_multiplier = 1 + (self.churn_risk_trend * self.months_since_creation * 0.1)
        
        # Behavioral factors
        if self.login_trend < -0.5:
            risk_multiplier *= 1.5
        if self.support_trend > 0.5:
            risk_multiplier *= 1.3
        if self.spend_trend < -0.3:
            risk_multiplier *= 1.4
        if self.product_trend < -0.2:
            risk_multiplier *= 1.2
            
        # Cap the probability
        return min(base_prob * risk_multiplier, 0.15)  # Max 15% monthly churn
    
    def _evolve_customer_data(self) -> Dict:
        """Evolve customer data based on trends and random factors."""
        evolved = {}
        
        # Update dates
        evolved['FIRST_OF_MONTH'] = datetime.strptime(self.data['FIRST_OF_MONTH'], '%Y-%m-%d') + relativedelta(months=1)
        evolved['FIRST_OF_MONTH'] = evolved['FIRST_OF_MONTH'].strftime('%Y-%m-%d')
        
        # Evolution trends (gradual drift)
        self.login_trend += np.random.normal(0, 0.1)
        self.support_trend += np.random.normal(0, 0.1)
        self.spend_trend += np.random.normal(0, 0.05)
        self.product_trend += np.random.normal(0, 0.03)
        
        # Clamp trends
        self.login_trend = np.clip(self.login_trend, -1, 1)
        self.support_trend = np.clip(self.support_trend, -0.5, 1)
        self.spend_trend = np.clip(self.spend_trend, -0.5, 0.5)
        self.product_trend = np.clip(self.product_trend, -0.3, 0.3)
        
        # Evolve login activity
        login_factor = 1 + (self.login_trend * 0.2)
        evolved['SUCCESS_LOGIN'] = max(0, int(self.data['SUCCESS_LOGIN'] * login_factor * np.random.uniform(0.8, 1.2)))
        evolved['TOTAL_LOGIN'] = max(evolved['SUCCESS_LOGIN'], int(self.data['TOTAL_LOGIN'] * login_factor * np.random.uniform(0.8, 1.2)))
        
        # Evolve support contacts
        support_factor = 1 + (self.support_trend * 0.3)
        for contact_type in ['TOTAL_CONTACTS', 'BILLING_CONTACTS', 'RETENTION_CONTACTS', 
                           'WORDPRESS_CONTACTS', 'DOMAIN_CONTACTS', 'EMAIL_CONTACTS',
                           'CPANEL_CONTACTS', 'ACCOUNT_CONTACTS', 'SALES_CONTACTS', 
                           'SSL_CONTACTS', 'TOS_CONTACTS']:
            if contact_type in self.data:
                base_value = self.data[contact_type]
                if contact_type == 'BILLING_CONTACTS' and self.support_trend > 0.3:
                    # Billing contacts spike before churn
                    evolved[contact_type] = max(0, int(base_value * support_factor * np.random.uniform(1.0, 2.0)))
                elif contact_type == 'RETENTION_CONTACTS' and self.support_trend > 0.5:
                    # Retention calls appear before churn
                    evolved[contact_type] = max(0, int(base_value + np.random.poisson(1)))
                else:
                    evolved[contact_type] = max(0, int(base_value * support_factor * np.random.uniform(0.7, 1.3)))
        
        # Evolve spending and products
        spend_factor = 1 + (self.spend_trend * 0.1)
        product_factor = 1 + (self.product_trend * 0.1)
        
        # Update product counts and amounts
        for col in ['ALL_PRODUCT_CNT', 'HOSTING_PRODUCT_CNT', 'DOMAIN_PRODUCT_CNT',
                   'ECOMMERC_PRODUCT_CNT', 'ADV_HOSTING_PRODUCT_CNT']:
            if col in self.data:
                evolved[col] = max(0, int(self.data[col] * product_factor * np.random.uniform(0.8, 1.2)))
        
        # Update spending amounts
        for col in ['ALL_PRODUCT_NET_AMT', 'HOSTING_NET_AMT', 'DOMAIN_NET_AMT',
                   'ECOMMERC_NET_AMT', 'ADV_HOSTING_NET_AMT']:
            if col in self.data:
                evolved[col] = max(0, round(self.data[col] * spend_factor * np.random.uniform(0.8, 1.2), 2))
        
        # Update renewal count (occasionally increases)
        if np.random.random() < 0.1:  # 10% chance of renewal this month
            evolved['RENEWAL_COUNT'] = self.data['RENEWAL_COUNT'] + 1
        else:
            evolved['RENEWAL_COUNT'] = self.data['RENEWAL_COUNT']
        
        # Update churn risk trend
        if evolved['SUCCESS_LOGIN'] < self.data['SUCCESS_LOGIN'] * 0.7:
            self.churn_risk_trend += 0.1
        if evolved['BILLING_CONTACTS'] > self.data['BILLING_CONTACTS']:
            self.churn_risk_trend += 0.2
        if evolved['ALL_PRODUCT_CNT'] < self.data['ALL_PRODUCT_CNT']:
            self.churn_risk_trend += 0.15
            
        return evolved
    
    def _generate_final_month_data(self) -> Dict:
        """Generate data for the final month before churn (showing churn signals)."""
        final_data = self.data.copy()
        
        # Show clear churn signals
        final_data['SUCCESS_LOGIN'] = max(0, int(final_data['SUCCESS_LOGIN'] * 0.3))
        final_data['TOTAL_LOGIN'] = max(final_data['SUCCESS_LOGIN'], int(final_data['TOTAL_LOGIN'] * 0.5))
        
        # Spike in support contacts
        final_data['BILLING_CONTACTS'] = min(10, final_data['BILLING_CONTACTS'] + np.random.poisson(2))
        final_data['RETENTION_CONTACTS'] = min(5, final_data['RETENTION_CONTACTS'] + np.random.poisson(1))
        final_data['TOTAL_CONTACTS'] = final_data['BILLING_CONTACTS'] + final_data['RETENTION_CONTACTS'] + sum([
            final_data.get(f'{t}_CONTACTS', 0) for t in ['WORDPRESS', 'DOMAIN', 'EMAIL', 'CPANEL', 'ACCOUNT', 'SALES', 'SSL', 'TOS']
        ])
        
        # Reduce products (customer moving services elsewhere)
        reduction_factor = np.random.uniform(0.3, 0.8)
        for col in ['ALL_PRODUCT_CNT', 'HOSTING_PRODUCT_CNT', 'DOMAIN_PRODUCT_CNT']:
            if col in final_data:
                final_data[col] = max(0, int(final_data[col] * reduction_factor))
        
        return final_data


class TemporalBatchGenerator:
    """Generates temporal datasets with customer lifecycle tracking."""
    
    def __init__(self, initial_customers=1000, monthly_new_customers=50, 
                 random_seed=42, output_dir="temporal_datasets", start_month="2024-07"):
        """
        Initialize the temporal batch generator.
        
        Args:
            initial_customers (int): Number of customers in the first month
            monthly_new_customers (int): New customers added each month
            random_seed (int): Base random seed
            output_dir (str): Directory to save datasets
            start_month (str): Starting month in YYYY-MM format (default: 2024-07)
        """
        self.initial_customers = initial_customers
        self.monthly_new_customers = monthly_new_customers
        self.base_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_month = start_month
        
        # Customer tracking
        self.customers: Dict[int, CustomerState] = {}
        self.next_customer_id = 1
        self.current_month = start_month
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'temporal_generation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set initial seed
        if random_seed:
            np.random.seed(random_seed)
            Faker.seed(random_seed)
            
        self.fake = Faker()
        
    def get_month_list(self, num_months=12) -> List[str]:
        """Generate list of months for dataset generation."""
        start_date = datetime.strptime(self.start_month + "-01", "%Y-%m-%d")
        months = []
        
        for i in range(num_months):
            current_date = start_date + relativedelta(months=i)
            months.append(current_date.strftime("%Y-%m"))
            
        return months
    
    def create_new_customer(self, month: str) -> CustomerState:
        """Create a new customer for the given month."""
        customer_id = self.next_customer_id
        self.next_customer_id += 1
        
        # Generate initial customer data using the base generator (suppress verbose output)
        temp_generator = ChurnDatasetGenerator(n_customers=1, random_seed=None, 
                                             reference_date=f"{month}-01")
        with redirect_stdout(StringIO()):
            temp_df = temp_generator.generate_dataset()
        
        # Convert to dictionary
        initial_data = temp_df.iloc[0].to_dict()
        initial_data['ACCOUNT_ID'] = customer_id
        initial_data['FIRST_OF_MONTH'] = f"{month}-01"
        
        # Create customer state
        customer = CustomerState(customer_id, month, initial_data)
        
        # Set some customers as outliers (5% chance)
        if np.random.random() < 0.05:
            self._make_outlier_customer(customer)
            
        return customer
    
    def _make_outlier_customer(self, customer: CustomerState):
        """Create outlier behavior for a customer."""
        outlier_type = np.random.choice(['high_spender', 'support_heavy', 'login_addict', 'product_hoarder'])
        
        if outlier_type == 'high_spender':
            # High spending customer
            customer.data['ALL_PRODUCT_NET_AMT'] *= np.random.uniform(5, 20)
            customer.data['ALL_PRODUCT_CNT'] = np.random.randint(10, 25)
            customer.churn_risk_trend = 0.05  # Lower churn risk
            
        elif outlier_type == 'support_heavy':
            # Customer with lots of support contacts
            customer.data['TOTAL_CONTACTS'] = np.random.randint(15, 30)
            customer.data['BILLING_CONTACTS'] = np.random.randint(3, 8)
            customer.support_trend = 0.8
            customer.churn_risk_trend = 0.3  # Higher churn risk
            
        elif outlier_type == 'login_addict':
            # Very active login customer
            customer.data['SUCCESS_LOGIN'] = np.random.randint(50, 200)
            customer.data['TOTAL_LOGIN'] = customer.data['SUCCESS_LOGIN'] + np.random.randint(0, 20)
            customer.login_trend = 0.5
            customer.churn_risk_trend = 0.02  # Very low churn risk
            
        elif outlier_type == 'product_hoarder':
            # Customer with many products
            customer.data['ALL_PRODUCT_CNT'] = np.random.randint(15, 35)
            customer.data['HOSTING_PRODUCT_CNT'] = np.random.randint(5, 15)
            customer.data['DOMAIN_PRODUCT_CNT'] = np.random.randint(10, 25)
            customer.product_trend = 0.2
            customer.churn_risk_trend = 0.03  # Low churn risk
    
    def generate_monthly_dataset(self, month: str, month_index: int) -> pd.DataFrame:
        """Generate dataset for a specific month."""
        self.logger.info(f"📊 Generating dataset for {month}")
        
        monthly_data = []
        churned_customers = []
        
        # Evolve existing customers
        for customer_id, customer in list(self.customers.items()):
            if customer.is_active:
                evolved_data = customer.evolve_behavior(1)
                
                if customer.is_active:
                    monthly_data.append(evolved_data)
                else:
                    # Customer churned
                    churned_customers.append(customer_id)
                    monthly_data.append(evolved_data)  # Include final month data
        
        # Remove churned customers for next month
        for customer_id in churned_customers:
            del self.customers[customer_id]
        
        # Add new customers
        for _ in range(self.monthly_new_customers):
            new_customer = self.create_new_customer(month)
            self.customers[new_customer.customer_id] = new_customer
            monthly_data.append(new_customer.data.copy())
        
        # Create DataFrame
        df = pd.DataFrame(monthly_data)
        
        # Calculate statistics
        total_customers = len(monthly_data)
        churned_count = len(churned_customers)
        churn_rate = (churned_count / (total_customers + churned_count)) * 100 if total_customers + churned_count > 0 else 0
        
        self.logger.info(f"   👥 Total customers: {total_customers}")
        self.logger.info(f"   📉 Churned customers: {churned_count}")
        self.logger.info(f"   📊 Churn rate: {churn_rate:.2f}%")
        self.logger.info(f"   ✨ New customers: {self.monthly_new_customers}")
        
        return df
    
    def generate_all_datasets(self, num_months=12) -> List[str]:
        """Generate all monthly datasets."""
        months = self.get_month_list(num_months=num_months)
        generated_files = []
        
        self.logger.info("🚀 Starting temporal batch dataset generation")
        self.logger.info(f"📅 Generating datasets for {len(months)} months")
        self.logger.info(f"👥 Initial customers: {self.initial_customers}")
        self.logger.info(f"➕ New customers per month: {self.monthly_new_customers}")
        self.logger.info("-" * 60)
        
        total_start_time = datetime.now()
        
        for i, month in enumerate(months):
            try:
                # Calculate and display progress
                progress_pct = ((i) / len(months)) * 100
                completed = i
                remaining = len(months) - i
                
                # Progress bar visualization
                bar_length = 30
                filled_length = int(bar_length * i // len(months))
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # Calculate ETA
                if i > 0:
                    elapsed_time = (datetime.now() - total_start_time).total_seconds()
                    avg_time_per_month = elapsed_time / i
                    eta_seconds = avg_time_per_month * remaining
                    eta_minutes = eta_seconds / 60
                    
                    if eta_minutes > 1:
                        eta_str = f"{eta_minutes:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "calculating..."
                
                # Display progress header
                print(f"\n{'='*70}")
                print(f"📊 PROGRESS: [{bar}] {progress_pct:.1f}% ({completed}/{len(months)})")
                print(f"⏱️  ETA: {eta_str} remaining")
                print(f"{'='*70}")
                
                # Remove verbose logging during processing
                
                # For first month, create initial customers
                if i == 0:
                    print(f"   🏗️ Creating {self.initial_customers:,} initial customers...")
                    
                    for idx in range(self.initial_customers):
                        if idx % 25000 == 0 and idx > 0:  # Progress for large customer creation
                            init_progress = (idx / self.initial_customers) * 100
                            print(f"\r   ⏳ Customer initialization: {init_progress:.1f}% ({idx:,}/{self.initial_customers:,})", end="", flush=True)
                        
                        new_customer = self.create_new_customer(month)
                        self.customers[new_customer.customer_id] = new_customer
                    
                    print(f"\r   ✅ Initialized {self.initial_customers:,} customers" + " " * 20)
                
                # Generate dataset for this month
                filename = f"{month.replace('-', '')}.csv"
                print(f"   🔄 {filename}: Generating...", end="", flush=True)
                start_time = datetime.now()
                
                # Temporarily suppress logger output during generation
                original_level = self.logger.level
                self.logger.setLevel(logging.ERROR)
                
                df = self.generate_monthly_dataset(month, i)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                # Restore logger level
                self.logger.setLevel(original_level)
                
                # Save dataset
                print(f"\r   💾 {filename}: Saving...", end="", flush=True)
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False)
                
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                
                # Success summary
                print(f"\r   ✅ {filename}: {file_size_mb:.1f}MB, {df.shape[0]:,} customers ({generation_time:.1f}s)" + " " * 10)
                
                generated_files.append(str(filepath))
                
                # Overall progress update
                overall_progress = ((i + 1) / len(months)) * 100
                print(f"📈 Overall Progress: {overall_progress:.1f}% complete")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to generate dataset for {month}: {str(e)}")
                print(f"   ❌ Error processing {month}")
                continue
        
        total_time = (datetime.now() - total_start_time).total_seconds()
        
        # Final completion progress
        print(f"\n{'='*70}")
        print(f"🎉 GENERATION COMPLETE! 100.0% ({len(generated_files)}/{len(months)})")
        print(f"{'='*70}")
        
        # Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 TEMPORAL GENERATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Generated: {len(generated_files)}/{len(months)} datasets")
        self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        # Performance stats
        if len(generated_files) > 0:
            avg_time_per_file = total_time / len(generated_files)
            self.logger.info(f"⚡ Average time per file: {avg_time_per_file:.1f} seconds")
        
        self.logger.info(f"📁 Output directory: {self.output_dir.absolute()}")
        
        if generated_files:
            total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in generated_files)
            total_customers = sum(len(pd.read_csv(f)) for f in generated_files[:3])  # Sample first 3 for speed
            avg_file_size = total_size_mb / len(generated_files)
            
            self.logger.info(f"💾 Total data size: {total_size_mb:.1f} MB")
            self.logger.info(f"📊 Average file size: {avg_file_size:.1f} MB")
            
            print(f"\n🎯 GENERATION STATS:")
            print(f"   📂 Files created: {len(generated_files)}")
            print(f"   💾 Total size: {total_size_mb:.1f} MB")
            print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
            print(f"   ⚡ Avg per file: {avg_time_per_file:.1f}s")
            
            self.logger.info("\n📁 Generated files:")
            for filepath in generated_files:
                filename = os.path.basename(filepath)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                self.logger.info(f"   • {filename} ({size_mb:.1f} MB)")
        
        return generated_files


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate temporal churn datasets')
    parser.add_argument('--initial-customers', type=int, default=1000,
                       help='Number of customers in first month (default: 1000)')
    parser.add_argument('--monthly-new', type=int, default=50,
                       help='New customers added each month (default: 50)')
    parser.add_argument('--months', type=int, default=12,
                       help='Number of months to generate (default: 12)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='temporal_datasets',
                       help='Output directory (default: temporal_datasets)')
    parser.add_argument('--start-month', type=str, default='2024-07',
                       help='Starting month in YYYY-MM format (default: 2024-07)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TemporalBatchGenerator(
        initial_customers=args.initial_customers,
        monthly_new_customers=args.monthly_new,
        random_seed=args.seed,
        output_dir=args.output_dir,
        start_month=args.start_month
    )
    
    try:
        # Generate all datasets
        generated_files = generator.generate_all_datasets(num_months=args.months)
        return generated_files
        
    except KeyboardInterrupt:
        generator.logger.info("\n⚠️  Generation interrupted by user")
        return []
    except Exception as e:
        generator.logger.error(f"❌ Temporal generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    generated_files = main() 