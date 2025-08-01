#!/usr/bin/env python3
"""
Customer Churn Dataset Generator (Pattern-Based)

Generates synthetic customer data for a hosting/domain service business
with churn labels driven by configurable behavioral patterns.
"""

import pandas as pd
import numpy as np
from faker import Faker
import uuid
from datetime import datetime, timedelta
import argparse
import os


class ChurnDatasetGenerator:
    """Generates synthetic customer churn data with realistic behavioral patterns."""
    
    def __init__(self, n_customers=100, random_seed=None):
        """
        Initialize the generator.
        
        Args:
            n_customers (int): Number of customer records to generate
            random_seed (int, optional): Seed for reproducible results
        """
        self.n_customers = n_customers
        self.random_seed = random_seed
        
        # Set seeds for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            Faker.seed(random_seed)
        
        self.fake = Faker()
        
    def generate_basic_info(self):
        """Generate basic customer information."""
        return {
            'customer_id': [str(uuid.uuid4()) for _ in range(self.n_customers)],
            'customer_name': [self.fake.name() for _ in range(self.n_customers)],
            'customer_email': [self.fake.email() for _ in range(self.n_customers)]
        }
    
    def generate_support_metrics(self):
        """Generate support-related metrics."""
        support_tickets = np.random.poisson(lam=2, size=self.n_customers)
        support_tickets = np.clip(support_tickets, 0, 10)
        
        # Resolution time correlates with ticket volume
        base_resolution = np.random.exponential(scale=24, size=self.n_customers)
        ticket_multiplier = 1 + (support_tickets / 10)
        avg_resolution_time = base_resolution * ticket_multiplier
        avg_resolution_time = np.clip(avg_resolution_time, 1, 200)
        
        # Critical tickets are subset of total tickets
        critical_tickets_sla_breach = np.random.binomial(
            n=np.minimum(support_tickets, 4), 
            p=0.3, 
            size=self.n_customers
        )
        
        return {
            'support_tickets': support_tickets,
            'avg_resolution_time': np.round(avg_resolution_time, 2),
            'critical_tickets_sla_breach': critical_tickets_sla_breach
        }
    
    def generate_product_metrics(self):
        """Generate product and subscription metrics."""
        # Tenure follows realistic distribution (many new, some long-term)
        tenure_months = np.random.gamma(shape=2, scale=8, size=self.n_customers)
        tenure_months = np.clip(tenure_months, 1, 48).astype(int)
        
        # Product renewals depend on tenure
        renewal_probability = np.minimum(tenure_months / 24, 0.8)
        product_renewals = np.random.binomial(n=5, p=renewal_probability)
        
        # Monthly spend follows log-normal distribution
        monthly_spend = np.random.lognormal(mean=3.5, sigma=0.8, size=self.n_customers)
        monthly_spend = np.clip(monthly_spend, 5, 150)
        
        # Total products correlates with spend
        spend_tier = np.digitize(monthly_spend, bins=[0, 25, 50, 100, 150])
        total_products = np.random.poisson(lam=spend_tier, size=self.n_customers)
        total_products = np.clip(total_products, 1, 10)
        
        # Products transferred out (churn indicator)
        transfer_probability = np.random.beta(a=1, b=9, size=self.n_customers)
        products_transferred_out = np.random.binomial(
            n=np.minimum(total_products, 3), 
            p=transfer_probability
        )
        
        return {
            'product_renewals': product_renewals,
            'tenure_months': tenure_months,
            'monthly_spend': np.round(monthly_spend, 2),
            'total_products': total_products,
            'products_transferred_out': products_transferred_out
        }
    
    def generate_performance_metrics(self):
        """Generate system performance metrics."""
        # Load time follows log-normal (most fast, some very slow)
        avg_load_time = np.random.lognormal(mean=1.2, sigma=0.8, size=self.n_customers)
        avg_load_time = np.clip(avg_load_time, 0.5, 15)
        
        # Downtime follows exponential distribution
        downtime_minutes = np.random.exponential(scale=30, size=self.n_customers)
        downtime_minutes = np.clip(downtime_minutes, 0, 300).astype(int)
        
        # Product usage - beta distribution creates realistic usage patterns
        product_usage_percent = np.random.beta(a=2, b=2, size=self.n_customers) * 100
        product_usage_percent = np.clip(product_usage_percent, 0, 100)
        
        return {
            'avg_load_time': np.round(avg_load_time, 2),
            'downtime_minutes': downtime_minutes,
            'product_usage_percent': np.round(product_usage_percent, 1)
        }
    
    def generate_last_login(self, tenure_months):
        """Generate last login dates based on tenure."""
        today = datetime.now()
        last_logins = []
        
        for tenure in tenure_months:
            # Recent customers login more frequently
            if tenure < 6:
                days_ago = np.random.exponential(scale=10)
            elif tenure < 12:
                days_ago = np.random.exponential(scale=20)
            else:
                days_ago = np.random.exponential(scale=35)
            
            days_ago = min(days_ago, tenure * 30)  # Can't login before becoming customer
            last_login = today - timedelta(days=int(days_ago))
            last_logins.append(last_login.strftime('%Y-%m-%d'))
        
        return last_logins
    
    def calculate_churn_score(self, df):
        """
        Calculate churn score based on the rule-based logic from PRD.
        
        Args:
            df (pd.DataFrame): DataFrame with customer data
            
        Returns:
            np.array: Churn scores for each customer
        """
        scores = np.zeros(len(df))
        
        # Calculate days since last login
        today = datetime.now()
        last_login_dates = pd.to_datetime(df['last_login'])
        days_since_login = (today - last_login_dates).dt.days
        
        # Apply scoring rules from PRD
        conditions = [
            df['support_tickets'] > 4,                    # +1
            df['avg_resolution_time'] > 48,               # +1
            df['critical_tickets_sla_breach'] > 1,        # +1
            df['product_renewals'] < 2,                   # +1
            df['tenure_months'] < 6,                      # +1
            df['monthly_spend'] < 20,                     # +1
            days_since_login > 45,                        # +1
            df['products_transferred_out'] > 1,           # +1
            df['avg_load_time'] > 5,                      # +1
            df['downtime_minutes'] > 120,                 # +1
            df['product_usage_percent'] < 30              # +1
        ]
        
        # Sum up the scores
        for condition in conditions:
            scores += condition.astype(int)
        
        return scores
    
    def assign_churn_status(self, scores):
        """
        Assign churn status based on scores with 10% noise.
        
        Args:
            scores (np.array): Churn scores
            
        Returns:
            list: Customer status ('active' or 'inactive')
        """
        # Base assignment: score >= 5 means inactive
        base_status = ['inactive' if score >= 5 else 'active' for score in scores]
        
        # Add 10% noise (random flips)
        noise_indices = np.random.choice(
            len(base_status), 
            size=int(0.1 * len(base_status)), 
            replace=False
        )
        
        final_status = base_status.copy()
        for idx in noise_indices:
            final_status[idx] = 'inactive' if final_status[idx] == 'active' else 'active'
        
        return final_status
    
    def generate_dataset(self):
        """Generate the complete synthetic dataset."""
        print(f"Generating {self.n_customers} customer records...")
        
        # Generate all data components
        basic_info = self.generate_basic_info()
        support_metrics = self.generate_support_metrics()
        product_metrics = self.generate_product_metrics()
        performance_metrics = self.generate_performance_metrics()
        
        # Combine into DataFrame
        data = {**basic_info, **support_metrics, **product_metrics, **performance_metrics}
        df = pd.DataFrame(data)
        
        # Generate last login based on tenure
        df['last_login'] = self.generate_last_login(df['tenure_months'])
        
        # Calculate churn scores and assign status
        churn_scores = self.calculate_churn_score(df)
        df['customer_status'] = self.assign_churn_status(churn_scores)
        
        # Add churn score for analysis (not in final output)
        df['_churn_score'] = churn_scores
        
        # Reorder columns to match PRD schema
        column_order = [
            'customer_id', 'customer_name', 'customer_email',
            'support_tickets', 'avg_resolution_time', 'critical_tickets_sla_breach',
            'product_renewals', 'tenure_months', 'monthly_spend',
            'last_login', 'total_products', 'products_transferred_out',
            'avg_load_time', 'downtime_minutes', 'product_usage_percent',
            'customer_status'
        ]
        
        return df[column_order]
    
    def save_and_preview(self, df, filename='churn_dataset_realistic.csv'):
        """Save dataset and show preview."""
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\n✅ Dataset saved to: {filename}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Show churn distribution
        churn_dist = df['customer_status'].value_counts()
        churn_rate = (churn_dist.get('inactive', 0) / len(df)) * 100
        print(f"📈 Churn rate: {churn_rate:.1f}% ({churn_dist.get('inactive', 0)} inactive / {len(df)} total)")
        
        # Preview first 5 rows
        print(f"\n📋 Preview (first 5 rows):")
        print(df.head().to_string(index=False))
        
        return filename


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Generate synthetic customer churn dataset')
    parser.add_argument('--rows', '-n', type=int, default=100, 
                       help='Number of customer records to generate (default: 100)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results')
    parser.add_argument('--output', '-o', type=str, default='churn_dataset_realistic.csv',
                       help='Output filename (default: churn_dataset_realistic.csv)')
    
    args = parser.parse_args()
    
    # Generate dataset
    start_time = datetime.now()
    generator = ChurnDatasetGenerator(n_customers=args.rows, random_seed=args.seed)
    dataset = generator.generate_dataset()
    generation_time = (datetime.now() - start_time).total_seconds()
    
    # Save and preview
    generator.save_and_preview(dataset, args.output)
    print(f"⚡ Generation time: {generation_time:.2f} seconds")
    
    return dataset


if __name__ == "__main__":
    dataset = main() 