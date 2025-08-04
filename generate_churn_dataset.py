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
    
    def __init__(self, n_customers=1000, random_seed=None, reference_date="2024-12-01"):
        """
        Initialize the generator.
        
        Args:
            n_customers (int): Number of customer records to generate
            random_seed (int, optional): Seed for reproducible results
            reference_date (str): Reference date for data generation
        """
        self.n_customers = n_customers
        self.random_seed = random_seed
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        
        # Set seeds for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            Faker.seed(random_seed)
        
        self.fake = Faker()
        
    def generate_basic_info(self):
        """Generate basic customer and service information."""
        # Generate service creation dates (spread over past 3 years)
        max_days_ago = 3 * 365
        days_ago = np.random.randint(1, max_days_ago, self.n_customers)
        service_created_dates = [
            (self.reference_date - timedelta(days=int(d))).date()
            for d in days_ago
        ]
        
        # Generate expiration dates (1-24 months after creation)
        expiration_dates = []
        deletion_dates = []
        
        for created_date in service_created_dates:
            # Term length (typically 1, 12, or 24 months)
            term_choices = [1, 3, 6, 12, 24]
            term = np.random.choice(term_choices, p=[0.1, 0.2, 0.2, 0.4, 0.1])
            expiration_date = created_date + timedelta(days=int(term * 30))
            expiration_dates.append(expiration_date)
            
            # Some services are deleted (10% chance)
            if np.random.random() < 0.1:
                deletion_date = created_date + timedelta(days=int(np.random.randint(1, term * 30)))
                deletion_dates.append(deletion_date)
            else:
                deletion_dates.append(None)
        
        return {
            'FIRST_OF_MONTH': [self.reference_date.replace(day=1).date()] * self.n_customers,
            'ACCOUNT_ID': [10000 + i for i in range(self.n_customers)],
            'PERSON_ORG_ID': [20000 + i for i in range(self.n_customers)],
            'PP_SERVICE_CREATED_DATE': service_created_dates,
            'PP_SERVICE_DELETION_DATE': deletion_dates,
            'PP_EXPIRATION_DATE': expiration_dates,
            'PP_PREMIUM_FLAG': np.random.choice(['Y', 'N'], self.n_customers, p=[0.15, 0.85]),
            'PP_BUNDLE_FLAG': np.random.choice(['Y', 'N'], self.n_customers, p=[0.3, 0.7]),
            'STATE': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH'], self.n_customers)
        }
    
    def generate_financial_metrics(self, bundle_flags, premium_flags):
        """Generate financial and product metrics."""
        # Base pricing varies by product type
        base_amounts = np.random.lognormal(mean=3.2, sigma=0.8, size=self.n_customers)
        base_amounts = np.clip(base_amounts, 10, 500)
        
        # Premium domains cost more
        premium_multiplier = np.where(np.array(premium_flags) == 'Y', 
                                     np.random.uniform(2, 10, self.n_customers), 1)
        base_amounts *= premium_multiplier
        
        # Bundle discounts
        bundle_discount = np.where(np.array(bundle_flags) == 'Y',
                                  np.random.uniform(0.1, 0.3, self.n_customers), 0)
        
        pp_net_amt = base_amounts
        pp_discount_amt = base_amounts * bundle_discount
        
        # Term lengths
        term_choices = [1, 3, 6, 12, 24]
        term_probs = [0.1, 0.2, 0.2, 0.4, 0.1]
        terms = np.random.choice(term_choices, self.n_customers, p=term_probs)
        
        # Renewal counts based on service age
        renewal_counts = np.random.poisson(lam=2, size=self.n_customers)
        renewal_counts = np.clip(renewal_counts, 0, 10)
        
        return {
            'PP_DISCOUNT_AMT': np.round(pp_discount_amt, 2),
            'PP_NET_AMT': np.round(pp_net_amt, 2),
            'TERM_IN_MONTHS': terms,
            'RENEWAL_COUNT': renewal_counts,
        }
    
    def generate_product_counts_and_amounts(self):
        """Generate product counts and corresponding amounts."""
        # All products
        all_product_cnt = np.random.poisson(lam=3, size=self.n_customers)
        all_product_cnt = np.clip(all_product_cnt, 1, 15)
        all_product_net_amt = all_product_cnt * np.random.lognormal(3, 0.5, self.n_customers)
        
        # Individual product categories (each is subset of all products)
        product_categories = [
            'ECOMMERC', 'ADV_HOSTING', 'HOSTING', 'DIY_WEBSIT', 'PRO_SERVICES',
            'HOSTING_ADD_ONS', 'EMAIL_PRODUCTIVITY', 'SECURITY', 'PREMIUM_DOMAIN', 
            'DOMAIN_VAS', 'DOMAIN'
        ]
        
        product_data = {}
        
        for category in product_categories:
            # Each category gets 0 to some fraction of all products
            max_for_category = np.maximum(1, all_product_cnt // 3)
            counts = np.random.poisson(lam=0.8, size=self.n_customers)
            counts = np.minimum(counts, max_for_category)
            
            # Amounts correlate with counts
            amounts = counts * np.random.lognormal(2.5, 0.8, self.n_customers)
            
            product_data[f'{category}_PRODUCT_CNT'] = counts
            product_data[f'{category}_NET_AMT'] = np.round(amounts, 2)
        
        product_data['ALL_PRODUCT_CNT'] = all_product_cnt
        product_data['ALL_PRODUCT_NET_AMT'] = np.round(all_product_net_amt, 2)
        
        # Add DOMAIN_NET_AMT as sum of domain-related amounts
        product_data['DOMAIN_NET_AMT'] = np.round(
            product_data['DOMAIN_PRODUCT_CNT'] * np.random.lognormal(2.5, 0.8, self.n_customers), 2
        )
        
        # Generate flags
        product_data['HAS_ECOMMERCE'] = np.where(product_data['ECOMMERC_PRODUCT_CNT'] > 0, 'Y', 'N')
        product_data['HAS_WORDPRESS'] = np.random.choice(['Y', 'N'], self.n_customers, p=[0.4, 0.6])
        
        return product_data
    
    def generate_nps_and_support_metrics(self):
        """Generate NPS scores and support contact metrics."""
        # NPS scores
        nps_promoter = np.random.poisson(lam=0.3, size=self.n_customers)
        nps_detractor = np.random.poisson(lam=0.2, size=self.n_customers)
        
        # Support contacts by category
        support_categories = [
            'TOTAL', 'WORDPRESS', 'DOMAIN', 'EMAIL', 'CPANEL', 'ACCOUNT', 
            'BILLING', 'RETENTION', 'SALES', 'SSL', 'TOS'
        ]
        
        support_data = {}
        
        # Generate total contacts first
        total_contacts = np.random.poisson(lam=2, size=self.n_customers)
        total_contacts = np.clip(total_contacts, 0, 20)
        support_data['TOTAL_CONTACTS'] = total_contacts
        
        # Other contact types are subsets of total
        for category in support_categories[1:]:  # Skip TOTAL
            if category == 'BILLING':
                # Billing contacts are more critical indicator
                contacts = np.random.poisson(lam=0.5, size=self.n_customers)
            elif category == 'RETENTION':
                # Retention contacts are red flag
                contacts = np.random.poisson(lam=0.2, size=self.n_customers)
            else:
                # Regular support contacts
                contacts = np.random.poisson(lam=0.8, size=self.n_customers)
            
            # Ensure subset of total
            contacts = np.minimum(contacts, total_contacts)
            support_data[f'{category}_CONTACTS'] = contacts
        
        support_data['NPS_PROMOTER_COUNT'] = nps_promoter
        support_data['NPS_DETRACTOR_COUNT'] = nps_detractor
        
        return support_data
    
    def generate_login_metrics(self):
        """Generate login activity metrics."""
        # Total login attempts
        total_logins = np.random.poisson(lam=10, size=self.n_customers)
        total_logins = np.clip(total_logins, 0, 100)
        
        # Success rate varies (most customers have good success rate)
        success_rates = np.random.beta(a=8, b=2, size=self.n_customers)
        successful_logins = np.round(total_logins * success_rates).astype(int)
        
        return {
            'SUCCESS_LOGIN': successful_logins,
            'TOTAL_LOGIN': total_logins
        }
    
    def calculate_churn_score(self, df):
        """
        Calculate churn score based on the rule-based logic from PRD.
        
        Args:
            df (pd.DataFrame): DataFrame with customer data
            
        Returns:
            np.array: Churn scores for each customer
        """
        scores = np.zeros(len(df))
        
        # Calculate service age
        today = self.reference_date
        service_ages = (today - pd.to_datetime(df['PP_SERVICE_CREATED_DATE'])).dt.days
        
        # Calculate days to expiration
        days_to_expiration = (pd.to_datetime(df['PP_EXPIRATION_DATE']) - today).dt.days
        
        # Apply scoring rules from PRD
        conditions = [
            ('Service Age < 90 days', service_ages < 90, 2),
            ('Service Expiring Soon (< 30 days)', days_to_expiration < 30, 3),
            ('Low Product Diversity (< 2)', df['ALL_PRODUCT_CNT'] < 2, 1),
            ('Low Spend (< $50)', df['ALL_PRODUCT_NET_AMT'] < 50, 2),
            ('No Renewals', df['RENEWAL_COUNT'] == 0, 2),
            ('High Support Contact (> 5)', df['TOTAL_CONTACTS'] > 5, 1),
            ('Billing Issues (> 2)', df['BILLING_CONTACTS'] > 2, 2),
            ('Retention Calls (> 0)', df['RETENTION_CONTACTS'] > 0, 1),
            ('Low Login Activity (0 success OR < 5 total)', 
             (df['SUCCESS_LOGIN'] == 0) | (df['TOTAL_LOGIN'] < 5), 2),
            ('NPS Detractor (> 0)', df['NPS_DETRACTOR_COUNT'] > 0, 1),
            ('Premium Domain Issues', 
             (df['PP_PREMIUM_FLAG'] == 'Y') & (df['PREMIUM_DOMAIN_PRODUCT_CNT'] == 0), 1),
            ('Short Term Contract (≤ 3 months)', df['TERM_IN_MONTHS'] <= 3, 1),
            ('High Discount Dependency (> 50%)', 
             df['PP_DISCOUNT_AMT'] / np.maximum(df['PP_NET_AMT'], 1) > 0.5, 1),
            ('No Core Products', 
             (df['HOSTING_PRODUCT_CNT'] == 0) & (df['DOMAIN_PRODUCT_CNT'] == 0), 2),
            ('Technical Support Issues (> 3)', 
             df['CPANEL_CONTACTS'] + df['EMAIL_CONTACTS'] > 3, 1)
        ]
        
        # Sum up the scores
        for rule_name, condition, weight in conditions:
            scores += (condition.astype(int) * weight)
        
        return scores

    
    def generate_dataset(self):
        """Generate the complete synthetic dataset."""
        print(f"Generating {self.n_customers} customer records...")
        
        # Generate all data components
        basic_info = self.generate_basic_info()
        financial_metrics = self.generate_financial_metrics(
            basic_info['PP_BUNDLE_FLAG'], 
            basic_info['PP_PREMIUM_FLAG']
        )
        product_data = self.generate_product_counts_and_amounts()
        support_data = self.generate_nps_and_support_metrics()
        login_data = self.generate_login_metrics()
        
        # Combine into DataFrame
        data = {**basic_info, **financial_metrics, **product_data, **support_data, **login_data}
        df = pd.DataFrame(data)
        
        # Calculate churn scores (no longer assigning status)
        churn_scores = self.calculate_churn_score(df)
        
        # Add churn score for analysis (not in final output)
        df['_churn_score'] = churn_scores
        
        # Define column order to match PRD schema
        column_order = [
            'FIRST_OF_MONTH', 'ACCOUNT_ID', 'PERSON_ORG_ID',
            'PP_SERVICE_CREATED_DATE', 'PP_SERVICE_DELETION_DATE', 'PP_EXPIRATION_DATE',
            'PP_PREMIUM_FLAG', 'PP_BUNDLE_FLAG', 'PP_DISCOUNT_AMT', 'PP_NET_AMT',
            'TERM_IN_MONTHS', 'RENEWAL_COUNT', 'ALL_PRODUCT_CNT', 'ALL_PRODUCT_NET_AMT',
            'ECOMMERC_PRODUCT_CNT', 'ECOMMERC_NET_AMT',
            'ADV_HOSTING_PRODUCT_CNT', 'ADV_HOSTING_NET_AMT',
            'HOSTING_PRODUCT_CNT', 'HOSTING_NET_AMT',
            'DIY_WEBSIT_PRODUCT_CNT', 'DIY_WEBSIT_NET_AMT',
            'PRO_SERVICES_PRODUCT_CNT', 'PRO_SERVICES_NET_AMT',
            'HOSTING_ADD_ONS_PRODUCT_CNT', 'HOSTING_ADD_ONS_NET_AMT',
            'EMAIL_PRODUCTIVITY_PRODUCT_CNT', 'EMAIL_PRODUCTIVITY_NET_AMT',
            'SECURITY_PRODUCT_CNT', 'SECURITY_NET_AMT',
            'PREMIUM_DOMAIN_PRODUCT_CNT', 'PREMIUM_DOMAIN_NET_AMT',
            'DOMAIN_VAS_PRODUCT_CNT', 'DOMAIN_VAS_NET_AMT',
            'DOMAIN_PRODUCT_CNT', 'DOMAIN_NET_AMT', 'STATE', 'HAS_ECOMMERCE', 'HAS_WORDPRESS',
            'NPS_PROMOTER_COUNT', 'NPS_DETRACTOR_COUNT',
            'TOTAL_CONTACTS', 'WORDPRESS_CONTACTS', 'DOMAIN_CONTACTS', 'EMAIL_CONTACTS',
            'CPANEL_CONTACTS', 'ACCOUNT_CONTACTS', 'BILLING_CONTACTS', 'RETENTION_CONTACTS',
            'SALES_CONTACTS', 'SSL_CONTACTS', 'TOS_CONTACTS',
            'SUCCESS_LOGIN', 'TOTAL_LOGIN'
        ]
        
        return df[column_order]
    
    def save_and_preview(self, df, filename='synthetic_churn_dataset.csv'):
        """Save dataset and show preview."""
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\n✅ Dataset saved to: {filename}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Churn status column removed - no longer showing churn rate
        
        # Show column count
        print(f"📋 Total columns: {len(df.columns)}")
        
        # Preview first 5 rows (key columns only)
        preview_cols = [
            'ACCOUNT_ID', 'PP_SERVICE_CREATED_DATE', 'ALL_PRODUCT_CNT', 
            'ALL_PRODUCT_NET_AMT', 'TOTAL_CONTACTS'
        ]
        print(f"\n📋 Preview (first 5 rows, key columns):")
        print(df[preview_cols].head().to_string(index=False))
        
        return filename


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Generate synthetic customer churn dataset')
    parser.add_argument('--rows', '-n', type=int, default=1000, 
                       help='Number of customer records to generate (default: 1000)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results')
    parser.add_argument('--output', '-o', type=str, default='synthetic_churn_dataset.csv',
                       help='Output filename (default: synthetic_churn_dataset.csv)')
    parser.add_argument('--reference-date', '-d', type=str, default='2024-12-01',
                       help='Reference date for data generation (default: 2024-12-01)')
    
    args = parser.parse_args()
    
    # Generate dataset
    start_time = datetime.now()
    generator = ChurnDatasetGenerator(
        n_customers=args.rows, 
        random_seed=args.seed,
        reference_date=args.reference_date
    )
    dataset = generator.generate_dataset()
    generation_time = (datetime.now() - start_time).total_seconds()
    
    # Save and preview
    generator.save_and_preview(dataset, args.output)
    print(f"⚡ Generation time: {generation_time:.2f} seconds")
    
    return dataset


if __name__ == "__main__":
    dataset = main() 