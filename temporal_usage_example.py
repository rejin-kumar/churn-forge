#!/usr/bin/env python3
"""
Temporal Churn Dataset Generator - Usage Examples

This script demonstrates how to use the temporal batch generator
and shows expected outputs and behavior patterns.
"""

import os
import pandas as pd
from temporal_batch_generator import TemporalBatchGenerator


def example_basic_usage():
    """Basic usage example with default settings."""
    print("🚀 EXAMPLE 1: Basic Temporal Generation")
    print("=" * 50)
    print("Generating 6 months of data with customer evolution...")
    
    # Create generator with smaller dataset for demo
    generator = TemporalBatchGenerator(
        initial_customers=100,     # Start with 100 customers
        monthly_new_customers=10,  # Add 10 new customers each month
        random_seed=42,
        output_dir="demo_temporal"
    )
    
    # Generate 6 months of data
    files = generator.generate_all_datasets(num_months=6)
    
    print(f"\n✅ Generated {len(files)} files:")
    for file in files:
        print(f"   📁 {os.path.basename(file)}")
    
    return files


def example_analyze_churn_patterns(files):
    """Analyze churn patterns across the generated datasets."""
    print("\n\n🔍 EXAMPLE 2: Analyzing Churn Patterns")
    print("=" * 50)
    
    monthly_stats = []
    customer_lifecycles = {}
    
    for file in files:
        month = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        
        # Track customer counts
        customers_this_month = set(df['ACCOUNT_ID'])
        
        # Calculate stats
        stats = {
            'month': month,
            'total_customers': len(df),
            'new_customers': len(customers_this_month - set(customer_lifecycles.keys())),
            'avg_login_success': df['SUCCESS_LOGIN'].mean(),
            'avg_support_contacts': df['TOTAL_CONTACTS'].mean(),
            'avg_billing_contacts': df['BILLING_CONTACTS'].mean(),
            'avg_product_count': df['ALL_PRODUCT_CNT'].mean(),
            'avg_spend': df['ALL_PRODUCT_NET_AMT'].mean()
        }
        
        # Calculate churn from previous month
        if monthly_stats:
            prev_customers = set()
            if len(monthly_stats) > 0:
                prev_file = files[len(monthly_stats)-1]
                prev_df = pd.read_csv(prev_file)
                prev_customers = set(prev_df['ACCOUNT_ID'])
            
            churned = prev_customers - customers_this_month
            stats['churned_customers'] = len(churned)
            stats['churn_rate'] = (len(churned) / len(prev_customers) * 100) if prev_customers else 0
        else:
            stats['churned_customers'] = 0
            stats['churn_rate'] = 0
        
        monthly_stats.append(stats)
        
        # Update customer lifecycle tracking
        for customer_id in customers_this_month:
            if customer_id not in customer_lifecycles:
                customer_lifecycles[customer_id] = []
            customer_lifecycles[customer_id].append(month)
    
    # Display analysis
    print("📊 Monthly Statistics:")
    print("-" * 100)
    print(f"{'Month':<8} {'Customers':<10} {'New':<5} {'Churned':<8} {'Churn%':<8} {'AvgLogin':<9} {'AvgSupport':<10} {'AvgSpend':<10}")
    print("-" * 100)
    
    for stats in monthly_stats:
        print(f"{stats['month']:<8} {stats['total_customers']:<10} "
              f"{stats['new_customers']:<5} {stats['churned_customers']:<8} "
              f"{stats['churn_rate']:<8.1f} {stats['avg_login_success']:<9.1f} "
              f"{stats['avg_support_contacts']:<10.1f} {stats['avg_spend']:<10.1f}")
    
    # Analyze customer lifecycles
    lifecycle_lengths = [len(months) for months in customer_lifecycles.values()]
    
    print(f"\n👥 Customer Lifecycle Analysis:")
    print(f"   Total unique customers: {len(customer_lifecycles)}")
    print(f"   Average lifecycle length: {sum(lifecycle_lengths)/len(lifecycle_lengths):.1f} months")
    print(f"   Shortest lifecycle: {min(lifecycle_lengths)} months")
    print(f"   Longest lifecycle: {max(lifecycle_lengths)} months")
    
    # Find customers who churned quickly (outliers)
    quick_churners = [cid for cid, months in customer_lifecycles.items() if len(months) <= 2]
    long_survivors = [cid for cid, months in customer_lifecycles.items() if len(months) >= 5]
    
    print(f"   Quick churners (≤2 months): {len(quick_churners)}")
    print(f"   Long survivors (≥5 months): {len(long_survivors)}")


def example_churn_behavior_analysis(files):
    """Analyze specific churn behavior patterns."""
    print("\n\n🎯 EXAMPLE 3: Churn Behavior Pattern Analysis")
    print("=" * 50)
    
    # Load all data
    all_data = []
    for file in files:
        month = os.path.basename(file).replace('.csv', '')
        df = pd.read_csv(file)
        df['month'] = month
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Track customer behavior over time
    customer_evolution = {}
    
    for _, row in combined_df.iterrows():
        customer_id = row['ACCOUNT_ID']
        month = row['month']
        
        if customer_id not in customer_evolution:
            customer_evolution[customer_id] = {}
        
        customer_evolution[customer_id][month] = {
            'success_login': row['SUCCESS_LOGIN'],
            'total_contacts': row['TOTAL_CONTACTS'],
            'billing_contacts': row['BILLING_CONTACTS'],
            'product_count': row['ALL_PRODUCT_CNT'],
            'spend': row['ALL_PRODUCT_NET_AMT']
        }
    
    # Identify churned customers (appeared in early months but not recent)
    months = sorted(combined_df['month'].unique())
    early_customers = set(combined_df[combined_df['month'] == months[0]]['ACCOUNT_ID'])
    recent_customers = set(combined_df[combined_df['month'] == months[-1]]['ACCOUNT_ID'])
    churned_customers = early_customers - recent_customers
    
    print(f"🔍 Analyzing behavior patterns for {len(churned_customers)} churned customers:")
    
    # Analyze patterns before churn
    churn_signals = {
        'decreasing_logins': 0,
        'increasing_support': 0,
        'increasing_billing': 0,
        'decreasing_products': 0,
        'decreasing_spend': 0
    }
    
    for customer_id in list(churned_customers)[:10]:  # Sample first 10
        customer_data = customer_evolution[customer_id]
        customer_months = sorted(customer_data.keys())
        
        if len(customer_months) >= 2:
            first_month = customer_data[customer_months[0]]
            last_month = customer_data[customer_months[-1]]
            
            # Check for churn signals
            if last_month['success_login'] < first_month['success_login'] * 0.7:
                churn_signals['decreasing_logins'] += 1
            
            if last_month['total_contacts'] > first_month['total_contacts'] * 1.3:
                churn_signals['increasing_support'] += 1
            
            if last_month['billing_contacts'] > first_month['billing_contacts']:
                churn_signals['increasing_billing'] += 1
            
            if last_month['product_count'] < first_month['product_count'] * 0.8:
                churn_signals['decreasing_products'] += 1
            
            if last_month['spend'] < first_month['spend'] * 0.8:
                churn_signals['decreasing_spend'] += 1
    
    print("\n📈 Churn Signal Analysis (% of churned customers showing pattern):")
    sample_size = min(10, len(churned_customers))
    for signal, count in churn_signals.items():
        percentage = (count / sample_size) * 100
        print(f"   {signal.replace('_', ' ').title()}: {percentage:.1f}% ({count}/{sample_size})")


def example_outlier_detection(files):
    """Detect and analyze outlier customers."""
    print("\n\n🎭 EXAMPLE 4: Outlier Customer Detection")
    print("=" * 50)
    
    # Load latest month data
    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    
    # Define outlier criteria
    outliers = {
        'high_spenders': df[df['ALL_PRODUCT_NET_AMT'] > df['ALL_PRODUCT_NET_AMT'].quantile(0.95)],
        'login_addicts': df[df['SUCCESS_LOGIN'] > df['SUCCESS_LOGIN'].quantile(0.95)],
        'support_heavy': df[df['TOTAL_CONTACTS'] > df['TOTAL_CONTACTS'].quantile(0.95)],
        'product_hoarders': df[df['ALL_PRODUCT_CNT'] > df['ALL_PRODUCT_CNT'].quantile(0.95)]
    }
    
    print("🔍 Outlier Customer Analysis:")
    print("-" * 60)
    
    for outlier_type, outlier_df in outliers.items():
        if len(outlier_df) > 0:
            print(f"\n{outlier_type.replace('_', ' ').title()} ({len(outlier_df)} customers):")
            
            if outlier_type == 'high_spenders':
                avg_spend = outlier_df['ALL_PRODUCT_NET_AMT'].mean()
                print(f"   Average spend: ${avg_spend:.2f}")
                print(f"   Spend range: ${outlier_df['ALL_PRODUCT_NET_AMT'].min():.2f} - ${outlier_df['ALL_PRODUCT_NET_AMT'].max():.2f}")
            
            elif outlier_type == 'login_addicts':
                avg_logins = outlier_df['SUCCESS_LOGIN'].mean()
                print(f"   Average successful logins: {avg_logins:.1f}")
                print(f"   Login range: {outlier_df['SUCCESS_LOGIN'].min()} - {outlier_df['SUCCESS_LOGIN'].max()}")
            
            elif outlier_type == 'support_heavy':
                avg_contacts = outlier_df['TOTAL_CONTACTS'].mean()
                avg_billing = outlier_df['BILLING_CONTACTS'].mean()
                print(f"   Average total contacts: {avg_contacts:.1f}")
                print(f"   Average billing contacts: {avg_billing:.1f}")
            
            elif outlier_type == 'product_hoarders':
                avg_products = outlier_df['ALL_PRODUCT_CNT'].mean()
                print(f"   Average product count: {avg_products:.1f}")
                print(f"   Product range: {outlier_df['ALL_PRODUCT_CNT'].min()} - {outlier_df['ALL_PRODUCT_CNT'].max()}")


def main():
    """Run all examples."""
    print("🎯 TEMPORAL CHURN DATASET GENERATOR - EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the temporal dataset generator")
    print("and analyzes the realistic churn patterns it creates.")
    print("=" * 60)
    
    # Run basic generation
    files = example_basic_usage()
    
    if files:
        # Run analysis examples
        example_analyze_churn_patterns(files)
        example_churn_behavior_analysis(files)
        example_outlier_detection(files)
        
        print("\n\n🎉 EXAMPLES COMPLETED!")
        print("=" * 40)
        print("Key takeaways:")
        print("✅ Customers show realistic evolution over time")
        print("✅ Churn patterns are observable in behavior changes")
        print("✅ Monthly churn rate stays around 2-3%")
        print("✅ New customers are added each month")
        print("✅ Outlier behaviors are included (~5% of customers)")
        print("\nCheck the 'demo_temporal' directory for generated files!")
    
    else:
        print("❌ No files were generated. Check for errors above.")


if __name__ == "__main__":
    main() 