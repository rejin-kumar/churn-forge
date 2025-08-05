#!/usr/bin/env python3
"""
Configuration-Based Temporal Dataset Generator

Generates temporal synthetic datasets with realistic customer lifecycle patterns
using JSON configuration files. Extends the base config generator with temporal logic.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from config_dataset_generator import ConfigDatasetGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        
        # Lifecycle stage
        self.lifecycle_stage = "new"
        self.segment = "standard"
        
    def evolve_behavior(self, temporal_config: Dict, current_month: str) -> Dict:
        """Evolve customer behavior for the current month."""
        behavior_config = temporal_config.get('behavioral_evolution', {})
        
        # Increment months since creation
        self.months_since_creation += 1
        
        # Update lifecycle stage
        self._update_lifecycle_stage(temporal_config)
        
        # Apply behavioral evolution patterns
        evolved_data = self.data.copy()
        
        # Login activity evolution
        login_config = behavior_config.get('login_activity', {})
        if self.lifecycle_stage == "at_risk":
            # Degrade login activity for at-risk customers
            degradation = login_config.get('degradation_rate', 0.15)
            evolved_data['SUCCESS_LOGIN'] = max(0, int(evolved_data.get('SUCCESS_LOGIN', 0) * (1 - degradation)))
            evolved_data['TOTAL_LOGIN'] = max(evolved_data['SUCCESS_LOGIN'], 
                                            int(evolved_data.get('TOTAL_LOGIN', 0) * (1 - degradation)))
        elif self.lifecycle_stage == "loyal":
            # Improve login activity for loyal customers
            improvement = login_config.get('improvement_rate', 0.05)
            evolved_data['SUCCESS_LOGIN'] = int(evolved_data.get('SUCCESS_LOGIN', 0) * (1 + improvement))
            evolved_data['TOTAL_LOGIN'] = int(evolved_data.get('TOTAL_LOGIN', 0) * (1 + improvement))
        
        # Support contacts evolution
        support_config = behavior_config.get('support_contacts', {})
        if self.lifecycle_stage == "at_risk":
            # Increase support contacts for at-risk customers
            increase = support_config.get('pre_churn_increase', 2.0)
            evolved_data['TOTAL_CONTACTS'] = int(evolved_data.get('TOTAL_CONTACTS', 0) * increase)
            evolved_data['BILLING_CONTACTS'] = min(8, int(evolved_data.get('BILLING_CONTACTS', 0) * increase))
        
        # Spending pattern evolution
        spending_config = behavior_config.get('spending_patterns', {})
        if self.lifecycle_stage == "at_risk":
            # Reduce spending for at-risk customers
            reduction = spending_config.get('churn_reduction_rate', 0.25)
            evolved_data['ALL_PRODUCT_NET_AMT'] = evolved_data.get('ALL_PRODUCT_NET_AMT', 0) * (1 - reduction)
            evolved_data['PP_NET_AMT'] = evolved_data.get('PP_NET_AMT', 0) * (1 - reduction)
        elif self.lifecycle_stage == "loyal":
            # Increase spending for loyal customers
            increase = spending_config.get('loyalty_increase_rate', 0.1)
            evolved_data['ALL_PRODUCT_NET_AMT'] = evolved_data.get('ALL_PRODUCT_NET_AMT', 0) * (1 + increase)
        
        # Product adoption evolution
        product_config = behavior_config.get('product_adoption', {})
        if self.lifecycle_stage == "growing" and np.random.random() < product_config.get('expansion_probability', 0.15):
            # Add products for growing customers
            evolved_data['ALL_PRODUCT_CNT'] = min(15, evolved_data.get('ALL_PRODUCT_CNT', 1) + 1)
        elif self.lifecycle_stage == "at_risk" and np.random.random() < product_config.get('contraction_probability', 0.08):
            # Remove products for at-risk customers
            evolved_data['ALL_PRODUCT_CNT'] = max(1, evolved_data.get('ALL_PRODUCT_CNT', 1) - 1)
        
        # Add temporal-specific fields
        evolved_data['MONTHS_SINCE_SIGNUP'] = self.months_since_creation
        evolved_data['CUSTOMER_LIFECYCLE_STAGE'] = self.lifecycle_stage
        evolved_data['CUSTOMER_SEGMENT'] = self.segment
        
        # Calculate churn probability score
        evolved_data['CHURN_PROBABILITY_SCORE'] = self._calculate_churn_probability(temporal_config, evolved_data)
        
        # Calculate change score from previous month
        evolved_data['PREV_MONTH_CHANGE_SCORE'] = self._calculate_change_score()
        
        # Update stored data
        self.data = evolved_data
        return evolved_data
    
    def _update_lifecycle_stage(self, temporal_config: Dict):
        """Update customer lifecycle stage based on behavior and time."""
        segments = temporal_config.get('customer_segments', {})
        
        # Determine lifecycle stage
        if self.months_since_creation <= segments.get('new_customer', {}).get('onboarding_period_months', 3):
            self.lifecycle_stage = "new"
        elif self.months_since_creation >= segments.get('loyal', {}).get('loyalty_threshold_months', 12):
            self.lifecycle_stage = "loyal"
        elif self.churn_risk_trend > 0.6:
            self.lifecycle_stage = "at_risk"
        elif self.data.get('ALL_PRODUCT_CNT', 1) > self.data.get('_prev_product_count', 1):
            self.lifecycle_stage = "growing"
        else:
            self.lifecycle_stage = "mature"
        
        # Determine segment
        high_value = segments.get('high_value', {})
        if (self.data.get('ALL_PRODUCT_NET_AMT', 0) > high_value.get('spend_threshold', 200) and
            self.data.get('ALL_PRODUCT_CNT', 0) >= high_value.get('product_count_min', 5)):
            self.segment = "high_value"
        elif self.lifecycle_stage == "at_risk":
            self.segment = "at_risk"
        elif self.lifecycle_stage == "new":
            self.segment = "new_customer"
        elif self.lifecycle_stage == "loyal":
            self.segment = "loyal"
        else:
            self.segment = "standard"
    
    def _calculate_churn_probability(self, temporal_config: Dict, current_data: Dict) -> float:
        """Calculate ML-ready churn probability score."""
        score = 0.0
        indicators = temporal_config.get('churn_prediction_features', {}).get('leading_indicators', [])
        
        for indicator in indicators:
            feature = indicator['feature']
            weight = indicator.get('weight', 0.5)
            
            if feature == "login_frequency_decline":
                prev_logins = self.data.get('SUCCESS_LOGIN', 0)
                curr_logins = current_data.get('SUCCESS_LOGIN', 0)
                if prev_logins > 0:
                    decline_rate = (prev_logins - curr_logins) / prev_logins
                    if decline_rate > abs(indicator.get('threshold_change', -0.4)):
                        score += weight
            
            elif feature == "support_contact_spike":
                prev_contacts = self.data.get('TOTAL_CONTACTS', 0)
                curr_contacts = current_data.get('TOTAL_CONTACTS', 0)
                if curr_contacts - prev_contacts >= indicator.get('threshold_increase', 3):
                    score += weight
            
            elif feature == "billing_issues_frequency":
                billing_contacts = current_data.get('BILLING_CONTACTS', 0)
                if billing_contacts >= indicator.get('threshold_count', 2):
                    score += weight
        
        # Apply lifecycle stage modifiers
        if self.lifecycle_stage == "at_risk":
            score += 0.3
        elif self.lifecycle_stage == "loyal":
            score = max(0, score - 0.2)
        elif self.lifecycle_stage == "new":
            score = max(0, score - 0.1)
        
        return min(1.0, score)
    
    def _calculate_change_score(self) -> float:
        """Calculate behavioral change score from previous month."""
        # Simple implementation - could be enhanced
        if not hasattr(self, '_prev_data'):
            return 0.0
        
        changes = 0
        total_features = 0
        
        key_features = ['SUCCESS_LOGIN', 'TOTAL_CONTACTS', 'ALL_PRODUCT_NET_AMT', 'ALL_PRODUCT_CNT']
        for feature in key_features:
            if feature in self.data and feature in self._prev_data:
                old_val = self._prev_data[feature] or 0
                new_val = self.data[feature] or 0
                if old_val > 0:
                    change_pct = abs((new_val - old_val) / old_val)
                    changes += change_pct
                total_features += 1
        
        return changes / max(1, total_features)
    
    def should_churn(self, temporal_config: Dict, current_month: str) -> bool:
        """Determine if customer should churn this month."""
        lifecycle_config = temporal_config.get('lifecycle_behavior', {})
        monthly_churn_rate = lifecycle_config.get('monthly_churn_rate', 0.025)
        
        # Apply lifecycle stage modifiers
        churn_probability = monthly_churn_rate
        
        if self.lifecycle_stage == "new":
            protection = temporal_config.get('customer_segments', {}).get('new_customer', {}).get('initial_churn_protection', 0.8)
            churn_probability *= (1 - protection)
        elif self.lifecycle_stage == "loyal":
            reduction = temporal_config.get('customer_segments', {}).get('loyal', {}).get('churn_rate_reduction', 0.6)
            churn_probability *= (1 - reduction)
        elif self.lifecycle_stage == "at_risk":
            churn_probability *= 3  # 3x higher churn for at-risk customers
        
        # Apply seasonal effects
        seasonal_effects = lifecycle_config.get('seasonal_effects', {})
        if seasonal_effects.get('enabled', False):
            month_num = int(current_month.split('-')[1])
            effect_strength = seasonal_effects.get('effect_strength', 0.3)
            
            if month_num in seasonal_effects.get('high_churn_months', []):
                churn_probability *= (1 + effect_strength)
            elif month_num in seasonal_effects.get('low_churn_months', []):
                churn_probability *= (1 - effect_strength)
        
        return np.random.random() < churn_probability


class ConfigTemporalGenerator:
    """Generates temporal datasets using configuration files."""
    
    def __init__(self, temporal_config_path: str):
        """
        Initialize the temporal generator.
        
        Args:
            temporal_config_path (str): Path to temporal configuration JSON file
        """
        self.temporal_config_path = temporal_config_path
        self.temporal_config = self._load_temporal_config()
        
        # Load base configuration
        base_config_path = self.temporal_config.get('base_config_path', 'template_config.json')
        self.base_generator = ConfigDatasetGenerator(base_config_path)
        
        # Temporal parameters
        temporal_params = self.temporal_config.get('temporal_params', {})
        self.initial_customers = temporal_params.get('initial_customers', 1000)
        self.monthly_new_customers = temporal_params.get('monthly_new_customers', 50)
        self.num_months = temporal_params.get('num_months', 12)
        self.start_month = temporal_params.get('start_month', '2024-07')
        self.output_dir = temporal_params.get('output_dir', 'temporal_datasets')
        
        # Customer state tracking
        self.customers: Dict[int, CustomerState] = {}
        self.next_customer_id = 1
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        logger.info(f"Initialized temporal generator: {self.temporal_config['name']}")
        logger.info(f"Will generate {self.num_months} months starting from {self.start_month}")
    
    def _load_temporal_config(self) -> Dict:
        """Load and validate the temporal configuration file."""
        try:
            with open(self.temporal_config_path, 'r') as f:
                config = json.load(f)
            return config['temporal_config']
        except FileNotFoundError:
            raise FileNotFoundError(f"Temporal config file not found: {self.temporal_config_path}")
        except KeyError:
            raise ValueError("Temporal config file must contain 'temporal_config' key")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in temporal config file: {e}")
    
    def generate_initial_customer_base(self, month: str) -> pd.DataFrame:
        """Generate the initial customer base for the first month."""
        logger.info(f"Generating initial customer base: {self.initial_customers} customers")
        
        # Generate base dataset
        self.base_generator.n_records = self.initial_customers
        df = self.base_generator.generate_dataset()
        
        # Create customer states
        for idx, row in df.iterrows():
            customer_id = self.next_customer_id
            self.next_customer_id += 1
            
            # Update customer ID in dataframe
            df.at[idx, 'ACCOUNT_ID'] = customer_id
            
            # Create customer state
            customer_data = row.to_dict()
            customer_state = CustomerState(customer_id, month, customer_data)
            self.customers[customer_id] = customer_state
        
        # Add temporal columns
        df = self._add_temporal_columns(df, month)
        
        return df
    
    def generate_monthly_dataset(self, month: str, is_first_month: bool = False) -> pd.DataFrame:
        """Generate dataset for a specific month."""
        logger.info(f"Generating dataset for {month}")
        
        if is_first_month:
            return self.generate_initial_customer_base(month)
        
        # Evolve existing customers
        active_customers = []
        churned_customers = []
        
        for customer_id, customer_state in list(self.customers.items()):
            if customer_state.is_active:
                # Check if customer should churn
                if customer_state.should_churn(self.temporal_config, month):
                    customer_state.is_active = False
                    customer_state.churn_month = month
                    churned_customers.append(customer_id)
                    logger.debug(f"Customer {customer_id} churned in {month}")
                else:
                    # Evolve customer behavior
                    evolved_data = customer_state.evolve_behavior(self.temporal_config, month)
                    active_customers.append(evolved_data)
        
        logger.info(f"Month {month}: {len(active_customers)} active, {len(churned_customers)} churned")
        
        # Add new customers
        new_customers = self._generate_new_customers(month)
        all_customers = active_customers + new_customers
        
        # Convert to DataFrame
        df = pd.DataFrame(all_customers)
        
        # Add temporal columns
        df = self._add_temporal_columns(df, month)
        
        return df
    
    def _generate_new_customers(self, month: str) -> List[Dict]:
        """Generate new customers for the month."""
        if self.monthly_new_customers == 0:
            return []
        
        logger.info(f"Adding {self.monthly_new_customers} new customers")
        
        # Generate new customer data
        self.base_generator.n_records = self.monthly_new_customers
        new_df = self.base_generator.generate_dataset()
        
        new_customers = []
        for idx, row in new_df.iterrows():
            customer_id = self.next_customer_id
            self.next_customer_id += 1
            
            # Update customer data
            customer_data = row.to_dict()
            customer_data['ACCOUNT_ID'] = customer_id
            
            # Create customer state
            customer_state = CustomerState(customer_id, month, customer_data)
            self.customers[customer_id] = customer_state
            
            new_customers.append(customer_data)
        
        return new_customers
    
    def _add_temporal_columns(self, df: pd.DataFrame, month: str) -> pd.DataFrame:
        """Add temporal-specific columns to the dataset."""
        # Add month identifier
        df['FIRST_OF_MONTH'] = datetime.strptime(f"{month}-01", "%Y-%m-%d").date()
        
        # Ensure temporal columns exist (they should be added by customer evolution)
        temporal_columns = {
            'MONTHS_SINCE_SIGNUP': 1,
            'CUSTOMER_LIFECYCLE_STAGE': 'new',
            'CHURN_PROBABILITY_SCORE': 0.1,
            'CUSTOMER_SEGMENT': 'standard',
            'PREV_MONTH_CHANGE_SCORE': 0.0
        }
        
        for col, default_val in temporal_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
    
    def generate_all_datasets(self) -> List[str]:
        """Generate all monthly datasets."""
        logger.info(f"Starting temporal generation: {self.num_months} months")
        
        generated_files = []
        start_date = datetime.strptime(f"{self.start_month}-01", "%Y-%m-%d")
        
        for month_idx in range(self.num_months):
            current_date = start_date + relativedelta(months=month_idx)
            month_str = current_date.strftime("%Y-%m")
            
            # Generate monthly dataset
            df = self.generate_monthly_dataset(month_str, is_first_month=(month_idx == 0))
            
            # Save dataset
            filename = f"{month_str}.csv"
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            generated_files.append(filepath)
            
            logger.info(f"Saved {month_str}.csv ({len(df)} records)")
        
        # Generate summary statistics
        summary_data = []
        for file_path in generated_files:
            month = os.path.basename(file_path).replace('.csv', '')
            df = pd.read_csv(file_path)
            
            # Find spending column (flexible detection)
            spending_cols = [col for col in df.columns if 'spend' in col.lower() or 'amount' in col.lower()]
            avg_spending = df[spending_cols[0]].mean() if spending_cols else 0
            
            # Find product count column (flexible detection)  
            product_cols = [col for col in df.columns if 'product' in col.lower() and 'cnt' in col.lower()]
            if not product_cols:
                product_cols = [col for col in df.columns if 'product' in col.lower() and 'count' in col.lower()]
            avg_products = df[product_cols[0]].mean() if product_cols else 0
            
            # Find churn probability column (flexible detection)
            churn_prob_cols = [col for col in df.columns if 'churn' in col.lower() and ('prob' in col.lower() or 'score' in col.lower())]
            if not churn_prob_cols:
                churn_prob_cols = [col for col in df.columns if 'risk' in col.lower() and 'score' in col.lower()]
            
            avg_churn_prob = df[churn_prob_cols[0]].mean() if churn_prob_cols else 0
            high_risk = len(df[df[churn_prob_cols[0]] > 0.7]) if churn_prob_cols else 0
            
            # Find lifecycle stage column (flexible detection)
            lifecycle_cols = [col for col in df.columns if 'lifecycle' in col.lower() or 'stage' in col.lower()]
            new_customers = len(df[df[lifecycle_cols[0]] == 'new']) if lifecycle_cols else 0
            
            summary = {
                'month': month,
                'total_customers': len(df),
                'new_customers': new_customers,
                'churned_customers': len([c for c in self.customers.values() 
                                        if c.churn_month == month]),
                'avg_churn_probability': avg_churn_prob,
                'high_risk_customers': high_risk,
                'avg_spending': avg_spending,
                'avg_products': avg_products
            }
            summary_data.append(summary)
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'temporal_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Generated summary statistics: {summary_path}")
        
        # Print key metrics
        total_generated = summary_df['total_customers'].sum()
        total_churned = summary_df['churned_customers'].sum()
        
        print(f"\n📊 Temporal Generation Summary")
        print(f"Files generated: {len(generated_files)}")
        print(f"Total customer records: {total_generated:,}")
        print(f"Total churned: {total_churned}")
        print(f"📁 Files saved to: {self.output_dir}")
        
        logger.info(f"✅ Generated {len(generated_files)} temporal datasets")
        return generated_files


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Generate temporal datasets from configuration file')
    parser.add_argument('--config', '-c', type=str, default='template_config.json',
                        help='Path to the configuration file')
    parser.add_argument('--months', '-m', type=int, default=None,
                       help='Number of months to generate (overrides config)')
    parser.add_argument('--initial-customers', type=int, default=None,
                       help='Initial customer count (overrides config)')
    parser.add_argument('--monthly-new', type=int, default=None,
                       help='New customers per month (overrides config)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # Generate temporal datasets
    start_time = datetime.now()
    generator = ConfigTemporalGenerator(args.config)
    
    # Override config with command line arguments if provided
    if args.months is not None:
        generator.num_months = args.months
    if args.initial_customers is not None:
        generator.initial_customers = args.initial_customers
    if args.monthly_new is not None:
        generator.monthly_new_customers = args.monthly_new
    if args.output_dir is not None:
        generator.output_dir = args.output_dir
        Path(generator.output_dir).mkdir(exist_ok=True)
    
    generated_files = generator.generate_all_datasets()
    generation_time = (datetime.now() - start_time).total_seconds()
    
    print(f"⚡ Total generation time: {generation_time:.2f} seconds")
    print(f"📁 Files saved to: {generator.output_dir}")
    
    return generated_files


if __name__ == "__main__":
    generated_files = main() 