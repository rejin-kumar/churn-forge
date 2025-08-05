#!/usr/bin/env python3
"""
Configuration-Based Synthetic Dataset Generator

Generates synthetic datasets for any domain using JSON configuration files.
Supports flexible schemas, business rules, and realistic patterns.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
import re
from datetime import datetime, timedelta
from faker import Faker
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigDatasetGenerator:
    """Generates synthetic datasets based on JSON configuration files."""
    
    def __init__(self, config_path: str):
        """
        Initialize the generator with a configuration file.
        
        Args:
            config_path (str): Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.dataset_config = self.config['dataset_config']
        
        # Extract generation parameters
        gen_params = self.dataset_config['generation_params']
        self.n_records = gen_params.get('n_records', 1000)
        self.random_seed = gen_params.get('random_seed')
        self.reference_date = datetime.strptime(
            gen_params.get('reference_date', '2024-12-01'), "%Y-%m-%d"
        )
        self.output_filename = gen_params.get('output_filename', 'synthetic_dataset.csv')
        
        # Set seeds for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            Faker.seed(self.random_seed)
        
        self.fake = Faker()
        self.data = {}
        
        logger.info(f"Initialized generator for {self.n_records} records")
        logger.info(f"Config: {self.dataset_config['name']}")
    
    def _load_config(self) -> Dict:
        """Load and validate the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def _parse_formula(self, formula: str, row_data: Dict) -> Any:
        """
        Parse and evaluate formula expressions.
        
        Args:
            formula (str): Formula string with column references and functions
            row_data (Dict): Current row data for column references
            
        Returns:
            Any: Evaluated result
        """
        # Replace column references with actual values
        formula_eval = formula
        
        # Handle column references
        for col_name, value in row_data.items():
            if col_name in formula_eval:
                if isinstance(value, str):
                    formula_eval = formula_eval.replace(col_name, f"'{value}'")
                else:
                    formula_eval = formula_eval.replace(col_name, str(value))
        
        # Handle built-in functions
        formula_eval = self._replace_builtin_functions(formula_eval, row_data)
        
        try:
            # Use eval carefully with restricted scope
            allowed_names = {
                "random": np.random.random,
                "choice": np.random.choice,
                "randint": np.random.randint,
                "normal": np.random.normal,
                "uniform": np.random.uniform,
                "exponential": np.random.exponential,
                "beta": np.random.beta,
                "__builtins__": {}
            }
            return eval(formula_eval, allowed_names)
        except Exception as e:
            logger.warning(f"Formula evaluation failed: {formula} -> {e}")
            return 0
    
    def _replace_builtin_functions(self, formula: str, row_data: Dict) -> str:
        """Replace custom built-in functions with their implementations."""
        
        # Random function: random(min, max)
        formula = re.sub(
            r'random\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
            r'(uniform(\1, \2))',
            formula
        )
        
        # Days since function: days_since(column_name)
        def replace_days_since(match):
            col_name = match.group(1)
            if col_name in row_data and isinstance(row_data[col_name], (datetime, pd.Timestamp)):
                days = (self.reference_date - pd.to_datetime(row_data[col_name])).days
                return str(days)
            return "0"
        
        formula = re.sub(r'days_since\((\w+)\)', replace_days_since, formula)
        
        # Days until function: days_until(column_name)
        def replace_days_until(match):
            col_name = match.group(1)
            if col_name in row_data and isinstance(row_data[col_name], (datetime, pd.Timestamp)):
                days = (pd.to_datetime(row_data[col_name]) - self.reference_date).days
                return str(max(0, days))
            return "0"
        
        formula = re.sub(r'days_until\((\w+)\)', replace_days_until, formula)
        
        # Months since function: months_since(column_name)
        def replace_months_since(match):
            col_name = match.group(1)
            if col_name in row_data and isinstance(row_data[col_name], (datetime, pd.Timestamp)):
                months = ((self.reference_date.year - pd.to_datetime(row_data[col_name]).year) * 12 + 
                         (self.reference_date.month - pd.to_datetime(row_data[col_name]).month))
                return str(max(0, months))
            return "0"
        
        formula = re.sub(r'months_since\((\w+)\)', replace_months_since, formula)
        
        return formula
    
    def _generate_column_data(self, column_config: Dict) -> List[Any]:
        """Generate data for a single column based on its configuration."""
        name = column_config['name']
        col_type = column_config['type']
        method = column_config['generation_method']
        params = column_config.get('generation_params', {})
        
        logger.info(f"Generating column: {name} ({method})")
        
        if method == 'sequential':
            return self._generate_sequential(params)
        elif method == 'random':
            return self._generate_random(col_type, params)
        elif method == 'lookup':
            return self._generate_lookup(params)
        elif method == 'calculated':
            # Calculated columns are handled later after dependencies are resolved
            return [None] * self.n_records
        elif method == 'correlated':
            # Correlated columns are handled later after target column exists
            return [None] * self.n_records
        else:
            raise ValueError(f"Unknown generation method: {method}")
    
    def _generate_sequential(self, params: Dict) -> List[int]:
        """Generate sequential values (IDs, etc.)."""
        start = params.get('min', 1)
        return list(range(start, start + self.n_records))
    
    def _generate_random(self, col_type: str, params: Dict) -> List[Any]:
        """Generate random values based on type and parameters."""
        distribution = params.get('distribution', 'uniform')
        
        if col_type == 'int':
            min_val = params.get('min', 0)
            max_val = params.get('max', 100)
            
            if distribution == 'uniform':
                return np.random.randint(min_val, max_val + 1, self.n_records).tolist()
            elif distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                values = np.random.normal(mean, std, self.n_records)
                return np.clip(values, min_val, max_val).astype(int).tolist()
            elif distribution == 'exponential':
                scale = (max_val - min_val) / 3
                values = np.random.exponential(scale, self.n_records) + min_val
                return np.clip(values, min_val, max_val).astype(int).tolist()
        
        elif col_type == 'float':
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            
            if distribution == 'uniform':
                return np.random.uniform(min_val, max_val, self.n_records).tolist()
            elif distribution == 'normal':
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                values = np.random.normal(mean, std, self.n_records)
                return np.clip(values, min_val, max_val).tolist()
            elif distribution == 'exponential':
                scale = (max_val - min_val) / 3
                values = np.random.exponential(scale, self.n_records) + min_val
                return np.clip(values, min_val, max_val).tolist()
        
        elif col_type == 'date':
            date_range = params.get('date_range', {})
            start_date = datetime.strptime(date_range.get('start', '2020-01-01'), '%Y-%m-%d')
            end_date = datetime.strptime(date_range.get('end', '2024-12-01'), '%Y-%m-%d')
            
            days_between = (end_date - start_date).days
            random_days = np.random.randint(0, days_between + 1, self.n_records)
            return [(start_date + timedelta(days=int(d))).date() for d in random_days]
        
        elif col_type == 'boolean':
            prob_true = params.get('probability_true', 0.5)
            return np.random.choice([True, False], self.n_records, p=[prob_true, 1-prob_true]).tolist()
        
        elif col_type == 'string':
            choices = params.get('choices', ['A', 'B', 'C'])
            probabilities = params.get('probabilities')
            return np.random.choice(choices, self.n_records, p=probabilities).tolist()
        
        return [None] * self.n_records
    
    def _generate_lookup(self, params: Dict) -> List[Any]:
        """Generate values from predefined choices with probabilities."""
        choices = params.get('choices', ['A', 'B', 'C'])
        probabilities = params.get('probabilities')
        return np.random.choice(choices, self.n_records, p=probabilities).tolist()
    
    def _resolve_calculated_columns(self):
        """Resolve calculated columns that depend on other columns."""
        schema = self.dataset_config['schema']
        calculated_columns = [
            col for col in schema['columns'] 
            if col['generation_method'] in ['calculated', 'correlated']
        ]
        
        for col_config in calculated_columns:
            name = col_config['name']
            params = col_config.get('generation_params', {})
            formula = params.get('formula', '')
            
            logger.info(f"Resolving calculated column: {name}")
            
            new_values = []
            for i in range(self.n_records):
                # Get current row data
                row_data = {col: self.data[col][i] for col in self.data.keys() if self.data[col][i] is not None}
                
                # Evaluate formula for this row
                try:
                    value = self._parse_formula(formula, row_data)
                    new_values.append(value)
                except Exception as e:
                    logger.warning(f"Failed to evaluate formula for {name}, row {i}: {e}")
                    new_values.append(0)
            
            self.data[name] = new_values
    
    def _apply_target_logic(self) -> List[Any]:
        """Apply scoring rules and generate target variable."""
        target_logic = self.dataset_config.get('target_logic', {})
        if not target_logic:
            return [0] * self.n_records
        
        scoring_rules = target_logic.get('scoring_rules', [])
        override_rules = target_logic.get('override_rules', [])
        noise_percentage = target_logic.get('noise_percentage', 0.1)
        thresholds = target_logic.get('threshold', {'high_risk': 6, 'medium_risk': 3, 'low_risk': 0})
        
        logger.info(f"Applying {len(scoring_rules)} scoring rules")
        
        scores = []
        for i in range(self.n_records):
            # Get row data
            row_data = {col: self.data[col][i] for col in self.data.keys()}
            
            # Calculate base score
            score = 0
            for rule in scoring_rules:
                condition = rule.get('condition', '')
                score_impact = rule.get('score_impact', 0)
                
                try:
                    # Parse condition
                    if self._evaluate_condition(condition, row_data):
                        score += score_impact
                except Exception as e:
                    logger.warning(f"Failed to evaluate rule condition: {condition} -> {e}")
            
            # Apply override rules
            for override in override_rules:
                override_type = override.get('type', '')
                condition = override.get('condition', '')
                
                try:
                    if self._evaluate_condition(condition, row_data):
                        if override_type == 'force_positive':
                            score = max(score, thresholds['high_risk'])
                        elif override_type == 'force_negative':
                            score = 0
                except Exception as e:
                    logger.warning(f"Failed to evaluate override condition: {condition} -> {e}")
            
            scores.append(score)
        
        # Add noise
        if noise_percentage > 0:
            noise_count = int(self.n_records * noise_percentage)
            noise_indices = np.random.choice(self.n_records, noise_count, replace=False)
            for idx in noise_indices:
                scores[idx] = np.random.randint(0, thresholds['high_risk'] + 1)
        
        return scores
    
    def _evaluate_condition(self, condition: str, row_data: Dict) -> bool:
        """Evaluate a condition string against row data."""
        # Replace column references
        condition_eval = condition
        for col_name, value in row_data.items():
            if col_name in condition_eval:
                if isinstance(value, str):
                    condition_eval = condition_eval.replace(col_name, f"'{value}'")
                elif value is None:
                    condition_eval = condition_eval.replace(col_name, "None")
                else:
                    condition_eval = condition_eval.replace(col_name, str(value))
        
        # Handle custom functions
        condition_eval = self._replace_builtin_functions(condition_eval, row_data)
        
        # Handle "is null" and "is not null"
        condition_eval = re.sub(r'(\w+)\s+is\s+null', r'\1 == None', condition_eval)
        condition_eval = re.sub(r'(\w+)\s+is\s+not\s+null', r'\1 != None', condition_eval)
        
        try:
            allowed_names = {
                "None": None,
                "__builtins__": {}
            }
            return bool(eval(condition_eval, allowed_names))
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition} -> {e}")
            return False
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete dataset."""
        logger.info("Starting dataset generation...")
        
        schema = self.dataset_config['schema']
        columns = schema['columns']
        
        # Generate basic columns first (non-calculated)
        for col_config in columns:
            if col_config['generation_method'] not in ['calculated', 'correlated']:
                self.data[col_config['name']] = self._generate_column_data(col_config)
        
        # Resolve calculated and correlated columns
        self._resolve_calculated_columns()
        
        # Apply target logic if configured
        target_logic = self.dataset_config.get('target_logic', {})
        if target_logic:
            target_column = target_logic.get('target_column', 'target_score')
            self.data[target_column] = self._apply_target_logic()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data)
        
        # Apply data quality issues
        df = self._apply_data_quality(df)
        
        logger.info(f"Generated dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _apply_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic data quality issues."""
        data_quality = self.dataset_config.get('data_quality', {})
        
        missing_pct = data_quality.get('missing_data_percentage', 0.0)
        outlier_pct = data_quality.get('outlier_percentage', 0.0)
        duplicate_pct = data_quality.get('duplicate_percentage', 0.0)
        
        if missing_pct > 0:
            # Add missing values to nullable columns
            schema = self.dataset_config['schema']
            nullable_columns = [
                col['name'] for col in schema['columns'] 
                if col.get('nullable', False)
            ]
            
            for col in nullable_columns:
                if col in df.columns:
                    n_missing = int(len(df) * missing_pct)
                    missing_indices = np.random.choice(len(df), n_missing, replace=False)
                    df.loc[missing_indices, col] = None
        
        if duplicate_pct > 0:
            # Add duplicate rows
            n_duplicates = int(len(df) * duplicate_pct)
            duplicate_indices = np.random.choice(len(df), n_duplicates, replace=True)
            duplicate_rows = df.iloc[duplicate_indices].copy()
            df = pd.concat([df, duplicate_rows], ignore_index=True)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Save the dataset to CSV file."""
        output_file = filename or self.output_filename
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset saved to {output_file}")
        return output_file
    
    def print_summary(self, df: pd.DataFrame):
        """Print dataset summary statistics."""
        print(f"\n📊 Dataset Summary: {self.dataset_config['name']}")
        print(f"{'='*60}")
        print(f"Records: {len(df):,}")
        print(f"Features: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target variable distribution if it exists
        target_logic = self.dataset_config.get('target_logic', {})
        if target_logic:
            target_column = target_logic.get('target_column', 'target_score')
            if target_column in df.columns:
                print(f"\n🎯 Target Variable: {target_column}")
                print(df[target_column].describe())
                
                # Risk categories
                thresholds = target_logic.get('threshold', {})
                if thresholds:
                    high_threshold = thresholds.get('high_risk', 6)
                    medium_threshold = thresholds.get('medium_risk', 3)
                    
                    high_risk = (df[target_column] >= high_threshold).sum()
                    medium_risk = ((df[target_column] >= medium_threshold) & 
                                  (df[target_column] < high_threshold)).sum()
                    low_risk = (df[target_column] < medium_threshold).sum()
                    
                    print(f"\n📈 Risk Distribution:")
                    print(f"High Risk: {high_risk:,} ({high_risk/len(df)*100:.1f}%)")
                    print(f"Medium Risk: {medium_risk:,} ({medium_risk/len(df)*100:.1f}%)")
                    print(f"Low Risk: {low_risk:,} ({low_risk/len(df)*100:.1f}%)")
        
        print(f"\n✅ Generation complete! File: {self.output_filename}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Generate synthetic dataset from configuration file')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to the JSON configuration file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output filename (overrides config file setting)')
    parser.add_argument('--rows', '-n', type=int, default=None,
                       help='Number of records to generate (overrides config file setting)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                       help='Random seed for reproducible results (overrides config file setting)')
    
    args = parser.parse_args()
    
    # Generate dataset
    start_time = datetime.now()
    generator = ConfigDatasetGenerator(args.config)
    
    # Override config with command line arguments if provided
    if args.rows is not None:
        generator.n_records = args.rows
    if args.seed is not None:
        generator.random_seed = args.seed
        np.random.seed(args.seed)
        Faker.seed(args.seed)
    if args.output is not None:
        generator.output_filename = args.output
    
    dataset = generator.generate_dataset()
    generation_time = (datetime.now() - start_time).total_seconds()
    
    # Save and summarize
    generator.save_dataset(dataset)
    generator.print_summary(dataset)
    print(f"⚡ Generation time: {generation_time:.2f} seconds")
    
    return dataset


if __name__ == "__main__":
    dataset = main() 