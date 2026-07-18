# generate_datasets.py - Standalone dataset generator for PredictMaint
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_data(
    n_machines: int = 10, 
    n_steps: int = 5000,
    seed: int = 42,
    failure_ratio: float = 0.3,
    noise_level: float = 0.05
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for predictive maintenance
    
    Parameters:
    -----------
    n_machines: Number of machines to simulate
    n_steps: Number of time steps per machine
    seed: Random seed for reproducibility
    failure_ratio: Proportion of machines that will experience failure
    noise_level: Amount of noise in sensor readings
    
    Returns:
    --------
    DataFrame with sensor data and labels
    """
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Machine configurations
    machine_types = ['Conveyor Motor', 'Compressor', 'CNC Spindle', 'Pump', 'Fan', 
                     'Gearbox', 'Turbine', 'Mixer', 'Conveyor Belt', 'Hydraulic Pump']
    
    base_params = {
        'Conveyor Motor': {'temp': 65, 'vib': 2.5, 'pressure': 3.0, 'current': 15, 'rpm': 1200, 'power': 5.5},
        'Compressor': {'temp': 80, 'vib': 3.5, 'pressure': 7.5, 'current': 25, 'rpm': 900, 'power': 15.0},
        'CNC Spindle': {'temp': 45, 'vib': 1.5, 'pressure': 0.5, 'current': 10, 'rpm': 4500, 'power': 7.5},
        'Pump': {'temp': 55, 'vib': 2.0, 'pressure': 5.0, 'current': 20, 'rpm': 600, 'power': 11.0},
        'Fan': {'temp': 40, 'vib': 4.0, 'pressure': 1.0, 'current': 5, 'rpm': 1800, 'power': 2.5},
        'Gearbox': {'temp': 70, 'vib': 3.0, 'pressure': 4.0, 'current': 18, 'rpm': 750, 'power': 8.0},
        'Turbine': {'temp': 90, 'vib': 2.8, 'pressure': 8.0, 'current': 30, 'rpm': 1500, 'power': 20.0},
        'Mixer': {'temp': 60, 'vib': 3.2, 'pressure': 2.0, 'current': 22, 'rpm': 450, 'power': 12.0},
        'Conveyor Belt': {'temp': 35, 'vib': 4.5, 'pressure': 1.5, 'current': 8, 'rpm': 300, 'power': 3.0},
        'Hydraulic Pump': {'temp': 75, 'vib': 2.2, 'pressure': 6.5, 'current': 28, 'rpm': 550, 'power': 16.0}
    }
    
    # Operating conditions
    conditions = ['Normal', 'High Load', 'Low Load', 'Startup', 'Shutdown']
    condition_multipliers = {
        'Normal': {'temp': 1.0, 'vib': 1.0, 'pressure': 1.0, 'current': 1.0, 'rpm': 1.0},
        'High Load': {'temp': 1.15, 'vib': 1.2, 'pressure': 1.1, 'current': 1.2, 'rpm': 1.05},
        'Low Load': {'temp': 0.85, 'vib': 0.8, 'pressure': 0.7, 'current': 0.7, 'rpm': 0.8},
        'Startup': {'temp': 0.9, 'vib': 1.1, 'pressure': 0.8, 'current': 1.3, 'rpm': 0.5},
        'Shutdown': {'temp': 0.7, 'vib': 0.5, 'pressure': 0.4, 'current': 0.4, 'rpm': 0.3}
    }
    
    data = []
    
    for machine_id in range(1, n_machines + 1):
        # Select machine type
        m_type = random.choice(machine_types[:min(len(machine_types), n_machines)])
        base = base_params[m_type]
        
        # Determine if this machine will fail
        will_fail = random.random() < failure_ratio
        
        # Random failure parameters
        if will_fail:
            failure_point = random.randint(int(n_steps * 0.5), int(n_steps * 0.9))
            failure_severity = random.uniform(0.6, 1.0)
            degradation_start = random.randint(int(n_steps * 0.3), int(failure_point * 0.8))
            degradation_rate = random.uniform(0.2, 0.8)
        else:
            failure_point = n_steps + 1000  # No failure within simulation
            failure_severity = 0
            degradation_start = n_steps + 1000
            degradation_rate = 0
        
        # Machine-specific noise
        machine_noise = {
            'temp': random.uniform(0.02, 0.08),
            'vib': random.uniform(0.03, 0.12),
            'pressure': random.uniform(0.02, 0.06),
            'current': random.uniform(0.03, 0.10),
            'rpm': random.uniform(0.02, 0.05)
        }
        
        # Generate time series for this machine
        current_condition = 'Normal'
        condition_duration = 0
        
        for step in range(n_steps):
            # Change operating condition periodically
            condition_duration += 1
            if condition_duration > random.randint(50, 200):
                current_condition = random.choice(conditions)
                condition_duration = 0
            
            # Get multipliers for current condition
            mult = condition_multipliers[current_condition]
            
            # Normal operation with noise and condition effects
            temp = base['temp'] * mult['temp'] * (1 + np.random.normal(0, noise_level * machine_noise['temp']))
            vib = base['vib'] * mult['vib'] * (1 + np.random.normal(0, noise_level * machine_noise['vib'] * 2))
            pressure = base['pressure'] * mult['pressure'] * (1 + np.random.normal(0, noise_level * machine_noise['pressure']))
            current = base['current'] * mult['current'] * (1 + np.random.normal(0, noise_level * machine_noise['current'] * 1.5))
            rpm = base['rpm'] * mult['rpm'] * (1 + np.random.normal(0, noise_level * machine_noise['rpm']))
            
            # Gradual degradation (if machine will fail)
            if will_fail and step > degradation_start:
                degradation_progress = (step - degradation_start) / (failure_point - degradation_start)
                degradation_factor = degradation_progress * degradation_rate * failure_severity
                
                vib *= (1 + degradation_factor * 0.6)
                temp *= (1 + degradation_factor * 0.3)
                current *= (1 + degradation_factor * 0.4)
                pressure *= (1 + degradation_factor * 0.2)
            
            # Sudden failure spikes (near failure point)
            if will_fail and step > failure_point * 0.9 and random.random() < 0.02:
                spike_type = random.choice(['vibration', 'temperature', 'current', 'pressure'])
                spike_magnitude = random.uniform(0.3, 0.8)
                
                if spike_type == 'vibration':
                    vib *= (1 + spike_magnitude)
                elif spike_type == 'temperature':
                    temp *= (1 + spike_magnitude * 0.5)
                elif spike_type == 'current':
                    current *= (1 + spike_magnitude * 0.6)
                elif spike_type == 'pressure':
                    pressure *= (1 + spike_magnitude * 0.3)
            
            # Add occasional sensor glitches
            if random.random() < 0.001:  # 0.1% chance
                glitch_type = random.choice(['spike', 'dropout', 'offset'])
                if glitch_type == 'spike':
                    vib *= random.uniform(1.5, 3.0)
                elif glitch_type == 'dropout':
                    temp = 0
                elif glitch_type == 'offset':
                    current += random.uniform(-5, 5)
            
            # Determine health label
            if will_fail and step >= failure_point:
                health_label = 'Critical'
                rul_hours = max(0, (failure_point - step) * 0.1)
            elif will_fail and step >= failure_point * 0.7:
                health_label = 'Warning'
                rul_hours = (failure_point - step) * 0.1
            else:
                health_label = 'Normal'
                rul_hours = (failure_point - step) * 0.1 if will_fail else 200.0
            
            # Cumulative runtime
            runtime_hours = step * 0.1
            
            # Calculate power consumption
            power = base.get('power', 5.0) * (current / base['current']) * (rpm / base['rpm'])
            
            data.append({
                'timestamp': step,
                'machine_id': f'M{machine_id:03d}',
                'machine_type': m_type,
                'temperature_c': round(temp, 2),
                'vibration_mm_s': round(vib, 2),
                'pressure_bar': round(pressure, 2),
                'current_a': round(current, 2),
                'rpm': round(rpm, 0),
                'power_kw': round(power, 2),
                'runtime_hours': round(runtime_hours, 1),
                'operating_condition': current_condition,
                'health_label': health_label,
                'rul_hours': round(rul_hours, 1),
                'will_fail': will_fail
            })
    
    return pd.DataFrame(data)

def generate_dataset_with_specific_scenarios():
    """Generate multiple datasets for different scenarios"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    scenarios = {
        'small': {'n_machines': 5, 'n_steps': 2000, 'failure_ratio': 0.2, 'noise_level': 0.03},
        'medium': {'n_machines': 10, 'n_steps': 5000, 'failure_ratio': 0.3, 'noise_level': 0.05},
        'large': {'n_machines': 20, 'n_steps': 10000, 'failure_ratio': 0.4, 'noise_level': 0.07},
        'high_noise': {'n_machines': 10, 'n_steps': 5000, 'failure_ratio': 0.3, 'noise_level': 0.15},
        'low_noise': {'n_machines': 10, 'n_steps': 5000, 'failure_ratio': 0.3, 'noise_level': 0.02},
        'high_failure': {'n_machines': 10, 'n_steps': 5000, 'failure_ratio': 0.6, 'noise_level': 0.05},
        'low_failure': {'n_machines': 10, 'n_steps': 5000, 'failure_ratio': 0.1, 'noise_level': 0.05},
    }
    
    datasets = {}
    
    for name, params in scenarios.items():
        print(f"Generating {name} dataset...")
        df = generate_synthetic_data(
            n_machines=params['n_machines'],
            n_steps=params['n_steps'],
            failure_ratio=params['failure_ratio'],
            noise_level=params['noise_level'],
            seed=42 + list(scenarios.keys()).index(name)
        )
        
        # Save dataset
        filename = f"data/machine_sensor_log_{name}.csv"
        df.to_csv(filename, index=False)
        datasets[name] = df
        
        print(f"  - Saved {filename} ({len(df)} rows, {df['machine_id'].nunique()} machines)")
        print(f"  - Health distribution: {df['health_label'].value_counts().to_dict()}")
        print(f"  - Failure ratio: {df['will_fail'].mean():.2%}")
        print()
    
    return datasets

def generate_training_test_split(dataset_name='medium'):
    """Generate train/test split for the specified dataset"""
    
    # Load dataset
    df = pd.read_csv(f'data/machine_sensor_log_{dataset_name}.csv')
    
    # Get unique machines
    machines = df['machine_id'].unique()
    
    # Split machines (80% train, 20% test)
    np.random.seed(42)
    train_machines = np.random.choice(machines, size=int(0.8 * len(machines)), replace=False)
    test_machines = [m for m in machines if m not in train_machines]
    
    # Create train and test sets
    train_data = df[df['machine_id'].isin(train_machines)]
    test_data = df[df['machine_id'].isin(test_machines)]
    
    # Save splits
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    
    print(f"Train data: {len(train_data)} rows ({len(train_machines)} machines)")
    print(f"Test data: {len(test_data)} rows ({len(test_machines)} machines)")
    
    return train_data, test_data

def generate_feature_importance_dataset():
    """Generate a dataset specifically for feature importance analysis"""
    
    print("Generating dataset for feature importance analysis...")
    
    # Create a dataset with controlled patterns
    np.random.seed(123)
    n_samples = 10000
    
    # Generate base sensors with different patterns
    temp = np.random.normal(65, 10, n_samples)
    vib = np.random.normal(2.5, 0.8, n_samples)
    pressure = np.random.normal(3.0, 0.6, n_samples)
    current = np.random.normal(15, 3, n_samples)
    rpm = np.random.normal(1200, 100, n_samples)
    runtime = np.random.uniform(0, 500, n_samples)
    
    # Introduce degradation patterns
    degradation = np.random.uniform(0, 1, n_samples)
    temp = temp + degradation * 20
    vib = vib + degradation * 3
    pressure = pressure + degradation * 2
    current = current + degradation * 8
    rpm = rpm - degradation * 200
    
    # Create labels based on sensor values
    health_score = (
        0.3 * (temp / 100) + 
        0.4 * (vib / 5) + 
        0.2 * (current / 30) + 
        0.1 * (pressure / 10)
    )
    
    labels = np.where(health_score < 0.3, 'Normal', 
                     np.where(health_score < 0.6, 'Warning', 'Critical'))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': range(n_samples),
        'machine_id': 'M001',
        'machine_type': 'Test Machine',
        'temperature_c': temp,
        'vibration_mm_s': vib,
        'pressure_bar': pressure,
        'current_a': current,
        'rpm': rpm,
        'runtime_hours': runtime,
        'health_label': labels,
        'rul_hours': 100 - degradation * 90
    })
    
    df.to_csv('data/feature_importance_demo.csv', index=False)
    print(f"Feature importance demo dataset saved: data/feature_importance_demo.csv")
    
    return df

def main():
    """Main function to generate all datasets"""
    
    print("=" * 60)
    print("🏭 PredictMaint - Dataset Generator")
    print("=" * 60)
    print()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets for different scenarios
    print("Generating datasets for different scenarios...")
    print("-" * 40)
    datasets = generate_dataset_with_specific_scenarios()
    
    print()
    print("Generating train/test splits...")
    print("-" * 40)
    train_data, test_data = generate_training_test_split('medium')
    
    print()
    print("Generating feature importance demo dataset...")
    print("-" * 40)
    feature_data = generate_feature_importance_dataset()
    
    # Generate summary statistics
    print()
    print("=" * 60)
    print("📊 Dataset Summary")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Machines: {df['machine_id'].nunique()}")
        print(f"  - Health distribution:")
        for label, count in df['health_label'].value_counts().items():
            print(f"      {label}: {count} ({count/len(df)*100:.1f}%)")
        print(f"  - Failure machines: {df['will_fail'].sum()}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - File: data/machine_sensor_log_{name}.csv")
    
    print("\n" + "=" * 60)
    print("✅ All datasets generated successfully!")
    print("=" * 60)
    print("\nAvailable datasets:")
    print("  - data/machine_sensor_log_small.csv")
    print("  - data/machine_sensor_log_medium.csv")
    print("  - data/machine_sensor_log_large.csv")
    print("  - data/machine_sensor_log_high_noise.csv")
    print("  - data/machine_sensor_log_low_noise.csv")
    print("  - data/machine_sensor_log_high_failure.csv")
    print("  - data/machine_sensor_log_low_failure.csv")
    print("  - data/train_data.csv")
    print("  - data/test_data.csv")
    print("  - data/feature_importance_demo.csv")

if __name__ == "__main__":
    main()