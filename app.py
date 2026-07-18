# app.py - PredictMaint with Complete Error Handling
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PredictMaint - Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CREATE SAMPLE CSV =============
def create_sample_csv():
    """Create a sample CSV with proper format"""
    np.random.seed(42)
    
    n_samples = 500
    machine_ids = ['M001', 'M002', 'M003']
    machine_types = ['Conveyor Motor', 'Compressor', 'CNC Spindle']
    
    data = []
    for mid, mtype in zip(machine_ids, machine_types):
        base_temp = {'Conveyor Motor': 65, 'Compressor': 80, 'CNC Spindle': 45}[mtype]
        base_vib = {'Conveyor Motor': 2.5, 'Compressor': 3.5, 'CNC Spindle': 1.5}[mtype]
        base_pressure = {'Conveyor Motor': 3.0, 'Compressor': 7.5, 'CNC Spindle': 0.5}[mtype]
        base_current = {'Conveyor Motor': 15, 'Compressor': 25, 'CNC Spindle': 10}[mtype]
        base_rpm = {'Conveyor Motor': 1200, 'Compressor': 900, 'CNC Spindle': 4500}[mtype]
        
        for step in range(n_samples):
            temp = base_temp + np.random.normal(0, 3)
            vib = base_vib + np.random.normal(0, 0.3)
            pressure = base_pressure + np.random.normal(0, 0.2)
            current = base_current + np.random.normal(0, 1.5)
            rpm = base_rpm + np.random.normal(0, 100)
            runtime = step * 0.1
            
            if step > n_samples * 0.7:
                degradation = (step - n_samples * 0.7) / (n_samples * 0.3)
                vib = vib * (1 + degradation * 0.5)
                temp = temp * (1 + degradation * 0.3)
                current = current * (1 + degradation * 0.4)
            
            if step > n_samples * 0.85:
                health = 'Critical'
                rul = max(0, (n_samples - step) * 0.1)
            elif step > n_samples * 0.7:
                health = 'Warning'
                rul = (n_samples - step) * 0.1
            else:
                health = 'Normal'
                rul = (n_samples - step) * 0.1
            
            data.append({
                'machine_id': mid,
                'machine_type': mtype,
                'timestamp': step,
                'temperature_c': round(temp, 2),
                'vibration_mm_s': round(vib, 2),
                'pressure_bar': round(pressure, 2),
                'current_a': round(current, 2),
                'rpm': round(rpm, 0),
                'runtime_hours': round(runtime, 1),
                'health_label': health,
                'rul_hours': round(rul, 1)
            })
    
    return pd.DataFrame(data)

# ============= DATA GENERATOR =============
@st.cache_data
def generate_synthetic_data(n_machines: int = 10, n_steps: int = 5000) -> pd.DataFrame:
    """Generate synthetic sensor data"""
    
    machine_types = ['Conveyor Motor', 'Compressor', 'CNC Spindle', 'Pump', 'Fan']
    base_params = {
        'Conveyor Motor': {'temp': 65, 'vib': 2.5, 'pressure': 3.0, 'current': 15, 'rpm': 1200},
        'Compressor': {'temp': 80, 'vib': 3.5, 'pressure': 7.5, 'current': 25, 'rpm': 900},
        'CNC Spindle': {'temp': 45, 'vib': 1.5, 'pressure': 0.5, 'current': 10, 'rpm': 4500},
        'Pump': {'temp': 55, 'vib': 2.0, 'pressure': 5.0, 'current': 20, 'rpm': 600},
        'Fan': {'temp': 40, 'vib': 4.0, 'pressure': 1.0, 'current': 5, 'rpm': 1800}
    }
    
    data = []
    for machine_id in range(1, n_machines + 1):
        m_type = random.choice(machine_types)
        base = base_params[m_type]
        
        failure_point = random.randint(int(n_steps * 0.6), int(n_steps * 0.95))
        failure_severity = random.uniform(0.5, 1.0)
        
        for step in range(n_steps):
            noise_factor = 0.05
            temp = base['temp'] * (1 + np.random.normal(0, noise_factor))
            vib = base['vib'] * (1 + np.random.normal(0, noise_factor * 2))
            pressure = base['pressure'] * (1 + np.random.normal(0, noise_factor))
            current = base['current'] * (1 + np.random.normal(0, noise_factor * 1.5))
            rpm = base['rpm'] * (1 + np.random.normal(0, noise_factor))
            
            if step > failure_point * 0.8:
                degradation = ((step - failure_point * 0.8) / (failure_point * 0.2)) * failure_severity
                vib *= (1 + degradation * 0.5)
                temp *= (1 + degradation * 0.3)
                current *= (1 + degradation * 0.4)
            
            if step > failure_point * 0.9 and random.random() < 0.01:
                spike = 1 + random.uniform(0.3, 0.8)
                vib *= spike
                temp *= spike * 0.5
            
            if step >= failure_point:
                health_label = 'Critical'
                rul_hours = max(0, (failure_point - step) * 0.1)
            elif step >= failure_point * 0.8:
                health_label = 'Warning'
                rul_hours = (failure_point - step) * 0.1
            else:
                health_label = 'Normal'
                rul_hours = (failure_point - step) * 0.1
            
            runtime_hours = step * 0.1
            
            data.append({
                'timestamp': step,
                'machine_id': f'M{machine_id:02d}',
                'machine_type': m_type,
                'temperature_c': round(temp, 2),
                'vibration_mm_s': round(vib, 2),
                'pressure_bar': round(pressure, 2),
                'current_a': round(current, 2),
                'rpm': round(rpm, 0),
                'runtime_hours': round(runtime_hours, 1),
                'health_label': health_label,
                'rul_hours': round(rul_hours, 1)
            })
    
    return pd.DataFrame(data)

# ============= DATA PREPROCESSING =============
def preprocess_data(data: pd.DataFrame, fit_mode: bool = False, known_categories: list = None) -> tuple:
    """Feature engineering for predictive maintenance"""
    
    df = data.copy()
    
    # Required columns
    required_cols = ['machine_id', 'temperature_c', 'vibration_mm_s', 'pressure_bar', 
                     'current_a', 'rpm', 'runtime_hours', 'health_label', 'rul_hours']
    
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, None
    
    # Rolling statistics
    df['temp_rolling_mean'] = df.groupby('machine_id')['temperature_c'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['vib_rolling_mean'] = df.groupby('machine_id')['vibration_mm_s'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Rate of change
    df['temp_rate'] = df.groupby('machine_id')['temperature_c'].diff().fillna(0)
    df['vib_rate'] = df.groupby('machine_id')['vibration_mm_s'].diff().fillna(0)
    df['current_rate'] = df.groupby('machine_id')['current_a'].diff().fillna(0)
    
    # Deviation from baseline
    df['temp_deviation'] = df.groupby('machine_id')['temperature_c'].transform(
        lambda x: x - x.rolling(100, min_periods=1).mean()
    ).fillna(0)
    
    # RUL trend
    df['rul_trend'] = df.groupby('machine_id')['rul_hours'].diff().fillna(0)
    
    # One-hot encoding
    if fit_mode:
        if 'machine_type' in df.columns:
            categories = sorted(df['machine_type'].unique())
        else:
            categories = ['Unknown']
            df['machine_type'] = 'Unknown'
        return df, categories
    else:
        if known_categories:
            if 'machine_type' not in df.columns:
                df['machine_type'] = 'Unknown'
            for category in known_categories:
                df[f'type_{category}'] = (df['machine_type'] == category).astype(int)
        return df, None

# ============= MODEL TRAINING =============
@st.cache_resource
def train_models(data: pd.DataFrame):
    """Train Random Forest models"""
    
    data_preprocessed, categories = preprocess_data(data, fit_mode=True)
    
    if data_preprocessed is None:
        return None, None, None, None, None, None, None, None, None, None
    
    data_final, _ = preprocess_data(data_preprocessed, fit_mode=False, known_categories=categories)
    
    base_features = ['temperature_c', 'vibration_mm_s', 'pressure_bar', 'current_a', 'rpm',
                     'runtime_hours', 'temp_rolling_mean', 'vib_rolling_mean',
                     'temp_rate', 'vib_rate', 'current_rate', 'temp_deviation', 'rul_trend']
    
    type_columns = [f'type_{cat}' for cat in categories]
    feature_cols = base_features + type_columns
    
    X = data_final[feature_cols].copy().fillna(0)
    
    y_class = data_final['health_label']
    label_map = {'Normal': 0, 'Warning': 1, 'Critical': 2}
    y_class_encoded = y_class.map(label_map)
    y_reg = data_final['rul_hours']
    
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class_encoded, y_reg, test_size=0.2, random_state=42, stratify=y_class_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train_scaled, y_class_train)
    
    reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg.fit(X_train_scaled, y_reg_train)
    
    y_class_pred = clf.predict(X_test_scaled)
    y_reg_pred = reg.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_class_test, y_class_pred),
        'precision': precision_score(y_class_test, y_class_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_class_test, y_class_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_class_test, y_class_pred, average='weighted', zero_division=0),
        'mae': mean_absolute_error(y_reg_test, y_reg_pred),
        'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
        'confusion_matrix': confusion_matrix(y_class_test, y_class_pred)
    }
    
    return clf, reg, scaler, feature_cols, label_map, metrics, categories, X_test_scaled, y_class_test, y_reg_test

# ============= MODEL INFERENCE =============
def run_model_or_algorithm(data: pd.DataFrame, model_choice: str, 
                           clf=None, reg=None, scaler=None,
                           feature_cols=None, label_map=None,
                           categories=None) -> dict:
    """Run ML model or rule-based baseline"""
    
    if model_choice == 'Machine Learning Model' and clf is not None:
        data_preprocessed, _ = preprocess_data(data, fit_mode=False, known_categories=categories)
        
        if data_preprocessed is None:
            return None
        
        for col in feature_cols:
            if col not in data_preprocessed.columns:
                data_preprocessed[col] = 0
        
        X = data_preprocessed[feature_cols].copy().fillna(0)
        X_scaled = scaler.transform(X)
        
        class_pred = clf.predict(X_scaled)
        class_proba = clf.predict_proba(X_scaled)
        rul_pred = reg.predict(X_scaled)
        
        inv_label_map = {v: k for k, v in label_map.items()}
        health_label = inv_label_map[int(class_pred[0])]
        failure_prob = max(class_proba[0])
        
        feature_importance = dict(zip(feature_cols, clf.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'health_label': health_label,
            'failure_probability': failure_prob,
            'rul_hours': rul_pred[0],
            'feature_importance': feature_importance,
            'top_features': top_features,
            'method': 'ML'
        }
    
    else:
        temp_warning = data['temperature_c'].iloc[0] > 75
        vib_warning = data['vibration_mm_s'].iloc[0] > 3.5
        pressure_warning = data['pressure_bar'].iloc[0] > 6.0
        current_warning = data['current_a'].iloc[0] > 20
        
        warning_count = sum([temp_warning, vib_warning, pressure_warning, current_warning])
        
        if warning_count >= 3:
            health_label = 'Critical'
            failure_prob = 0.85
            rul_hours = max(0, 50 - data['runtime_hours'].iloc[0] % 100)
        elif warning_count >= 1:
            health_label = 'Warning'
            failure_prob = 0.45
            rul_hours = max(0, 100 - data['runtime_hours'].iloc[0] % 100)
        else:
            health_label = 'Normal'
            failure_prob = 0.05
            rul_hours = max(0, 200 - data['runtime_hours'].iloc[0] % 100)
        
        feature_importance = {
            'temperature_c': 0.25 if temp_warning else 0.1,
            'vibration_mm_s': 0.35 if vib_warning else 0.1,
            'pressure_bar': 0.20 if pressure_warning else 0.1,
            'current_a': 0.20 if current_warning else 0.1
        }
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'health_label': health_label,
            'failure_probability': failure_prob,
            'rul_hours': rul_hours,
            'feature_importance': feature_importance,
            'top_features': top_features,
            'method': 'Rule-Based'
        }

# ============= EXPLANATION =============
def generate_explanation(result: dict) -> str:
    """Generate natural language explanation"""
    
    health_label = result['health_label']
    top_features = result['top_features']
    failure_prob = result['failure_probability']
    method = result['method']
    
    explanation_parts = []
    
    if health_label == 'Critical':
        explanation_parts.append("🚨 CRITICAL: Immediate action required!")
    elif health_label == 'Warning':
        explanation_parts.append("⚠️ WARNING: Machine needs attention soon.")
    else:
        explanation_parts.append("✅ NORMAL: Machine is operating within expected parameters.")
    
    if top_features:
        feature_names = {
            'temperature_c': 'temperature',
            'vibration_mm_s': 'vibration',
            'pressure_bar': 'pressure',
            'current_a': 'current draw',
            'rpm': 'rotational speed',
            'runtime_hours': 'runtime'
        }
        top_feature_name, importance = top_features[0]
        if top_feature_name.startswith('type_'):
            top_feature_name = top_feature_name.replace('type_', '')
        feature_name = feature_names.get(top_feature_name, top_feature_name)
        importance_percent = importance * 100
        
        explanation_parts.append(f"Main factor: {feature_name} ({importance_percent:.1f}% influence)")
        
        if len(top_features) > 1:
            other_features = []
            for f, _ in top_features[1:3]:
                if f.startswith('type_'):
                    f = f.replace('type_', '')
                other_features.append(feature_names.get(f, f))
            explanation_parts.append(f"Also contributing: {', '.join(other_features)}")
    
    explanation_parts.append(f"Failure probability: {failure_prob:.1%}")
    explanation_parts.append(f"Method: {method}")
    explanation_parts.append(f"RUL: {result['rul_hours']:.1f} hours")
    
    return " | ".join(explanation_parts)

# ============= VISUALIZATIONS =============
def create_visuals(data: pd.DataFrame, result: dict, machine_id: str):
    """Create visualizations"""
    
    machine_data = data[data['machine_id'] == machine_id]
    timestamps = machine_data['timestamp'].values if 'timestamp' in machine_data.columns else range(len(machine_data))
    
    fig_trend = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Temperature', 'Vibration', 'Pressure', 'Current', 'RPM', 'Health Status'),
        shared_xaxes=True
    )
    
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=machine_data['temperature_c'], mode='lines', name='Temp', line=dict(color='red')),
        row=1, col=1
    )
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=machine_data['vibration_mm_s'], mode='lines', name='Vibration', line=dict(color='orange')),
        row=1, col=2
    )
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=machine_data['pressure_bar'], mode='lines', name='Pressure', line=dict(color='blue')),
        row=2, col=1
    )
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=machine_data['current_a'], mode='lines', name='Current', line=dict(color='purple')),
        row=2, col=2
    )
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=machine_data['rpm'], mode='lines', name='RPM', line=dict(color='green')),
        row=3, col=1
    )
    
    health_colors = {'Normal': 'green', 'Warning': 'yellow', 'Critical': 'red'}
    colors = [health_colors.get(h, 'gray') for h in machine_data['health_label'].values]
    fig_trend.add_trace(
        go.Scatter(x=timestamps, y=[1]*len(timestamps), mode='markers', 
                   marker=dict(color=colors, size=5), name='Health Status'),
        row=3, col=2
    )
    
    fig_trend.update_layout(height=500, showlegend=True)
    
    if 'feature_importance' in result:
        features = []
        importance = []
        for f, imp in sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]:
            if f.startswith('type_'):
                f = f.replace('type_', '')
            features.append(f)
            importance.append(imp)
        
        fig_importance = go.Figure(data=[
            go.Bar(x=features, y=importance, marker_color='coral')
        ])
        fig_importance.update_layout(
            title='Top 10 Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance',
            height=300
        )
    else:
        fig_importance = None
    
    return {
        'trend_chart': fig_trend,
        'importance_chart': fig_importance,
        'health_color': health_colors.get(result['health_label'], 'gray')
    }

# ============= VALIDATE CSV =============
def validate_csv(df: pd.DataFrame) -> tuple:
    """Validate CSV format and return (is_valid, error_message, required_columns)"""
    
    required_cols = ['machine_id', 'temperature_c', 'vibration_mm_s', 'pressure_bar', 
                     'current_a', 'rpm', 'runtime_hours', 'health_label', 'rul_hours']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}", required_cols
    
    return True, "All required columns present", required_cols

# ============= MAIN UI =============
def main():
    """Main application"""
    
    st.title("🏭 PredictMaint — AI-Powered Predictive Maintenance")
    st.markdown("---")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'categories' not in st.session_state:
        st.session_state.categories = None
    if 'selected_machine' not in st.session_state:
        st.session_state.selected_machine = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "Machine Learning Model"
    if 'speed' not in st.session_state:
        st.session_state.speed = 5
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_error' not in st.session_state:
        st.session_state.data_error = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # ===== DATA SOURCE =====
        st.subheader("📊 Data Source")
        
        data_source = st.radio(
            "Choose option:",
            ["Generate Sample Data", "Upload CSV"],
            index=0,
            key="data_source"
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "📁 Upload CSV File",
                type=['csv'],
                help="Upload a CSV file with sensor data"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate CSV
                    is_valid, message, required_cols = validate_csv(df)
                    
                    if not is_valid:
                        st.error(f"❌ {message}")
                        st.info(f"Required columns: {', '.join(required_cols)}")
                        st.warning(f"Your columns: {', '.join(df.columns)}")
                        st.session_state.data_error = message
                        st.session_state.data_loaded = False
                    else:
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.session_state.models_trained = False
                        st.session_state.data_error = None
                        st.success(f"✅ Data loaded successfully!")
                        st.info(f"📊 {len(df)} rows, {len(df.columns)} columns")
                        st.info(f"🏭 {df['machine_id'].nunique()} machines")
                        
                        with st.expander("📊 Data Preview"):
                            st.dataframe(df.head(10))
                        
                        # Show health distribution
                        health_counts = df['health_label'].value_counts()
                        st.write("Health Status Distribution:")
                        for label, count in health_counts.items():
                            st.write(f"  - {label}: {count} ({count/len(df)*100:.1f}%)")
                            
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.session_state.data_error = str(e)
                    st.session_state.data_loaded = False
        
        # ===== SAMPLE DATA =====
        if data_source == "Generate Sample Data":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Generate Sample Data", use_container_width=True):
                    with st.spinner("Generating synthetic data..."):
                        st.session_state.data = generate_synthetic_data(n_machines=10, n_steps=5000)
                        st.session_state.data_loaded = True
                        st.session_state.models_trained = False
                        st.session_state.data_error = None
                    st.success("✅ Sample data generated!")
                    st.rerun()
            
            with col2:
                sample_df = create_sample_csv()
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample CSV",
                    data=csv,
                    file_name="sample_maintenance_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # ===== DATA INFO =====
        if st.session_state.data is not None and st.session_state.data_loaded:
            st.divider()
            st.subheader("📊 Data Info")
            
            # Check if machine_id exists
            if 'machine_id' not in st.session_state.data.columns:
                st.error("❌ Missing 'machine_id' column!")
            else:
                st.success(f"✅ {st.session_state.data.shape[0]} rows")
                st.info(f"🏭 {st.session_state.data['machine_id'].nunique()} machines")
                
                # Machine selection
                machines = st.session_state.data['machine_id'].unique()
                st.session_state.selected_machine = st.selectbox(
                    "🔧 Select Machine",
                    machines,
                    index=0 if len(machines) > 0 else None
                )
                
                # Model selection
                st.session_state.model_choice = st.radio(
                    "🧠 Analysis Method",
                    ["Machine Learning Model", "Rule-Based Baseline"],
                    index=0
                )
                
                # Simulation controls
                st.divider()
                st.subheader("▶️ Simulation Controls")
                st.session_state.speed = st.slider("Speed", 1, 10, 5)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("▶️ Play", use_container_width=True):
                        st.session_state.simulation_running = True
                with col2:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state.simulation_running = False
                with col3:
                    if st.button("⏹️ Reset", use_container_width=True):
                        st.session_state.current_step = 0
                        st.session_state.simulation_running = False
                
                # Train models
                if not st.session_state.models_trained:
                    st.divider()
                    if st.button("🎯 Train ML Models", use_container_width=True, type="primary"):
                        with st.spinner("Training ML models on your data..."):
                            try:
                                clf, reg, scaler, feature_cols, label_map, metrics, categories, X_test, y_class_test, y_reg_test = train_models(st.session_state.data)
                                
                                if clf is not None:
                                    st.session_state.clf = clf
                                    st.session_state.reg = reg
                                    st.session_state.scaler = scaler
                                    st.session_state.feature_cols = feature_cols
                                    st.session_state.label_map = label_map
                                    st.session_state.metrics = metrics
                                    st.session_state.categories = categories
                                    st.session_state.models_trained = True
                                    
                                    st.success("✅ Models trained successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to train models. Check your data format.")
                            except Exception as e:
                                st.error(f"❌ Training error: {str(e)}")
    
    # ===== MAIN CONTENT =====
    if st.session_state.data is not None and st.session_state.data_loaded and st.session_state.models_trained:
        if st.session_state.selected_machine is None:
            st.warning("Please select a machine from the sidebar.")
            return
        
        # Check if machine_id exists
        if 'machine_id' not in st.session_state.data.columns:
            st.error("❌ Data is missing 'machine_id' column. Please reload with correct format.")
            return
        
        machine_data = st.session_state.data[
            st.session_state.data['machine_id'] == st.session_state.selected_machine
        ]
        
        if len(machine_data) == 0:
            st.warning(f"No data found for machine {st.session_state.selected_machine}")
            return
        
        current_step = st.session_state.current_step
        max_step = len(machine_data) - 1
        
        if current_step > max_step:
            current_step = max_step
            st.session_state.current_step = current_step
        
        current_data = machine_data.iloc[current_step:current_step+1] if current_step < len(machine_data) else machine_data.iloc[[-1]]
        
        if len(current_data) > 0:
            result = run_model_or_algorithm(
                current_data,
                st.session_state.model_choice,
                clf=st.session_state.clf,
                reg=st.session_state.reg,
                scaler=st.session_state.scaler,
                feature_cols=st.session_state.feature_cols,
                label_map=st.session_state.label_map,
                categories=st.session_state.categories
            )
            
            if result is None:
                st.error("Error in prediction. Please check your data.")
                return
            
            historical_data = machine_data.iloc[:current_step+1]
            visuals = create_visuals(historical_data, result, st.session_state.selected_machine) if len(historical_data) > 0 else None
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Health Status", result['health_label'], 
                         delta=f"{result['failure_probability']:.1%} prob", delta_color="inverse")
            with col2:
                st.metric("RUL", f"{result['rul_hours']:.1f} hrs")
            with col3:
                st.metric("Progress", f"{current_step}/{max_step}")
            with col4:
                st.metric("Method", result['method'])
            
            # Gauge
            health_values = {'Normal': 33, 'Warning': 66, 'Critical': 100}
            health_value = health_values.get(result['health_label'], 0)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_value,
                title={'text': "Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': visuals['health_color'] if visuals else 'gray'},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "lightyellow"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Top features
            st.subheader("📊 Top Contributing Factors")
            cols = st.columns(min(5, len(result['top_features'])))
            for idx, (feature, importance) in enumerate(result['top_features']):
                if idx < 5:
                    display_name = feature.replace('_', ' ').title()
                    if display_name.startswith('Type'):
                        display_name = display_name.replace('Type', 'Machine Type:')
                    with cols[idx]:
                        st.metric(display_name, f"{importance:.1%}")
            
            # Explanation
            st.subheader("💡 AI Explanation")
            explanation = generate_explanation(result)
            if result['health_label'] == 'Critical':
                st.error(explanation)
            elif result['health_label'] == 'Warning':
                st.warning(explanation)
            else:
                st.success(explanation)
            
            # Charts
            if visuals:
                st.plotly_chart(visuals['trend_chart'], use_container_width=True)
                if visuals['importance_chart']:
                    st.plotly_chart(visuals['importance_chart'], use_container_width=True)
            
            # Evaluation
            with st.expander("📈 View Model Evaluation Metrics"):
                st.subheader("Classification Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.2%}")
                with col2:
                    st.metric("Precision", f"{st.session_state.metrics['precision']:.2%}")
                with col3:
                    st.metric("Recall", f"{st.session_state.metrics['recall']:.2%}")
                with col4:
                    st.metric("F1-Score", f"{st.session_state.metrics['f1']:.2%}")
                
                st.subheader("Regression Metrics (RUL)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE", f"{st.session_state.metrics['mae']:.2f} hrs")
                with col2:
                    st.metric("RMSE", f"{st.session_state.metrics['rmse']:.2f} hrs")
                
                cm = st.session_state.metrics['confusion_matrix']
                labels = list(st.session_state.label_map.keys())
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels, colorscale='RdBu',
                    text=cm, texttemplate="%{text}", textfont={"size": 16}
                ))
                fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Auto-advance
            if st.session_state.simulation_running and current_step < max_step:
                time.sleep(0.1 * (10 / st.session_state.speed))
                st.session_state.current_step += 1
                st.rerun()
    
    elif st.session_state.data is not None and st.session_state.data_loaded and not st.session_state.models_trained:
        if 'machine_id' in st.session_state.data.columns:
            st.info("👈 Click 'Train ML Models' in the sidebar to start the AI analysis.")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.data.head(20))
        else:
            st.error("❌ Data is missing 'machine_id' column!")
    
    elif st.session_state.data is not None and not st.session_state.data_loaded:
        st.error(f"❌ Invalid data format: {st.session_state.data_error}")
        st.info("""
        Please make sure your CSV has these required columns:
        - **machine_id**: Machine identifier (e.g., 'M01')
        - **temperature_c**: Temperature in Celsius
        - **vibration_mm_s**: Vibration in mm/s RMS
        - **pressure_bar**: Pressure in bar
        - **current_a**: Current in Amperes
        - **rpm**: Rotational speed
        - **runtime_hours**: Cumulative runtime
        - **health_label**: 'Normal', 'Warning', or 'Critical'
        - **rul_hours**: Remaining Useful Life in hours
        """)
    
    else:
        st.info("👈 Generate or upload data using the sidebar controls to begin.")
        
        st.markdown("""
        ### 📋 Required CSV Format
        
        Your CSV must have these columns:
        
        | Column | Description | Example |
        |--------|-------------|---------|
        | `machine_id` | Machine identifier | M01 |
        | `temperature_c` | Temperature in Celsius | 65.2 |
        | `vibration_mm_s` | Vibration in mm/s RMS | 2.4 |
        | `pressure_bar` | Pressure in bar | 3.1 |
        | `current_a` | Current in Amperes | 15.3 |
        | `rpm` | Rotational speed | 1205 |
        | `runtime_hours` | Cumulative runtime | 0.0 |
        | `health_label` | Health status | Normal |
        | `rul_hours` | Remaining Useful Life | 100.0 |
        
        ### 🚀 Quick Start
        
        1. Click **"Download Sample CSV"** in the sidebar
        2. Open the file in Excel to see the format
        3. Upload your own CSV file
        4. Train the model and start the simulation
        """)

if __name__ == "__main__":
    main()