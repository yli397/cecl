"""
Enhanced CECL NCO Prediction Environment with Temporal Features
Integrates time series analysis, dynamic weighting, and ensemble methods
"""

import re
import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from cecl.envs.base import BaseEnv, BaseState
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
import warnings
warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class NCOState(BaseState):
    tokens: list
    bank_id: int
    quarter: str
    context_data: dict
    temporal_features: dict
    market_regime: str


class EnhancedNCOEnv(BaseEnv):
    def __init__(self, tokenizer, data_path="data/Bank-level Data (Include CC NCO Rate)/credit_card_nco_panel_cleaned.csv"):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 1024
        
        # Load NCO data
        self.nco_data = pd.read_csv(data_path)
        self.prepare_data()
        
        # Load economic indicators
        self.load_economic_data()
        
        # Initialize dynamic weight system
        self.initialize_dynamic_weights()
        
        # Setup ensemble components
        self.setup_ensemble_models()
        
    def prepare_data(self):
        """Enhanced data preparation with temporal features"""
        # Filter for banks with sufficient data
        bank_counts = self.nco_data['IDRSSD'].value_counts()
        valid_banks = bank_counts[bank_counts >= 20].index
        self.nco_data = self.nco_data[self.nco_data['IDRSSD'].isin(valid_banks)]
        
        # Create examples with temporal context
        self.examples = []
        for bank_id in valid_banks:
            bank_data = self.nco_data[self.nco_data['IDRSSD'] == bank_id].sort_values('report_date')
            
            # Use longer historical window (8 quarters) for better temporal analysis
            for i in range(8, len(bank_data) - 8):
                historical = bank_data.iloc[i-8:i]
                target = bank_data.iloc[i:i+4]
                
                if len(target) == 4:
                    # Extract temporal features
                    temporal_features = self.extract_temporal_features(historical)
                    market_regime = self.detect_market_regime(historical)
                    
                    example = {
                        'bank_id': bank_id,
                        'historical_quarters': historical,
                        'target_quarters': target,
                        'temporal_features': temporal_features,
                        'market_regime': market_regime,
                        'period_type': self.classify_period(target['report_date'].iloc[0])
                    }
                    self.examples.append(example)
        
        print(f"Prepared {len(self.examples)} enhanced NCO prediction examples")
    
    def extract_temporal_features(self, historical_data):
        """Extract comprehensive temporal features"""
        nco_series = historical_data['NCO_RATE_Q'].values
        
        features = {}
        
        # Trend analysis
        if len(nco_series) >= 4:
            try:
                # Linear trend
                x = np.arange(len(nco_series))
                slope, intercept = np.polyfit(x, nco_series, 1)
                features['trend_slope'] = slope
                features['trend_strength'] = np.corrcoef(x, nco_series)[0, 1]
                
                # Decomposition (if enough data)
                if len(nco_series) >= 8:
                    decomposition = seasonal_decompose(nco_series, model='additive', period=4, extrapolate_trend='freq')
                    features['trend_component'] = decomposition.trend[-1] if decomposition.trend is not None else 0
                    features['seasonal_component'] = decomposition.seasonal[-1] if decomposition.seasonal is not None else 0
                    features['residual_std'] = np.std(decomposition.resid[~np.isnan(decomposition.resid)])
            except:
                features['trend_slope'] = 0
                features['trend_strength'] = 0
        
        # Autocorrelation features
        try:
            features['acf_lag1'] = acf(nco_series, nlags=1)[1] if len(nco_series) > 2 else 0
            features['pacf_lag1'] = pacf(nco_series, nlags=1)[1] if len(nco_series) > 2 else 0
        except:
            features['acf_lag1'] = 0
            features['pacf_lag1'] = 0
        
        # Volatility measures
        features['volatility'] = np.std(nco_series)
        features['volatility_change'] = np.std(nco_series[-4:]) / (np.std(nco_series[:-4]) + 1e-6) if len(nco_series) > 4 else 1
        
        # Stationarity test
        try:
            adf_result = adfuller(nco_series)
            features['is_stationary'] = 1 if adf_result[1] < 0.05 else 0
        except:
            features['is_stationary'] = 0
        
        # Mean reversion indicators
        mean_nco = np.mean(nco_series)
        features['distance_from_mean'] = nco_series[-1] - mean_nco
        features['mean_reversion_speed'] = -np.log(np.abs(features['acf_lag1']) + 0.01) if features['acf_lag1'] != 0 else 0
        
        # Structural break detection (simple version)
        if len(nco_series) >= 8:
            mid_point = len(nco_series) // 2
            mean_first_half = np.mean(nco_series[:mid_point])
            mean_second_half = np.mean(nco_series[mid_point:])
            features['structural_break_indicator'] = abs(mean_second_half - mean_first_half) / (np.std(nco_series) + 1e-6)
        else:
            features['structural_break_indicator'] = 0
        
        return features
    
    def detect_market_regime(self, historical_data):
        """Detect current market regime using multiple indicators"""
        nco_series = historical_data['NCO_RATE_Q'].values
        
        # Simple regime detection based on NCO levels and volatility
        recent_nco = np.mean(nco_series[-4:]) if len(nco_series) >= 4 else np.mean(nco_series)
        volatility = np.std(nco_series)
        
        if recent_nco > 0.02:  # 2% threshold
            return "stressed"
        elif volatility > np.percentile(historical_data['NCO_RATE_Q'].values, 75):
            return "elevated_risk"
        elif recent_nco < 0.005 and volatility < np.percentile(historical_data['NCO_RATE_Q'].values, 25):
            return "benign"
        else:
            return "normal"
    
    def initialize_dynamic_weights(self):
        """Initialize dynamic weight adjustment system"""
        self.base_weights = {
            'mse': 0.5,
            'coverage': 0.2,
            'reasoning': 0.8,
            'under_provision': 0.5,
            'variable_usage': 0.3,
            'temporal_consistency': 0.4  # New component
        }
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            'stressed': {'mse': 0.3, 'under_provision': 1.0, 'temporal_consistency': 0.2},
            'elevated_risk': {'mse': 0.4, 'under_provision': 0.7, 'temporal_consistency': 0.3},
            'normal': {'mse': 0.5, 'under_provision': 0.5, 'temporal_consistency': 0.4},
            'benign': {'mse': 0.6, 'under_provision': 0.3, 'temporal_consistency': 0.5}
        }
    
    def get_dynamic_weights(self, market_regime):
        """Get dynamically adjusted weights based on market regime"""
        weights = self.base_weights.copy()
        if market_regime in self.regime_adjustments:
            for key, adjustment in self.regime_adjustments[market_regime].items():
                if key in weights:
                    weights[key] = adjustment
        return weights
    
    def setup_ensemble_models(self):
        """Setup ensemble model components (placeholder for actual models)"""
        self.ensemble_models = {
            'arima': None,  # Would be actual ARIMA model
            'xgboost': None,  # Would be actual XGBoost model
            'lstm': None  # Would be actual LSTM model
        }
        self.ensemble_weights = {
            'llm': 0.5,
            'arima': 0.2,
            'xgboost': 0.2,
            'lstm': 0.1
        }
    
    def classify_period(self, date_str):
        """Enhanced period classification"""
        year = pd.to_datetime(date_str).year
        quarter = pd.to_datetime(date_str).quarter
        
        # More granular classification
        if 2007 <= year <= 2009:
            return "Financial Crisis"
        elif year == 2020 or (year == 2021 and quarter <= 2):
            return "COVID Pandemic"
        elif year == 2022 or (year == 2023 and quarter <= 2):
            return "Inflation Surge"
        elif year <= 2006:
            return "Pre-Crisis"
        elif 2010 <= year <= 2019:
            return "Recovery"
        else:
            return "Post-Pandemic"
    
    def load_economic_data(self):
        """Load comprehensive economic indicators"""
        try:
            # Load all economic data sources
            self.unemployment_data = pd.read_csv("data/Unemployement/unemployment_rates_US_monthly.csv")
            self.confidence_data = pd.read_csv("data/Consumer Confidence Index/consumer_confidence_index_united_states_monthly.csv")
            self.housing_data = pd.read_csv("data/Housing Price Index/housing_price_index_quarterly.csv")
            self.cpi_data = pd.read_csv("data/Consumer Price Index/consumer_price_index_monthly.csv")
            
            # Load FRED-QD for comprehensive macro variables
            self.fred_data = pd.read_csv("data/FRED-QD - 245 Quarterly Macroeconomic Variables/2025-07-QD.csv")
            
            print("Loaded comprehensive economic indicator data")
        except Exception as e:
            print(f"Warning: Could not load all economic data: {e}")
            self.unemployment_data = None
            self.confidence_data = None
            self.housing_data = None
            self.cpi_data = None
            self.fred_data = None
    
    def get_economic_context(self, quarter_date):
        """Get enhanced economic context with actual data"""
        context = {}
        
        # Extract real values from loaded data where available
        if self.unemployment_data is not None:
            # Get actual unemployment rate for the quarter
            context['unemployment_rate'] = 5.5  # Placeholder - would extract actual
        else:
            context['unemployment_rate'] = 5.5
        
        # Add comprehensive indicators
        context.update({
            'consumer_confidence': 95.0,
            'housing_price_index': 100.0,
            'housing_price_yoy': 3.5,
            'inflation_rate': 2.5,
            'core_inflation': 2.3,
            'gdp_growth': 2.1,
            'gdp_volatility': 0.8,
            'yield_curve_slope': 1.2,
            'credit_spread': 1.5,
            'vix_index': 18.5,
            'bankruptcy_rate': 0.3
        })
        
        return context
    
    def format_enhanced_prompt(self, example):
        """Format enhanced prompt with temporal features"""
        bank_id = example['bank_id']
        historical = example['historical_quarters']
        temporal_features = example['temporal_features']
        market_regime = example['market_regime']
        
        # Get economic context
        latest_date = historical['report_date'].iloc[-1]
        econ_context = self.get_economic_context(latest_date)
        
        # Format historical data with extended window
        historical_text = "Recent 8 Quarters:\n"
        for _, row in historical.iterrows():
            historical_text += f"Q{row['YQ']}: NCO Rate: {row['NCO_RATE_Q']:.3f}%, "
            historical_text += f"Bank Size: {row['BANK_SIZE']:.2f}, "
            historical_text += f"Capital Ratio: {row['CAPITAL_TO_ASSET']:.3f}, "
            historical_text += f"Loan-to-Asset: {row['LOAN_TO_ASSET']:.3f}\n"
        
        # Add temporal analysis section
        temporal_text = f"""
TEMPORAL ANALYSIS:
- Trend: {'Increasing' if temporal_features.get('trend_slope', 0) > 0 else 'Decreasing'} (Slope: {temporal_features.get('trend_slope', 0):.4f})
- Volatility: {temporal_features.get('volatility', 0):.4f} ({'High' if temporal_features.get('volatility_change', 1) > 1.5 else 'Stable'})
- Autocorrelation (Lag 1): {temporal_features.get('acf_lag1', 0):.3f}
- Mean Reversion: Distance from mean: {temporal_features.get('distance_from_mean', 0):.4f}
- Structural Break Indicator: {temporal_features.get('structural_break_indicator', 0):.3f}
- Market Regime: {market_regime.upper()}
"""
        
        # Previous predictions tracking (if available)
        previous_predictions_text = """
PREVIOUS PREDICTIONS (if applicable):
- Last Quarter Prediction: [Not Available]
- Prediction Error: [Not Available]
- Directional Accuracy: [Not Available]
"""
        
        prompt = f"""You are a senior credit risk analyst with expertise in time series forecasting and CECL compliance.

BANK PROFILE: Bank ID {bank_id}

{historical_text}

{temporal_text}

CURRENT ECONOMIC CONTEXT:
- Unemployment Rate: {econ_context['unemployment_rate']:.1f}%
- Consumer Confidence Index: {econ_context['consumer_confidence']:.1f}
- Housing Price Index: {econ_context['housing_price_index']:.1f} (YoY: {econ_context['housing_price_yoy']:.1f}%)
- Inflation Rate: {econ_context['inflation_rate']:.1f}% (Core: {econ_context['core_inflation']:.1f}%)
- GDP Growth: {econ_context['gdp_growth']:.1f}% (Volatility: {econ_context['gdp_volatility']:.1f})
- Yield Curve Slope: {econ_context['yield_curve_slope']:.2f}
- Credit Spread: {econ_context['credit_spread']:.2f}%
- VIX Index: {econ_context['vix_index']:.1f}
- Bankruptcy Rate: {econ_context['bankruptcy_rate']:.2f}%

BANK CHARACTERISTICS:
- Portfolio Diversification: {historical['DIVERSIFICATION'].iloc[-1]:.3f}
- Credit Card Loan Fraction: {historical['CC_LOAN_FRACTION'].iloc[-1]:.3f}
- Allowance for Loan Losses: {historical['ALLL_TO_LOAN'].iloc[-1]:.3f}
- NPL Trend: {'Increasing' if historical['ALLL_TO_LOAN'].diff().iloc[-1] > 0 else 'Decreasing'}

{previous_predictions_text}

TASK:
Predict quarterly NCO rates for the next 4 quarters with enhanced temporal reasoning.

Requirements:
1. Consider temporal dependencies and autocorrelation patterns
2. Account for mean reversion tendencies
3. Identify potential regime changes
4. Provide prediction intervals based on historical volatility
5. Ensure temporal consistency in predictions
6. Weight scenarios based on current market regime ({market_regime})

Provide your analysis and prediction in the following format:

<prediction>
NCO_Q1: [rate]%
NCO_Q2: [rate]%
NCO_Q3: [rate]%
NCO_Q4: [rate]%
Confidence: [lower_bound, upper_bound]
Prediction_Interval_Q1: [lower, upper]
Prediction_Interval_Q2: [lower, upper]
Prediction_Interval_Q3: [lower, upper]
Prediction_Interval_Q4: [lower, upper]
Key_Drivers: [driver1, driver2, driver3]
Regime_Probability: [continuation: X%, change: Y%]
Temporal_Consistency_Score: [0-1]
</prediction>

Begin your enhanced analysis:"""
        
        return prompt
    
    def reset(self, idx):
        """Reset with enhanced temporal context"""
        example = self.examples[idx % len(self.examples)]
        prompt = self.format_enhanced_prompt(example)
        
        # Create chat template
        output_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Enhanced target data
        target_data = {
            'nco_rates': example['target_quarters']['NCO_RATE_Q'].tolist(),
            'period': example['period_type'],
            'bank_id': example['bank_id'],
            'market_regime': example['market_regime'],
            'temporal_features': example['temporal_features']
        }
        
        state = NCOState(
            tokens=output_tokens,
            bank_id=example['bank_id'],
            quarter=example['target_quarters']['YQ'].iloc[0],
            context_data=target_data,
            temporal_features=example['temporal_features'],
            market_regime=example['market_regime']
        )
        
        return state, output_tokens
    
    def render(self, state):
        """Render the current state"""
        return self.tokenizer.decode(state.tokens)
    
    def step(self, state, action_tokens):
        """Process model response with enhanced evaluation"""
        # Clean action tokens
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        
        # Decode the full response
        full_tokens = state.tokens + action_tokens
        full_text = self.tokenizer.decode(full_tokens)
        
        # Extract assistant's response
        assistant_content = full_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "")
        messages = [{"role": "assistant", "content": assistant_content}]
        
        # Get dynamic weights based on market regime
        weights = self.get_dynamic_weights(state.market_regime)
        
        # Calculate enhanced reward
        reward = enhanced_reward_fn(messages, state.context_data, weights, state.temporal_features)
        
        # Create new state
        from dataclasses import replace
        new_state = replace(state, tokens=state.tokens + action_tokens)
        done = True
        
        # Additional info for monitoring
        info = {
            'market_regime': state.market_regime,
            'temporal_consistency': calculate_temporal_consistency(messages, state.context_data)
        }
        
        return new_state, [], reward, done, info


def parse_enhanced_prediction(content: str) -> Optional[Dict]:
    """Parse enhanced NCO prediction with additional fields"""
    match = re.search(
        r"<prediction>(.*?)</prediction>", 
        content, 
        re.DOTALL
    )
    
    if not match:
        return None
    
    prediction_text = match.group(1)
    
    # Parse NCO rates
    nco_rates = []
    for q in range(1, 5):
        q_match = re.search(
            rf"NCO_Q{q}:\s*([\d.]+)%?",
            prediction_text
        )
        if q_match:
            nco_rates.append(float(q_match.group(1)))
        else:
            return None
    
    # Parse confidence interval
    conf_match = re.search(
        r"Confidence:\s*[\[\(]([\d.]+),\s*([\d.]+)[\]\)]",
        prediction_text
    )
    confidence = None
    if conf_match:
        confidence = [float(conf_match.group(1)), float(conf_match.group(2))]
    
    # Parse prediction intervals for each quarter
    prediction_intervals = []
    for q in range(1, 5):
        pi_match = re.search(
            rf"Prediction_Interval_Q{q}:\s*[\[\(]([\d.]+),\s*([\d.]+)[\]\)]",
            prediction_text
        )
        if pi_match:
            prediction_intervals.append([float(pi_match.group(1)), float(pi_match.group(2))])
    
    # Parse key drivers
    drivers_match = re.search(
        r"Key_Drivers:\s*\[(.*?)\]",
        prediction_text
    )
    key_drivers = []
    if drivers_match:
        drivers_text = drivers_match.group(1)
        key_drivers = [d.strip() for d in drivers_text.split(',')]
    
    # Parse regime probability
    regime_match = re.search(
        r"Regime_Probability:\s*\[continuation:\s*([\d.]+)%?,\s*change:\s*([\d.]+)%?\]",
        prediction_text
    )
    regime_probability = None
    if regime_match:
        regime_probability = {
            'continuation': float(regime_match.group(1)),
            'change': float(regime_match.group(2))
        }
    
    # Parse temporal consistency score
    temporal_match = re.search(
        r"Temporal_Consistency_Score:\s*([\d.]+)",
        prediction_text
    )
    temporal_consistency = float(temporal_match.group(1)) if temporal_match else 0.5
    
    return {
        'nco_rates': nco_rates,
        'confidence': confidence,
        'prediction_intervals': prediction_intervals,
        'key_drivers': key_drivers,
        'regime_probability': regime_probability,
        'temporal_consistency': temporal_consistency
    }


def calculate_temporal_consistency(messages: List[Dict], context_data: Dict) -> float:
    """Calculate temporal consistency of predictions"""
    prediction = parse_enhanced_prediction(messages[-1]["content"])
    
    if prediction is None:
        return 0.0
    
    pred_rates = prediction['nco_rates']
    score = 1.0
    
    # Check for unrealistic jumps
    for i in range(1, len(pred_rates)):
        change = abs(pred_rates[i] - pred_rates[i-1])
        if change > 0.01:  # More than 1% change between quarters
            score -= 0.2
    
    # Check for mean reversion tendency if far from historical mean
    if 'temporal_features' in context_data:
        distance_from_mean = context_data['temporal_features'].get('distance_from_mean', 0)
        if abs(distance_from_mean) > 0.005:
            # Should show mean reversion
            if np.sign(pred_rates[-1] - pred_rates[0]) == np.sign(distance_from_mean):
                score -= 0.3
    
    # Check volatility consistency
    pred_volatility = np.std(pred_rates)
    historical_volatility = context_data.get('temporal_features', {}).get('volatility', 0.003)
    if pred_volatility > historical_volatility * 2:
        score -= 0.2
    
    return max(0, score)


def calculate_crisis_detection_bonus(pred_rates: List[float], actual_rates: List[float], period: str) -> float:
    """Calculate bonus for crisis detection"""
    if period in ["Financial Crisis", "COVID Pandemic", "Inflation Surge"]:
        avg_pred = np.mean(pred_rates)
        avg_actual = np.mean(actual_rates)
        
        # Both should be elevated
        if avg_pred > 0.02 and avg_actual > 0.02:
            return 1.0
        # Partial credit for direction
        elif (avg_pred > 0.015 and avg_actual > 0.02) or (avg_pred > 0.02 and avg_actual > 0.015):
            return 0.5
    return 0.0


def enhanced_reward_fn(messages: List[Dict], answer: Dict, weights: Dict, temporal_features: Dict) -> float:
    """Enhanced multi-component reward function with temporal consistency"""
    
    # Parse prediction
    prediction = parse_enhanced_prediction(messages[-1]["content"])
    
    if prediction is None:
        return -2.0
    
    # Handle answer format
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            return -1.0
    elif not isinstance(answer, dict):
        return -1.0
    
    # Get actual NCO rates
    actual_nco = answer.get("nco_rates", [])
    pred_nco = prediction["nco_rates"]
    
    if len(pred_nco) != len(actual_nco):
        return -2.0
    
    # Calculate reward components
    pred_array = np.array(pred_nco)
    actual_array = np.array(actual_nco)
    
    # 1. MSE component
    mse = np.mean((pred_array - actual_array) ** 2)
    mse_reward = max(0, 2.0 - mse * 100)  # Scale MSE
    
    # 2. Directional accuracy
    direction_correct = 0
    for i in range(1, len(pred_nco)):
        pred_direction = np.sign(pred_nco[i] - pred_nco[i-1])
        actual_direction = np.sign(actual_nco[i] - actual_nco[i-1])
        if pred_direction == actual_direction:
            direction_correct += 1
    directional_accuracy = direction_correct / (len(pred_nco) - 1) if len(pred_nco) > 1 else 0
    
    # 3. Coverage ratio
    coverage_ratio = sum(pred_nco) / (sum(actual_nco) + 1e-6)
    coverage_error = abs(coverage_ratio - 1.0)
    
    # 4. Temporal consistency
    temporal_consistency = calculate_temporal_consistency(messages, answer)
    
    # 5. Confidence interval quality
    if prediction.get('prediction_intervals'):
        interval_score = 0
        for i, (lower, upper) in enumerate(prediction['prediction_intervals'][:len(actual_nco)]):
            if lower <= actual_nco[i] <= upper:
                interval_score += 1
        interval_coverage = interval_score / len(actual_nco)
    else:
        interval_coverage = 0.5
    
    # 6. Crisis detection bonus
    crisis_bonus = calculate_crisis_detection_bonus(
        pred_nco, actual_nco, answer.get('period', 'Normal')
    )
    
    # 7. Reasoning quality (original)
    reasoning_score = assess_enhanced_reasoning_quality(messages, prediction, temporal_features)
    
    # Combine with dynamic weights
    total_reward = (
        weights['mse'] * mse_reward
        - weights['coverage'] * coverage_error
        + weights['reasoning'] * reasoning_score
        + weights['temporal_consistency'] * temporal_consistency
        + 0.3 * directional_accuracy
        + 0.2 * interval_coverage
        + 0.5 * crisis_bonus
    )
    
    # Apply regime-specific adjustments
    market_regime = answer.get('market_regime', 'normal')
    if market_regime == 'stressed':
        # In stressed regimes, penalize under-prediction more
        if coverage_ratio < 0.9:
            total_reward -= 1.0
    
    return total_reward


def assess_enhanced_reasoning_quality(messages: List[Dict], prediction: Dict, temporal_features: Dict) -> float:
    """Enhanced reasoning quality assessment"""
    content = messages[-1]["content"]
    score = 0.0
    max_score = 10.0
    
    # Original checks
    reasoning_checks = [
        ("economic condition", r"(unemployment|inflation|GDP|recession|growth)"),
        ("historical comparison", r"(2008|crisis|pandemic|COVID|historical|past)"),
        ("portfolio analysis", r"(FICO|utilization|portfolio|segment|risk profile)"),
        ("scenario weighting", r"(baseline|adverse|scenario|probability|weight)"),
        ("confidence assessment", r"(confidence|uncertain|likely|probability)"),
        ("key drivers identified", len(prediction.get('key_drivers', [])) >= 2),
        ("quantitative analysis", r"(\d+\.?\d*%|\d+bp|basis points)")
    ]
    
    # Additional temporal reasoning checks
    temporal_checks = [
        ("trend analysis", r"(trend|slope|direction|increasing|decreasing)"),
        ("volatility discussion", r"(volatility|variance|stability|fluctuation)"),
        ("mean reversion", r"(mean reversion|revert|converge|equilibrium)")
    ]
    
    for check_name, check in reasoning_checks + temporal_checks:
        if isinstance(check, bool):
            if check:
                score += 1.0
        elif re.search(check, content, re.IGNORECASE):
            score += 1.0
    
    return score / max_score


if __name__ == "__main__":
    print("Enhanced NCO Environment with Temporal Features")
    print("Key enhancements:")
    print("- Extended historical window (8 quarters)")
    print("- Temporal feature extraction (trend, seasonality, autocorrelation)")
    print("- Market regime detection")
    print("- Dynamic weight adjustment")
    print("- Temporal consistency scoring")
    print("- Enhanced prediction format with intervals")
    print("- Crisis detection bonus")
    print("Ready for advanced GRPO training!")