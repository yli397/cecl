"""
Single-Quarter NCO Prediction Environment for CECL
One-step ahead prediction with rich historical context
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
    target_quarter: str
    context_data: dict
    temporal_features: dict
    market_regime: str
    historical_quarters: list


class SingleQuarterNCOEnv(BaseEnv):
    """Environment for single-quarter NCO prediction"""
    
    def __init__(self, tokenizer, data_path="data/Bank-level Data (Include CC NCO Rate)/credit_card_nco_panel_cleaned.csv"):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 768  # Shorter for single prediction
        
        # Load NCO data
        self.nco_data = pd.read_csv(data_path)
        self.prepare_data()
        
        # Load economic indicators
        self.load_economic_data()
        
        # Initialize dynamic weight system
        self.initialize_dynamic_weights()
        
    def prepare_data(self):
        """Prepare single-quarter prediction examples"""
        # Filter for banks with sufficient history
        bank_counts = self.nco_data['IDRSSD'].value_counts()
        valid_banks = bank_counts[bank_counts >= 20].index
        self.nco_data = self.nco_data[self.nco_data['IDRSSD'].isin(valid_banks)]
        
        # Create single-quarter prediction examples
        self.examples = []
        for bank_id in valid_banks:
            bank_data = self.nco_data[self.nco_data['IDRSSD'] == bank_id].sort_values('report_date')
            
            # Use 12 quarters of history to predict next quarter
            for i in range(12, len(bank_data)):
                historical = bank_data.iloc[i-12:i]  # 12 quarters of history
                target = bank_data.iloc[i:i+1]  # Single quarter target
                
                if len(target) == 1:
                    # Extract temporal features from historical data
                    temporal_features = self.extract_temporal_features(historical)
                    market_regime = self.detect_market_regime(historical)
                    
                    # Get last 4 quarters for detailed analysis
                    recent_quarters = historical.tail(4)
                    
                    example = {
                        'bank_id': bank_id,
                        'historical_quarters': historical,
                        'recent_quarters': recent_quarters,
                        'target_quarter': target,
                        'target_nco': target['NCO_RATE_Q'].iloc[0],
                        'target_date': target['YQ'].iloc[0],
                        'temporal_features': temporal_features,
                        'market_regime': market_regime,
                        'period_type': self.classify_period(target['report_date'].iloc[0])
                    }
                    self.examples.append(example)
        
        print(f"Prepared {len(self.examples)} single-quarter NCO prediction examples")
        print(f"Using 12 quarters of history to predict 1 quarter ahead")
    
    def extract_temporal_features(self, historical_data):
        """Extract comprehensive temporal features from historical data"""
        nco_series = historical_data['NCO_RATE_Q'].values
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(nco_series)
        features['std'] = np.std(nco_series)
        features['recent_mean'] = np.mean(nco_series[-4:])  # Last year average
        features['recent_std'] = np.std(nco_series[-4:])
        
        # Trend analysis
        if len(nco_series) >= 4:
            x = np.arange(len(nco_series))
            slope, intercept = np.polyfit(x, nco_series, 1)
            features['trend_slope'] = slope
            features['trend_strength'] = np.corrcoef(x, nco_series)[0, 1]
            
            # Recent trend (last 4 quarters)
            recent_x = np.arange(4)
            recent_slope, _ = np.polyfit(recent_x, nco_series[-4:], 1)
            features['recent_trend'] = recent_slope
        
        # Momentum indicators
        features['momentum_3q'] = nco_series[-1] - nco_series[-4] if len(nco_series) >= 4 else 0
        features['momentum_6q'] = nco_series[-1] - nco_series[-7] if len(nco_series) >= 7 else 0
        
        # Autocorrelation features
        try:
            acf_values = acf(nco_series, nlags=min(4, len(nco_series)-1))
            features['acf_lag1'] = acf_values[1] if len(acf_values) > 1 else 0
            features['acf_lag4'] = acf_values[4] if len(acf_values) > 4 else 0
        except:
            features['acf_lag1'] = 0
            features['acf_lag4'] = 0
        
        # Volatility measures
        features['volatility'] = np.std(nco_series)
        features['volatility_ratio'] = np.std(nco_series[-4:]) / (np.std(nco_series[:-4]) + 1e-6) if len(nco_series) > 4 else 1
        
        # Mean reversion indicators
        features['distance_from_mean'] = nco_series[-1] - features['mean']
        features['zscore'] = features['distance_from_mean'] / (features['std'] + 1e-6)
        
        # Seasonal component (quarterly)
        if len(nco_series) >= 8:
            try:
                decomposition = seasonal_decompose(nco_series, model='additive', period=4, extrapolate_trend='freq')
                features['seasonal_component'] = decomposition.seasonal[-1]
                features['trend_component'] = decomposition.trend[-1]
                features['residual_component'] = decomposition.resid[-1]
            except:
                features['seasonal_component'] = 0
                features['trend_component'] = features['mean']
                features['residual_component'] = 0
        
        # Level shifts
        if len(nco_series) >= 8:
            first_half_mean = np.mean(nco_series[:len(nco_series)//2])
            second_half_mean = np.mean(nco_series[len(nco_series)//2:])
            features['level_shift'] = second_half_mean - first_half_mean
        
        return features
    
    def detect_market_regime(self, historical_data):
        """Detect current market regime from recent data"""
        nco_series = historical_data['NCO_RATE_Q'].values
        recent_nco = np.mean(nco_series[-4:]) if len(nco_series) >= 4 else np.mean(nco_series)
        volatility = np.std(nco_series[-4:]) if len(nco_series) >= 4 else np.std(nco_series)
        
        # Trend direction
        if len(nco_series) >= 4:
            recent_trend = nco_series[-1] - nco_series[-4]
        else:
            recent_trend = 0
        
        # Classify regime
        if recent_nco > 0.02:  # 2% threshold
            return "stressed"
        elif recent_nco > 0.01 and recent_trend > 0:
            return "deteriorating"
        elif volatility > np.percentile(historical_data['NCO_RATE_Q'].values, 75):
            return "elevated_risk"
        elif recent_nco < 0.005 and volatility < np.percentile(historical_data['NCO_RATE_Q'].values, 25):
            return "benign"
        else:
            return "normal"
    
    def initialize_dynamic_weights(self):
        """Initialize dynamic weight adjustment for single prediction"""
        self.base_weights = {
            'mse': 0.6,  # Higher weight on accuracy for single prediction
            'directional': 0.3,  # Direction is important
            'reasoning': 0.5,
            'temporal_consistency': 0.4,
            'confidence_calibration': 0.2
        }
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            'stressed': {'mse': 0.4, 'directional': 0.4, 'confidence_calibration': 0.3},
            'deteriorating': {'mse': 0.5, 'directional': 0.5, 'temporal_consistency': 0.3},
            'elevated_risk': {'mse': 0.5, 'directional': 0.4, 'temporal_consistency': 0.3},
            'normal': {'mse': 0.6, 'directional': 0.3, 'temporal_consistency': 0.4},
            'benign': {'mse': 0.7, 'directional': 0.2, 'temporal_consistency': 0.5}
        }
    
    def get_dynamic_weights(self, market_regime):
        """Get dynamically adjusted weights based on market regime"""
        weights = self.base_weights.copy()
        if market_regime in self.regime_adjustments:
            for key, adjustment in self.regime_adjustments[market_regime].items():
                if key in weights:
                    weights[key] = adjustment
        return weights
    
    def classify_period(self, date_str):
        """Classify economic period"""
        year = pd.to_datetime(date_str).year
        quarter = pd.to_datetime(date_str).quarter
        
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
        """Load economic indicator data"""
        try:
            self.unemployment_data = pd.read_csv("data/Unemployement/unemployment_rates_US_monthly.csv")
            self.confidence_data = pd.read_csv("data/Consumer Confidence Index/consumer_confidence_index_united_states_monthly.csv")
            self.housing_data = pd.read_csv("data/Housing Price Index/housing_price_index_quarterly.csv")
            print("Loaded economic indicator data")
        except Exception as e:
            print(f"Warning: Could not load economic data: {e}")
            self.unemployment_data = None
            self.confidence_data = None
            self.housing_data = None
    
    def get_economic_context(self, quarter_date):
        """Get economic context for the prediction quarter"""
        # Placeholder - would extract actual values
        context = {
            'unemployment_rate': 5.5,
            'unemployment_change': 0.2,
            'consumer_confidence': 95.0,
            'confidence_change': -2.0,
            'housing_price_index': 100.0,
            'housing_price_yoy': 3.5,
            'inflation_rate': 2.5,
            'gdp_growth': 2.1,
            'yield_curve_slope': 1.2,
            'credit_spread': 1.5
        }
        return context
    
    def format_single_quarter_prompt(self, example):
        """Format prompt for single-quarter prediction"""
        bank_id = example['bank_id']
        historical = example['historical_quarters']
        recent = example['recent_quarters']
        temporal_features = example['temporal_features']
        market_regime = example['market_regime']
        target_date = example['target_date']
        
        # Get economic context
        econ_context = self.get_economic_context(target_date)
        
        # Format full historical context (12 quarters)
        historical_summary = f"""
HISTORICAL CONTEXT (12 Quarters):
- Average NCO: {temporal_features['mean']:.3f}%
- Std Dev: {temporal_features['std']:.3f}%
- Trend: {'Increasing' if temporal_features.get('trend_slope', 0) > 0 else 'Decreasing'} ({temporal_features.get('trend_slope', 0):.4f})
- Recent Momentum (3Q): {temporal_features.get('momentum_3q', 0):.3f}%
- Current Distance from Mean: {temporal_features.get('distance_from_mean', 0):.3f}% (Z-score: {temporal_features.get('zscore', 0):.2f})
"""
        
        # Format recent 4 quarters detail
        recent_text = "RECENT 4 QUARTERS DETAIL:\n"
        for _, row in recent.iterrows():
            recent_text += f"Q{row['YQ']}: NCO={row['NCO_RATE_Q']:.3f}%, "
            recent_text += f"Capital={row['CAPITAL_TO_ASSET']:.3f}, "
            recent_text += f"NPL={row['ALLL_TO_LOAN']:.3f}\n"
        
        # Temporal analysis
        temporal_text = f"""
TEMPORAL ANALYSIS:
- Last Value: {historical['NCO_RATE_Q'].iloc[-1]:.3f}%
- Recent Trend: {'Up' if temporal_features.get('recent_trend', 0) > 0 else 'Down'} ({temporal_features.get('recent_trend', 0):.4f})
- Volatility Ratio: {temporal_features.get('volatility_ratio', 1):.2f}
- ACF(1): {temporal_features.get('acf_lag1', 0):.3f} (persistence)
- Seasonal Component: {temporal_features.get('seasonal_component', 0):.3f}%
- Market Regime: {market_regime.upper()}
"""
        
        prompt = f"""You are a senior credit risk analyst specializing in quarterly NCO forecasting.

TARGET: Predict NCO rate for Q{target_date} (next quarter)
BANK ID: {bank_id}

{historical_summary}

{recent_text}

{temporal_text}

ECONOMIC INDICATORS (Current):
- Unemployment: {econ_context['unemployment_rate']:.1f}% (Δ: {econ_context['unemployment_change']:+.1f}%)
- Consumer Confidence: {econ_context['consumer_confidence']:.1f} (Δ: {econ_context['confidence_change']:+.1f})
- Housing Price Index: {econ_context['housing_price_index']:.1f} (YoY: {econ_context['housing_price_yoy']:+.1f}%)
- Inflation: {econ_context['inflation_rate']:.1f}%
- GDP Growth: {econ_context['gdp_growth']:.1f}%

BANK CHARACTERISTICS:
- Portfolio Diversification: {historical['DIVERSIFICATION'].iloc[-1]:.3f}
- Credit Card Fraction: {historical['CC_LOAN_FRACTION'].iloc[-1]:.3f}
- Capital Ratio: {historical['CAPITAL_TO_ASSET'].iloc[-1]:.3f}

TASK: Predict the NCO rate for the NEXT QUARTER ONLY.

Consider:
1. Historical trend and momentum
2. Mean reversion tendencies (current z-score: {temporal_features.get('zscore', 0):.2f})
3. Autocorrelation patterns (persistence)
4. Economic leading indicators
5. Bank-specific risk factors
6. Current market regime ({market_regime})

Provide your analysis and single-quarter prediction:

<prediction>
NCO_Next_Quarter: [rate]%
Confidence_Interval: [lower, upper]
Expected_Direction: [increase/decrease/stable]
Key_Driver: [primary factor influencing prediction]
Confidence_Level: [high/medium/low]
</prediction>

Begin your analysis:"""
        
        return prompt
    
    def reset(self, idx):
        """Reset with single-quarter prediction setup"""
        example = self.examples[idx % len(self.examples)]
        prompt = self.format_single_quarter_prompt(example)
        
        # Create chat template
        output_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Target data for reward calculation
        target_data = {
            'nco_actual': example['target_nco'],
            'previous_nco': example['historical_quarters']['NCO_RATE_Q'].iloc[-1],
            'period': example['period_type'],
            'bank_id': example['bank_id'],
            'market_regime': example['market_regime'],
            'temporal_features': example['temporal_features'],
            'historical_nco': example['historical_quarters']['NCO_RATE_Q'].tolist()
        }
        
        state = NCOState(
            tokens=output_tokens,
            bank_id=example['bank_id'],
            target_quarter=example['target_date'],
            context_data=target_data,
            temporal_features=example['temporal_features'],
            market_regime=example['market_regime'],
            historical_quarters=example['recent_quarters']['YQ'].tolist()
        )
        
        return state, output_tokens
    
    def render(self, state):
        """Render the current state"""
        return self.tokenizer.decode(state.tokens)
    
    def step(self, state, action_tokens):
        """Process model response for single-quarter prediction"""
        # Clean action tokens
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        
        # Decode the full response
        full_tokens = state.tokens + action_tokens
        full_text = self.tokenizer.decode(full_tokens)
        
        # Extract assistant's response
        assistant_content = full_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "")
        messages = [{"role": "assistant", "content": assistant_content}]
        
        # Get dynamic weights
        weights = self.get_dynamic_weights(state.market_regime)
        
        # Calculate reward for single prediction
        reward = single_quarter_reward_fn(messages, state.context_data, weights)
        
        # Create new state
        from dataclasses import replace
        new_state = replace(state, tokens=state.tokens + action_tokens)
        done = True
        
        # Additional info - empty dict to avoid JAX string handling issues
        info = {}
        
        return new_state, [], reward, done, info


def parse_single_prediction(content: str) -> Optional[Dict]:
    """Parse single-quarter NCO prediction"""
    match = re.search(
        r"<prediction>(.*?)</prediction>", 
        content, 
        re.DOTALL
    )
    
    if not match:
        return None
    
    prediction_text = match.group(1)
    
    # Parse NCO prediction
    nco_match = re.search(
        r"NCO_Next_Quarter:\s*([\d.]+)%?",
        prediction_text
    )
    if not nco_match:
        return None
    
    prediction = {
        'nco_rate': float(nco_match.group(1))
    }
    
    # Parse confidence interval
    conf_match = re.search(
        r"Confidence_Interval:\s*\[([\d.]+),\s*([\d.]+)\]",
        prediction_text
    )
    if conf_match:
        prediction['confidence_interval'] = [
            float(conf_match.group(1)), 
            float(conf_match.group(2))
        ]
    
    # Parse expected direction
    dir_match = re.search(
        r"Expected_Direction:\s*(\w+)",
        prediction_text
    )
    if dir_match:
        prediction['direction'] = dir_match.group(1).lower()
    
    # Parse key driver
    driver_match = re.search(
        r"Key_Driver:\s*([^\n]+)",
        prediction_text
    )
    if driver_match:
        prediction['key_driver'] = driver_match.group(1).strip()
    
    # Parse confidence level
    conf_level_match = re.search(
        r"Confidence_Level:\s*(\w+)",
        prediction_text
    )
    if conf_level_match:
        prediction['confidence_level'] = conf_level_match.group(1).lower()
    
    return prediction


def single_quarter_reward_fn(messages: List[Dict], context_data: Dict, weights: Dict) -> float:
    """Reward function for single-quarter prediction"""
    
    # Parse prediction
    prediction = parse_single_prediction(messages[-1]["content"])
    
    if prediction is None:
        return -2.0  # Invalid format penalty
    
    pred_nco = prediction['nco_rate']
    actual_nco = context_data['nco_actual']
    previous_nco = context_data['previous_nco']
    
    # 1. Accuracy component (MSE)
    mse = (pred_nco - actual_nco) ** 2
    accuracy_reward = max(0, 2.0 - mse * 1000)  # Scale for percentage values
    
    # 2. Directional accuracy
    actual_direction = np.sign(actual_nco - previous_nco)
    pred_direction = np.sign(pred_nco - previous_nco)
    
    # Also check if prediction matches stated direction
    if prediction.get('direction'):
        if prediction['direction'] == 'increase' and pred_direction > 0:
            direction_consistent = True
        elif prediction['direction'] == 'decrease' and pred_direction < 0:
            direction_consistent = True
        elif prediction['direction'] == 'stable' and abs(pred_direction) < 0.001:
            direction_consistent = True
        else:
            direction_consistent = False
    else:
        direction_consistent = True
    
    direction_reward = 1.0 if pred_direction == actual_direction else -0.5
    if not direction_consistent:
        direction_reward -= 0.5
    
    # 3. Confidence calibration
    if prediction.get('confidence_interval'):
        lower, upper = prediction['confidence_interval']
        interval_width = upper - lower
        
        # Check if actual falls within interval
        in_interval = lower <= actual_nco <= upper
        
        # Reward narrower intervals that contain the actual
        if in_interval:
            # Narrower intervals get higher reward
            calibration_reward = 1.0 - min(interval_width / 0.01, 1.0)  # Normalize by 1%
        else:
            # Penalty for missing the actual
            distance_to_interval = min(abs(actual_nco - lower), abs(actual_nco - upper))
            calibration_reward = -distance_to_interval * 100
    else:
        calibration_reward = 0
    
    # 4. Temporal consistency
    # Check if prediction is reasonable given historical volatility
    historical_nco = context_data.get('historical_nco', [])
    if historical_nco:
        historical_std = np.std(historical_nco)
        deviation = abs(pred_nco - previous_nco)
        
        # Penalize predictions that deviate too much from historical patterns
        if deviation > 3 * historical_std:
            temporal_penalty = -1.0
        elif deviation > 2 * historical_std:
            temporal_penalty = -0.5
        else:
            temporal_penalty = 0
    else:
        temporal_penalty = 0
    
    # 5. Reasoning quality
    reasoning_score = assess_single_prediction_reasoning(messages, prediction, context_data)
    
    # 6. Crisis detection bonus (if applicable)
    crisis_bonus = 0
    if context_data.get('period') in ["Financial Crisis", "COVID Pandemic"]:
        if actual_nco > 0.02 and pred_nco > 0.015:
            crisis_bonus = 0.5  # Correctly identified elevated risk
    
    # Combine components with weights
    total_reward = (
        weights.get('mse', 0.6) * accuracy_reward +
        weights.get('directional', 0.3) * direction_reward +
        weights.get('confidence_calibration', 0.2) * calibration_reward +
        weights.get('temporal_consistency', 0.4) * temporal_penalty +
        weights.get('reasoning', 0.5) * reasoning_score +
        crisis_bonus
    )
    
    return total_reward


def assess_single_prediction_reasoning(messages: List[Dict], prediction: Dict, context_data: Dict) -> float:
    """Assess reasoning quality for single prediction"""
    content = messages[-1]["content"]
    score = 0.0
    max_score = 8.0
    
    # Check for key reasoning components
    checks = [
        ("trend analysis", r"(trend|momentum|direction)"),
        ("mean reversion", r"(mean reversion|z-score|deviation from mean)"),
        ("economic factors", r"(unemployment|confidence|housing|GDP)"),
        ("persistence analysis", r"(autocorrelation|ACF|persistence|lag)"),
        ("regime consideration", r"(regime|stressed|elevated|normal)"),
        ("bank specifics", r"(capital|diversification|portfolio)"),
        ("confidence justification", prediction.get('confidence_level') is not None),
        ("key driver identified", prediction.get('key_driver') is not None)
    ]
    
    for check_name, check in checks:
        if isinstance(check, bool):
            if check:
                score += 1.0
        elif re.search(check, content, re.IGNORECASE):
            score += 1.0
    
    return score / max_score


def assess_prediction_quality(messages: List[Dict], context_data: Dict) -> Dict:
    """Assess overall prediction quality"""
    prediction = parse_single_prediction(messages[-1]["content"])
    
    if prediction is None:
        return {'quality': 'invalid'}
    
    quality_metrics = {}
    
    # Check prediction reasonableness
    pred_nco = prediction['nco_rate']
    if 0 <= pred_nco <= 10:  # Reasonable range for NCO
        quality_metrics['reasonable_range'] = True
    else:
        quality_metrics['reasonable_range'] = False
    
    # Check confidence interval quality
    if prediction.get('confidence_interval'):
        lower, upper = prediction['confidence_interval']
        interval_width = upper - lower
        if 0.001 <= interval_width <= 0.02:  # Reasonable interval width
            quality_metrics['reasonable_interval'] = True
        else:
            quality_metrics['reasonable_interval'] = False
    
    # Check reasoning depth
    content = messages[-1]["content"]
    reasoning_length = len(content)
    if reasoning_length > 500:
        quality_metrics['detailed_reasoning'] = True
    else:
        quality_metrics['detailed_reasoning'] = False
    
    return quality_metrics


if __name__ == "__main__":
    print("Single-Quarter NCO Prediction Environment")
    print("-" * 50)
    print("Key Features:")
    print("- One-step ahead prediction (next quarter only)")
    print("- 12 quarters of historical context")
    print("- Rich temporal feature extraction")
    print("- Dynamic weight adjustment by regime")
    print("- Directional accuracy emphasis")
    print("- Confidence interval calibration")
    print("- Temporal consistency checking")
    print("\nReady for GRPO training!")