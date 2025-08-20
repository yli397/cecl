"""
CECL NCO Prediction Environment with GRPO Reward Function
Multi-component reward function for optimizing NCO predictions
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


def parse_prediction(content: str) -> Optional[Dict]:
    """Parse NCO prediction from LLM output"""
    match = re.search(
        r"<prediction>(.*?)</prediction>", 
        content, 
        re.DOTALL
    )
    
    if not match:
        return None
    
    prediction_text = match.group(1)
    
    # Parse NCO rates for each quarter
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
    
    # Parse confidence interval - accept both [] and () formats
    conf_match = re.search(
        r"Confidence:\s*[\[\(]([\d.]+),\s*([\d.]+)[\]\)]",
        prediction_text
    )
    confidence = None
    if conf_match:
        confidence = [float(conf_match.group(1)), float(conf_match.group(2))]
    
    # Parse key drivers
    drivers_match = re.search(
        r"Key_Drivers:\s*\[(.*?)\]",
        prediction_text
    )
    key_drivers = []
    if drivers_match:
        drivers_text = drivers_match.group(1)
        key_drivers = [d.strip() for d in drivers_text.split(',')]
    
    return {
        'nco_rates': nco_rates,
        'confidence': confidence,
        'key_drivers': key_drivers
    }


def calculate_mse(pred: List[float], actual: List[float]) -> float:
    """Calculate Mean Squared Error"""
    pred = np.array(pred)
    actual = np.array(actual)
    return np.mean((pred - actual) ** 2)


def calculate_coverage_ratio(pred: List[float], actual: List[float]) -> float:
    """
    Calculate coverage ratio (predicted NCO / actual NCO)
    Ideal ratio is 1.0
    """
    pred_sum = sum(pred)
    actual_sum = sum(actual)
    if actual_sum == 0:
        return 1.0
    return pred_sum / actual_sum


def assess_reasoning_quality(messages: List[Dict], prediction: Dict) -> float:
    """
    Assess the quality of reasoning in the response
    Returns score between 0 and 1
    """
    content = messages[-1]["content"]
    score = 0.0
    max_score = 7.0
    
    # Check for key reasoning components
    reasoning_checks = [
        ("economic condition", r"(unemployment|inflation|GDP|recession|growth)"),
        ("historical comparison", r"(2008|crisis|pandemic|COVID|historical|past)"),
        ("portfolio analysis", r"(FICO|utilization|portfolio|segment|risk profile)"),
        ("scenario weighting", r"(baseline|adverse|scenario|probability|weight)"),
        ("confidence assessment", r"(confidence|uncertain|likely|probability)"),
        ("key drivers identified", len(prediction.get('key_drivers', [])) >= 2),
        ("quantitative analysis", r"(\d+\.?\d*%|\d+bp|basis points)")
    ]
    
    for check_name, check in reasoning_checks:
        if isinstance(check, bool):
            if check:
                score += 1.0
        elif re.search(check, content, re.IGNORECASE):
            score += 1.0
    
    return score / max_score


def calculate_under_provision_penalty(pred: List[float], actual: List[float]) -> float:
    """
    Penalize under-provisioning more heavily than over-provisioning
    This aligns with regulatory preferences for conservative estimates
    """
    penalty = 0.0
    for p, a in zip(pred, actual):
        if p < a:  # Under-provisioned
            penalty += (a - p) ** 2 * 2.5  # Higher penalty
        else:  # Over-provisioned
            penalty += (p - a) ** 2 * 0.5  # Lower penalty
    return penalty / len(pred)


def assess_variable_usage(messages: List[Dict]) -> float:
    """
    Score based on how many of the 17+ variables are considered
    Returns score between 0 and 1
    """
    content = messages[-1]["content"]
    
    # List of key variables to check
    variables = [
        "unemployment", "UNRATE",
        "housing", "HPI", "house price",
        "consumer confidence", "CCI", "sentiment",
        "inflation", "CPI",
        "bankruptcy",
        "debt service", "debt ratio",
        "capital ratio",
        "NPL", "non-performing",
        "ROA", "return on assets",
        "diversification",
        "FICO",
        "utilization",
        "uncertainty",
        "state", "regional",
        "size", "assets"
    ]
    
    used_count = 0
    for var in variables:
        if re.search(var, content, re.IGNORECASE):
            used_count += 1
    
    # Normalize to 0-1
    return min(used_count / 15.0, 1.0)


def reward_fn(messages: List[Dict], answer) -> float:
    """
    Multi-component reward function for CECL NCO prediction
    
    Components:
    - MSE accuracy (α₁ = 1.0)
    - Coverage ratio accuracy (α₂ = 0.3)
    - Reasoning quality (α₃ = 0.2)
    - Under-provision penalty (α₄ = 2.5)
    - Variable usage score (α₅ = 0.1)
    """
    
    # Parse prediction from LLM response
    prediction = parse_prediction(messages[-1]["content"])
    
    if prediction is None:
        return -5.0  # Reduced penalty for invalid format
    
    # Handle answer format - could be dict or string
    if isinstance(answer, str):
        # If answer is a JSON string, parse it
        try:
            answer = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, return minimal reward
            return -3.0
    elif not isinstance(answer, dict):
        # If answer is neither string nor dict, something is wrong
        return -3.0
    
    # Get actual NCO rates
    actual_nco = answer.get("nco_rates", [])
    pred_nco = prediction["nco_rates"]
    
    if len(pred_nco) != len(actual_nco):
        return -5.0  # Reduced penalty for wrong number of predictions
    
    # Calculate reward components
    mse = calculate_mse(pred_nco, actual_nco)
    coverage_error = abs(calculate_coverage_ratio(pred_nco, actual_nco) - 1.0)
    reasoning_score = assess_reasoning_quality(messages, prediction)
    under_provision = calculate_under_provision_penalty(pred_nco, actual_nco)
    variable_score = assess_variable_usage(messages)
    
    # Combine with balanced weights
    # Reduced penalties, increased positive rewards
    α1, α2, α3, α4, α5 = 0.5, 0.2, 0.8, 0.5, 0.3
    
    reward = (
        -α1 * mse
        -α2 * coverage_error
        +α3 * reasoning_score
        -α4 * under_provision
        +α5 * variable_score
    )
    
    # Bonus for crisis detection (if applicable)
    if answer.get("period") == "Financial Crisis":
        # Check if model predicted elevated NCO
        avg_pred = np.mean(pred_nco)
        avg_actual = np.mean(actual_nco)
        if avg_pred > 2.0 and avg_actual > 2.0:  # Both elevated
            reward += 0.5  # Bonus for crisis detection
    
    return reward


def interact(messages: List[Dict]) -> List[Dict]:
    """
    Multi-turn interaction for additional data requests
    This could query external APIs or databases
    """
    content = messages[-1]["content"]
    
    # Check if model requests additional data
    if "<request_data>" in content:
        match = re.search(
            r"<request_data>(.*?)</request_data>",
            content
        )
        if match:
            data_request = match.group(1)
            
            # Simulate fetching additional data
            # In production, this would query actual data sources
            additional_data = fetch_additional_data(data_request)
            
            return [
                {"role": "tool", "content": json.dumps(additional_data)}
            ]
    
    return []


def fetch_additional_data(request: str) -> Dict:
    """
    Fetch additional data based on request
    In production, this would connect to FRED API, etc.
    """
    # Simulated response
    if "historical" in request.lower():
        return {
            "historical_nco": {
                "2008Q1": 2.45,
                "2008Q2": 3.12,
                "2008Q3": 4.23,
                "2008Q4": 5.67
            },
            "context": "Financial crisis period"
        }
    elif "peer" in request.lower():
        return {
            "peer_comparison": {
                "bank_a": 1.23,
                "bank_b": 1.45,
                "bank_c": 1.67,
                "industry_avg": 1.55
            }
        }
    else:
        return {"data": "No additional data available"}


def evaluate_test_performance(predictions: List[Dict], 
                             actuals: List[Dict]) -> Dict:
    """
    Evaluate model performance on test set
    """
    all_mse = []
    all_coverage = []
    crisis_detection = []
    
    for pred, actual in zip(predictions, actuals):
        pred_rates = pred["nco_rates"]
        actual_rates = actual["nco_rates"]
        
        mse = calculate_mse(pred_rates, actual_rates)
        coverage = calculate_coverage_ratio(pred_rates, actual_rates)
        
        all_mse.append(mse)
        all_coverage.append(coverage)
        
        # Crisis detection
        if actual.get("period") in ["Financial Crisis", "COVID Pandemic"]:
            avg_pred = np.mean(pred_rates)
            avg_actual = np.mean(actual_rates)
            detected = (avg_pred > 2.0 and avg_actual > 2.0)
            crisis_detection.append(detected)
    
    return {
        "rmse": np.sqrt(np.mean(all_mse)),
        "coverage_ratio_mean": np.mean(all_coverage),
        "coverage_ratio_std": np.std(all_coverage),
        "crisis_detection_rate": np.mean(crisis_detection) if crisis_detection else None
    }


if __name__ == "__main__":
    # Test the reward function
    test_messages = [
        {"role": "user", "content": "[ECONOMIC CONTEXT]..."},
        {"role": "assistant", "content": """
        Analyzing the economic indicators, I observe elevated unemployment at 4.5% 
        with an upward trend. The housing market shows weakness with HPI declining 
        year-over-year. Consumer confidence has dropped below historical averages.
        
        Comparing to the 2008 financial crisis, current conditions show similar 
        early warning signs but with lower severity. The bank's portfolio has 
        35% subprime exposure (FICO <620) which increases vulnerability.
        
        Weighting scenarios:
        - Baseline (50%): Gradual economic recovery, NCO peaks at 2.1%
        - Adverse (35%): Recession materializes, NCO reaches 3.5%
        - Severely Adverse (15%): Crisis conditions, NCO could hit 5.0%
        
        <prediction>
        NCO_Q1: 1.85%
        NCO_Q2: 2.10%
        NCO_Q3: 2.25%
        NCO_Q4: 2.05%
        Confidence: [1.5, 2.8]
        Key_Drivers: [unemployment, housing_decline, consumer_confidence]
        </prediction>
        """}
    ]
    
    test_answer = {
        "nco_rates": [1.95, 2.20, 2.35, 2.15],
        "period": "Normal Market"
    }
    
    reward = reward_fn(test_messages, test_answer)
    print(f"Test reward: {reward:.4f}")