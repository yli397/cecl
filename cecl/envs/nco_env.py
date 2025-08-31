"""
CECL NCO Prediction Environment for cecl Training
Integrates with cecl framework for reinforcement learning training
"""

import re
import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from cecl.envs.base import BaseEnv, BaseState


@dataclass(frozen=True)
class NCOState(BaseState):
    tokens: list
    bank_id: int
    quarter: str
    context_data: dict


class NCOEnv(BaseEnv):
    def __init__(
        self,
        tokenizer,
        data_path="data/Bank-level Data (Include CC NCO Rate)/credit_card_nco_panel_cleaned.csv",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 1024  # Longer responses for complex analysis

        # Load NCO data
        self.nco_data = pd.read_csv(data_path)
        self.prepare_data()

        # Load economic indicators
        self.load_economic_data()

    def prepare_data(self):
        """Prepare training examples from NCO data"""
        # Filter for banks with sufficient data
        bank_counts = self.nco_data["IDRSSD"].value_counts()
        valid_banks = bank_counts[bank_counts >= 20].index
        self.nco_data = self.nco_data[self.nco_data["IDRSSD"].isin(valid_banks)]

        # Create examples
        self.examples = []
        for bank_id in valid_banks:
            bank_data = self.nco_data[self.nco_data["IDRSSD"] == bank_id].sort_values(
                "report_date"
            )

            # Create sliding windows of 4 quarters for prediction
            for i in range(len(bank_data) - 4):
                historical = bank_data.iloc[i : i + 4]
                target = (
                    bank_data.iloc[i + 4 : i + 8] if i + 8 <= len(bank_data) else None
                )

                if target is not None and len(target) == 4:
                    example = {
                        "bank_id": bank_id,
                        "historical_quarters": historical,
                        "target_quarters": target,
                        "period_type": self.classify_period(
                            target["report_date"].iloc[0]
                        ),
                    }
                    self.examples.append(example)

        print(f"Prepared {len(self.examples)} NCO prediction examples")

    def classify_period(self, date_str):
        """Classify the economic period"""
        year = pd.to_datetime(date_str).year
        if 2007 <= year <= 2009:
            return "Financial Crisis"
        elif 2020 <= year <= 2021:
            return "COVID Pandemic"
        elif year <= 2006 or 2010 <= year <= 2019:
            return "Normal Market"
        else:
            return "Post-Pandemic"

    def load_economic_data(self):
        """Load economic indicators"""
        try:
            # Load unemployment data
            self.unemployment_data = pd.read_csv(
                "data/Unemployement/unemployment_rates_US_monthly.csv"
            )

            # Load consumer confidence
            self.confidence_data = pd.read_csv(
                "data/Consumer Confidence Index/consumer_confidence_index_united_states_monthly.csv"
            )

            # Load housing price index
            self.housing_data = pd.read_csv(
                "data/Housing Price Index/housing_price_index_quarterly.csv"
            )

            print("Loaded economic indicator data")
        except Exception as e:
            print(f"Warning: Could not load economic data: {e}")
            self.unemployment_data = None
            self.confidence_data = None
            self.housing_data = None

    def get_economic_context(self, quarter_date):
        """Get economic context for a specific quarter"""
        context = {}

        if self.unemployment_data is not None:
            # Find closest unemployment rate
            try:
                date_obj = pd.to_datetime(quarter_date)
                # Simplified - use a representative rate
                context["unemployment_rate"] = 5.5  # Default value
            except:
                context["unemployment_rate"] = 5.5

        # Add other economic indicators (simplified for demo)
        context.update(
            {
                "consumer_confidence": 95.0,
                "housing_price_index": 100.0,
                "inflation_rate": 2.5,
                "gdp_growth": 2.1,
            }
        )

        return context

    def format_prompt(self, example):
        """Format the prediction prompt"""
        bank_id = example["bank_id"]
        historical = example["historical_quarters"]

        # Get latest quarter economic context
        latest_date = historical["report_date"].iloc[-1]
        econ_context = self.get_economic_context(latest_date)

        # Format historical data
        historical_text = ""
        for _, row in historical.iterrows():
            historical_text += f"Q{row['YQ']}: NCO Rate: {row['NCO_RATE_Q']:.3f}%, "
            historical_text += f"Bank Size: {row['BANK_SIZE']:.2f}, "
            historical_text += f"Capital Ratio: {row['CAPITAL_TO_ASSET']:.3f}, "
            historical_text += f"Loan-to-Asset: {row['LOAN_TO_ASSET']:.3f}\\n"

        prompt = f"""You are a senior credit risk analyst tasked with predicting Net Charge-Off (NCO) rates for Bank ID {bank_id}.

HISTORICAL PERFORMANCE (Last 4 Quarters):
{historical_text}

CURRENT ECONOMIC CONTEXT:
- Unemployment Rate: {econ_context["unemployment_rate"]:.1f}%
- Consumer Confidence Index: {econ_context["consumer_confidence"]:.1f}
- Housing Price Index: {econ_context["housing_price_index"]:.1f}
- Inflation Rate: {econ_context["inflation_rate"]:.1f}%
- GDP Growth: {econ_context["gdp_growth"]:.1f}%

BANK CHARACTERISTICS:
- Portfolio Diversification: {historical["DIVERSIFICATION"].iloc[-1]:.3f}
- Credit Card Loan Fraction: {historical["CC_LOAN_FRACTION"].iloc[-1]:.3f}
- Allowance for Loan Losses: {historical["ALLL_TO_LOAN"].iloc[-1]:.3f}

TASK:
Predict the quarterly NCO rates for the next 4 quarters. Consider:
1. Historical trends and patterns
2. Current economic conditions and outlook
3. Bank-specific risk factors
4. Industry trends and peer comparisons
5. Regulatory environment and stress testing scenarios

Provide your analysis and prediction in the following format:

<prediction>
NCO_Q1: [rate]%
NCO_Q2: [rate]%
NCO_Q3: [rate]%
NCO_Q4: [rate]%
Confidence: [lower_bound, upper_bound]
Key_Drivers: [driver1, driver2, driver3]
</prediction>

Begin your analysis:"""

        return prompt

    def reset(self, idx):
        """Reset environment with a specific example"""
        example = self.examples[idx % len(self.examples)]
        prompt = self.format_prompt(example)

        # Create chat template
        output_tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Store example data for reward calculation
        target_data = {
            "nco_rates": example["target_quarters"]["NCO_RATE_Q"].tolist(),
            "period": example["period_type"],
            "bank_id": example["bank_id"],
        }

        state = NCOState(
            tokens=output_tokens,
            bank_id=example["bank_id"],
            quarter=example["target_quarters"]["YQ"].iloc[0],
            context_data=target_data,
        )

        return state, output_tokens

    def render(self, state):
        """Render the current state"""
        return self.tokenizer.decode(state.tokens)

    def step(self, state, action_tokens):
        """Process model response and calculate reward"""
        # Clean action tokens (remove after EOS)
        action_tokens = self.clean_action(
            action_tokens, self.tokenizer.get_eos_token_id()
        )

        # Decode the full response
        full_tokens = state.tokens + action_tokens
        full_text = self.tokenizer.decode(full_tokens)

        # Extract the assistant's response
        assistant_content = full_text.split("<|im_start|>assistant")[-1].replace(
            "<|im_end|>", ""
        )
        messages = [{"role": "assistant", "content": assistant_content}]

        # Calculate reward using the imported reward function
        reward = reward_fn(messages, state.context_data)

        # Create new state with action tokens
        from dataclasses import replace

        new_state = replace(state, tokens=state.tokens + action_tokens)
        done = True  # Single-turn environment

        return new_state, [], reward, done, {}


# Import reward functions from nco.py
def parse_prediction(content: str) -> Optional[Dict]:
    """Parse NCO prediction from LLM output"""
    match = re.search(r"<prediction>(.*?)</prediction>", content, re.DOTALL)

    if not match:
        return None

    prediction_text = match.group(1)

    # Parse NCO rates for each quarter
    nco_rates = []
    for q in range(1, 5):
        q_match = re.search(rf"NCO_Q{q}:\\s*([\\d.]+)%?", prediction_text)
        if q_match:
            nco_rates.append(float(q_match.group(1)))
        else:
            return None

    # Parse confidence interval
    conf_match = re.search(
        r"Confidence:\\s*[\\[\\(]([\\d.]+),\\s*([\\d.]+)[\\]\\)]", prediction_text
    )
    confidence = None
    if conf_match:
        confidence = [float(conf_match.group(1)), float(conf_match.group(2))]

    # Parse key drivers
    drivers_match = re.search(r"Key_Drivers:\\s*\\[(.*?)\\]", prediction_text)
    key_drivers = []
    if drivers_match:
        drivers_text = drivers_match.group(1)
        key_drivers = [d.strip() for d in drivers_text.split(",")]

    return {
        "nco_rates": nco_rates,
        "confidence": confidence,
        "key_drivers": key_drivers,
    }


def calculate_mse(pred: List[float], actual: List[float]) -> float:
    """Calculate Mean Squared Error"""
    pred = np.array(pred)
    actual = np.array(actual)
    return np.mean((pred - actual) ** 2)


def assess_reasoning_quality(messages: List[Dict], prediction: Dict) -> float:
    """Assess the quality of reasoning in the response"""
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
        ("key drivers identified", len(prediction.get("key_drivers", [])) >= 2),
        ("quantitative analysis", r"(\\d+\\.?\\d*%|\\d+bp|basis points)"),
    ]

    for check_name, check in reasoning_checks:
        if isinstance(check, bool):
            if check:
                score += 1.0
        elif re.search(check, content, re.IGNORECASE):
            score += 1.0

    return score / max_score


def reward_fn(messages: List[Dict], answer) -> float:
    """Multi-component reward function for CECL NCO prediction"""

    # Parse prediction from LLM response
    prediction = parse_prediction(messages[-1]["content"])

    if prediction is None:
        return -2.0  # Penalty for invalid format

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
    mse = calculate_mse(pred_nco, actual_nco)
    reasoning_score = assess_reasoning_quality(messages, prediction)

    # Simplified reward function
    # Higher reward for lower MSE and better reasoning
    mse_reward = max(0, 2.0 - mse)  # Scale MSE penalty
    reasoning_reward = reasoning_score * 1.0

    # Bonus for having key drivers
    driver_bonus = 0.2 if len(prediction.get("key_drivers", [])) >= 2 else 0

    total_reward = mse_reward + reasoning_reward + driver_bonus

    return total_reward


if __name__ == "__main__":
    # Test the environment
    from cecl.models.tokenizer import Tokenizer

    # This would need to be tested with actual tokenizer
    print("NCO Environment created successfully")
    print("Ready for integration with cecl training")
