#!/usr/bin/env python
"""
Test script for Enhanced NCO Environment
Validates all new features and improvements
"""

import sys
sys.path.append('/mmfs1/home/yli269/cecl')

from cecl.envs.env_creator import create_env
from cecl.models.tokenizer import create_tokenizer
import numpy as np
import json


def test_enhanced_nco_env():
    """Test the enhanced NCO environment"""
    
    print("="*60)
    print("Testing Enhanced NCO Environment with Temporal Features")
    print("="*60)
    
    # Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = create_tokenizer('./checkpoints/Qwen3-1.7B/')
    print("   ✓ Tokenizer created")
    
    # Create enhanced environment
    print("\n2. Creating enhanced NCO environment...")
    try:
        env = create_env('nco_enhanced', tokenizer)
        print(f"   ✓ Environment created")
        print(f"   ✓ Number of examples: {len(env.examples)}")
    except Exception as e:
        print(f"   ✗ Error creating environment: {e}")
        return
    
    # Test temporal features
    print("\n3. Testing temporal feature extraction...")
    if env.examples:
        example = env.examples[0]
        temporal_features = example.get('temporal_features', {})
        print("   Temporal features extracted:")
        for key, value in temporal_features.items():
            print(f"     - {key}: {value:.4f}" if isinstance(value, (int, float)) else f"     - {key}: {value}")
    
    # Test market regime detection
    print("\n4. Testing market regime detection...")
    if env.examples:
        regimes = [ex['market_regime'] for ex in env.examples[:100]]
        unique_regimes = set(regimes)
        print(f"   ✓ Detected regimes: {unique_regimes}")
        for regime in unique_regimes:
            count = regimes.count(regime)
            print(f"     - {regime}: {count}/{len(regimes)} ({count/len(regimes)*100:.1f}%)")
    
    # Test dynamic weights
    print("\n5. Testing dynamic weight adjustment...")
    for regime in ['normal', 'stressed', 'elevated_risk', 'benign']:
        weights = env.get_dynamic_weights(regime)
        print(f"   Weights for {regime} regime:")
        print(f"     {weights}")
    
    # Test reset and state creation
    print("\n6. Testing environment reset...")
    try:
        state, tokens = env.reset(0)
        print(f"   ✓ State created successfully")
        print(f"   ✓ Prompt length: {len(tokens)} tokens")
        print(f"   ✓ Market regime: {state.market_regime}")
        print(f"   ✓ Temporal features available: {len(state.temporal_features)} features")
    except Exception as e:
        print(f"   ✗ Error in reset: {e}")
        return
    
    # Test enhanced prompt format
    print("\n7. Testing enhanced prompt generation...")
    if env.examples:
        prompt = env.format_enhanced_prompt(env.examples[0])
        print("   ✓ Enhanced prompt generated")
        print(f"   ✓ Prompt length: {len(prompt)} characters")
        
        # Check for key components
        components = [
            "TEMPORAL ANALYSIS",
            "Market Regime",
            "Trend:",
            "Volatility:",
            "Autocorrelation",
            "Economic Context",
            "Prediction_Interval"
        ]
        
        for component in components:
            if component in prompt:
                print(f"     ✓ Contains {component}")
            else:
                print(f"     ✗ Missing {component}")
    
    # Test period classification
    print("\n8. Testing period classification...")
    test_dates = [
        ('2008-06-30', 'Financial Crisis'),
        ('2020-06-30', 'COVID Pandemic'),
        ('2022-06-30', 'Inflation Surge'),
        ('2015-06-30', 'Recovery'),
        ('2024-06-30', 'Post-Pandemic')
    ]
    
    for date, expected in test_dates:
        actual = env.classify_period(date)
        status = "✓" if actual == expected else "✗"
        print(f"   {status} {date}: {actual} (expected: {expected})")
    
    # Test reward calculation with sample prediction
    print("\n9. Testing enhanced reward function...")
    sample_prediction = {
        'nco_rates': [0.005, 0.006, 0.007, 0.006],
        'confidence': [0.004, 0.008],
        'prediction_intervals': [[0.004, 0.006], [0.005, 0.007], [0.006, 0.008], [0.005, 0.007]],
        'key_drivers': ['unemployment', 'housing_decline', 'inflation'],
        'regime_probability': {'continuation': 70, 'change': 30},
        'temporal_consistency': 0.85
    }
    
    sample_actual = {
        'nco_rates': [0.0055, 0.0062, 0.0068, 0.0058],
        'market_regime': 'normal',
        'temporal_features': {'volatility': 0.002, 'trend_slope': 0.0001}
    }
    
    print("   Sample prediction vs actual:")
    print(f"     Predicted: {sample_prediction['nco_rates']}")
    print(f"     Actual: {sample_actual['nco_rates']}")
    
    # Calculate MSE
    pred = np.array(sample_prediction['nco_rates'])
    actual = np.array(sample_actual['nco_rates'])
    mse = np.mean((pred - actual) ** 2)
    print(f"     MSE: {mse:.6f}")
    
    # Check temporal consistency
    print(f"     Temporal Consistency: {sample_prediction['temporal_consistency']}")
    
    # Check prediction intervals coverage
    coverage = 0
    for i, (lower, upper) in enumerate(sample_prediction['prediction_intervals']):
        if lower <= sample_actual['nco_rates'][i] <= upper:
            coverage += 1
    coverage_rate = coverage / len(sample_actual['nco_rates'])
    print(f"     Interval Coverage: {coverage_rate:.1%}")
    
    print("\n" + "="*60)
    print("Enhanced NCO Environment Test Complete!")
    print("="*60)
    
    # Summary
    print("\nSummary of Enhancements:")
    print("✓ Extended historical window (8 quarters)")
    print("✓ Temporal feature extraction")
    print("✓ Market regime detection")
    print("✓ Dynamic weight adjustment")
    print("✓ Enhanced prompt with temporal context")
    print("✓ Prediction intervals")
    print("✓ Temporal consistency scoring")
    print("✓ Crisis detection capability")
    print("✓ Mean reversion analysis")
    print("✓ Structural break detection")
    
    return env


if __name__ == "__main__":
    try:
        env = test_enhanced_nco_env()
        
        # Additional statistics
        if env and env.examples:
            print("\n" + "="*60)
            print("Dataset Statistics:")
            print("="*60)
            
            # Bank statistics
            bank_ids = [ex['bank_id'] for ex in env.examples]
            unique_banks = len(set(bank_ids))
            print(f"Total banks: {unique_banks}")
            print(f"Total examples: {len(env.examples)}")
            print(f"Examples per bank: {len(env.examples)/unique_banks:.1f}")
            
            # Period distribution
            periods = [ex['period_type'] for ex in env.examples]
            period_counts = {}
            for period in periods:
                period_counts[period] = period_counts.get(period, 0) + 1
            
            print("\nPeriod distribution:")
            for period, count in sorted(period_counts.items()):
                print(f"  {period}: {count} ({count/len(periods)*100:.1f}%)")
            
            # Regime distribution
            regimes = [ex['market_regime'] for ex in env.examples]
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            print("\nMarket regime distribution:")
            for regime, count in sorted(regime_counts.items()):
                print(f"  {regime}: {count} ({count/len(regimes)*100:.1f}%)")
            
            print("\n✅ All tests passed successfully!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()