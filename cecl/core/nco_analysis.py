"""
NCO Prediction Analysis and Monitoring Tools
For evaluating GRPO training progress and model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
import json
import os


class NCOAnalyzer:
    """Comprehensive analysis tools for NCO predictions"""
    
    def __init__(self, predictions_path: str = None):
        self.predictions = []
        self.actuals = []
        self.rewards = []
        self.regimes = []
        self.temporal_scores = []
        
        if predictions_path and os.path.exists(predictions_path):
            self.load_predictions(predictions_path)
    
    def load_predictions(self, path: str):
        """Load predictions from training logs"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.predictions = data.get('predictions', [])
            self.actuals = data.get('actuals', [])
            self.rewards = data.get('rewards', [])
            self.regimes = data.get('regimes', [])
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.predictions or not self.actuals:
            return {}
        
        metrics = {}
        
        # Convert to arrays
        pred_array = np.array(self.predictions)
        actual_array = np.array(self.actuals)
        
        # Basic metrics
        metrics['rmse'] = np.sqrt(np.mean((pred_array - actual_array) ** 2))
        metrics['mae'] = np.mean(np.abs(pred_array - actual_array))
        metrics['mape'] = np.mean(np.abs((actual_array - pred_array) / (actual_array + 1e-6))) * 100
        
        # Directional accuracy
        if len(pred_array.shape) == 2 and pred_array.shape[1] > 1:
            direction_correct = 0
            total_directions = 0
            for i in range(pred_array.shape[0]):
                for j in range(1, pred_array.shape[1]):
                    pred_dir = np.sign(pred_array[i, j] - pred_array[i, j-1])
                    actual_dir = np.sign(actual_array[i, j] - actual_array[i, j-1])
                    if pred_dir == actual_dir:
                        direction_correct += 1
                    total_directions += 1
            metrics['directional_accuracy'] = direction_correct / total_directions if total_directions > 0 else 0
        
        # Coverage ratio statistics
        coverage_ratios = []
        for pred, actual in zip(self.predictions, self.actuals):
            if sum(actual) > 0:
                coverage_ratios.append(sum(pred) / sum(actual))
        
        if coverage_ratios:
            metrics['coverage_mean'] = np.mean(coverage_ratios)
            metrics['coverage_std'] = np.std(coverage_ratios)
            metrics['coverage_median'] = np.median(coverage_ratios)
        
        # Regime-specific metrics
        if self.regimes:
            regime_metrics = {}
            unique_regimes = list(set(self.regimes))
            for regime in unique_regimes:
                regime_idx = [i for i, r in enumerate(self.regimes) if r == regime]
                if regime_idx:
                    regime_pred = pred_array[regime_idx] if len(pred_array.shape) == 2 else pred_array[regime_idx]
                    regime_actual = actual_array[regime_idx] if len(actual_array.shape) == 2 else actual_array[regime_idx]
                    regime_metrics[regime] = {
                        'rmse': np.sqrt(np.mean((regime_pred - regime_actual) ** 2)),
                        'count': len(regime_idx)
                    }
            metrics['regime_metrics'] = regime_metrics
        
        # Crisis detection metrics
        crisis_periods = ['Financial Crisis', 'COVID Pandemic', 'Inflation Surge']
        if self.regimes:
            crisis_idx = [i for i, r in enumerate(self.regimes) if r in crisis_periods]
            if crisis_idx:
                crisis_pred = pred_array[crisis_idx] if len(pred_array.shape) == 2 else pred_array[crisis_idx]
                crisis_actual = actual_array[crisis_idx] if len(actual_array.shape) == 2 else actual_array[crisis_idx]
                
                # Check if both predicted and actual are elevated
                crisis_detected = 0
                for p, a in zip(crisis_pred, crisis_actual):
                    if np.mean(p) > 0.02 and np.mean(a) > 0.02:
                        crisis_detected += 1
                
                metrics['crisis_detection_rate'] = crisis_detected / len(crisis_idx) if crisis_idx else 0
        
        # Temporal consistency
        if self.temporal_scores:
            metrics['temporal_consistency_mean'] = np.mean(self.temporal_scores)
            metrics['temporal_consistency_std'] = np.std(self.temporal_scores)
        
        # Reward progression
        if self.rewards:
            metrics['reward_mean'] = np.mean(self.rewards)
            metrics['reward_std'] = np.std(self.rewards)
            metrics['reward_trend'] = np.polyfit(range(len(self.rewards)), self.rewards, 1)[0] if len(self.rewards) > 1 else 0
        
        return metrics
    
    def plot_performance_dashboard(self, save_path: str = None):
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('NCO Prediction Performance Dashboard', fontsize=16)
        
        # 1. Predicted vs Actual scatter
        ax = axes[0, 0]
        if self.predictions and self.actuals:
            flat_pred = np.array(self.predictions).flatten()
            flat_actual = np.array(self.actuals).flatten()
            ax.scatter(flat_actual, flat_pred, alpha=0.5)
            ax.plot([0, max(flat_actual)], [0, max(flat_actual)], 'r--', label='Perfect prediction')
            ax.set_xlabel('Actual NCO Rate')
            ax.set_ylabel('Predicted NCO Rate')
            ax.set_title('Predicted vs Actual')
            ax.legend()
        
        # 2. Residuals distribution
        ax = axes[0, 1]
        if self.predictions and self.actuals:
            residuals = flat_pred - flat_actual
            ax.hist(residuals, bins=30, edgecolor='black')
            ax.axvline(0, color='r', linestyle='--')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Residuals Distribution (μ={np.mean(residuals):.4f})')
        
        # 3. Reward progression
        ax = axes[0, 2]
        if self.rewards:
            ax.plot(self.rewards, alpha=0.7)
            z = np.polyfit(range(len(self.rewards)), self.rewards, 1)
            p = np.poly1d(z)
            ax.plot(range(len(self.rewards)), p(range(len(self.rewards))), "r--", label=f'Trend: {z[0]:.4f}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Progression')
            ax.legend()
        
        # 4. Coverage ratio distribution
        ax = axes[1, 0]
        if self.predictions and self.actuals:
            coverage_ratios = []
            for pred, actual in zip(self.predictions, self.actuals):
                if sum(actual) > 0:
                    coverage_ratios.append(sum(pred) / sum(actual))
            if coverage_ratios:
                ax.hist(coverage_ratios, bins=20, edgecolor='black')
                ax.axvline(1.0, color='r', linestyle='--', label='Ideal (1.0)')
                ax.set_xlabel('Coverage Ratio')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Coverage Ratio (μ={np.mean(coverage_ratios):.3f})')
                ax.legend()
        
        # 5. Regime-specific performance
        ax = axes[1, 1]
        if hasattr(self, 'regime_metrics'):
            metrics = self.calculate_metrics()
            if 'regime_metrics' in metrics:
                regimes = list(metrics['regime_metrics'].keys())
                rmses = [metrics['regime_metrics'][r]['rmse'] for r in regimes]
                ax.bar(regimes, rmses)
                ax.set_xlabel('Market Regime')
                ax.set_ylabel('RMSE')
                ax.set_title('Performance by Regime')
                ax.tick_params(axis='x', rotation=45)
        
        # 6. Temporal consistency scores
        ax = axes[1, 2]
        if self.temporal_scores:
            ax.plot(self.temporal_scores, alpha=0.7)
            ax.axhline(np.mean(self.temporal_scores), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(self.temporal_scores):.3f}')
            ax.set_xlabel('Prediction Instance')
            ax.set_ylabel('Temporal Consistency Score')
            ax.set_title('Temporal Consistency')
            ax.legend()
        
        # 7. Q-Q plot for residuals
        ax = axes[2, 0]
        if self.predictions and self.actuals:
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot (Normality Check)')
        
        # 8. Autocorrelation of residuals
        ax = axes[2, 1]
        if self.predictions and self.actuals and len(residuals) > 10:
            try:
                autocorr = acf(residuals, nlags=min(10, len(residuals)-1))
                ax.bar(range(len(autocorr)), autocorr)
                ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Lag')
                ax.set_ylabel('ACF')
                ax.set_title('Residuals Autocorrelation')
            except:
                pass
        
        # 9. Performance metrics summary
        ax = axes[2, 2]
        metrics = self.calculate_metrics()
        if metrics:
            text = "Performance Summary:\n\n"
            text += f"RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
            text += f"MAE: {metrics.get('mae', 'N/A'):.4f}\n"
            text += f"MAPE: {metrics.get('mape', 'N/A'):.2f}%\n"
            text += f"Dir. Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2%}\n"
            text += f"Coverage μ±σ: {metrics.get('coverage_mean', 'N/A'):.3f}±{metrics.get('coverage_std', 'N/A'):.3f}\n"
            text += f"Crisis Detection: {metrics.get('crisis_detection_rate', 'N/A'):.2%}\n"
            text += f"Reward Trend: {metrics.get('reward_trend', 'N/A'):.4f}"
            ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='center', fontfamily='monospace')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def compare_with_baseline(self, baseline_predictions: List) -> Dict:
        """Compare GRPO predictions with baseline model"""
        if not self.predictions or not self.actuals or not baseline_predictions:
            return {}
        
        comparison = {}
        
        # Calculate metrics for both
        grpo_pred = np.array(self.predictions)
        baseline_pred = np.array(baseline_predictions)
        actual = np.array(self.actuals)
        
        # RMSE comparison
        grpo_rmse = np.sqrt(np.mean((grpo_pred - actual) ** 2))
        baseline_rmse = np.sqrt(np.mean((baseline_pred - actual) ** 2))
        comparison['rmse_improvement'] = (baseline_rmse - grpo_rmse) / baseline_rmse * 100
        
        # Diebold-Mariano test for statistical significance
        try:
            grpo_errors = grpo_pred - actual
            baseline_errors = baseline_pred - actual
            diff = grpo_errors**2 - baseline_errors**2
            dm_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)))
            comparison['dm_statistic'] = dm_stat
            comparison['dm_pvalue'] = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        except:
            comparison['dm_statistic'] = None
            comparison['dm_pvalue'] = None
        
        return comparison
    
    def export_results(self, filepath: str):
        """Export analysis results to file"""
        results = {
            'metrics': self.calculate_metrics(),
            'predictions': self.predictions,
            'actuals': self.actuals,
            'rewards': self.rewards,
            'regimes': self.regimes,
            'temporal_scores': self.temporal_scores
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results = convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {filepath}")


def monitor_training_progress(log_file: str, interval: int = 60):
    """Monitor live training progress"""
    import time
    
    analyzer = NCOAnalyzer()
    
    while True:
        try:
            # Read latest predictions from log
            if os.path.exists(log_file):
                analyzer.load_predictions(log_file)
                
                # Calculate and display metrics
                metrics = analyzer.calculate_metrics()
                
                print("\n" + "="*50)
                print(f"NCO Training Progress - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*50)
                
                if metrics:
                    print(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                    print(f"Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2%}")
                    print(f"Coverage Ratio: {metrics.get('coverage_mean', 'N/A'):.3f} ± {metrics.get('coverage_std', 'N/A'):.3f}")
                    print(f"Crisis Detection Rate: {metrics.get('crisis_detection_rate', 'N/A'):.2%}")
                    print(f"Temporal Consistency: {metrics.get('temporal_consistency_mean', 'N/A'):.3f}")
                    print(f"Reward Trend: {metrics.get('reward_trend', 'N/A'):.4f}")
                    
                    if 'regime_metrics' in metrics:
                        print("\nRegime-specific RMSE:")
                        for regime, regime_data in metrics['regime_metrics'].items():
                            print(f"  {regime}: {regime_data['rmse']:.4f} (n={regime_data['count']})")
                
                # Generate dashboard
                analyzer.plot_performance_dashboard(save_path='nco_training_dashboard.png')
                print("\nDashboard saved to nco_training_dashboard.png")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    # Example usage
    print("NCO Analysis Tools")
    print("-" * 50)
    print("Available functions:")
    print("1. NCOAnalyzer() - Main analysis class")
    print("2. monitor_training_progress(log_file) - Live monitoring")
    print("3. analyzer.plot_performance_dashboard() - Visualization")
    print("4. analyzer.compare_with_baseline() - Model comparison")
    print("5. analyzer.export_results() - Export results")
    
    # If predictions exist, analyze them
    if os.path.exists('nco_predictions.json'):
        analyzer = NCOAnalyzer('nco_predictions.json')
        metrics = analyzer.calculate_metrics()
        print("\nCurrent Performance Metrics:")
        for key, value in metrics.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")