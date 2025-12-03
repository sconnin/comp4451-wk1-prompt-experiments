"""
Report generator for experiment results.
"""
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate

from database import Database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports and comparisons from experiment results."""
    
    def __init__(self):
        """Initialize report generator."""
        self.db = Database()
    
    def list_experiments(self):
        """List all experiments."""
        experiments = self.db.get_all_experiments()
        
        if not experiments:
            print("\nNo experiments found in database.")
            return
        
        print("\n" + "=" * 80)
        print("Available Experiments")
        print("=" * 80)
        
        data = []
        for exp in experiments:
            data.append([
                exp['id'],
                exp['name'],
                exp['created_at']
            ])
        
        print(tabulate(data, headers=['ID', 'Name', 'Created At'], tablefmt='grid'))
        print()
    
    def show_experiment_results(self, experiment_id: int, verbose: bool = False):
        """Display results for a specific experiment."""
        results = self.db.get_experiment_results(experiment_id)
        
        if not results:
            print(f"\nNo results found for experiment ID: {experiment_id}")
            return
        
        print("\n" + "=" * 80)
        print(f"Experiment Results: {results[0]['experiment_name']}")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Template: {result['template_type']}")
            print(f"Model: {result['model']}")
            print(f"Tokens: {result['tokens_used']}")
            print(f"Response Time: {result['response_time']:.2f}s")
            
            print("\nScores:")
            scores_data = [
                ['Relevance', f"{result['relevance_score']:.2f}" if result['relevance_score'] else 'N/A'],
                ['Accuracy', f"{result['accuracy_score']:.2f}" if result['accuracy_score'] else 'N/A'],
                ['Completeness', f"{result['completeness_score']:.2f}" if result['completeness_score'] else 'N/A'],
                ['Consistency', f"{result['consistency_score']:.2f}" if result['consistency_score'] else 'N/A'],
                ['Efficiency', f"{result['efficiency_score']:.2f}" if result['efficiency_score'] else 'N/A'],
                ['Bias', f"{result['bias_score']:.2f}" if result['bias_score'] else 'N/A']
            ]
            print(tabulate(scores_data, tablefmt='simple'))
            
            if verbose:
                print(f"\nPrompt:\n{result['prompt_text'][:200]}...")
                print(f"\nResponse:\n{result['response_text'][:300]}...")
            
            print("-" * 80)
    
    def compare_templates(self):
        """Compare performance across different template types."""
        comparison = self.db.get_template_comparison()
        
        if not comparison:
            print("\nNo data available for template comparison.")
            return
        
        print("\n" + "=" * 80)
        print("Template Performance Comparison")
        print("=" * 80)
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(comparison)
        
        # Format numeric columns
        numeric_cols = ['avg_response_time', 'avg_tokens', 'avg_relevance', 'avg_accuracy',
                       'avg_completeness', 'avg_consistency', 'avg_efficiency', 'avg_bias']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        print("\nOverall Metrics:")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        
        if not df.empty:
            best_relevance = df.loc[df['avg_relevance'].idxmax(), 'template_type']
            best_efficiency = df.loc[df['avg_efficiency'].idxmax(), 'template_type']
            fastest = df.loc[df['avg_response_time'].idxmin(), 'template_type']
            
            print(f"Best Relevance: {best_relevance}")
            print(f"Best Efficiency: {best_efficiency}")
            print(f"Fastest Response: {fastest}")
    
    def export_to_csv(self, experiment_id: int, output_path: str):
        """Export experiment results to CSV."""
        results = self.db.get_experiment_results(experiment_id)
        
        if not results:
            print(f"\nNo results found for experiment ID: {experiment_id}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Results exported to: {output_file}")
        print(f"\n✓ Results exported to: {output_file}")
    
    def export_comparison_to_csv(self, output_path: str):
        """Export template comparison to CSV."""
        comparison = self.db.get_template_comparison()
        
        if not comparison:
            print("\nNo data available for template comparison.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(comparison)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"Comparison exported to: {output_file}")
        print(f"\n✓ Comparison exported to: {output_file}")
    
    def cleanup(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate reports from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python report_generator.py --list
  
  # Show results for experiment 1
  python report_generator.py --experiment 1
  
  # Show detailed results
  python report_generator.py --experiment 1 --verbose
  
  # Compare all templates
  python report_generator.py --compare
  
  # Export experiment results to CSV
  python report_generator.py --experiment 1 --export results/exp1.csv
  
  # Export comparison to CSV
  python report_generator.py --compare --export results/comparison.csv
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all experiments'
    )
    
    parser.add_argument(
        '--experiment',
        type=int,
        help='Show results for specific experiment ID'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare performance across template types'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output including prompts and responses'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    
    args = parser.parse_args()
    
    # Require at least one action
    if not any([args.list, args.experiment, args.compare]):
        parser.print_help()
        sys.exit(1)
    
    generator = ReportGenerator()
    
    try:
        if args.list:
            generator.list_experiments()
        
        if args.experiment:
            generator.show_experiment_results(args.experiment, args.verbose)
            
            if args.export:
                generator.export_to_csv(args.experiment, args.export)
        
        if args.compare:
            generator.compare_templates()
            
            if args.export:
                generator.export_comparison_to_csv(args.export)
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)
    finally:
        generator.cleanup()


if __name__ == '__main__':
    main()
