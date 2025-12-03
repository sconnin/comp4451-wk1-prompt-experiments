"""
Main experiment runner for LLM prompt experiments.
"""
import argparse
import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime

from database import Database
from prompt_loader import PromptLoader
from llm_client import LLMClient
from evaluator import ResponseEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates experiment execution."""
    
    def __init__(self, config_path: str):
        """Initialize experiment runner with config file."""
        self.config = self._load_config(config_path)
        self.db = Database()
        self.prompt_loader = PromptLoader()
        self.llm_client = LLMClient(model=self.config.get('model', 'gpt-3.5-turbo'))
        self.evaluator = ResponseEvaluator()
        
        logger.info(f"Initialized experiment: {self.config.get('experiment_name', 'unnamed')}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load experiment configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def run(self):
        """Execute the experiment."""
        logger.info("=" * 60)
        logger.info(f"Starting experiment: {self.config.get('experiment_name')}")
        logger.info("=" * 60)
        
        # Create experiment record
        experiment_id = self.db.create_experiment(
            name=self.config.get('experiment_name', f'experiment_{datetime.now().isoformat()}'),
            config=self.config
        )
        
        # Process each prompt configuration
        prompts = self.config.get('prompts', [])
        
        for i, prompt_config in enumerate(prompts, 1):
            logger.info(f"\nProcessing prompt {i}/{len(prompts)}")
            self._process_prompt(experiment_id, prompt_config)
        
        logger.info("=" * 60)
        logger.info(f"Experiment completed: {self.config.get('experiment_name')}")
        logger.info(f"Results saved to database (Experiment ID: {experiment_id})")
        logger.info("=" * 60)
        
        return experiment_id
    
    def _process_prompt(self, experiment_id: int, prompt_config: dict):
        """Process a single prompt configuration."""
        template_type = prompt_config.get('template')
        variables = prompt_config.get('variables', {})
        
        # Render prompt from template
        prompt_text = self.prompt_loader.render_template(template_type, variables)
        
        if not prompt_text:
            logger.error(f"Failed to render template: {template_type}")
            return
        
        logger.info(f"Template: {template_type}")
        logger.debug(f"Prompt: {prompt_text[:100]}...")
        
        # Create prompt record
        prompt_id = self.db.create_prompt(
            experiment_id=experiment_id,
            template_type=template_type,
            prompt_text=prompt_text,
            variables=variables
        )
        
        # Generate response
        try:
            result = self.llm_client.generate_response(
                prompt=prompt_text,
                temperature=prompt_config.get('temperature'),
                max_tokens=prompt_config.get('max_tokens')
            )
            
            logger.info(f"Response generated: {result['tokens_used']} tokens, {result['response_time']:.2f}s")
            
            # Create response record
            response_id = self.db.create_response(
                prompt_id=prompt_id,
                response_text=result['response_text'],
                model=result['model'],
                tokens_used=result['tokens_used'],
                response_time=result['response_time']
            )
            
            # Evaluate response
            scores = self.evaluator.evaluate(
                prompt=prompt_text,
                response=result['response_text'],
                response_time=result['response_time'],
                tokens_used=result['tokens_used']
            )
            
            logger.info(f"Evaluation scores: {scores}")
            
            # Create evaluation record
            self.db.create_evaluation(
                response_id=response_id,
                scores=scores,
                notes=f"Automated evaluation for {template_type}"
            )
            
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.db.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run LLM prompt experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config experiments/example.yaml
  python run_experiment.py --config experiments/comparison.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    try:
        experiment_id = runner.run()
        logger.info(f"\nTo view results, run:")
        logger.info(f"  python report_generator.py --experiment {experiment_id}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
