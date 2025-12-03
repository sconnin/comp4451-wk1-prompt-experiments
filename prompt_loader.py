"""
Prompt template loader and manager.
"""
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages prompt templates."""
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize prompt loader with templates directory."""
        self.templates_dir = templates_dir
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Load all prompt templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.txt'):
                template_name = filename.replace('.txt', '')
                template_path = os.path.join(self.templates_dir, filename)
                
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        self.templates[template_name] = f.read()
                    logger.info(f"Loaded template: {template_name}")
                except Exception as e:
                    logger.error(f"Error loading template {template_name}: {e}")
    
    def get_template(self, template_name: str) -> Optional[str]:
        """Get a specific template by name."""
        template = self.templates.get(template_name)
        if not template:
            logger.warning(f"Template not found: {template_name}")
        return template
    
    def render_template(self, template_name: str, variables: Dict) -> Optional[str]:
        """Render a template with provided variables."""
        template = self.get_template(template_name)
        if not template:
            return None
        
        try:
            # Simple string formatting with variables
            rendered = template.format(**variables)
            logger.debug(f"Rendered template: {template_name}")
            return rendered
        except KeyError as e:
            logger.error(f"Missing variable {e} for template {template_name}")
            return None
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return None
    
    def list_templates(self) -> list:
        """List all available templates."""
        return list(self.templates.keys())
