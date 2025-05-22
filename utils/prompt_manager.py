# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# read the prompt template from a file


import os
from pathlib import Path
import yaml
from typing import Dict, Optional
import re

class PromptManager:
    """Manager for handling prompt templates with variable substitution."""
    
    def __init__(self, prompt_path: Optional[str] = None):
        if prompt_path is None:
            root_dir = Path(__file__).parent.parent
            prompt_path = root_dir / "src"/ "openllm_ocr_annotator" / "config" / "prompt_templates.yaml"
            
        self.config_path = prompt_path
        self.load_config()
        self.default_variables = self.config.get('variables', {})
    
    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _replace_variables(self, template: str, variables: Dict[str, str]) -> str:
        """Replace template variables with actual values.
        
        Args:
            template: Template string containing {{variable}} placeholders
            variables: Dictionary of variable names and their values
            
        Returns:
            String with variables replaced
        """
        for key, value in variables.items():
            pattern = r'\{\{' + re.escape(key) + r'\}\}'
            template = re.sub(pattern, value, template)
        return template
    
    def get_prompt(self, model: str, task: str, variables: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get prompt template and substitute variables.
        
        Args:
            model: Model name (e.g., 'openai')
            task: Task name (e.g., 'vision_extraction')
            variables: Dictionary of variable names and values for substitution
            
        Returns:
            Dict containing processed system and user prompts
        """
        # Get base template
        if model in self.config and task in self.config[model]:
            template = self.config[model][task]
        elif task in self.config['default']:
            template = self.config['default'][task]
        else:
            raise ValueError(f"No template found for model={model}, task={task}")
        
        # merge default variables with custom variables
        merged_variables = self.default_variables.copy()
        if variables:
            merged_variables.update(variables)
        
        # Process variables in both system and user prompts
        return {
            "system": self._replace_variables(template["system"], merged_variables),
            "user": self._replace_variables(template["user"], merged_variables)
        }
