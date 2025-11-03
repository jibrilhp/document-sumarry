import yaml
from pathlib import Path
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate


class PromptLoader:
    """Utility class to load prompts from YAML files"""
    
    def __init__(self, prompts_file_path: str = "prompts.yaml"):
        """
        Initialize the PromptLoader with a path to the prompts YAML file.
        
        Args:
            prompts_file_path: Path to the YAML file containing prompts
        """
        self.prompts_file_path = Path(prompts_file_path)
        self._prompts: Dict[str, Any] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from the YAML file"""
        if not self.prompts_file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file_path}")
        
        with open(self.prompts_file_path, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)
    
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            
        Returns:
            The prompt template string
        """
        if prompt_name not in self._prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found in prompts file")
        
        prompt_config = self._prompts[prompt_name]
        
        # Handle both 'template' and 'prefix' keys
        if 'template' in prompt_config:
            return prompt_config['template']
        elif 'prefix' in prompt_config:
            return prompt_config['prefix']
        else:
            raise ValueError(f"Prompt '{prompt_name}' must have either 'template' or 'prefix' key")
    
    def create_chat_prompt(self, prompt_name: str, role: str = "human") -> ChatPromptTemplate:
        """
        Create a ChatPromptTemplate from a prompt name.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            role: The role for the prompt (e.g., 'human', 'system', 'assistant')
            
        Returns:
            A ChatPromptTemplate instance
        """
        template = self.get_prompt(prompt_name)
        return ChatPromptTemplate([(role, template)])
    
    def reload(self):
        """Reload prompts from the YAML file (useful for hot-reloading)"""
        self._load_prompts()
    
    def get_all_prompt_names(self) -> list[str]:
        """Get a list of all available prompt names"""
        return list(self._prompts.keys())