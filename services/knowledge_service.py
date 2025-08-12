"""
Knowledge Service Module (knowledge_service.py) - Version 1.0

This module serves as the "authoritative knowledge base" and "single source of truth"
for the entire migration workflow. It loads, manages, and provides all code rules
for converting from PySpark to Snowpark, known as "Code Recipes".
"""

import json
from services.llm_service import CortexLLMService


class KnowledgeServiceError(Exception):
    """Custom exception class for KnowledgeService errors."""
    pass


class KnowledgeService:
    """
    Core knowledge service that manages migration rules and provides recipe lookup functionality.
    """

    def __init__(self, knowledge_base_path: str, llm_service: CortexLLMService = None):
        """
        Initialize the knowledge service instance and load the knowledge base.

        Args:
            knowledge_base_path (str): Complete path to the knowledge_base.json file
            llm_service (CortexLLMService, optional): LLM service instance for semantic discovery

        Raises:
            KnowledgeServiceError: If the knowledge base file cannot be loaded or parsed
        """
        self.llm_service = llm_service
        self.recipes = {}
        self._load_knowledge_base(knowledge_base_path)

    def _load_knowledge_base(self, knowledge_base_path: str):
        """Load and parse the knowledge base JSON file."""
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Handle two actual JSON structures: list or single object
            recipes_list = data if isinstance(data, list) else [data]

            # Index recipes by ID
            for recipe in recipes_list:
                if isinstance(recipe, dict) and 'id' in recipe:
                    self.recipes[recipe['id']] = recipe

        except FileNotFoundError:
            raise KnowledgeServiceError(f"Knowledge base file not found: {knowledge_base_path}")
        except json.JSONDecodeError as e:
            raise KnowledgeServiceError(f"Failed to parse knowledge base JSON: {str(e)}")

    def get_recipes_from_suggested_patterns(self, pattern_ids: list) -> list:
        """
        Core method to retrieve complete code recipes based on pattern ID list.

        Args:
            pattern_ids (list): List of pattern ID strings

        Returns:
            list: List of complete recipe dictionary objects
        """
        return [self.recipes[pid] for pid in pattern_ids if pid in self.recipes]

    def discover_patterns(self, keyword: str, top_k: int = 3) -> list:
        """
        Advanced method for semantic pattern discovery using natural language keywords.

        Args:
            keyword (str): Natural language search term
            top_k (int): Number of most relevant patterns to return

        Returns:
            list: List of the most relevant pattern ID strings

        Raises:
            NotImplementedError: If no LLM service was provided during initialization
        """
        if self.llm_service is None:
            raise NotImplementedError(
                "Semantic pattern discovery requires an LLM service. "
                "Please provide llm_service parameter during initialization."
            )

        # Build context from all available recipes
        recipe_context = [
            {
                "id": recipe_id,
                "description": recipe.get("description", ""),
                "usage_context": recipe.get("usage_context", "")
            }
            for recipe_id, recipe in self.recipes.items()
        ]

        prompt = f"""
        Given the following PySpark to Snowpark migration patterns and a user keyword,
        identify the {top_k} most relevant pattern IDs that match the user's intent.

        Available patterns:
        {json.dumps(recipe_context, indent=2)}

        User keyword: "{keyword}"

        Please return the {top_k} most relevant pattern IDs in order of relevance.
        """

        json_schema = {
            "type": "object",
            "properties": {
                "pattern_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": top_k
                }
            },
            "required": ["pattern_ids"]
        }

        try:
            response = self.llm_service.get_json_completion(
                prompt=prompt,
                json_schema=json_schema
            )
            return response.get("pattern_ids", [])

        except Exception as e:
            raise KnowledgeServiceError(f"Semantic discovery failed: {str(e)}")