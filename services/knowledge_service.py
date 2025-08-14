"""
Knowledge Service Module (knowledge_service.py) - Enhanced Version

This module serves as the "authoritative knowledge base" and "single source of truth"
for Python data processing to Snowpark migration workflow. It uses LLM to intelligently
match Python functions (PySpark/Pandas/native Python) to appropriate Snowpark recipes.
"""

import json
from typing import List, Dict, Any, Optional
from services.llm_service import CortexLLMService
import os
from datetime import datetime


class KnowledgeServiceError(Exception):
    """Custom exception class for KnowledgeService errors."""
    pass


class KnowledgeService:
    """
    Core knowledge service that manages Snowpark recipes and provides intelligent
    matching for Python data processing functions using LLM.
    """

    def __init__(self, knowledge_base_path: str, llm_service: CortexLLMService = None):
        """
        Initialize the knowledge service instance and load the knowledge base.

        Args:
            knowledge_base_path (str): Complete path to the knowledge_base.json file
            llm_service (CortexLLMService, optional): LLM service instance for intelligent matching

        Raises:
            KnowledgeServiceError: If the knowledge base file cannot be loaded or parsed
        """
        self.llm_service = llm_service
        self.recipes = {}
        self._load_knowledge_base(knowledge_base_path)

    def _debug_save_prompt_and_context(self, prompt, python_functions, top_k):
        """调试用：保存prompt和context到文件"""
        try:
            # 确保debug目录存在
            debug_dir = "./debug_output"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存完整的知识库context
            context_file = os.path.join(debug_dir, f"knowledge_base_context_{timestamp}.txt")
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write("KNOWLEDGE BASE - ALL RECIPES\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total recipes: {len(self.recipes)}\n\n")

                for i, (recipe_id, recipe) in enumerate(self.recipes.items(), 1):
                    f.write(f"[{i:2d}] ID: {recipe_id}\n")
                    f.write(f"     Description: {recipe.get('description', 'No description')}\n")
                    # usage = recipe.get('usage_context', '')
                    # if usage:
                    #     f.write(f"     Usage: {usage}\n")
                    f.write("\n")

            # 保存完整的prompt
            prompt_file = os.path.join(debug_dir, f"llm_prompt_{timestamp}.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write("COMPLETE LLM PROMPT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Input functions: {python_functions}\n")
                f.write(f"Top K: {top_k}\n")
                f.write(f"Prompt length: {len(prompt)} chars\n")
                f.write("=" * 50 + "\n\n")
                f.write(prompt)

            print(f"Debug files saved: {context_file}, {prompt_file}")
        except Exception as e:
            print(f"Debug save failed (non-critical): {e}")

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

    def discover_patterns(self, python_functions: List[str], top_k: Optional[int] = None) -> List[str]:
        """
        Intelligently discover relevant Snowpark recipe IDs for given Python functions using LLM.

        Args:
            python_functions (List[str]): List of Python function names (PySpark/Pandas/native Python)
            top_k (Optional[int]): Maximum number of recipe IDs to return. If None, uses adaptive strategy.

        Returns:
            List[str]: List of the most relevant Snowpark recipe ID strings

        Raises:
            NotImplementedError: If no LLM service was provided during initialization
            KnowledgeServiceError: If LLM analysis fails
        """
        if self.llm_service is None:
            raise NotImplementedError(
                "Intelligent pattern discovery requires an LLM service. "
                "Please provide llm_service parameter during initialization."
            )

        # Adaptive top_k strategy
        if top_k is None:
            top_k = max(3, min(12, len(python_functions) * 2))

        if not python_functions:
            return []

        try:
            # Build LLM prompt with all available recipes as context
            prompt = self._build_discovery_prompt(python_functions, top_k)

            #self._debug_save_prompt_and_context(prompt, python_functions, top_k)

            # Define JSON schema for LLM response
            json_schema = {
                "type": "object",
                "properties": {
                    "recipe_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": top_k
                    }
                },
                "required": ["recipe_ids"]
            }

            # Get LLM analysis
            response = self.llm_service.get_json_completion(
                prompt=prompt,
                json_schema=json_schema
            )

            # Extract and validate recipe IDs
            recipe_ids = response.get("recipe_ids", [])
            validated_ids = self._validate_recipe_ids(recipe_ids)

            return validated_ids[:top_k]

        except Exception as e:
            # Simple fallback: keyword-based matching
            fallback_ids = self._keyword_fallback(python_functions, top_k)
            if fallback_ids:
                return fallback_ids
            raise KnowledgeServiceError(f"Pattern discovery failed: {str(e)}")

    def get_recipes_from_suggested_patterns(self, recipe_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve complete recipe information based on recipe ID list.

        Args:
            recipe_ids (List[str]): List of recipe ID strings

        Returns:
            List[Dict[str, Any]]: List of complete recipe dictionary objects
        """
        recipes = []
        for recipe_id in recipe_ids:
            if recipe_id in self.recipes:
                recipes.append(self.recipes[recipe_id])
        return recipes

    def _build_discovery_prompt(self, python_functions: List[str], top_k: int) -> str:
        """Build the LLM prompt for intelligent pattern discovery."""

        # Build context from all available recipes
        recipes_context = self._build_recipes_context()

        # Format Python functions
        functions_str = ", ".join(python_functions)

        prompt = f"""You are an expert in migrating Python data processing code to Snowflake Snowpark.

Your task: Analyze the given Python functions and recommend the most relevant Snowpark recipes for migration.

AVAILABLE SNOWPARK RECIPES:
{recipes_context}

PYTHON FUNCTIONS TO MIGRATE:
{functions_str}

ANALYSIS INSTRUCTIONS:
1. Understand the data processing intent of each Python function
2. Consider these function types:
   - PySpark functions (spark.table, DataFrame.select, etc.)
   - Pandas functions (pandas.read_csv, DataFrame.groupby, etc.)  
   - Native Python data processing functions
3. Match to the most appropriate Snowpark recipes from the available list
4. Consider these Snowpark approaches:
   - Snowpark DataFrame API for distributed processing
   - pandas on Snowflake for pandas-like operations
   - Snowpark Session functions for data access
   - Snowpark UDF patterns for custom logic

SELECTION CRITERIA:
- Prioritize functional equivalence
- Consider performance characteristics
- Prefer simpler, more direct mappings
- Include complementary recipes for complete workflows

Return the top {top_k} most relevant recipe IDs, ordered by relevance (most relevant first).

OUTPUT FORMAT:
{{"recipe_ids": ["recipe_id_1", "recipe_id_2", ...]}}
"""
        return prompt

    def _build_recipes_context(self) -> str:
        """Build formatted context of all recipes for LLM prompt."""
        context_blocks = []

        for recipe_id, recipe in self.recipes.items():
            description = recipe.get('description', 'No description available')
            #usage_context = recipe.get('usage_context', '')

            context_block = f"ID: {recipe_id}\nDescription: {description}"
            #if usage_context:
            #   context_block += f"\nUsage Context: {usage_context}"

            context_blocks.append(context_block)

        return "\n\n".join(context_blocks)

    def _validate_recipe_ids(self, recipe_ids: List[str]) -> List[str]:
        """Validate and filter recipe IDs to ensure they exist in knowledge base."""
        valid_ids = []
        for recipe_id in recipe_ids:
            if isinstance(recipe_id, str) and recipe_id in self.recipes:
                valid_ids.append(recipe_id)
        return valid_ids

    def _keyword_fallback(self, python_functions: List[str], top_k: int) -> List[str]:
        """Simple keyword-based fallback matching when LLM fails."""
        matches = []

        # Extract keywords from Python functions
        keywords = []
        for func in python_functions:
            # Simple keyword extraction: split by dots and underscores
            parts = func.lower().replace('.', ' ').replace('_', ' ').split()
            keywords.extend(parts)

        # Score recipes based on keyword matches
        recipe_scores = {}
        for recipe_id, recipe in self.recipes.items():
            score = 0
            recipe_text = (
                recipe.get('description', '') + ' ' +
                recipe.get('usage_context', '') + ' ' +
                recipe_id
            ).lower()

            for keyword in keywords:
                if keyword in recipe_text:
                    score += 1

            if score > 0:
                recipe_scores[recipe_id] = score

        # Sort by score and return top_k
        sorted_recipes = sorted(recipe_scores.items(), key=lambda x: x[1], reverse=True)
        return [recipe_id for recipe_id, _ in sorted_recipes[:top_k]]