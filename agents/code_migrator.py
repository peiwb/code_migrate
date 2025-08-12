"""
Code Migrator Module (code_migrator.py)

Core execution engine for PySpark to Snowpark code migration using LLM assistance.
"""

from typing import List, Dict, Any
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService


class MigrationError(Exception):
    """Custom exception for migration-related errors."""
    pass


class CodeMigrator:
    """Core code migration engine that transforms PySpark functions to Snowpark equivalents."""

    def __init__(self, llm_service: CortexLLMService):
        """Initialize the code migrator instance."""
        self.llm_service = llm_service

    def migrate_function(
            self,
            source_code: str,
            function_analysis: dict,
            knowledge_service: KnowledgeService
    ) -> str:
        """
        Migrate a PySpark function to Snowpark equivalent.

        Args:
            source_code: Function code with inline migration guidance
            function_analysis: Analysis data from code_analyzer
            knowledge_service: Knowledge service instance

        Returns:
            str: Migrated Snowpark function code
        """
        try:
            # Get suggested patterns and recipes
            suggested_patterns = function_analysis.get('suggested_patterns', [])
            recipes = knowledge_service.get_recipes_from_suggested_patterns(suggested_patterns)

            # Build migration prompt and get LLM response
            prompt = self._build_migration_prompt(source_code, recipes)
            response = self.llm_service.get_text_completion(prompt)

            return response.strip()

        except Exception as e:
            raise MigrationError(f"Failed to migrate function: {str(e)}") from e

    def _build_migration_prompt(self, function_code: str, recipes: list) -> str:
        """Build the complete migration prompt for the LLM."""
        formatted_recipes = self._format_recipes(recipes)

        return f"""You are a top-tier software engineer specializing in migrating Python/PySpark functions to equivalent Snowpark functions.

Before starting, learn from these verified PySpark to Snowpark conversion examples:

---[REFERENCE EXAMPLES]---
{formatted_recipes}
---[END REFERENCE EXAMPLES]---

Your task: Translate the **entire** PySpark function below into functionally equivalent Snowpark code.

CRITICAL: The function contains inline comments with specific migration guidance. Read and follow these comments carefully - they provide crucial context for accurate migration.

---[FUNCTION TO MIGRATE]---
```python
{function_code}
```
---[END FUNCTION]---

MIGRATION RULES:
1. Preserve original function signature (name and parameters)
2. Follow ALL inline migration guidance comments carefully
3. Keep docstrings; remove migration guidance comments after applying them
4. Use reference examples to translate PySpark APIs to Snowpark equivalents
5. Maintain pure Python logic unchanged (loops, conditions, variables)
6. For unknown APIs not covered in examples/guidance: add "# TODO: [MANUAL MIGRATION REQUIRED]"
7. Output ONLY the migrated function code - no explanations or markdown

Begin migration now."""

    def _format_recipes(self, recipes: list) -> str:
        """Format recipes into readable text blocks for the prompt."""
        if not recipes:
            return "No specific recipes available."

        formatted_blocks = []
        for recipe in recipes:
            recipe_id = recipe.get('id', 'Unknown Pattern')
            description = recipe.get('description', 'No description')
            snowpark_code = recipe.get('snowpark_code', 'No implementation')

            formatted_blocks.append(f"""# Pattern: {recipe_id}
# Description: {description}
# Snowpark Implementation:
{snowpark_code}""")

        return '\n\n'.join(formatted_blocks)