"""
Code Reviewer Module (code_reviewer.py) - V1.1

This module serves as the fourth and final executor in the migration workflow,
acting as an Automated Reviewer and Corrector. It performs a two-step process:
1. Review: Intelligent review of migrated Snowpark functions against original PySpark code
2. Correct: Automatic correction based on review findings to produce optimized final code
"""

import json
from typing import Dict, List
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService


class ReviewError(Exception):
    """Custom exception for code review and correction errors."""
    pass


class CodeReviewer:
    """
    Automated code reviewer and corrector for PySpark to Snowpark migrations.

    Uses LLM-driven intelligence through two carefully designed prompts to perform
    comprehensive review and correction of migrated code.
    """

    def __init__(self, llm_service: CortexLLMService):
        """Initialize the code reviewer instance."""
        self.llm_service = llm_service

    def review_and_correct_migration(
            self,
            original_function_code: str,
            migrated_function_code: str,
            knowledge_service: KnowledgeService,
            function_analysis: dict
    ) -> dict:
        """
        Main public entry point for the complete review-correction workflow.

        Returns:
            dict: Contains 'review_report' (dict) and 'corrected_code' (str)
        """
        try:
            # Extract suggested patterns and get recipes
            suggested_patterns = function_analysis.get('suggested_patterns', [])
            recipes = knowledge_service.get_recipes_from_suggested_patterns(suggested_patterns)

            # Phase 1: Review
            review_report = self._get_review_report(
                original_function_code, migrated_function_code, recipes
            )

            # Phase 2: Correct
            corrected_code = self._apply_corrections(migrated_function_code, review_report)

            return {
                'review_report': review_report,
                'corrected_code': corrected_code
            }

        except Exception as e:
            raise ReviewError(f"Failed to complete review and correction process: {str(e)}")

    def _get_review_report(self, original_code: str, migrated_code: str, recipes: list) -> dict:
        """Execute the review phase to generate structured review report."""
        try:
            prompt = self._build_review_prompt(original_code, migrated_code, recipes)
            return self.llm_service.get_json_completion(
                prompt=prompt,
                json_schema=self._get_review_json_schema()
            )
        except Exception as e:
            raise ReviewError(f"Failed to generate review report: {str(e)}")

    def _apply_corrections(self, migrated_code: str, review_report: dict) -> str:
        """Execute the correction phase based on review findings."""
        try:
            prompt = self._build_correction_prompt(migrated_code, review_report)
            corrected_code = self.llm_service.get_text_completion(prompt=prompt)
            return self._clean_code_output(corrected_code)
        except Exception as e:
            raise ReviewError(f"Failed to apply corrections: {str(e)}")

    def _build_review_prompt(self, original_code: str, migrated_code: str, recipes: list) -> str:
        """Build the comprehensive review prompt for LLM evaluation."""
        formatted_recipes = self._format_recipes(recipes)

        return f"""You are an experienced Principal Software Engineer tasked with conducting a rigorous code review of a PySpark to Snowpark migration.

You will receive three materials:
1. **Original PySpark Function**: The pre-migration code.
2. **Migrated Snowpark Function**: The post-migration code, which is your review target.
3. **Authoritative Reference Materials**: Code recipes that represent expected transformation rules guiding the migration process.

Your review must be objective, rigorous, and output in structured JSON format.

---[Authoritative Reference Materials]---
{formatted_recipes}
---[End of Authoritative Reference Materials]---

---[Original PySpark Function]---
```python
{original_code}
```
---[End of Original PySpark Function]---

---[Migrated Snowpark Function]---
```python
{migrated_code}
```
---[End of Migrated Snowpark Function]---

Please evaluate the "Migrated Snowpark Function" according to the following review checklist and generate a JSON-formatted review report.

【Review Checklist】
1. Logic Equivalence: Is the migrated code functionally and logically equivalent to the original code?
2. Recipe Adherence: Does the migration correctly apply the code patterns defined in the "Authoritative Reference Materials"?
3. Documentation & Comment Preservation: Are function signatures, docstrings, and all inline comments completely preserved?
4. TODO Flag Check: Does it contain # TODO: [MANUAL MIGRATION REQUIRED] markers? If so, explicitly note this in the report.
5. Overall Quality Assessment: Provide an overall evaluation of migration quality from code readability, conciseness, and Snowpark best practices perspectives.

【Output Format】
Your final output must be and can only be a well-formatted JSON object without any additional explanations."""

    def _build_correction_prompt(self, migrated_code: str, review_report: dict) -> str:
        """Build the correction prompt for applying review suggestions."""
        review_json = json.dumps(review_report, indent=2, ensure_ascii=False)

        return f"""You are a professional software engineer tasked with modifying and improving code based on a code review report.

You will receive two materials:
1. **Code to be Corrected**: The initial version of code that needs your modifications.
2. **Review Report**: A JSON object containing specific modification comments and suggestions.

Your task is to carefully read every comment in the review report and apply all suggestions to the code, ultimately producing corrected, higher-quality code.

---[Code to be Corrected]---
```python
{migrated_code}
```
---[End of Code to be Corrected]---

---[Review Report]---
{review_json}
---[End of Review Report]---

Please apply all suggestions from the review report to correct the code.

【Output Format】
Your final output must be and can only be the corrected, complete Python function code. Do not include any additional explanations, preambles, or Markdown markers."""

    def _format_recipes(self, recipes: list) -> str:
        """Format recipes into readable text blocks for the prompt."""
        if not recipes:
            return "No specific recipes available for this migration pattern."

        formatted_blocks = []
        for i, recipe in enumerate(recipes, 1):
            # Handle different recipe structures
            recipe_id = recipe.get('pattern_name') or recipe.get('id', f'Recipe {i}')
            description = recipe.get('description', 'No description')

            block = f"Recipe {i}: {recipe_id}\nDescription: {description}\n"

            # Add PySpark example
            pyspark_code = recipe.get('pyspark_example') or recipe.get('pyspark_code')
            if pyspark_code:
                block += f"PySpark Example:\n```python\n{pyspark_code}\n```\n"

            # Add Snowpark example
            snowpark_code = recipe.get('snowpark_example') or recipe.get('snowpark_code')
            if snowpark_code:
                block += f"Snowpark Example:\n```python\n{snowpark_code}\n```\n"

            # Add notes if available
            if recipe.get('notes'):
                block += f"Notes: {recipe['notes']}\n"

            formatted_blocks.append(block)

        return '\n'.join(formatted_blocks)

    def _get_review_json_schema(self) -> dict:
        """Get the JSON schema for structured review report output."""
        return {
            "type": "object",
            "properties": {
                "migration_confidence_score": {
                    "type": "number",
                    "description": "A float from 0.0 to 1.0 representing your overall confidence in this migration quality. 1.0 represents perfection."
                },
                "summary": {
                    "type": "string",
                    "description": "A summary review comment, e.g., 'High-quality migration, requires only minor modifications.' or 'Migration has major logical issues, requires manual refactoring.'"
                },
                "review_comments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "A category name, e.g., 'Logic', 'Style', 'Completeness', 'TODO'"
                            },
                            "comment": {
                                "type": "string",
                                "description": "Specific review comment."
                            },
                            "suggestion": {
                                "type": "string",
                                "description": "(Optional) Modification suggestion for this comment."
                            }
                        },
                        "required": ["category", "comment"]
                    }
                }
            },
            "required": ["migration_confidence_score", "summary", "review_comments"]
        }

    def _clean_code_output(self, code_text: str) -> str:
        """Clean and format the corrected code output from LLM."""
        code_text = code_text.strip()

        # Remove markdown code blocks
        if code_text.startswith('```python'):
            code_text = code_text[9:]
        elif code_text.startswith('```'):
            code_text = code_text[3:]

        if code_text.endswith('```'):
            code_text = code_text[:-3]

        # Clean up whitespace while preserving structure
        lines = [line.rstrip() for line in code_text.split('\n')]

        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)