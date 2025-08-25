"""
Code Reviewer Module (code_reviewer.py) - V2.0

This module serves as the final quality gate in the migration workflow,
implementing an "Auditor + Surgeon" two-phase approach:
1. Auditor: Rigorous review and issue identification with structured JSON report
2. Surgeon: Precise, mechanical correction application based on audit findings
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
    Automated code reviewer and corrector implementing the Auditor + Surgeon pattern.

    Phase 1 (Auditor): Identifies issues and generates structured JSON audit report
    Phase 2 (Surgeon): Mechanically applies corrections based on audit findings
    """

    def __init__(self, llm_service: CortexLLMService):
        """Initialize the code reviewer with LLM service."""
        self.llm_service = llm_service

    def review_and_correct_migration(
        self,
        original_function_code: str,
        migrated_function_code: str,
        knowledge_service: KnowledgeService,
        function_analysis: dict
    ) -> dict:
        """
        Main entry point for the complete auditor-surgeon workflow.

        Returns:
            dict: Contains 'review_report' (dict) and 'corrected_code' (str)
        """
        try:
            # Extract suggested patterns and get recipes
            suggested_patterns = function_analysis.get('suggested_patterns', [])
            recipes = knowledge_service.get_recipes_from_suggested_patterns(suggested_patterns)

            # Phase 1: Auditor - Generate review report
            review_report = self._generate_review_report(
                original_function_code, migrated_function_code, recipes
            )

            # Check audit results and determine next action
            status = review_report['overall_assessment']['status']

            if status == 'PERFECT':
                # No corrections needed
                corrected_code = migrated_function_code
            elif status == 'NEEDS_REFINEMENT':
                # Phase 2: Surgeon - Apply corrections
                corrected_code = self._apply_corrections(migrated_function_code, review_report)
            else:
                raise ReviewError(f"Invalid audit status: {status}")

            return {
                'review_report': review_report,
                'corrected_code': corrected_code
            }

        except Exception as e:
            raise ReviewError(f"Failed to complete review and correction process: {str(e)}")

    # === Phase 1: Auditor Implementation ===

    def _generate_review_report(self, original_code: str, migrated_code: str, recipes: list) -> dict:
        """Execute audit phase to generate structured review report."""
        try:
            prompt = self._build_review_prompt(original_code, migrated_code, recipes)
            return self.llm_service.get_json_completion(
                prompt=prompt,
                json_schema=self._get_review_json_schema()
            )
        except Exception as e:
            raise ReviewError(f"Failed to generate review report: {str(e)}")

    def _get_review_json_schema(self) -> dict:
        """Define structured JSON schema for machine-executable audit report."""
        return {
            "type": "object",
            "properties": {
                "overall_assessment": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["PERFECT", "NEEDS_REFINEMENT"],
                            "description": "PERFECT if code is flawless; NEEDS_REFINEMENT otherwise."
                        },
                        "summary": {
                            "type": "string",
                            "description": "Overall textual summary of migration quality."
                        }
                    },
                    "required": ["status", "summary"]
                },
                "findings": {
                    "type": "array",
                    "description": "List of specific issue findings. Empty if status is PERFECT.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["API_MISUSE", "LOGIC_DIVERGENCE", "BEST_PRACTICE_VIOLATION", "STYLE_ISSUE"],
                                "description": "Classification of the issue."
                            },
                            "faulty_code_snippet": {
                                "type": "string",
                                "description": "Exact problematic code snippet from the reviewed code."
                            },
                            "issue_description": {
                                "type": "string",
                                "description": "Detailed description of the issue."
                            },
                            "suggested_correction": {
                                "type": "string",
                                "description": "Ready-to-use code snippet that directly replaces faulty_code_snippet."
                            }
                        },
                        "required": ["category", "faulty_code_snippet", "issue_description", "suggested_correction"]
                    }
                }
            },
            "required": ["overall_assessment", "findings"]
        }

    def _build_review_prompt(self, original_code: str, migrated_code: str, recipes: list) -> str:
        """Build comprehensive audit prompt for rigorous code review."""
        formatted_recipes = self._format_recipes(recipes)

        return f"""You are a meticulous and detail-oriented Snowpark code auditor with expertise in migrations from both PySpark and Pandas. Your task is to rigorously audit the "Migrated Snowpark Function" against the "Original Source Function" and provide a structured JSON report with precise, actionable corrections.
Critical Audit Protocol:
1.Identify Source Dialect: First, examine the "Original Source Function" to determine if its primary dialect is PySpark or Pandas. This is essential for the next step.
2.Select Correct Reference: Based on the identified dialect, you MUST use ONLY the corresponding set of reference materials for your audit (either PySpark-to-Snowpark or Pandas-to-Snowpark).
3.Perform Audit: Conduct a thorough review based on the checklist below, using the original function as the source of truth for intent and the correct reference materials as the standard for correctness.

---[Reference Materials]---
{formatted_recipes}
---[End of Reference Materials]---

---[Original PySpark Function (Source of Truth for Intent)]---
```python
{original_code}
```
---[End of Original PySpark Function]---

---[Migrated Snowpark Function (Target for Audit)]---
```python
{migrated_code}
```
---[End of Migrated Snowpark Function]---

**Audit Checklist & Instructions:**

1. Correct API Translation: Does the migrated code use the correct Snowpark APIs as defined in the relevant reference materials?

2. Logical Equivalence: Does the Snowpark code's logic perfectly match the intent of the original PySpark/Pandas code? Pay close attention to nuances (e.g., Pandas' in-memory vs. Snowpark's lazy execution).

3. Best Practice Adherence: Does the code follow Snowpark best practices? Are there more efficient ways to achieve the same result?

4. Completeness: Are all functionalities from the original function present? Are docstrings and meaningful comments preserved?

5. TODO Flag Check: Does the code contain # TODO: [MANUAL MIGRATION REQUIRED] markers? If so, this is a BEST_PRACTICE_VIOLATION.

**Output Requirements:**
Generate a JSON report based on the provided schema. For each finding, you MUST provide both faulty_code_snippet and suggested_correction. If the code is perfect, the findings array must be empty and the status must be PERFECT.

Your response must be valid JSON only, with no additional text or explanations."""

    # === Phase 2: Surgeon Implementation ===

    def _apply_corrections(self, migrated_code: str, review_report: dict) -> str:
        """Execute correction phase with mechanical precision based on audit findings."""
        try:
            prompt = self._build_correction_prompt(migrated_code, review_report)
            corrected_code = self.llm_service.get_text_completion(prompt=prompt)
            return self._clean_code_output(corrected_code)
        except Exception as e:
            raise ReviewError(f"Failed to apply corrections: {str(e)}")

    def _build_correction_prompt(self, migrated_code: str, review_report: dict) -> str:
        """Build precise correction prompt for mechanical code modification."""
        review_json_string = json.dumps(review_report, indent=2, ensure_ascii=False)

        return f"""You are a code modification robot. Your only task is to perform a series of precise "find and replace" operations on a given piece of code based on a JSON instruction list.

---[Code to Modify]---
```python
{migrated_code}
```
---[End of Code to Modify]---

---[Modification Instruction List (JSON)]---
{review_json_string}
---[End of Modification Instruction List]---

**Instructions:**
1. Iterate through each object in the "findings" array of the JSON list.
2. For each object, find the exact code snippet specified in "faulty_code_snippet" within the "Code to Modify".
3. Replace it precisely with the code from "suggested_correction".
4. Do not perform any other changes, additions, or creative modifications.
5. After applying all replacements, output the complete, final, modified code.

**Your output must ONLY be the Python code itself, without any explanations or markdown formatting.**"""

    # === Helper Methods ===

    def _format_recipes(self, recipes: list) -> str:
        """Format recipes into readable text blocks for prompt context."""
        if not recipes:
            return "No specific recipes available for this migration pattern."

        formatted_blocks = []
        for i, recipe in enumerate(recipes, 1):
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