"""
Code Enricher Module (code_enricher.py) - Version 1.0

This module serves as the second executor in the migration workflow.
It receives "raw" segmented PySpark function code and performs enrichment processing,
including code documentation and test case generation.
"""

from typing import Dict
from services.llm_service import CortexLLMService


class EnrichmentError(Exception):
    """Custom exception raised when code enrichment operations fail."""
    pass


class CodeEnricher:
    """
    Code enricher that uses LLM to add documentation and generate unit tests for PySpark functions.

    This class is responsible for:
    1. Adding Google-style docstrings and inline comments to PySpark functions
    2. Generating complete pytest unit tests for the functions
    """

    def __init__(self, llm_service: CortexLLMService):
        """
        Initialize the code enricher instance.

        Args:
            llm_service: An instantiated CortexLLMService object for LLM interactions.
        """
        self.llm_service = llm_service
        self._enrichment_schema = {
            "type": "object",
            "properties": {
                "enriched_code": {
                    "type": "string",
                    "description": "The original function code enriched with a Google-style docstring and inline comments."
                },
                "test_function": {
                    "type": "string",
                    "description": "A complete, runnable pytest function to test the original PySpark function."
                }
            },
            "required": ["enriched_code", "test_function"]
        }

    def enrich_function(self, function_code: str) -> Dict[str, str]:
        """
        The main public entry point for code enrichment.

        Takes a single PySpark function code and returns a dictionary containing
        the enriched code with documentation and a generated test function.

        Args:
            function_code: A string containing the code of a single function to be processed.

        Returns:
            A Python dictionary with two keys:
            - 'enriched_code': Function code with added documentation and comments
            - 'test_function': Complete pytest test function code for the original function

        Raises:
            EnrichmentError: When LLM service call fails or returns invalid data
        """
        try:
            # Build the enrichment prompt using the function code
            prompt = self._build_enrichment_prompt(function_code)

            # Call LLM service to get structured JSON response
            result = self.llm_service.get_json_completion(
                prompt=prompt,
                json_schema=self._enrichment_schema
            )

            # Validate that required keys are present
            if not isinstance(result, dict):
                raise EnrichmentError("LLM service returned invalid response format")

            if 'enriched_code' not in result or 'test_function' not in result:
                raise EnrichmentError("LLM service response missing required keys")

            return result

        except Exception as e:
            raise EnrichmentError(f"Code enrichment failed: {str(e)}") from e

    def _build_enrichment_prompt(self, function_code: str) -> str:
        """
        Dynamically assemble a high-quality prompt for code enrichment and test generation.

        Args:
            function_code: The PySpark function code to be enriched

        Returns:
            The complete prompt string ready for LLM processing
        """
        prompt_template = """You are a top-tier PySpark development expert and technical documentation engineer. Your task is to process the single PySpark function provided below and simultaneously complete two tasks: 1. Write high-quality documentation for it. 2. Write a complete unit test for it.

**Task Details:**

**1. Code Documentation:**
   - Add a detailed `docstring` that follows Google Python style guidelines, clearly describing the function's purpose, parameters (Args), and return values (Returns).
   - Add concise inline comments that reveal the business intent, performance considerations, or hidden assumptions behind the code. These comments should act as instructions for a subsequent AI agent that will migrate this code to another platform, highlighting anything that is not obvious from the code itself.

**2. Unit Test Generation:**
   - Write a complete test function that can be directly run with `pytest`.
   - The test function must include:
     a. Create a local SparkSession for testing.
     b. Generate realistic input data for testing (e.g., a PySpark DataFrame).
     c. Define expected output data (e.g., another PySpark DataFrame) that should be the correct result after the input data is processed by the function.
     d. Call the function under test with the input data.
     e. Use assertions to strictly compare actual output with expected output to verify function correctness.

**Function to Process:**
```python
{function_code}
```

Output Format:
Your final output must be and can only be a well-formatted JSON object without any additional explanations. The JSON object must contain two keys: "enriched_code" and "test_function"."""

        return prompt_template.format(function_code=function_code)