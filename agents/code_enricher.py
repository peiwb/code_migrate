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
        prompt_template = """You are a senior Python developer with deep expertise in both PySpark and Pandas data manipulation libraries. Your task is to analyze the provided Python function, and then perform two tasks: 1. Write comprehensive documentation. 2. Generate a correct and runnable unit test.

**Task Details:**

**1. Code Documentation:**
- Add a detailed `docstring` following the Google Python style guide, explaining the function's purpose, arguments (Args), and return values (Returns).
- Add concise inline comments to guide a future AI migration agent.
- **For mixed-dialect functions, it is CRITICAL that you add inline comments at the points of interaction (e.g., `.toPandas()` or `spark.createDataFrame(pandas_df)`), explaining the performance implications of moving data between distributed and in-memory contexts.**

**2. Unit Test Generation:**
- You must first inspect the function's code to determine which libraries are used (PySpark, Pandas, standard Python, or a mix).
- Then, generate a complete `pytest` test function according to the most appropriate rule below.

**Rule A: If the function uses ONLY PySpark:**
- The test must create a local SparkSession.
- Input and expected output data should be PySpark DataFrames.
- Assert the equality of the output DataFrame's content against the expected data.

**Rule B: If the function uses ONLY Pandas:**
- The test must `import pandas as pd`.
- Input and expected output data should be Pandas DataFrames.
- Use an appropriate method (e.g., `pd.testing.assert_frame_equal`) to assert DataFrame equality.

**Rule C: If the function is ONLY standard Python:**
- Use standard Python data structures (e.g., lists of dicts) for input and expected output.
- Use standard pytest assertions to check the result.

**Rule D: If the function uses a MIX of PySpark and Pandas:**
- The test must be an **integration test** that validates the end-to-end logic.
- It must create a local SparkSession.
- It should create the initial input data in the format required by the function's signature.
- It must assert the final output, which is typically a Spark DataFrame, against the expected data.

**Function to Process:**

```python
{function_code}
```

**Output Format:** Your final output must be a single, well-formatted JSON object. The JSON object must contain two keys: "enriched_code" and "test_function", and nothing else.
"""
        return prompt_template.format(function_code=function_code)