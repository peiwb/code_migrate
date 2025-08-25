"""
Code Analyzer Module (code_analyzer.py) - V1.2

This module provides the CodeAnalyzer class which performs comprehensive analysis
of PySpark scripts through a three-stage analysis pipeline.
"""

import ast
import json
from datetime import datetime
from typing import Dict, List, Any
from services.llm_service import CortexLLMService


class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass


class CodeAnalyzer:
    """
    A comprehensive PySpark script analyzer that performs three-stage analysis:
    1. Package Analysis - Categorizes import statements
    2. Function Analysis - Analyzes individual functions for dependencies and patterns
    3. Conversion Order - Determines optimal function conversion sequence
    """

    def __init__(self, llm_service: CortexLLMService):
        """
        Initialize the CodeAnalyzer instance.

        Args:
            llm_service: An instantiated CortexLLMService object
        """
        self.llm_service = llm_service

    def analyze_script(self, script_content: str, source_file_name: str) -> Dict[str, Any]:
        """
        Main public entry point. Executes the complete three-stage analysis pipeline.

        Args:
            script_content: The complete PySpark script content as string
            source_file_name: Name of the original source file

        Returns:
            dict: Structured analysis report containing all analysis results

        Raises:
            AnalysisError: If any analysis stage fails
        """
        try:
            # Stage 1: Package Analysis
            package_analysis_result = self._analyze_packages(script_content)

            # Stage 2: Function Analysis
            function_analysis_result = self._analyze_functions(script_content)

            # Stage 3: Determine Conversion Order
            conversion_order_result = self._determine_conversion_order(function_analysis_result)

            # Combine all results into final report
            return {
                "source_file_name": source_file_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "package_analysis": package_analysis_result,
                "function_analysis": function_analysis_result,
                "conversion_order": conversion_order_result
            }

        except Exception as e:
            raise AnalysisError(f"Script analysis failed: {str(e)}")

    def _llm_call_with_error_handling(self, operation_name: str, prompt: str, schema: dict):
        """Centralized LLM call with error handling."""
        try:
            return self.llm_service.get_json_completion(prompt, schema)
        except Exception as e:
            raise AnalysisError(f"{operation_name} failed: {str(e)}")

    def _analyze_packages(self, script_content: str) -> Dict[str, List[Dict[str, str]]]:
        """Execute Stage 1: Package Analysis."""
        prompt = f"""You are a Python code analysis expert. Your task is to analyze the Python script provided below, which may contain a mix of standard libraries, PySpark, Pandas, and other data science packages. Please identify all import statements and categorize them.

Your goal is to create a complete inventory of the script's dependencies.

Please return your analysis results in the specified JSON format without any additional explanations or comments.

**Script to analyze:**
```python
{script_content}
```"""

        package_schema = {
            "type": "object",
            "properties": {
                "standard_libs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "import_statement": {"type": "string"},
                            "purpose": {"type": "string"}
                        },
                        "required": ["import_statement", "purpose"]
                    }
                },
                "third_party": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "import_statement": {"type": "string"},
                            "purpose": {"type": "string"}
                        },
                        "required": ["import_statement", "purpose"]
                    }
                },
                "custom_modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "import_statement": {"type": "string"},
                            "purpose": {"type": "string"}
                        },
                        "required": ["import_statement", "purpose"]
                    }
                }
            },
            "required": ["standard_libs", "third_party", "custom_modules"]
        }

        return self._llm_call_with_error_handling("Package analysis", prompt, package_schema)

    def _analyze_functions(self, script_content: str) -> List[Dict[str, Any]]:
        """Execute Stage 2: Function Analysis."""
        # Extract function blocks using AST
        function_blocks = []
        try:
            tree = ast.parse(script_content)
            script_lines = script_content.split('\n')

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(script_lines)
                    function_code = '\n'.join(script_lines[start_line:end_line])
                    function_blocks.append(function_code)
        except SyntaxError:
            # Fallback: treat entire script as one function
            function_blocks = [script_content]

        # Analyze each function
        results = []
        function_schema = {
            "type": "object",
            "properties": {
                "function_name": {"type": "string", "description": "Function name"},
                "function_dialect": {
                    "type": "string",
                    "enum": ["pyspark", "pandas", "python", "mixed"],
                    "description": "The primary programming dialect identified in the function."
                },
                "dependencies": {
                    "type": "object",
                    "properties": {
                        "internal_functions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of other functions called within this function"
                        },
                        "external_packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "External package names used by this function, e.g., pyspark.sql.functions"
                        }
                    },
                    "required": ["internal_functions", "external_packages"]
                },
                "suggested_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified key library pattern IDs, e.g., 'pyspark.withColumn', 'pandas.apply'."
                }
            },
            "required": ["function_name", "dependencies", "suggested_patterns"]
        }

        for function_code in function_blocks:
            prompt = f"""You are a Python Data Engineering and Migration Expert, specializing in migrating code from various sources like PySpark, Pandas, and pure Python to Snowpark.

Your task is to perform a deep analysis on the **single** Python function provided below and extract three key pieces of information:

1.  **Primary Dialect**: Identify and label the function's primary programming paradigm. Is it mainly using PySpark APIs, Pandas APIs, or is it pure Python logic?
2.  **Dependencies**: List all internal functions it calls and external packages it uses (e.g., `pyspark.sql.functions`, `pandas`, `numpy`).
3.  **Key Patterns**: Identify the most important code patterns or API calls that will be critical for migration (e.g., `pyspark.withColumn`, `pandas.apply`, `sklearn.fit`).

Please strictly return your analysis results in the specified JSON format without any additional explanations.

**Function to analyze:**
```python
{function_code}
```"""

            result = self._llm_call_with_error_handling("Function analysis", prompt, function_schema)
            results.append(result)

        return results

    def _determine_conversion_order(self, function_analysis: List[Dict[str, Any]]) -> List[str]:
        """Execute Stage 3: Determine Conversion Order."""
        # Extract dependencies for topological sorting
        dependencies_data = []
        for func_analysis in function_analysis:
            dependencies_data.append({
                "function_name": func_analysis.get("function_name", ""),
                "internal_dependencies": func_analysis.get("dependencies", {}).get("internal_functions", [])
            })

        dependencies_json = json.dumps(dependencies_data, indent=2)

        prompt = f"""You are a dependency analysis expert. Your task is to perform topological sorting based on the function dependency list provided below.

You need to return a function name array, ordered from "no dependencies" functions to "dependent" functions.

IMPORTANT: Only include functions that are defined within the script itself. Ignore external/imported function dependencies.

**Function Dependencies (JSON format):**
```json
{dependencies_json}
```"""

        order_schema = {
            "type": "object",
            "properties": {
                "conversion_order": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["conversion_order"]
        }

        result = self._llm_call_with_error_handling("Conversion order determination", prompt, order_schema)
        return result.get("conversion_order", [])