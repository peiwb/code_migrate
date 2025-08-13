"""
LLM Service JSON Debug Test Suite (test_llm_service_debug.py)

Comprehensive test suite specifically designed to debug JSON parsing issues
in the CortexLLMService. This suite captures detailed information about
Cortex API responses and parsing behavior.
"""

import pytest
import json
import traceback
from typing import Dict, Any, List
from services.llm_service import CortexLLMService, LLMServiceError


class TestLLMServiceJSONDebug:
    """
    Comprehensive test suite to debug LLM Service JSON parsing issues.
    Focuses on capturing detailed response information and parsing behavior.
    """

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance for testing"""
        try:
            service = CortexLLMService()
            print(f"\n✓ LLM Service created successfully")
            print(f"  Session type: {type(service.session)}")
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService: {e}")

    def print_detailed_response_info(self, response, test_name: str):
        """Helper method to print detailed response information"""
        print(f"\n{'='*80}")
        print(f"DETAILED RESPONSE ANALYSIS - {test_name}")
        print(f"{'='*80}")

        print(f"\n1. RESPONSE TYPE AND BASIC INFO:")
        print(f"   Type: {type(response)}")
        print(f"   String representation length: {len(str(response))}")
        print(f"   Is string: {isinstance(response, str)}")

        print(f"\n2. RESPONSE CONTENT (first 500 chars):")
        response_str = str(response)
        print(f"   '{response_str[:500]}'")
        if len(response_str) > 500:
            print(f"   ... (truncated, total length: {len(response_str)})")

        print(f"\n3. RESPONSE ATTRIBUTES:")
        if hasattr(response, '__dict__'):
            print(f"   Attributes: {list(response.__dict__.keys())}")
        else:
            print(f"   No __dict__ attribute")

        print(f"   Has 'structured_output': {hasattr(response, 'structured_output')}")

        if hasattr(response, 'structured_output'):
            print(f"   structured_output value: {response.structured_output}")
            print(f"   structured_output type: {type(response.structured_output)}")

            if response.structured_output:
                print(f"   structured_output length: {len(response.structured_output)}")
                if len(response.structured_output) > 0:
                    print(f"   structured_output[0]: {response.structured_output[0]}")
                    print(f"   structured_output[0] type: {type(response.structured_output[0])}")

                    if isinstance(response.structured_output[0], dict):
                        print(f"   structured_output[0] keys: {response.structured_output[0].keys()}")
                        if 'raw_message' in response.structured_output[0]:
                            print(f"   raw_message: {response.structured_output[0]['raw_message']}")
                            print(f"   raw_message type: {type(response.structured_output[0]['raw_message'])}")

        print(f"\n4. OTHER POSSIBLE ATTRIBUTES:")
        common_attrs = ['content', 'message', 'data', 'result', 'output', 'choices']
        for attr in common_attrs:
            if hasattr(response, attr):
                attr_value = getattr(response, attr)
                print(f"   {attr}: {attr_value} (type: {type(attr_value)})")

        print(f"\n{'='*80}")

    def test_simple_json_schema(self, llm_service):
        """Test 1: Simple JSON schema with basic request"""
        print(f"\n{'='*60}")
        print("TEST 1: SIMPLE JSON SCHEMA")
        print(f"{'='*60}")

        simple_prompt = "Please return a JSON object with a field called 'result' set to 'success'"

        simple_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"]
        }

        print(f"\nInput:")
        print(f"  Prompt: {simple_prompt}")
        print(f"  Schema: {json.dumps(simple_schema, indent=2)}")

        try:
            # Capture the raw response before any processing
            print(f"\nCalling get_json_completion...")
            result = llm_service.get_json_completion(simple_prompt, simple_schema)

            print(f"\n✓ SUCCESS - Parsed result:")
            print(f"  Result type: {type(result)}")
            print(f"  Result content: {result}")

            # Validate the result
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "result" in result, f"Expected 'result' key in response"

        except Exception as e:
            print(f"\n✗ FAILED with exception:")
            print(f"  Exception type: {type(e)}")
            print(f"  Exception message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            raise

    def test_array_json_schema(self, llm_service):
        """Test 2: Array JSON schema similar to knowledge service"""
        print(f"\n{'='*60}")
        print("TEST 2: ARRAY JSON SCHEMA")
        print(f"{'='*60}")

        array_prompt = """Please return a JSON object with an array of strings.
The array should contain 3 sample pattern IDs like: pattern.1, pattern.2, pattern.3"""

        array_schema = {
            "type": "object",
            "properties": {
                "pattern_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                }
            },
            "required": ["pattern_ids"]
        }

        print(f"\nInput:")
        print(f"  Prompt: {array_prompt}")
        print(f"  Schema: {json.dumps(array_schema, indent=2)}")

        try:
            result = llm_service.get_json_completion(array_prompt, array_schema)

            print(f"\n✓ SUCCESS - Parsed result:")
            print(f"  Result type: {type(result)}")
            print(f"  Result content: {result}")

            # Validate the result
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "pattern_ids" in result, f"Expected 'pattern_ids' key in response"
            assert isinstance(result["pattern_ids"], list), f"Expected list for pattern_ids"

        except Exception as e:
            print(f"\n✗ FAILED with exception:")
            print(f"  Exception type: {type(e)}")
            print(f"  Exception message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            raise

    def test_complex_knowledge_service_scenario(self, llm_service):
        """Test 3: Complex scenario similar to knowledge service usage"""
        print(f"\n{'='*60}")
        print("TEST 3: COMPLEX KNOWLEDGE SERVICE SCENARIO")
        print(f"{'='*60}")

        # Simulate the kind of prompt knowledge service would send
        complex_prompt = """You are an expert in migrating Python data processing code to Snowflake Snowpark.

Your task: Analyze the given Python functions and recommend the most relevant Snowpark recipes.

AVAILABLE SNOWPARK RECIPES:
ID: snowpark.session.table
Description: Creates a DataFrame from a table or view in Snowflake

ID: snowpark.session.sql  
Description: Executes a SQL query and returns the result as a DataFrame

ID: snowpark.dataframe.select
Description: Select specific columns from a DataFrame

PYTHON FUNCTIONS TO MIGRATE:
spark.table, DataFrame.select

Return the most relevant recipe IDs in this exact format:
{"recipe_ids": ["recipe_id_1", "recipe_id_2"]}
"""

        complex_schema = {
            "type": "object",
            "properties": {
                "recipe_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5
                }
            },
            "required": ["recipe_ids"]
        }

        print(f"\nInput:")
        print(f"  Prompt length: {len(complex_prompt)} characters")
        print(f"  Prompt preview (first 200 chars): {complex_prompt[:200]}...")
        print(f"  Schema: {json.dumps(complex_schema, indent=2)}")

        try:
            result = llm_service.get_json_completion(complex_prompt, complex_schema)

            print(f"\n✓ SUCCESS - Parsed result:")
            print(f"  Result type: {type(result)}")
            print(f"  Result content: {result}")

            # Validate the result
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "recipe_ids" in result, f"Expected 'recipe_ids' key in response"
            assert isinstance(result["recipe_ids"], list), f"Expected list for recipe_ids"

        except Exception as e:
            print(f"\n✗ FAILED with exception:")
            print(f"  Exception type: {type(e)}")
            print(f"  Exception message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            raise

    def test_raw_response_inspection(self, llm_service):
        """Test 4: Raw response inspection - modify LLM service temporarily"""
        print(f"\n{'='*60}")
        print("TEST 4: RAW RESPONSE INSPECTION")
        print(f"{'='*60}")

        # We'll monkey patch the LLM service to capture raw response
        original_get_json_completion = llm_service.get_json_completion

        def capture_raw_response(prompt: str, json_schema: dict) -> dict:
            """Modified version that captures raw response"""
            try:
                from snowflake.cortex import complete

                prompt_array = [{"role": "user", "content": prompt}]
                options = {
                    "response_format": {
                        "type": "json",
                        "schema": json_schema
                    }
                }

                print(f"\nCalling Cortex API with:")
                print(f"  Model: claude-3-5-sonnet")
                print(f"  Prompt array length: {len(prompt_array)}")
                print(f"  Options: {options}")

                response = complete(
                    model="claude-3-5-sonnet",
                    prompt=prompt_array,
                    options=options,
                    session=llm_service.session
                )

                # Detailed response inspection
                self.print_detailed_response_info(response, "RAW CORTEX RESPONSE")

                # Now try the original parsing logic step by step
                print(f"\nPARSING ATTEMPT 1: structured_output path")
                if hasattr(response, 'structured_output') and response.structured_output:
                    try:
                        result1 = response.structured_output[0]['raw_message']
                        print(f"  ✓ SUCCESS via structured_output: {result1}")
                        print(f"  Result type: {type(result1)}")
                        return result1
                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")

                print(f"\nPARSING ATTEMPT 2: string parsing path")
                if isinstance(response, str):
                    try:
                        result2 = json.loads(response)
                        print(f"  ✓ SUCCESS via string parsing: {result2}")
                        return result2
                    except Exception as e:
                        print(f"  ✗ FAILED: {e}")

                print(f"\nPARSING ATTEMPT 3: str() conversion path")
                try:
                    result3 = json.loads(str(response))
                    print(f"  ✓ SUCCESS via str() conversion: {result3}")
                    return result3
                except Exception as e:
                    print(f"  ✗ FAILED: {e}")

                raise LLMServiceError("All parsing methods failed")

            except Exception as e:
                print(f"\nCORTEX API CALL FAILED:")
                print(f"  Exception type: {type(e)}")
                print(f"  Exception message: {str(e)}")
                traceback.print_exc()
                raise LLMServiceError(f"Failed to get JSON completion: {e}")

        # Temporarily replace the method
        llm_service.get_json_completion = capture_raw_response

        test_prompt = "Return JSON with field 'test' set to 'inspection'"
        test_schema = {
            "type": "object",
            "properties": {"test": {"type": "string"}},
            "required": ["test"]
        }

        try:
            result = llm_service.get_json_completion(test_prompt, test_schema)
            print(f"\n✓ FINAL SUCCESS: {result}")
        except Exception as e:
            print(f"\n✗ FINAL FAILURE: {e}")
            raise
        finally:
            # Restore original method
            llm_service.get_json_completion = original_get_json_completion

    def test_edge_cases(self, llm_service):
        """Test 5: Edge cases that might cause issues"""
        print(f"\n{'='*60}")
        print("TEST 5: EDGE CASES")
        print(f"{'='*60}")

        edge_cases = [
            {
                "name": "Empty array request",
                "prompt": "Return an empty array of pattern_ids",
                "schema": {
                    "type": "object",
                    "properties": {"pattern_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["pattern_ids"]
                }
            },
            {
                "name": "Large array request",
                "prompt": "Return 10 pattern IDs named pattern.1 through pattern.10",
                "schema": {
                    "type": "object",
                    "properties": {"pattern_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 15}},
                    "required": ["pattern_ids"]
                }
            },
            {
                "name": "Special characters",
                "prompt": "Return pattern IDs with special characters like dots, underscores, hyphens",
                "schema": {
                    "type": "object",
                    "properties": {"pattern_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["pattern_ids"]
                }
            }
        ]

        for i, case in enumerate(edge_cases, 1):
            print(f"\n--- Edge Case {i}: {case['name']} ---")
            print(f"Prompt: {case['prompt']}")

            try:
                result = llm_service.get_json_completion(case['prompt'], case['schema'])
                print(f"✓ SUCCESS: {result}")
            except Exception as e:
                print(f"✗ FAILED: {type(e).__name__}: {e}")

    def test_comparison_with_text_completion(self, llm_service):
        """Test 6: Compare JSON vs text completion for same request"""
        print(f"\n{'='*60}")
        print("TEST 6: JSON VS TEXT COMPLETION COMPARISON")
        print(f"{'='*60}")

        base_prompt = """Please return pattern IDs for these Python functions: spark.table, DataFrame.select
Format as JSON: {"pattern_ids": ["id1", "id2"]}"""

        # Test text completion first
        print(f"\n--- TEXT COMPLETION TEST ---")
        try:
            text_result = llm_service.get_text_completion(base_prompt)
            print(f"✓ Text completion SUCCESS:")
            print(f"  Type: {type(text_result)}")
            print(f"  Length: {len(text_result)}")
            print(f"  Content: {text_result}")

            # Try to parse the text result as JSON
            try:
                parsed_text = json.loads(text_result)
                print(f"  ✓ Text result is valid JSON: {parsed_text}")
            except:
                print(f"  ✗ Text result is not valid JSON")

        except Exception as e:
            print(f"✗ Text completion FAILED: {e}")

        # Test JSON completion
        print(f"\n--- JSON COMPLETION TEST ---")
        json_schema = {
            "type": "object",
            "properties": {"pattern_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["pattern_ids"]
        }

        try:
            json_result = llm_service.get_json_completion(base_prompt, json_schema)
            print(f"✓ JSON completion SUCCESS:")
            print(f"  Type: {type(json_result)}")
            print(f"  Content: {json_result}")
        except Exception as e:
            print(f"✗ JSON completion FAILED: {e}")
            traceback.print_exc()

    def run_all_tests(self):
        """Run all tests in sequence for comprehensive debugging"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE LLM SERVICE JSON DEBUG SESSION")
        print(f"{'='*80}")

        try:
            llm_service = CortexLLMService()

            # Run tests in order of complexity
            self.test_simple_json_schema(llm_service)
            self.test_array_json_schema(llm_service)
            self.test_complex_knowledge_service_scenario(llm_service)
            self.test_raw_response_inspection(llm_service)
            self.test_edge_cases(llm_service)
            self.test_comparison_with_text_completion(llm_service)

            print(f"\n{'='*80}")
            print("ALL TESTS COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")

        except Exception as e:
            print(f"\n{'='*80}")
            print("TEST SUITE FAILED")
            print(f"Error: {e}")
            print(f"{'='*80}")


# Standalone execution for debugging
if __name__ == "__main__":
    test_instance = TestLLMServiceJSONDebug()
    test_instance.run_all_tests()