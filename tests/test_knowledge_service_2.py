"""
Knowledge Service Test Module (test_knowledge_service.py) - Enhanced Version

This test module validates the complete functionality of the enhanced KnowledgeService,
focusing on its ability to intelligently match Python functions to Snowpark recipes
using LLM-based analysis.

Testing Philosophy (POC Phase):
- Deterministic testing: For core file loading and ID query functionality
- Real API calls: For the advanced discover_patterns functionality with Python functions
- Structure and reasonableness validation: For LLM returns focusing on data structure
- Knowledge base inspection: Output all available recipes for manual review
"""

import pytest
import json
import os
from typing import List, Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService, KnowledgeServiceError


class TestEnhancedKnowledgeService:
    """Enhanced KnowledgeService Test Class

    Contains all test cases for the enhanced KnowledgeService, validating both deterministic
    file operations and LLM-based Python function analysis functionality.
    """

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance

        Scope: session - created only once for the entire test session

        Returns:
            CortexLLMService: Configured LLM service instance

        Raises:
            Exception: If credentials configuration is invalid
        """
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService instance. Please check credentials: {e}")

    @pytest.fixture(scope="function")
    def knowledge_base_file(self):
        """Get path to the real knowledge base JSON file

        Returns the path to the actual knowledge_base.json file used in production.
        This ensures tests validate against real data structure and content.

        Returns:
            str: Path to the real knowledge_base.json file
        """
        # Get the directory where this test file is located
        test_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root, then into data directory
        project_root = os.path.dirname(test_dir)
        knowledge_base_path = os.path.join(project_root, "data", "knowledge_base.json")

        # Verify file exists
        if not os.path.exists(knowledge_base_path):
            pytest.fail(f"Real knowledge base file not found at: {knowledge_base_path}")

        return knowledge_base_path

    @pytest.fixture(scope="function")
    def knowledge_service(self, knowledge_base_file):
        """Create KnowledgeService instance without LLM service

        Args:
            knowledge_base_file: Path to knowledge base file

        Returns:
            KnowledgeService: KnowledgeService instance without LLM capabilities
        """
        return KnowledgeService(knowledge_base_path=knowledge_base_file)

    @pytest.fixture(scope="function")
    def knowledge_service_with_llm(self, knowledge_base_file, llm_service):
        """Create fully-featured KnowledgeService instance with LLM service

        Args:
            knowledge_base_file: Path to knowledge base file
            llm_service: CortexLLMService instance

        Returns:
            KnowledgeService: Fully-featured KnowledgeService instance with LLM capabilities
        """
        return KnowledgeService(knowledge_base_path=knowledge_base_file, llm_service=llm_service)

    def test_initialization_success(self, knowledge_service):
        """Test Case 1: Validate successful KnowledgeService initialization

        Verifies that KnowledgeService can successfully load a properly formatted
        JSON file and correctly index the data.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        # Verify internal state after successful initialization
        assert knowledge_service.recipes is not None, "Recipes should not be None after initialization"
        assert isinstance(knowledge_service.recipes, dict), "Recipes should be a dictionary"

        # Verify that we have loaded some recipes
        assert len(knowledge_service.recipes) > 0, "Should have loaded at least one recipe"

        # Verify the structure of loaded recipes using first available recipe
        first_recipe_id = next(iter(knowledge_service.recipes))
        first_recipe = knowledge_service.recipes[first_recipe_id]
        assert isinstance(first_recipe, dict), "Individual recipe should be a dictionary"
        assert "id" in first_recipe, "Recipe should contain 'id' field"
        assert "description" in first_recipe, "Recipe should contain 'description' field"

    def test_initialization_file_not_found(self):
        """Test Case 2: Validate proper error handling for non-existent files

        Verifies that when provided with a non-existent file path, KnowledgeService
        throws the expected KnowledgeServiceError.
        """
        # Test with non-existent file path
        non_existent_path = "/path/that/does/not/exist/knowledge_base.json"

        # Use pytest.raises context manager to verify expected exception
        with pytest.raises(KnowledgeServiceError):
            KnowledgeService(knowledge_base_path=non_existent_path)

    def test_print_all_available_recipes(self, knowledge_service):
        """Test Case 3: Output all available recipes for manual inspection

        Prints all recipe IDs and descriptions from the knowledge base for manual review.
        This helps understand what Snowpark capabilities are available.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        print("\n" + "=" * 100)
        print("KNOWLEDGE BASE CONTENT - ALL AVAILABLE SNOWPARK RECIPES")
        print("=" * 100)

        total_recipes = len(knowledge_service.recipes)
        print(f"\nTotal Recipes Available: {total_recipes}")
        print("-" * 50)

        for i, (recipe_id, recipe) in enumerate(knowledge_service.recipes.items(), 1):
            print(f"\n[{i:2d}] Recipe ID: {recipe_id}")
            print(f"     Description: {recipe.get('description', 'No description available')}")

            # Also show usage context if available
            usage_context = recipe.get('usage_context', '')
            if usage_context:
                print(f"     Usage Context: {usage_context}")

        print("\n" + "=" * 100)
        print("KNOWLEDGE BASE CONTENT INSPECTION COMPLETED")
        print("=" * 100 + "\n")

        # Basic validation
        assert total_recipes > 0, "Should have at least one recipe in knowledge base"

    def test_get_recipes_from_suggested_patterns(self, knowledge_service):
        """Test Case 4: Validate core recipe retrieval functionality

        Verifies that the core method get_recipes_from_suggested_patterns can
        accurately return recipes based on a list of IDs, handling existing IDs,
        non-existent IDs, and duplicate IDs appropriately.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        # Get available recipe IDs from the real knowledge base
        available_ids = list(knowledge_service.recipes.keys())

        if len(available_ids) < 2:
            pytest.skip("Need at least 2 recipes in knowledge base for this test")

        # Use real IDs from the knowledge base
        first_real_id = available_ids[0]
        second_real_id = available_ids[1]

        # Define test input list containing existing IDs, non-existent IDs, and duplicates
        test_pattern_ids = [
            first_real_id,  # existing ID
            "nonexistent.pattern",  # non-existent ID
            second_real_id,  # existing ID
            first_real_id,  # duplicate ID
            "another.nonexistent"  # another non-existent ID
        ]

        # Call the method under test
        recipes = knowledge_service.get_recipes_from_suggested_patterns(test_pattern_ids)

        # Verify return type and structure
        assert isinstance(recipes, list), "Returned recipes should be a list"

        # Should return 3 recipes: first_real_id (twice) + second_real_id (once)
        assert len(recipes) == 3, "Should return recipes for all valid IDs including duplicates"

        # Verify that returned recipes match expected IDs
        returned_ids = [recipe["id"] for recipe in recipes if "id" in recipe]
        assert returned_ids.count(first_real_id) == 2, f"Should include {first_real_id} recipe twice"
        assert second_real_id in returned_ids, f"Should include {second_real_id} recipe"

        # Verify that first recipe matches first valid ID in input
        assert recipes[0]["id"] == first_real_id, "First recipe should match first valid ID"

    def test_discover_patterns_without_llm_service(self, knowledge_service):
        """Test Case 5: Validate NotImplementedError when LLM service is not provided

        Verifies that discover_patterns raises NotImplementedError when called
        without an LLM service.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        test_python_functions = ["pandas.read_csv", "DataFrame.groupby"]

        with pytest.raises(NotImplementedError):
            knowledge_service.discover_patterns(python_functions=test_python_functions)

    def test_discover_patterns_with_pyspark_functions(self, knowledge_service_with_llm):
        """Test Case 6: Test discover_patterns with PySpark functions

        Validates that discover_patterns can handle PySpark function analysis.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with PySpark functions
        pyspark_functions = ["spark.table", "DataFrame.select", "DataFrame.groupBy"]

        result = knowledge_service_with_llm.discover_patterns(
            python_functions=pyspark_functions,
            top_k=5
        )

        # Verify return type and structure
        assert isinstance(result, list), "discover_patterns should return a list"
        assert len(result) <= 5, "Result should respect top_k parameter limit"

        # If list is not empty, verify all elements are strings
        if result:
            assert all(isinstance(item, str) for item in result), "All items in result should be strings"

        # Print results for inspection
        print(f"\n--- PySpark Functions Analysis ---")
        print(f"Input: {pyspark_functions}")
        print(f"Recommended Snowpark Recipe IDs: {result}")

    def test_discover_patterns_with_pandas_functions(self, knowledge_service_with_llm):
        """Test Case 7: Test discover_patterns with Pandas functions

        Validates that discover_patterns can handle Pandas function analysis.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with Pandas functions
        pandas_functions = ["pandas.read_csv", "DataFrame.groupby", "DataFrame.merge"]

        result = knowledge_service_with_llm.discover_patterns(
            python_functions=pandas_functions,
            top_k=4
        )

        # Verify return type and structure
        assert isinstance(result, list), "discover_patterns should return a list"
        assert len(result) <= 4, "Result should respect top_k parameter limit"

        # If list is not empty, verify all elements are strings
        if result:
            assert all(isinstance(item, str) for item in result), "All items in result should be strings"

        # Print results for inspection
        print(f"\n--- Pandas Functions Analysis ---")
        print(f"Input: {pandas_functions}")
        print(f"Recommended Snowpark Recipe IDs: {result}")

    def test_discover_patterns_with_mixed_functions(self, knowledge_service_with_llm):
        """Test Case 8: Test discover_patterns with mixed Python functions

        Validates that discover_patterns can handle a mix of PySpark, Pandas, and native Python functions.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with mixed functions
        mixed_functions = [
            "spark.sql",  # PySpark
            "pandas.read_csv",  # Pandas
            "DataFrame.join",  # Could be either PySpark or Pandas
            "numpy.mean",  # NumPy
            "session.table"  # Already looks like Snowpark
        ]

        result = knowledge_service_with_llm.discover_patterns(
            python_functions=mixed_functions
        )

        # Verify return type and structure
        assert isinstance(result, list), "discover_patterns should return a list"

        # With adaptive top_k, should be max(3, min(12, 5*2)) = 10
        assert len(result) <= 10, "Result should respect adaptive top_k limit"

        # Print results for inspection
        print(f"\n--- Mixed Functions Analysis ---")
        print(f"Input: {mixed_functions}")
        print(f"Recommended Snowpark Recipe IDs: {result}")

    def test_discover_patterns_adaptive_top_k(self, knowledge_service_with_llm):
        """Test Case 9: Test adaptive top_k functionality

        Validates that the adaptive top_k strategy works correctly for different input sizes.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with single function (should get top_k = 3)
        single_function = ["DataFrame.filter"]
        result_single = knowledge_service_with_llm.discover_patterns(python_functions=single_function)
        assert len(result_single) <= 3, "Single function should use adaptive top_k = 3"

        # Test with many functions (should get top_k = 12, capped)
        many_functions = [f"function_{i}" for i in range(10)]
        result_many = knowledge_service_with_llm.discover_patterns(python_functions=many_functions)
        assert len(result_many) <= 12, "Many functions should use adaptive top_k = 12 (capped)"

        print(f"\n--- Adaptive Top-K Testing ---")
        print(f"Single function result count: {len(result_single)} (expected: ≤3)")
        print(f"Many functions result count: {len(result_many)} (expected: ≤12)")

    def test_complete_workflow_inspection(self, knowledge_service_with_llm):
        """Test Case 10: Complete workflow inspection with detailed output

        Runs a complete workflow from Python functions to final recipes and prints
        detailed output for manual inspection of the entire process.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test a realistic scenario
        python_functions = [
            "spark.table",
            "DataFrame.select",
            "DataFrame.filter",
            "DataFrame.groupBy",
            "DataFrame.agg"
        ]

        print("\n" + "=" * 80)
        print("COMPLETE WORKFLOW INSPECTION")
        print("=" * 80)

        print(f"\nStep 1: Input Python Functions")
        print(f"Functions to migrate: {python_functions}")

        # Step 1: Discover patterns
        print(f"\nStep 2: LLM Analysis - Discovering Snowpark patterns...")
        pattern_ids = knowledge_service_with_llm.discover_patterns(
            python_functions=python_functions,
            top_k=6
        )

        print(f"Discovered Recipe IDs: {pattern_ids}")

        # Step 2: Get complete recipes
        print(f"\nStep 3: Retrieving complete recipes...")
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        print(f"Retrieved {len(recipes)} complete recipes")

        # Step 3: Display detailed recipes
        print(f"\nStep 4: Complete Recipe Details:")
        print("-" * 50)

        for i, recipe in enumerate(recipes, 1):
            print(f"\n[{i}] Recipe ID: {recipe.get('id', 'Unknown')}")
            print(f"    Description: {recipe.get('description', 'No description')}")

            # Show code snippet if available
            code_snippet = recipe.get('code_snippet', '')
            if code_snippet:
                print(f"    Code: {code_snippet}")

            # Show usage context if available
            usage_context = recipe.get('usage_context', '')
            if usage_context:
                print(f"    Usage: {usage_context}")

        print("\n" + "=" * 80)
        print("COMPLETE WORKFLOW INSPECTION COMPLETED")
        print("=" * 80 + "\n")

        # Validate the workflow
        assert len(pattern_ids) > 0, "Should discover at least one pattern"
        assert len(recipes) > 0, "Should retrieve at least one recipe"
        assert len(recipes) == len(pattern_ids), "Should get one recipe per pattern ID"

    def test_knowledge_service_handles_empty_functions_list(self, knowledge_service_with_llm):
        """Test Case 11: Validate handling of empty Python functions list

        Ensures that discover_patterns handles empty input lists gracefully.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with empty list
        empty_functions = []

        result = knowledge_service_with_llm.discover_patterns(python_functions=empty_functions)

        # Verify appropriate handling
        assert isinstance(result, list), "Should return list for empty input"

    def test_knowledge_service_handles_unknown_functions(self, knowledge_service_with_llm):
        """Test Case 12: Validate handling of unknown/invalid Python functions

        Tests the system's ability to handle completely unknown function names.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test with completely unknown functions
        unknown_functions = ["totally.unknown.function", "made.up.api", "nonexistent.method"]

        result = knowledge_service_with_llm.discover_patterns(python_functions=unknown_functions)

        # Should still return a list (might be empty or contain reasonable fallbacks)
        assert isinstance(result, list), "Should return list even for unknown functions"

        print(f"\n--- Unknown Functions Test ---")
        print(f"Input unknown functions: {unknown_functions}")
        print(f"LLM response: {result}")

    def test_recipe_structure_validation(self, knowledge_service):
        """Test Case 13: Validate structure of individual recipes

        Verifies that recipes returned by get_recipes_from_suggested_patterns
        have the expected structure and required fields.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        # Get available recipe IDs from the real knowledge base
        available_ids = list(knowledge_service.recipes.keys())

        if len(available_ids) == 0:
            pytest.skip("No recipes available in knowledge base for this test")

        # Get a real recipe
        test_id = available_ids[0]
        recipes = knowledge_service.get_recipes_from_suggested_patterns([test_id])

        assert len(recipes) == 1, "Should return exactly one recipe"

        recipe = recipes[0]

        # Verify essential fields that should exist in any recipe
        essential_fields = ["id", "description"]
        for field in essential_fields:
            assert field in recipe, f"Recipe should contain '{field}' field"

        # Verify field types
        assert isinstance(recipe["id"], str), "Recipe id should be string"
        assert isinstance(recipe["description"], str), "Recipe description should be string"

    def test_output_get_recipes_from_suggested_patterns_results(self, knowledge_service):
        """Test Case 14: Output detailed results from get_recipes_from_suggested_patterns

        This test case is specifically designed to inspect and output the complete
        results from get_recipes_from_suggested_patterns to understand the data structure
        and content returned by this method.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        print("\n" + "=" * 120)
        print("GET_RECIPES_FROM_SUGGESTED_PATTERNS DETAILED OUTPUT INSPECTION")
        print("=" * 120)

        # Get available recipe IDs from the real knowledge base
        available_ids = list(knowledge_service.recipes.keys())

        if len(available_ids) == 0:
            print("\nNo recipes available in knowledge base for this test")
            pytest.skip("No recipes available in knowledge base for this test")

        # Test with different scenarios to see various outputs
        test_scenarios = [
            {
                "name": "Single Valid Recipe ID",
                "pattern_ids": [available_ids[0]] if len(available_ids) > 0 else []
            },
            {
                "name": "Multiple Valid Recipe IDs",
                "pattern_ids": available_ids[:3] if len(available_ids) >= 3 else available_ids[:len(available_ids)]
            },
            {
                "name": "Mix of Valid and Invalid IDs",
                "pattern_ids": [
                    available_ids[0] if len(available_ids) > 0 else "valid.id",
                    "invalid.recipe.id",
                    available_ids[1] if len(available_ids) > 1 else "another.valid.id"
                ]
            },
            {
                "name": "Only Invalid IDs",
                "pattern_ids": ["completely.invalid.id", "another.invalid.id"]
            },
            {
                "name": "Empty List",
                "pattern_ids": []
            }
        ]

        for scenario in test_scenarios:
            print(f"\n{'-' * 80}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"{'-' * 80}")

            pattern_ids = scenario['pattern_ids']
            print(f"\nInput Pattern IDs: {pattern_ids}")

            try:
                # Call the method under test
                recipes = knowledge_service.get_recipes_from_suggested_patterns(pattern_ids)

                # Output basic information
                print(f"\nResult Type: {type(recipes)}")
                print(f"Result Length: {len(recipes)}")

                if len(recipes) == 0:
                    print("\nNo recipes returned (empty list)")
                else:
                    print(f"\nReturned {len(recipes)} recipe(s):")

                    # Output detailed information for each recipe
                    for i, recipe in enumerate(recipes, 1):
                        print(f"\n  [{i}] Recipe Details:")
                        print(f"      Type: {type(recipe)}")

                        if isinstance(recipe, dict):
                            # Show all available keys
                            print(f"      Available Keys: {list(recipe.keys())}")

                            # Show detailed content for each key
                            for key, value in recipe.items():
                                value_type = type(value).__name__

                                # Truncate long values for readability
                                if isinstance(value, str) and len(value) > 100:
                                    display_value = f"{value[:100]}... (truncated, full length: {len(value)} chars)"
                                else:
                                    display_value = value

                                print(f"      {key:15} ({value_type:8}): {display_value}")
                        else:
                            print(f"      Content: {recipe}")

                # Show JSON representation for complete structure visibility
                print(f"\nComplete JSON Structure:")
                try:
                    json_output = json.dumps(recipes, indent=2, ensure_ascii=False)
                    # Truncate if too long
                    if len(json_output) > 2000:
                        print(f"{json_output[:2000]}... (truncated, full length: {len(json_output)} chars)")
                    else:
                        print(json_output)
                except (TypeError, ValueError) as e:
                    print(f"Could not serialize to JSON: {e}")
                    print(f"Raw object: {recipes}")

            except Exception as e:
                print(f"\nERROR occurred during execution: {e}")
                print(f"Error type: {type(e).__name__}")

        print(f"\n{'=' * 120}")
        print("GET_RECIPES_FROM_SUGGESTED_PATTERNS DETAILED OUTPUT INSPECTION COMPLETED")
        print(f"{'=' * 120}\n")

        # Basic validation (at least one scenario should work)
        basic_test_ids = [available_ids[0]] if len(available_ids) > 0 else []
        if basic_test_ids:
            basic_recipes = knowledge_service.get_recipes_from_suggested_patterns(basic_test_ids)
            assert isinstance(basic_recipes, list), "Should always return a list"