"""
Knowledge Service Test Module (test_knowledge_service.py) - V1.0

This test module validates the complete functionality of KnowledgeService as a whole,
focusing on its ability to successfully load, parse, and query the knowledge_base.json
file, and provide accurate data through its public interfaces. Additionally, it aims
to validate its optional, LLM-based advanced discovery functionality.

Testing Philosophy (POC Phase):
- Deterministic testing: For core file loading and ID query functionality, tests must
  be completely deterministic and repeatable. This is achieved by using the real
  knowledge_base.json file.
- Real API calls: For the advanced discover_patterns functionality, tests execute real
  network requests to validate Prompt effectiveness.
- Structure and reasonableness validation: For LLM returns, our assertions focus on
  validating the correctness of data structure rather than exact text matching.
"""

import pytest
import json
import os
from typing import List, Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService, KnowledgeServiceError


class TestKnowledgeService:
    """KnowledgeService Test Class

    Contains all test cases for KnowledgeService, validating both deterministic
    file operations and LLM-based advanced functionality.
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

    def test_get_recipes_from_suggested_patterns(self, knowledge_service):
        """Test Case 3: Validate core recipe retrieval functionality

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
            first_real_id,            # existing ID
            "nonexistent.pattern",    # non-existent ID
            second_real_id,           # existing ID
            first_real_id,            # duplicate ID
            "another.nonexistent"     # another non-existent ID
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

    def test_discover_patterns_success(self, knowledge_service_with_llm):
        """Test Case 4: Validate advanced LLM-based discover_patterns functionality

        Verifies that the optional, LLM-based discover_patterns method can
        successfully return a properly structured list.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Call the discover_patterns method with test parameters
        result = knowledge_service_with_llm.discover_patterns(keyword="add a new column", top_k=2)

        # Verify return type and structure
        assert isinstance(result, list), "discover_patterns should return a list"
        assert len(result) <= 2, "Result should respect top_k parameter limit"

        # If list is not empty, verify all elements are strings
        if result:
            assert all(isinstance(item, str) for item in result), "All items in result should be strings"

    def test_discover_patterns_without_llm_service(self, knowledge_service):
        """Test Case: Validate NotImplementedError when LLM service is not provided

        Verifies that discover_patterns raises NotImplementedError when called
        without an LLM service.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        with pytest.raises(NotImplementedError):
            knowledge_service.discover_patterns(keyword="add a new column", top_k=2)

    def test_print_discover_patterns_for_inspection(self, knowledge_service_with_llm):
        """Test Case 5: Print discover_patterns results for manual inspection

        [For debugging and manual inspection] Runs the discover_patterns method
        and prints results to console for developers to carefully examine the
        quality of LLM-returned results.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Execute discover_patterns with test query
        result = knowledge_service_with_llm.discover_patterns(keyword="how to filter data based on a condition")

        # Verify result is not None
        assert result is not None, "discover_patterns result should not be None"

        # Print results in clear format for manual inspection
        print("\n" + "="*80)
        print("DISCOVER PATTERNS RESULT - MANUAL INSPECTION OUTPUT")
        print("="*80)

        print(f"\n--- Discovered patterns for 'how to filter data based on a condition' ---")
        print(json.dumps(result, indent=4))

        print("\n" + "-"*80)
        print("DISCOVER PATTERNS PRINTING COMPLETED")
        print("-"*80 + "\n")

    def test_knowledge_service_handles_empty_patterns_list(self, knowledge_service):
        """Additional Test Case: Validate handling of empty pattern list

        Ensures that get_recipes_from_suggested_patterns handles empty input
        lists gracefully.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        # Test with empty list
        empty_patterns = []

        result = knowledge_service.get_recipes_from_suggested_patterns(empty_patterns)

        # Verify appropriate handling
        assert isinstance(result, list), "Should return list for empty input"
        assert len(result) == 0, "Should return empty list for empty input"

    def test_knowledge_service_handles_all_invalid_patterns(self, knowledge_service):
        """Additional Test Case: Validate handling of all invalid pattern IDs

        Ensures that get_recipes_from_suggested_patterns handles cases where
        none of the provided IDs exist in the knowledge base.

        Args:
            knowledge_service: KnowledgeService instance without LLM
        """
        # Test with only non-existent IDs
        invalid_patterns = ["nonexistent.id1", "invalid.pattern", "missing.recipe"]

        result = knowledge_service.get_recipes_from_suggested_patterns(invalid_patterns)

        # Verify appropriate handling
        assert isinstance(result, list), "Should return list for invalid input"
        assert len(result) == 0, "Should return empty list when no valid IDs found"

    def test_knowledge_service_recipe_structure(self, knowledge_service):
        """Additional Test Case: Validate structure of individual recipes

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

    def test_discover_patterns_with_different_keywords(self, knowledge_service_with_llm):
        """Additional Test Case: Test discover_patterns with various keywords

        Validates that discover_patterns can handle different types of queries
        and keywords appropriately.

        Args:
            knowledge_service_with_llm: KnowledgeService instance with LLM capabilities
        """
        # Test different keyword types
        test_keywords = [
            "group data",
            "aggregation",
            "transform columns"
        ]

        for keyword in test_keywords:
            result = knowledge_service_with_llm.discover_patterns(keyword=keyword, top_k=1)

            # Basic validation for each keyword
            assert isinstance(result, list), f"Should return list for keyword: {keyword}"
            assert len(result) <= 1, f"Should respect top_k=1 for keyword: {keyword}"