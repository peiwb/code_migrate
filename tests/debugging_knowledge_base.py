"""
Comprehensive test suite for Knowledge Service debugging
This test suite helps identify exactly where the problem occurs in the discover_patterns method.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from services.llm_service import CortexLLMService, LLMServiceError
from services.knowledge_service import KnowledgeService, KnowledgeServiceError


class TestKnowledgeServiceDebug:
    """Comprehensive test suite to debug Knowledge Service issues"""

    @pytest.fixture
    def sample_knowledge_base(self):
        """Create a sample knowledge base for testing"""
        return [
            {
                "id": "test.pattern.1",
                "description": "Test pattern 1 for basic operations",
                "usage_context": "Used for simple data loading"
            },
            {
                "id": "test.pattern.2",
                "description": "Test pattern 2 for advanced operations",
                "usage_context": "Used for complex transformations"
            },
            {
                "id": "test.pattern.3",
                "description": "Test pattern 3 for filtering operations",
                "usage_context": "Used for data filtering and selection"
            }
        ]

    @pytest.fixture
    def temp_knowledge_file(self, sample_knowledge_base):
        """Create a temporary knowledge base file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_knowledge_base, f)
            temp_file_path = f.name

        yield temp_file_path

        # Cleanup
        os.unlink(temp_file_path)

    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service for controlled testing"""
        mock_service = Mock(spec=CortexLLMService)
        return mock_service

    @pytest.fixture
    def knowledge_service_without_llm(self, temp_knowledge_file):
        """Create knowledge service without LLM for basic testing"""
        return KnowledgeService(temp_knowledge_file)

    @pytest.fixture
    def knowledge_service_with_mock_llm(self, temp_knowledge_file, mock_llm_service):
        """Create knowledge service with mock LLM"""
        return KnowledgeService(temp_knowledge_file, mock_llm_service)

    def test_knowledge_base_loading(self, knowledge_service_without_llm):
        """Test 1: Verify knowledge base loads correctly"""
        print("\n=== Test 1: Knowledge Base Loading ===")

        ks = knowledge_service_without_llm
        assert len(ks.recipes) == 3
        assert "test.pattern.1" in ks.recipes
        assert "test.pattern.2" in ks.recipes
        assert "test.pattern.3" in ks.recipes

        print("✓ Knowledge base loaded successfully")
        print(f"✓ Loaded {len(ks.recipes)} recipes")
        for recipe_id in ks.recipes.keys():
            print(f"  - {recipe_id}")

    def test_get_recipes_from_suggested_patterns(self, knowledge_service_without_llm):
        """Test 2: Verify recipe retrieval works"""
        print("\n=== Test 2: Recipe Retrieval ===")

        ks = knowledge_service_without_llm
        pattern_ids = ["test.pattern.1", "test.pattern.3"]
        recipes = ks.get_recipes_from_suggested_patterns(pattern_ids)

        assert len(recipes) == 2
        assert recipes[0]["id"] == "test.pattern.1"
        assert recipes[1]["id"] == "test.pattern.3"

        print("✓ Recipe retrieval working correctly")
        print(f"✓ Retrieved {len(recipes)} recipes for pattern IDs: {pattern_ids}")

    def test_discover_patterns_without_llm(self, knowledge_service_without_llm):
        """Test 3: Verify discover_patterns fails gracefully without LLM"""
        print("\n=== Test 3: Discover Patterns Without LLM ===")

        ks = knowledge_service_without_llm

        with pytest.raises(NotImplementedError) as exc_info:
            ks.discover_patterns("test keyword")

        assert "Semantic pattern discovery requires an LLM service" in str(exc_info.value)
        print("✓ Correctly raises NotImplementedError without LLM service")

    def test_recipe_context_building(self, knowledge_service_with_mock_llm):
        """Test 4: Verify recipe context is built correctly"""
        print("\n=== Test 4: Recipe Context Building ===")

        ks = knowledge_service_with_mock_llm

        # Build context manually to inspect
        recipe_context = [
            {
                "id": recipe_id,
                "description": recipe.get("description", "")
            }
            for recipe_id, recipe in ks.recipes.items()
        ]

        assert len(recipe_context) == 3

        print("✓ Recipe context built successfully")
        print("Context structure:")
        for i, context in enumerate(recipe_context):
            print(f"  {i + 1}. ID: {context['id']}")
            print(f"     Description: {context['description'][:50]}...")
            print(f"     Description Length: {len(context['description'])}")

    def test_prompt_generation(self, knowledge_service_with_mock_llm):
        """Test 5: Test prompt generation and length"""
        print("\n=== Test 5: Prompt Generation ===")

        ks = knowledge_service_with_mock_llm
        keyword = "test keyword"
        top_k = 2

        # Build context
        recipe_context = [
            {
                "id": recipe_id,
                "description": recipe.get("description", "")
            }
            for recipe_id, recipe in ks.recipes.items()
        ]

        prompt = f"""
        Given the following PySpark to Snowpark migration patterns and a user keyword,
        identify the {top_k} most relevant pattern IDs that match the user's intent.

        Available patterns:
        {json.dumps(recipe_context, indent=2)}

        User keyword: "{keyword}"

        Please return the {top_k} most relevant pattern IDs in order of relevance.
        """

        print(f"✓ Prompt generated successfully")
        print(f"✓ Prompt length: {len(prompt)} characters")
        print(f"✓ Prompt preview (first 200 chars):")
        print(f"   {prompt[:200]}...")

        # Check if prompt is too long (typical API limits)
        if len(prompt) > 10000:
            print(f"⚠ WARNING: Prompt might be too long ({len(prompt)} chars)")
        else:
            print(f"✓ Prompt length is reasonable ({len(prompt)} chars)")

    def test_json_schema_generation(self, knowledge_service_with_mock_llm):
        """Test 6: Test JSON schema generation"""
        print("\n=== Test 6: JSON Schema Generation ===")

        top_k = 2
        json_schema = {
            "type": "object",
            "properties": {
                "pattern_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": top_k
                }
            },
            "required": ["pattern_ids"]
        }

        print("✓ JSON Schema generated successfully")
        print("Schema structure:")
        print(json.dumps(json_schema, indent=2))

    def test_mock_llm_successful_call(self, knowledge_service_with_mock_llm):
        """Test 7: Test with successful mock LLM response"""
        print("\n=== Test 7: Mock LLM Successful Call ===")

        ks = knowledge_service_with_mock_llm

        # Configure mock to return successful response
        expected_response = {"pattern_ids": ["test.pattern.1", "test.pattern.2"]}
        ks.llm_service.get_json_completion.return_value = expected_response

        result = ks.discover_patterns("test keyword", top_k=2)

        assert result == ["test.pattern.1", "test.pattern.2"]
        assert ks.llm_service.get_json_completion.called

        print("✓ Mock LLM call successful")
        print(f"✓ Returned pattern IDs: {result}")

    def test_mock_llm_error_handling(self, knowledge_service_with_mock_llm):
        """Test 8: Test LLM error handling"""
        print("\n=== Test 8: Mock LLM Error Handling ===")

        ks = knowledge_service_with_mock_llm

        # Configure mock to raise an exception
        ks.llm_service.get_json_completion.side_effect = Exception("Mock API Error")

        with pytest.raises(KnowledgeServiceError) as exc_info:
            ks.discover_patterns("test keyword")

        assert "Semantic discovery failed" in str(exc_info.value)
        print("✓ Error handling working correctly")
        print(f"✓ Exception message: {exc_info.value}")

    def test_real_llm_service_instantiation(self):
        """Test 9: Test real LLM service can be instantiated"""
        print("\n=== Test 9: Real LLM Service Instantiation ===")

        try:
            from services.llm_service import CortexLLMService
            llm_service = CortexLLMService()
            print("✓ Real LLM service instantiated successfully")
            print(f"✓ Session type: {type(llm_service.session)}")
            return llm_service
        except Exception as e:
            print(f"✗ Failed to instantiate real LLM service: {e}")
            pytest.skip(f"Cannot test with real LLM service: {e}")

    def test_real_llm_simple_call(self):
        """Test 10: Test simple call to real LLM service"""
        print("\n=== Test 10: Real LLM Simple Call ===")

        try:
            llm_service = self.test_real_llm_service_instantiation()

            # Simple schema for testing
            simple_schema = {
                "type": "object",
                "properties": {
                    "test_field": {"type": "string"}
                },
                "required": ["test_field"]
            }

            result = llm_service.get_json_completion(
                "Please return a JSON with test_field set to 'success'",
                simple_schema
            )

            print("✓ Real LLM simple call successful")
            print(f"✓ Response: {result}")

        except Exception as e:
            print(f"✗ Real LLM simple call failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

    def test_real_llm_with_actual_knowledge_base(self, temp_knowledge_file):
        """Test 11: Test with real LLM and actual knowledge base"""
        print("\n=== Test 11: Real LLM with Actual Knowledge Base ===")

        try:
            llm_service = self.test_real_llm_service_instantiation()
            ks = KnowledgeService(temp_knowledge_file, llm_service)

            result = ks.discover_patterns("data loading", top_k=2)

            print("✓ Real LLM with knowledge base successful")
            print(f"✓ Discovered patterns: {result}")

        except Exception as e:
            print(f"✗ Real LLM with knowledge base failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()

    def test_step_by_step_debug(self, temp_knowledge_file):
        """Test 12: Step-by-step debugging of the actual issue"""
        print("\n=== Test 12: Step-by-Step Debug ===")

        try:
            # Step 1: Create LLM service
            print("Step 1: Creating LLM service...")
            llm_service = CortexLLMService()
            print("✓ LLM service created")

            # Step 2: Create knowledge service
            print("Step 2: Creating knowledge service...")
            ks = KnowledgeService(temp_knowledge_file, llm_service)
            print("✓ Knowledge service created")

            # Step 3: Build context
            print("Step 3: Building recipe context...")
            recipe_context = [
                {
                    "id": recipe_id,
                    "description": recipe.get("description", "")
                }
                for recipe_id, recipe in ks.recipes.items()
            ]
            print(f"✓ Context built with {len(recipe_context)} recipes")

            # Step 4: Build prompt
            print("Step 4: Building prompt...")
            keyword = "test"
            top_k = 2
            prompt = f"""
Given the following PySpark to Snowpark migration patterns and a user keyword,
identify the {top_k} most relevant pattern IDs that match the user's intent.

Available patterns:
{json.dumps(recipe_context, indent=2)}

User keyword: "{keyword}"

Please return the {top_k} most relevant pattern IDs in order of relevance.
"""
            print(f"✓ Prompt built (length: {len(prompt)})")

            # Step 5: Build schema
            print("Step 5: Building JSON schema...")
            json_schema = {
                "type": "object",
                "properties": {
                    "pattern_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": top_k
                    }
                },
                "required": ["pattern_ids"]
            }
            print("✓ Schema built")

            # Step 6: Call LLM
            print("Step 6: Calling LLM service...")
            response = llm_service.get_json_completion(prompt, json_schema)
            print(f"✓ LLM call successful: {response}")

        except Exception as e:
            print(f"✗ Step-by-step debug failed at: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()


def run_comprehensive_debug():
    """Run all tests in sequence for comprehensive debugging"""
    print("=" * 60)
    print("COMPREHENSIVE KNOWLEDGE SERVICE DEBUG SESSION")
    print("=" * 60)

    # Create test instance
    test_instance = TestKnowledgeServiceDebug()

    # You can run individual tests or all of them
    # Uncomment the ones you want to run:

    # Basic functionality tests
    # test_instance.test_knowledge_base_loading()
    # test_instance.test_get_recipes_from_suggested_patterns()
    # test_instance.test_discover_patterns_without_llm()

    # Advanced debugging tests
    # test_instance.test_recipe_context_building()
    # test_instance.test_prompt_generation()
    # test_instance.test_json_schema_generation()

    # Mock LLM tests
    # test_instance.test_mock_llm_successful_call()
    # test_instance.test_mock_llm_error_handling()

    # Real LLM tests (these will show the actual error)
    # test_instance.test_real_llm_service_instantiation()
    # test_instance.test_real_llm_simple_call()
    # test_instance.test_real_llm_with_actual_knowledge_base()

    # The most comprehensive test
    # test_instance.test_step_by_step_debug()

    print("\n" + "=" * 60)
    print("DEBUG SESSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Run the debug session
    run_comprehensive_debug()

    # Or run with pytest:
    # pytest -v test_knowledge_service_debug.py -s