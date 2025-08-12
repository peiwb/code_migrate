"""
Targeted debugging for the specific issue in step 11
"""

import json
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService


def debug_step_11_issue(knowledge_base_path):
    """Debug the specific issue in step 11"""

    print("=== TARGETED DEBUG FOR STEP 11 ISSUE ===\n")

    try:
        # Create services
        llm_service = CortexLLMService()
        ks = KnowledgeService(knowledge_base_path, llm_service)

        print(f"✓ Services created successfully")
        print(f"✓ Loaded {len(ks.recipes)} recipes from knowledge base")

        # Build context and check size
        recipe_context = [
            {
                "id": recipe_id,
                "description": recipe.get("description", "")
            }
            for recipe_id, recipe in ks.recipes.items()
        ]

        context_json = json.dumps(recipe_context, indent=2)
        print(f"✓ Context built - {len(recipe_context)} recipes")
        print(f"✓ Context JSON size: {len(context_json)} characters")

        # Check individual recipe sizes
        print("\n--- Recipe Size Analysis ---")
        for i, context in enumerate(recipe_context[:5]):  # Show first 5
            desc_len = len(context['description'])
            print(f"Recipe {i + 1} ({context['id']}): description={desc_len} chars")
            if desc_len > 200:
                print(f"  ⚠ Long description: {context['description'][:100]}...")

        if len(recipe_context) > 5:
            print(f"... and {len(recipe_context) - 5} more recipes")

        # Build full prompt and check size
        keyword = "test"
        top_k = 2

        prompt = f"""
Given the following PySpark to Snowpark migration patterns and a user keyword,
identify the {top_k} most relevant pattern IDs that match the user's intent.

Available patterns:
{context_json}

User keyword: "{keyword}"

Please return the {top_k} most relevant pattern IDs in order of relevance.
"""

        print(f"\n--- Prompt Analysis ---")
        print(f"✓ Full prompt size: {len(prompt)} characters")
        print(f"✓ Prompt lines: {prompt.count(chr(10))} lines")

        # Check for API limits
        if len(prompt) > 100000:  # 100KB
            print("⚠ WARNING: Prompt is very large (>100KB)")
        elif len(prompt) > 50000:  # 50KB
            print("⚠ WARNING: Prompt is large (>50KB)")
        elif len(prompt) > 20000:  # 20KB
            print("⚠ CAUTION: Prompt is moderately large (>20KB)")
        else:
            print("✓ Prompt size seems reasonable")

        # Test with reduced context first
        print(f"\n--- Testing with Reduced Context ---")

        # Try with just 3 recipes
        small_context = recipe_context[:3]
        small_context_json = json.dumps(small_context, indent=2)

        small_prompt = f"""
Given the following PySpark to Snowpark migration patterns and a user keyword,
identify the {top_k} most relevant pattern IDs that match the user's intent.

Available patterns:
{small_context_json}

User keyword: "{keyword}"

Please return the {top_k} most relevant pattern IDs in order of relevance.
"""

        print(f"✓ Small prompt size: {len(small_prompt)} characters")

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

        print("Attempting small context test...")
        small_result = llm_service.get_json_completion(small_prompt, json_schema)
        print(f"✓ Small context test SUCCESSFUL: {small_result}")

        # If small context works, try progressively larger contexts
        print(f"\n--- Progressive Context Testing ---")

        context_sizes = [5, 10, 20, len(recipe_context)]

        for size in context_sizes:
            if size > len(recipe_context):
                size = len(recipe_context)

            print(f"\nTesting with {size} recipes...")

            test_context = recipe_context[:size]
            test_context_json = json.dumps(test_context, indent=2)

            test_prompt = f"""
Given the following PySpark to Snowpark migration patterns and a user keyword,
identify the {top_k} most relevant pattern IDs that match the user's intent.

Available patterns:
{test_context_json}

User keyword: "{keyword}"

Please return the {top_k} most relevant pattern IDs in order of relevance.
"""

            print(f"  Prompt size: {len(test_prompt)} characters")

            try:
                result = llm_service.get_json_completion(test_prompt, json_schema)
                print(f"  ✓ SUCCESS with {size} recipes: {result}")
            except Exception as e:
                print(f"  ✗ FAILED with {size} recipes: {e}")
                print(
                    f"  This suggests the limit is between {context_sizes[context_sizes.index(size) - 1] if context_sizes.index(size) > 0 else 0} and {size} recipes")
                break

            if size == len(recipe_context):
                break


    except Exception as e:
        print(f"✗ Debug failed: {e}")
        import traceback
        traceback.print_exc()


# Usage
if __name__ == "__main__":
    # Replace with your actual knowledge base path
    knowledge_base_path = "path/to/your/knowledge_base.json"
    debug_step_11_issue(knowledge_base_path)