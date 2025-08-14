"""
Realistic Scenario Test for Knowledge Service (test_realistic_scenarios.py)

This test module focuses on real-world migration scenarios to demonstrate
the actual output format and quality of the KnowledgeService recommendations.
"""

import pytest
import json
import os
from typing import List, Dict, Any

# Import modules under test
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService, KnowledgeServiceError


class TestRealisticMigrationScenarios:
    """
    Test class focused on realistic migration scenarios with actual output inspection.
    """

    @pytest.fixture(scope="session")
    def llm_service(self):
        """Create a real CortexLLMService instance"""
        try:
            service = CortexLLMService()
            return service
        except Exception as e:
            pytest.fail(f"Failed to create CortexLLMService: {e}")

    @pytest.fixture(scope="session")
    def knowledge_service_with_llm(self, llm_service):
        """Create KnowledgeService with LLM capabilities"""
        # Get the knowledge base file path
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(test_dir)
        knowledge_base_path = os.path.join(project_root, "data", "knowledge_base.json")

        if not os.path.exists(knowledge_base_path):
            pytest.fail(f"Knowledge base file not found: {knowledge_base_path}")

        return KnowledgeService(knowledge_base_path=knowledge_base_path, llm_service=llm_service)

    def print_migration_result(self, scenario_name: str, python_functions: List[str],
                               pattern_ids: List[str], recipes: List[Dict]):
        """Helper method to print migration results with RAW recipe data"""
        print(f"\n{'=' * 80}")
        print(f"MIGRATION SCENARIO: {scenario_name}")
        print(f"{'=' * 80}")

        print(f"\n📥 INPUT - Python Functions to Migrate:")
        for i, func in enumerate(python_functions, 1):
            print(f"  {i}. {func}")

        print(f"\n🎯 LLM RECOMMENDATIONS - Snowpark Recipe IDs:")
        if pattern_ids:
            for i, pattern_id in enumerate(pattern_ids, 1):
                print(f"  {i}. {pattern_id}")
        else:
            print("  (No recommendations returned)")

        print(f"\n📋 RAW COMPLETE RECIPES (Exact format for next agent):")
        print(f"Recipe count: {len(recipes)}")
        print(f"{'-' * 60}")

        if recipes:
            # Print the exact raw data structure
            print("RAW RECIPES LIST:")
            print(json.dumps(recipes, indent=2, ensure_ascii=False))
        else:
            print("[]")

        print(f"\n{'=' * 80}")

    def test_scenario_1_basic_pyspark_etl(self, knowledge_service_with_llm):
        """Test Scenario 1: Basic PySpark ETL Pipeline"""
        python_functions = [
            "spark.table",
            "DataFrame.select",
            "DataFrame.filter",
            "DataFrame.write"
        ]

        # Get LLM recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions, top_k=6)

        # Get complete recipes
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "Basic PySpark ETL Pipeline",
            python_functions,
            pattern_ids,
            recipes
        )

        # Basic validations
        assert isinstance(pattern_ids, list), "Should return list of pattern IDs"
        assert isinstance(recipes, list), "Should return list of recipes"
        assert len(recipes) == len(pattern_ids), "Should get one recipe per pattern ID"

    def test_scenario_2_pandas_data_analysis(self, knowledge_service_with_llm):
        """Test Scenario 2: Pandas Data Analysis Workflow"""
        python_functions = [
            "pandas.read_csv",
            "DataFrame.groupby",
            "DataFrame.agg",
            "DataFrame.merge",
            "DataFrame.to_csv"
        ]

        # Get recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions, top_k=7)
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "Pandas Data Analysis Workflow",
            python_functions,
            pattern_ids,
            recipes
        )

        # Validations
        assert len(pattern_ids) > 0, "Should recommend at least one pattern"
        assert all(isinstance(pid, str) for pid in pattern_ids), "All pattern IDs should be strings"

    def test_scenario_3_mixed_data_processing(self, knowledge_service_with_llm):
        """Test Scenario 3: Mixed Data Processing (PySpark + Pandas + SQL)"""
        python_functions = [
            "spark.sql",
            "DataFrame.join",
            "pandas.DataFrame",
            "numpy.sum",
            "DataFrame.pivot_table"
        ]

        # Get recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions)  # Use adaptive top_k
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "Mixed Data Processing (PySpark + Pandas + SQL)",
            python_functions,
            pattern_ids,
            recipes
        )

        # Show adaptive top_k behavior
        expected_top_k = max(3, min(12, len(python_functions) * 2))
        print(f"\n📊 ADAPTIVE TOP_K INFO:")
        print(f"  Input functions count: {len(python_functions)}")
        print(f"  Expected max top_k: {expected_top_k}")
        print(f"  Actual returned count: {len(pattern_ids)}")

    def test_scenario_4_data_transformation_pipeline(self, knowledge_service_with_llm):
        """Test Scenario 4: Complex Data Transformation Pipeline"""
        python_functions = [
            "spark.table",
            "DataFrame.withColumn",
            "DataFrame.when",
            "DataFrame.cast",
            "DataFrame.groupBy",
            "DataFrame.agg",
            "DataFrame.orderBy",
            "DataFrame.write.mode"
        ]

        # Get recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions, top_k=10)
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "Complex Data Transformation Pipeline",
            python_functions,
            pattern_ids,
            recipes
        )

        # Show recipe details analysis
        print(f"\n📊 RECIPE ANALYSIS:")
        print(f"  Total recommendations: {len(pattern_ids)}")
        print(f"  Recipe categories found:")

        categories = {}
        for recipe in recipes:
            recipe_id = recipe.get('id', '')
            if '.' in recipe_id:
                category = '.'.join(recipe_id.split('.')[:2])  # e.g., 'snowpark.session'
                categories[category] = categories.get(category, 0) + 1

        for category, count in categories.items():
            print(f"    - {category}: {count} recipes")

    def test_scenario_5_simple_data_loading(self, knowledge_service_with_llm):
        """Test Scenario 5: Simple Data Loading and Preview"""
        python_functions = [
            "spark.read.csv",
            "DataFrame.show",
            "DataFrame.count"
        ]

        # Get recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions, top_k=4)
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "Simple Data Loading and Preview",
            python_functions,
            pattern_ids,
            recipes
        )

    def test_scenario_6_string_processing(self, knowledge_service_with_llm):
        """Test Scenario 6: String Processing Operations"""
        python_functions = [
            "DataFrame.withColumn",
            "functions.upper",
            "functions.trim",
            "functions.regexp_replace",
            "functions.concat"
        ]

        # Get recommendations
        pattern_ids = knowledge_service_with_llm.discover_patterns(python_functions, top_k=6)
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        # Print results
        self.print_migration_result(
            "String Processing Operations",
            python_functions,
            pattern_ids,
            recipes
        )

    def test_complete_migration_workflow_demo(self, knowledge_service_with_llm):
        """Demo: Complete Migration Workflow with Real Examples"""

        print(f"\n{'=' * 100}")
        print(f"COMPLETE MIGRATION WORKFLOW DEMONSTRATION")
        print(f"{'=' * 100}")

        # Multiple realistic scenarios
        scenarios = [
            {
                "name": "E-commerce Analytics",
                "functions": ["spark.table", "DataFrame.join", "DataFrame.groupBy", "DataFrame.agg", "DataFrame.write"],
                "description": "Typical e-commerce data analysis: join customer and order data, aggregate sales metrics"
            },
            {
                "name": "Log Processing",
                "functions": ["spark.read.text", "DataFrame.filter", "functions.regexp_extract", "DataFrame.select"],
                "description": "Parse and filter log files, extract specific patterns"
            },
            {
                "name": "Data Quality Check",
                "functions": ["DataFrame.isNull", "DataFrame.count", "DataFrame.distinct", "DataFrame.summary"],
                "description": "Data quality assessment and basic statistics"
            }
        ]

        for scenario in scenarios:
            print(f"\n{'-' * 60}")
            print(f"SCENARIO: {scenario['name']}")
            print(f"USE CASE: {scenario['description']}")
            print(f"{'-' * 60}")

            # Process each scenario
            pattern_ids = knowledge_service_with_llm.discover_patterns(scenario['functions'], top_k=5)
            recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

            print(f"\nInput Functions: {scenario['functions']}")
            print(f"Recommended Snowpark Patterns: {pattern_ids}")
            print(f"Number of Complete Recipes Retrieved: {len(recipes)}")

            # Show RAW recipes data
            print(f"\nRAW RECIPES DATA (for next agent context):")
            print(json.dumps(recipes, indent=2, ensure_ascii=False))

        print(f"\n{'=' * 100}")
        print(f"WORKFLOW DEMONSTRATION COMPLETED")
        print(f"{'=' * 100}")

    def test_output_format_validation(self, knowledge_service_with_llm):
        """Test: Validate Output Format and Structure - Show RAW Data"""

        print(f"\n{'=' * 80}")
        print(f"RAW OUTPUT FORMAT INSPECTION")
        print(f"{'=' * 80}")

        test_functions = ["spark.table", "DataFrame.select", "DataFrame.filter"]

        # Test discover_patterns output
        pattern_ids = knowledge_service_with_llm.discover_patterns(test_functions, top_k=3)

        print(f"\n🔍 DISCOVER_PATTERNS RAW OUTPUT:")
        print(f"Type: {type(pattern_ids)}")
        print(f"Length: {len(pattern_ids)}")
        print(f"Raw content:")
        print(json.dumps(pattern_ids, indent=2))

        # Test get_recipes output - RAW FORMAT
        recipes = knowledge_service_with_llm.get_recipes_from_suggested_patterns(pattern_ids)

        print(f"\n📋 GET_RECIPES RAW OUTPUT (EXACT DATA FOR NEXT AGENT):")
        print(f"Type: {type(recipes)}")
        print(f"Length: {len(recipes)}")
        print(f"\nRAW RECIPES DATA STRUCTURE:")
        print(json.dumps(recipes, indent=2, ensure_ascii=False))

        print(f"\n🔍 INDIVIDUAL RECIPE INSPECTION:")
        for i, recipe in enumerate(recipes):
            print(f"\nRecipe {i + 1} keys: {list(recipe.keys())}")
            print(f"Recipe {i + 1} raw data:")
            print(json.dumps(recipe, indent=2, ensure_ascii=False))

        # Validate structure
        assert isinstance(pattern_ids, list), "Pattern IDs should be a list"
        assert isinstance(recipes, list), "Recipes should be a list"
        assert all(isinstance(pid, str) for pid in pattern_ids), "All pattern IDs should be strings"
        assert all(isinstance(r, dict) for r in recipes), "All recipes should be dictionaries"

        print(f"\n✅ RAW FORMAT VALIDATION COMPLETED")