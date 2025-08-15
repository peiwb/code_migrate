#!/usr/bin/env python3
"""
分步调试测试 - 找出 test_main.py 的具体问题
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_step1_basic_imports():
    """步骤1: 测试基本导入"""
    print("\n=== Step 1: Testing Basic Imports ===")

    try:
        from main import (
            parse_arguments, read_script_file, ensure_output_directory,
            extract_function_code, save_json_file, save_python_file
        )
        print("✓ Main module functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import main functions: {e}")
        return False


def test_step2_service_imports():
    """步骤2: 测试服务导入"""
    print("\n=== Step 2: Testing Service Imports ===")

    try:
        from services.llm_service import CortexLLMService
        print("✓ LLM Service imported successfully")
    except Exception as e:
        print(f"✗ Failed to import LLM Service: {e}")
        return False

    try:
        from services.knowledge_service import KnowledgeService
        print("✓ Knowledge Service imported successfully")
    except Exception as e:
        print(f"✗ Failed to import Knowledge Service: {e}")
        return False

    return True


def test_step3_agent_imports():
    """步骤3: 测试agent导入"""
    print("\n=== Step 3: Testing Agent Imports ===")

    agents = [
        ('code_analyzer', 'CodeAnalyzer'),
        ('code_enricher', 'CodeEnricher'),
        ('code_migrator', 'CodeMigrator'),
        ('code_reviewer', 'CodeReviewer')
    ]

    for module_name, class_name in agents:
        try:
            module = __import__(f'agents.{module_name}', fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"✓ {class_name} imported successfully")
        except Exception as e:
            print(f"✗ Failed to import {class_name}: {e}")
            return False

    return True


def test_step4_knowledge_base_file():
    """步骤4: 检查知识库文件"""
    print("\n=== Step 4: Testing Knowledge Base File ===")

    kb_paths = [
        "data/knowledge_base.json",
        "../data/knowledge_base.json",
        "./data/knowledge_base.json",
        "knowledge_base.json"
    ]

    for kb_path in kb_paths:
        if os.path.exists(kb_path):
            print(f"✓ Found knowledge base at: {kb_path}")
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ Knowledge base loaded successfully, keys: {list(data.keys())}")
                return kb_path
            except Exception as e:
                print(f"✗ Failed to load knowledge base: {e}")
        else:
            print(f"✗ Knowledge base not found at: {kb_path}")

    # Create a minimal knowledge base for testing
    print("Creating minimal knowledge base for testing...")
    kb_path = "data/knowledge_base.json"
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)

    minimal_kb = {
        "pyspark_to_snowpark_mappings": {
            "SparkSession": "Session",
            "DataFrame": "DataFrame"
        },
        "common_patterns": {
            "session_creation": "Session.builder.create()"
        }
    }

    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(minimal_kb, f, indent=2)

    print(f"✓ Created minimal knowledge base at: {kb_path}")
    return kb_path


def test_step5_service_initialization(kb_path):
    """步骤5: 测试服务初始化"""
    print("\n=== Step 5: Testing Service Initialization ===")

    # Test LLM Service
    try:
        from services.llm_service import CortexLLMService
        llm_service = CortexLLMService()
        print("✓ LLM Service initialized successfully")
        print(f"  LLM Service type: {type(llm_service)}")
        print(f"  LLM Service methods: {[method for method in dir(llm_service) if not method.startswith('_')]}")
    except Exception as e:
        print(f"✗ LLM Service initialization failed: {e}")
        print(f"  Error type: {type(e)}")
        return False, None

    # Test Knowledge Service
    try:
        from services.knowledge_service import KnowledgeService
        knowledge_service = KnowledgeService(knowledge_base_path=kb_path)
        print("✓ Knowledge Service initialized successfully")
        print(f"  Knowledge Service type: {type(knowledge_service)}")
    except Exception as e:
        print(f"✗ Knowledge Service initialization failed: {e}")
        return False, None

    return True, llm_service


def test_step6_agent_initialization(llm_service):
    """步骤6: 测试agent初始化"""
    print("\n=== Step 6: Testing Agent Initialization ===")

    agents_info = [
        ('agents.code_analyzer', 'CodeAnalyzer'),
        ('agents.code_enricher', 'CodeEnricher'),
        ('agents.code_migrator', 'CodeMigrator'),
        ('agents.code_reviewer', 'CodeReviewer')
    ]

    agents = {}

    for module_name, class_name in agents_info:
        try:
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            agent = agent_class(llm_service=llm_service)
            agents[class_name.lower()] = agent
            print(f"✓ {class_name} initialized successfully")
            print(f"  {class_name} methods: {[method for method in dir(agent) if not method.startswith('_')]}")
        except Exception as e:
            print(f"✗ {class_name} initialization failed: {e}")
            return False, None

    return True, agents


def test_step7_extract_function_fix():
    """步骤7: 测试修正后的函数提取"""
    print("\n=== Step 7: Testing Fixed Function Extraction ===")

    from main import extract_function_code

    # 修正的测试数据 - 使用真正的换行符
    valid_script = "def function1():\n    pass\n\ndef function2():\n    return True"

    try:
        # 测试存在的函数
        result = extract_function_code(valid_script, "function1")
        print(f"✓ Extracted function1: {repr(result)}")

        result = extract_function_code(valid_script, "function2")
        print(f"✓ Extracted function2: {repr(result)}")

        # 测试不存在的函数
        try:
            result = extract_function_code(valid_script, "nonexistent_function")
            print("✗ Should have raised exception for nonexistent function")
            return False
        except Exception as e:
            if "not found in script" in str(e):
                print(f"✓ Correctly raised exception for nonexistent function: {e}")
            else:
                print(f"✗ Wrong exception type: {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Function extraction test failed: {e}")
        return False


def test_step8_simple_analysis(agents):
    """步骤8: 测试简单的代码分析"""
    print("\n=== Step 8: Testing Simple Code Analysis ===")

    simple_script = '''
def create_spark_session():
    """Create a Spark session."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("test").getOrCreate()
    return spark

def process_data(spark):
    """Process some data."""
    df = spark.read.csv("data.csv")
    return df
'''

    try:
        analyzer = agents['codeanalyzer']
        result = analyzer.analyze_script(simple_script, "simple_test.py")
        print(f"✓ Analysis completed successfully")
        print(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if isinstance(result, dict) and 'function_analysis' in result:
            functions = result['function_analysis']
            print(f"  Found {len(functions)} functions")
            for func in functions:
                print(f"    - {func.get('function_name', 'unknown')}")

        return True, result

    except Exception as e:
        print(f"✗ Code analysis failed: {e}")
        print(f"  Error details: {str(e)[:200]}...")
        return False, None


def test_step9_simple_enrichment(agents, analysis_result):
    """步骤9: 测试简单的代码丰富"""
    print("\n=== Step 9: Testing Simple Code Enrichment ===")

    if not analysis_result or not isinstance(analysis_result, dict):
        print("✗ No valid analysis result to work with")
        return False

    simple_function_code = '''def create_spark_session():
    """Create a Spark session."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("test").getOrCreate()
    return spark'''

    # Get function analysis for the first function
    function_analysis = None
    if 'function_analysis' in analysis_result:
        functions = analysis_result['function_analysis']
        if functions:
            function_analysis = functions[0]

    try:
        enricher = agents['codeenricher']
        result = enricher.enrich_function(simple_function_code, function_analysis)
        print(f"✓ Code enrichment completed successfully")
        print(f"  Result type: {type(result)}")
        print(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        return True

    except Exception as e:
        print(f"✗ Code enrichment failed: {e}")
        print(f"  Error details: {str(e)[:200]}...")
        return False


def run_complete_debug():
    """运行完整的调试测试"""
    print("🚀 Starting Complete Debug Test Suite")
    print("=" * 60)

    # Step 1: Basic imports
    if not test_step1_basic_imports():
        return False

    # Step 2: Service imports
    if not test_step2_service_imports():
        return False

    # Step 3: Agent imports
    if not test_step3_agent_imports():
        return False

    # Step 4: Knowledge base
    kb_path = test_step4_knowledge_base_file()
    if not kb_path:
        return False

    # Step 5: Service initialization
    service_ok, llm_service = test_step5_service_initialization(kb_path)
    if not service_ok:
        return False

    # Step 6: Agent initialization
    agents_ok, agents = test_step6_agent_initialization(llm_service)
    if not agents_ok:
        return False

    # Step 7: Function extraction fix
    if not test_step7_extract_function_fix():
        return False

    # Step 8: Simple analysis
    analysis_ok, analysis_result = test_step8_simple_analysis(agents)
    if not analysis_ok:
        return False

    # Step 9: Simple enrichment
    if not test_step9_simple_enrichment(agents, analysis_result):
        return False

    print("\n" + "=" * 60)
    print("🎉 All debug tests passed! Services are working correctly.")
    print("✅ You can now proceed with the full integration test.")

    return True


if __name__ == "__main__":
    """直接运行调试测试"""
    success = run_complete_debug()

    if success:
        print("\n🔧 Next Steps:")
        print("1. Fix the string escaping issues in test_main.py")
        print("2. Fix the fixture scope issues")
        print("3. Run the full integration test")
    else:
        print("\n❌ Debug tests failed. Please fix the issues above before proceeding.")

    sys.exit(0 if success else 1)