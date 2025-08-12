#!/usr/bin/env python3
"""
PySpark to Snowpark Migration Tool - Main Entry Point
POC Version - Command Center
"""

import argparse
import os
import logging
import json
import ast
import sys
from pathlib import Path

# 导入services
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService

# 导入agents
from agents.code_analyzer import CodeAnalyzer
from agents.code_enricher import CodeEnricher
from agents.code_migrator import CodeMigrator
from agents.code_reviewer import CodeReviewer


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PySpark to Snowpark Migration Tool'
    )
    parser.add_argument(
        '--file',
        required=True,
        help='Path to the PySpark script file to migrate'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for generated files'
    )
    return parser.parse_args()


def read_script_file(file_path):
    """Read script file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise Exception(f"Input file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")


def ensure_output_directory(output_dir):
    """Ensure output directory exists"""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating output directory {output_dir}: {str(e)}")


def extract_function_code(script_content, function_name):
    """Extract specified function code from script content"""
    try:
        tree = ast.parse(script_content)
        lines = script_content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno - 1
                # Try to get end line, estimate if not supported
                if hasattr(node, 'end_lineno'):
                    end_line = node.end_lineno
                else:
                    # Simple estimation: find next same-level definition or end of file
                    end_line = len(lines)
                    for i in range(start_line + 1, len(lines)):
                        if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                            if lines[i].startswith('def ') or lines[i].startswith('class '):
                                end_line = i
                                break

                return '\n'.join(lines[start_line:end_line])

        raise Exception(f"Function '{function_name}' not found in script")
    except SyntaxError as e:
        raise Exception(f"Syntax error in script when extracting function {function_name}: {str(e)}")


def save_json_file(data, file_path):
    """Save JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Error saving JSON file {file_path}: {str(e)}")


def save_python_file(content, file_path):
    """Save Python file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise Exception(f"Error saving Python file {file_path}: {str(e)}")


def merge_imports_and_functions(package_analysis, functions):
    """Merge import statements and function code"""
    # Extract all import statements
    imports = []

    # Standard library imports
    for lib in package_analysis.get('standard_libs', []):
        imports.append(lib['import_statement'])

    # Third-party library imports (should be converted to Snowpark equivalents, but keep original for POC)
    for lib in package_analysis.get('third_party', []):
        imports.append(lib['import_statement'])

    # Custom module imports
    for lib in package_analysis.get('custom_modules', []):
        imports.append(lib['import_statement'])

    # Merge all content
    final_content = '\n'.join(imports) + '\n\n' + '\n\n'.join(functions)
    return final_content


def merge_test_functions(test_functions):
    """Merge test functions"""
    test_imports = [
        "import pytest",
        "# Add other test-related imports as needed"
    ]

    final_content = '\n'.join(test_imports) + '\n\n' + '\n\n'.join(test_functions)
    return final_content


def main():
    """Main function - Workflow orchestrator"""
    logger = setup_logging()

    try:
        logger.info("=== PySpark to Snowpark Migration Tool Started ===")

        # Step 1: Parse command line arguments
        args = parse_arguments()
        input_file = args.file
        output_dir = args.output

        logger.info(f"Input file: {input_file}")
        logger.info(f"Output directory: {output_dir}")

        # Validate input file exists
        if not os.path.exists(input_file):
            raise Exception(f"Input file does not exist: {input_file}")

        # Ensure output directory exists
        ensure_output_directory(output_dir)

        # Step 2: Initialize services
        logger.info("Initializing services...")
        knowledge_base_path = "data/knowledge_base.json"
        llm_service = CortexLLMService()
        knowledge_service = KnowledgeService(knowledge_base_path=knowledge_base_path)

        # Step 3: Instantiate agents (dependency injection)
        logger.info("Initializing agents...")
        analyzer = CodeAnalyzer(llm_service=llm_service)
        enricher = CodeEnricher(llm_service=llm_service)
        migrator = CodeMigrator(llm_service=llm_service)
        reviewer = CodeReviewer(llm_service=llm_service)

        # Step 4: Read source file
        logger.info("Reading source file...")
        script_content = read_script_file(input_file)
        source_file_name = os.path.basename(input_file)

        # Step 5: Execute analysis
        logger.info("Analyzing script...")
        analysis_report = analyzer.analyze_script(script_content, source_file_name)

        # Save analysis report (debug file for POC phase)
        analysis_report_path = os.path.join(output_dir, "analysis_report.json")
        save_json_file(analysis_report, analysis_report_path)
        logger.info(f"Analysis report saved to: {analysis_report_path}")

        # Step 6: Get conversion order
        conversion_order = analysis_report.get('conversion_order', [])
        if not conversion_order:
            # If no conversion_order, extract function names from function_analysis
            conversion_order = [func['function_name'] for func in analysis_report.get('function_analysis', [])]

        logger.info(f"Functions to process: {conversion_order}")

        # Step 7: First loop - Code enrichment phase
        logger.info("=== Phase 1: Code Enrichment ===")
        enriched_functions = {}
        test_functions = {}

        for function_name in conversion_order:
            logger.info(f"Enriching function: {function_name}")

            # Extract function code
            function_code = extract_function_code(script_content, function_name)

            # Get function analysis data
            function_analysis = None
            for func_data in analysis_report.get('function_analysis', []):
                if func_data['function_name'] == function_name:
                    function_analysis = func_data
                    break

            # Call enricher
            enrichment_result = enricher.enrich_function(function_code, function_analysis)

            # Store results
            enriched_functions[function_name] = enrichment_result.get('enriched_code', function_code)
            test_functions[function_name] = enrichment_result.get('test_function', '')

        # Step 8: Second loop - Main code migration
        logger.info("=== Phase 2: Main Code Migration ===")
        final_migrated_functions = []

        for function_name in conversion_order:
            logger.info(f"Migrating function: {function_name}")

            # Get enriched code
            enriched_code = enriched_functions[function_name]

            # Get function analysis data
            function_analysis = None
            for func_data in analysis_report.get('function_analysis', []):
                if func_data['function_name'] == function_name:
                    function_analysis = func_data
                    break

            # Call migrator
            migrated_code = migrator.migrate_function(enriched_code, function_analysis)

            # Call reviewer
            review_result = reviewer.review_and_correct_migration(migrated_code, function_analysis)
            corrected_code = review_result.get('corrected_code', migrated_code)

            # Collect results
            final_migrated_functions.append(corrected_code)

        # Step 9: Third loop - Test code migration
        logger.info("=== Phase 3: Test Code Migration ===")
        final_test_migrations = []

        for function_name in conversion_order:
            if test_functions[function_name]:  # Only process functions with test code
                logger.info(f"Migrating test for: {function_name}")

                test_code = test_functions[function_name]

                # Get function analysis data
                function_analysis = None
                for func_data in analysis_report.get('function_analysis', []):
                    if func_data['function_name'] == function_name:
                        function_analysis = func_data
                        break

                # Call migrator for test code (assuming migrator has test handling method)
                try:
                    migrated_test = migrator.migrate_function(test_code, function_analysis)
                    review_result = reviewer.review_and_correct_migration(migrated_test, function_analysis)
                    corrected_test = review_result.get('corrected_code', migrated_test)
                    final_test_migrations.append(corrected_test)
                except Exception as e:
                    logger.warning(f"Error migrating test for {function_name}: {str(e)}")
                    # If test migration fails, use original test code
                    final_test_migrations.append(test_code)

        # Step 10: Merge and save final files
        logger.info("=== Phase 4: Merging and Saving Final Files ===")

        # Merge main code
        package_analysis = analysis_report.get('package_analysis', {})
        final_script_content = merge_imports_and_functions(package_analysis, final_migrated_functions)

        # Save migrated script
        script_name = os.path.splitext(source_file_name)[0]
        migrated_script_path = os.path.join(output_dir, f"{script_name}_migrated.py")
        save_python_file(final_script_content, migrated_script_path)
        logger.info(f"Migrated script saved to: {migrated_script_path}")

        # Merge and save test code
        if final_test_migrations:
            final_test_content = merge_test_functions(final_test_migrations)
            test_script_path = os.path.join(output_dir, f"{script_name}_tests.py")
            save_python_file(final_test_content, test_script_path)
            logger.info(f"Test script saved to: {test_script_path}")

        logger.info("=== Migration completed successfully! ===")
        logger.info(f"Output files in: {output_dir}")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()