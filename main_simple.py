#!/usr/bin/env python3
"""
PySpark to Snowpark Migration Tool - Main Entry Point
Version 2.1 - Per-Function Atomic Processing with Simplified Migration
"""

import argparse
import os
import logging
import json
import ast
import sys
from pathlib import Path

# Import services
from services.llm_service import CortexLLMService
from services.knowledge_service import KnowledgeService

# Import agents
from agents.code_analyzer import CodeAnalyzer
from agents.code_enricher import CodeEnricher
from agents.code_migrator import CodeMigrator


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
        description='PySpark to Snowpark Migration Tool - Per-Function Processing'
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
    """Extract specified function code from script content using AST parsing"""
    try:
        tree = ast.parse(script_content)
        lines = script_content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno - 1

                # Get end line
                if hasattr(node, 'end_lineno'):
                    end_line = node.end_lineno
                else:
                    # Fallback: estimate end line
                    end_line = len(lines)
                    for i in range(start_line + 1, len(lines)):
                        if lines[i].strip() and not lines[i].startswith((' ', '\t')):
                            if lines[i].startswith(('def ', 'class ', '@')):
                                end_line = i
                                break

                return '\n'.join(lines[start_line:end_line])

        raise Exception(f"Function '{function_name}' not found in script")
    except SyntaxError as e:
        raise Exception(f"Syntax error in script when extracting function {function_name}: {str(e)}")


def save_artifact(content, file_path):
    """Universal save function for both JSON and text content"""
    try:
        ensure_output_directory(os.path.dirname(file_path))

        if isinstance(content, dict):
            # Save as formatted JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        elif isinstance(content, str):
            # Save as text file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            raise Exception(f"Unsupported content type: {type(content)}")

    except Exception as e:
        raise Exception(f"Error saving artifact to {file_path}: {str(e)}")


def get_function_analysis(analysis_report, function_name):
    """Extract function analysis data for specific function"""
    function_analyses = analysis_report.get('function_analysis', [])
    for func_data in function_analyses:
        if func_data.get('function_name') == function_name:
            return func_data
    return None


def process_single_function(function_name, script_content, analysis_report, output_dir,
                            enricher, migrator, knowledge_service, logger):
    """
    Process a single function through the simplified migration workflow

    Args:
        function_name (str): Name of the function to process
        script_content (str): Complete script content
        analysis_report (dict): Global analysis report
        output_dir (str): Base output directory
        enricher: CodeEnricher instance
        migrator: CodeMigrator instance
        knowledge_service: KnowledgeService instance
        logger: Logger instance
    """
    logger.info(f"Starting processing for function: {function_name}")

    # Create dedicated output directory for this function
    function_output_dir = os.path.join(output_dir, function_name)
    ensure_output_directory(function_output_dir)
    logger.info(f"Created output directory: {function_output_dir}")

    # Extract and enrich function code
    logger.info(f"Extracting and enriching function: {function_name}")
    function_code = extract_function_code(script_content, function_name)

    # Get function analysis data
    function_analysis = get_function_analysis(analysis_report, function_name)
    if not function_analysis:
        logger.warning(f"No analysis data found for function: {function_name}")

    # Enrich the function
    enrichment_result = enricher.enrich_function(function_code)
    enriched_code = enrichment_result.get('enriched_code', function_code)
    test_code = enrichment_result.get('test_function', '')

    # Save enrichment artifacts
    save_artifact(enriched_code, os.path.join(function_output_dir, '01_enriched_code.py'))
    if test_code:
        save_artifact(test_code, os.path.join(function_output_dir, '02_original_test.py'))

    # Track A: Process main business logic code
    logger.info(f"Starting Track A (main code) migration for: {function_name}")

    # Migrate main code - final output
    migrated_code = migrator.migrate_function(enriched_code, function_analysis, knowledge_service)
    save_artifact(migrated_code, os.path.join(function_output_dir, '03_final_migrated_code.py'))

    # Track B: Process unit test code (if exists)
    if test_code:
        logger.info(f"Starting Track B (test code) migration for: {function_name}")

        # Migrate test code - final output
        migrated_test = migrator.migrate_function(test_code, function_analysis, knowledge_service)
        save_artifact(migrated_test, os.path.join(function_output_dir, '04_final_migrated_test.py'))
    else:
        logger.info(f"No test code generated for function: {function_name}, skipping Track B")

    logger.info(f"Completed processing for function: {function_name}")


def main():
    """Main workflow orchestrator - Per-function atomic processing"""
    logger = setup_logging()

    try:
        logger.info("=== PySpark to Snowpark Migration Tool Started (Version 2.1) ===")

        # Phase 1: Global Preparation
        logger.info("=== Phase 1: Global Preparation ===")

        # Parse command line arguments
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

        # Initialize services
        logger.info("Initializing services...")
        knowledge_base_path = "data/knowledge_base.json"
        llm_service = CortexLLMService()
        knowledge_service = KnowledgeService(knowledge_base_path=knowledge_base_path)

        # Instantiate agents with dependency injection (removed reviewer)
        logger.info("Initializing agents...")
        analyzer = CodeAnalyzer(llm_service=llm_service)
        enricher = CodeEnricher(llm_service=llm_service)
        migrator = CodeMigrator(llm_service=llm_service)

        # Read source file
        logger.info("Reading source file...")
        script_content = read_script_file(input_file)
        source_file_name = os.path.basename(input_file)

        # Execute global analysis
        logger.info("Analyzing script...")
        analysis_report = analyzer.analyze_script(script_content, source_file_name)

        # Save global analysis report
        analysis_report_path = os.path.join(output_dir, "analysis_report.json")
        save_artifact(analysis_report, analysis_report_path)
        logger.info(f"Analysis report saved to: {analysis_report_path}")

        # Determine processing order and filter out non-existent functions
        conversion_order = analysis_report.get('conversion_order', [])
        if not conversion_order:
            # Fallback: extract function names from function_analysis
            conversion_order = [func['function_name'] for func in analysis_report.get('function_analysis', [])]

        # Filter functions that actually exist in the script
        functions_to_process = []
        for function_name in conversion_order:
            try:
                extract_function_code(script_content, function_name)
                functions_to_process.append(function_name)
            except Exception:
                logger.warning(f"Function '{function_name}' not found in script, skipping...")

        if not functions_to_process:
            logger.warning("No functions found to process. Migration complete.")
            return

        logger.info(f"Functions to process: {functions_to_process}")

        # Phase 2: Per-Function Processing Loop
        logger.info("=== Phase 2: Per-Function Processing Loop ===")

        for function_name in functions_to_process:
            try:
                process_single_function(
                    function_name=function_name,
                    script_content=script_content,
                    analysis_report=analysis_report,
                    output_dir=output_dir,
                    enricher=enricher,
                    migrator=migrator,
                    knowledge_service=knowledge_service,
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Error processing function {function_name}: {str(e)}")
                # Continue with other functions instead of stopping completely
                continue

        # Phase 3: Process Completion
        logger.info("=== Phase 3: Process Completion ===")
        logger.info("=== Migration completed successfully! ===")
        logger.info(f"All artifacts saved in: {output_dir}")
        logger.info(f"Processed {len(functions_to_process)} functions with individual output directories")

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()