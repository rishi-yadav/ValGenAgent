#!/usr/bin/env python3
"""
Enhanced Test Workflow Runner

This script coordinates the entire test workflow:
1. Accept feature input file (JSON) containing name and description fields
2. Load additional documentation from static input_dirs directory (docs/ and public_urls_testplan.txt)
3. Generate test plan document using combined feature info and additional documentation
4. Run test_automation_agent.py to create test code using code/ directory for reference
5. Execute the generated tests
"""

import os
import sys
import time
import json
import argparse
import subprocess
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import shutil
import importlib
import logging as logger
# Import the new OpenAI API key utility
from utils.logging_config import setup_logging

# Import the new document processor
from utils.document_processor import DocumentProcessor

# Import the modules directly
from test_plangenerator import generate_test_plan_files
from agents.codegen_agent import run_test_automation

logging = logger.getLogger("VGA") 

# Load environment variables
load_dotenv()

@dataclass
class TestResult:
    """Data class for test execution results"""
    test_name: str
    status: str
    execution_time: float
    error_message: Optional[str] = None

class TestWorkflowRunner:
    """Manages the end-to-end test workflow"""

    def __init__(self, args,
                 generate_plan: bool = True,
                 run_automation: bool = True,
                 code_agent_prompt: Optional[str] = None,
                 review_agent_prompt: Optional[str] = None,
                 test_coordinator_prompt: Optional[str] = None):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.verbose = args.verbose
        self.test_plan_file = args.test_plan
        self.feature_input_file = args.feature_input_file
        self.generate_plan = generate_plan
        self.run_automation = run_automation
        self.code_agent_prompt = code_agent_prompt
        self.review_agent_prompt = review_agent_prompt
        self.test_coordinator_prompt = test_coordinator_prompt

        # Static input directory for documents and URLs
        self.input_dirs_path = Path("input_dirs")

        # Initialize feature_name (will be set when loading feature info)
        self.feature_name = "sample"

        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            self.args,
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_feature_info(self) -> Dict:
        """Load feature information from feature input file."""
        if not self.feature_input_file or not os.path.exists(self.feature_input_file):
            logging.warning("No feature input file provided or file does not exist")
            return {}

        try:
            logging.debug(f"Loading feature information from: {self.feature_input_file}")

            with open(self.feature_input_file, 'r', encoding='utf-8') as f:
                feature_info = json.load(f)

            # Validate required fields
            if 'name' not in feature_info or 'description' not in feature_info:
                logging.error("Feature input file must contain 'name' and 'description' fields")
                return {}

            logging.info(f"Loaded feature: {feature_info['name']}")
            return feature_info

        except Exception as e:
            logging.error(f"Error loading feature info from file: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return {}

    def load_additional_docs_content(self) -> str:
        """Load additional documentation content from input_dirs directory."""
        if not self.input_dirs_path.exists():
            logging.warning(f"Static input directory {self.input_dirs_path} does not exist")
            return ""

        try:
            logging.debug(f"Loading additional documentation from: {self.input_dirs_path}")
            stage_start = time.time()

            # Load documents from docs directory and URLs
            doc_infos = self.doc_processor.load_documents_from_directory(self.input_dirs_path / "docs")

            if not doc_infos:
                logging.info("No additional documentation found")
                return ""

            # Prepare content for inclusion in test plan generation
            prepared_content = self.doc_processor.prepare_content(doc_infos)

            stage_time = time.time() - stage_start
            logging.info(f"Additional documentation loading completed in {stage_time:.2f} seconds")
            return prepared_content

        except Exception as e:
            logging.error(f"Error loading additional documentation: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return ""

    def generate_test_plan(self) -> Tuple[bool, str]:
        """Generate test plan document."""
        if not self.feature_input_file and self.args.human_feedback and not self.args.with_test_plan:
            logging.info("Interactive prompt with human feedback for code generation")
            return True, None
        if self.test_plan_file and os.path.exists(self.test_plan_file):
            logging.info(f"Using existing test plan: {self.test_plan_file}")
            return True, self.test_plan_file

        try:
            feature_info={}
            if not self.args.human_feedback:
                logging.info("Loading feature information...")
                feature_info = self.load_feature_info()

                if not feature_info:
                    logging.error("No valid feature information found")
                    return False, ""

                # Extract name from feature_info and create filename
                feature_name = feature_info['name']
                # Clean the name for use as filename (remove spaces, special chars)
                feature_name = "".join(c for c in feature_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                feature_name = feature_name.replace(' ', '_').lower()
            else:
                feature_name='human_feedback_feature'
            self.feature_name = feature_name  # Store for use in other methods

            test_plan_file = self.output_dir / f"{self.feature_name}_test_plan.docx"
            test_plan_json = self.output_dir / f"{self.feature_name}_test_plan.json"

            # Load additional documentation content
            additional_docs_content = self.load_additional_docs_content()

            # Combine feature info with additional documentation
            enhanced_feature_info = feature_info.copy() if feature_info else {}
            enhanced_feature_info = feature_info.copy()
            if additional_docs_content:
                enhanced_feature_info['additional_documentation'] = additional_docs_content
                logging.info("Added additional documentation content to feature info")

            # Save enhanced feature info to temporary file
            feature_info_path = self.output_dir / "temp_feature_info.json"
            with open(feature_info_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_feature_info, f, indent=2, ensure_ascii=False)

            logging.info(f"Temporary feature info saved to: {feature_info_path}")

            logging.debug("Generating test plan...")
            plan_start = time.time()

            success = generate_test_plan_files(
                self.args,
                output_file=str(test_plan_file),
                json_file=str(test_plan_json),
                feature_info_file=str(feature_info_path),
            )

            plan_time = time.time() - plan_start

            if not success:
                logging.error("Error generating test plan")
                return False, ""

            logging.info(f"Test plan generation completed in {plan_time:.2f} seconds")

            # Clean up temporary file
            if feature_info_path and feature_info_path.exists():
                feature_info_path.unlink()

            # Return the JSON file path for the test automation agent to use
            return True, str(test_plan_json)

        except Exception as e:
            logging.error(f"Error in test plan generation: {e}")
            import traceback
            traceback.print_exc()
            return False, ""

    def run_test_automation(self, test_plan_file: str) -> bool:
        """Run test automation agent to generate and execute tests."""
        logging.debug("Initializing test automation...")
        try:
            output_dir = self.output_dir

            logging.debug("Generating test code...")
            # Call the function directly instead of subprocess
            success = run_test_automation(
                self.args,
                test_plan_path=test_plan_file,
                output_dir=output_dir,
                max_retries=2,  # default value
                max_context=25,  # default value
                code_agent_prompt=self.code_agent_prompt,
                review_agent_prompt=self.review_agent_prompt,
                test_coordinator_prompt=self.test_coordinator_prompt,
            )

            if not success:
                logging.error("Error in test automation: Test generation failed")
                return False

            if self.args.execute_python or self.args.execute_cpp:
                logging.info("Test generation and execution completed")
            elif self.args.build:
                logging.info("Test generation and build completed")
            else:
                logging.info("Test code generation completed")

            return True

        except Exception as e:
            logging.error(f"Error running test automation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_test_results(self) -> List[TestResult]:
        """Collect results from test execution."""
        logging.debug("Parsing test results...")
        results_start = time.time()

        results = []
        xml_files = glob.glob(str(self.output_dir  / "*.xml"))

        for xml_file in xml_files:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for testcase in root.findall(".//testcase"):
                    result = TestResult(
                        test_name=testcase.get('name', 'Unknown'),
                        status='Failed' if testcase.findall('./failure') else 'Passed',
                        execution_time=float(testcase.get('time', 0)),
                        error_message=testcase.findall('./failure')[0].text if testcase.findall('./failure') else None
                    )
                    results.append(result)

            except Exception as e:
                logging.error(f"Error parsing test results from {xml_file}: {e}")

        results_time = time.time() - results_start
        logging.info(f"Results collection completed in {results_time:.2f} seconds")

        return results

    def save_results_to_excel(self, results: List[TestResult]) -> None:
        """Save test results to Excel file."""
        if not results:
            logging.info("No test results to save")
            return

        try:
            logging.debug("Creating Excel report...")
            excel_start = time.time()

            df = pd.DataFrame([
                {
                    'Test Name': r.test_name,
                    'Status': r.status,
                    'Execution Time (s)': r.execution_time,
                    'Error Message': r.error_message or ''
                }
                for r in results
            ])

            excel_file = self.output_dir / f"{self.feature_name}_test_results.xlsx"
            df.to_excel(excel_file, index=False)

            excel_time = time.time() - excel_start
            logging.info(f"Excel report saved to: {excel_file}")
            logging.info(f"Excel generation completed in {excel_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Error saving results to Excel: {e}")

    def print_workflow_summary(self):
        """Print a summary of the workflow configuration."""
        logging.info("=" * 60)
        logging.info(f"Test Workflow Configuration")
        logging.info("=" * 60)
        logging.info(f"Output Directory: {self.output_dir}")
        logging.info(f"Generate Test Plan: {'Yes' if self.generate_plan else 'No'}")
        logging.info(f"Run Test Automation: {'Yes' if self.run_automation else 'No'}")
        if self.args.execute_python:
            logging.info(f"Execute Python Tests: {'Yes' if self.args.execute_python else 'No'}")
        elif self.args.build:
            logging.info(f"Build CPP Tests: {'Yes' if self.args.build else "No"}")
            logging.info(f"Execute CPP Tests: {'Yes' if self.args.execute_cpp else "No"}")
        if self.test_plan_file:
            logging.info(f"Test Plan File: {self.test_plan_file}")
        if self.feature_input_file:
            logging.info(f"Feature Input File: {self.feature_input_file}")
        logging.info(f"Static Input Directory: {self.input_dirs_path}")
        logging.info(f"Docs Directory: {self.input_dirs_path / 'docs'}")
        logging.info(f"URLs File: {self.input_dirs_path / 'public_urls_testplan.txt'}")
        logging.info("=" * 60)

    def run(self) -> bool:
        """Run the complete test workflow."""
        # Print workflow configuration
        self.print_workflow_summary()

        workflow_start = time.time()
        test_plan_file = None

        # Step 1: Generate test plan (if enabled)
        if self.generate_plan:
            logging.info("="*60)
            logging.info("STAGE 1: TEST PLAN GENERATION")
            logging.info("="*60)
            stage1_start = time.time()

            success, test_plan_file = self.generate_test_plan()

            stage1_time = time.time() - stage1_start

            if not success:
                logging.error("FAILED: Failed to generate test plan.")
                return False

            logging.info(f"SUCCESS: Stage 1 completed successfully in {stage1_time:.2f} seconds")
            logging.info(f"Test plan saved: {test_plan_file}")
        else:
            # Use provided test plan file or look for existing one
            if self.test_plan_file and os.path.exists(self.test_plan_file):
                test_plan_file = self.test_plan_file
                logging.info(f"Using existing test plan: {test_plan_file}")
            else:
                # Look for existing test plan in output directory (prefer JSON for automation)
                possible_plans = [
                    self.output_dir / f"{self.feature_name}_test_plan.json",
                    self.output_dir / f"{self.feature_name}_test_plan.docx"
                ]
                for plan_file in possible_plans:
                    if plan_file.exists():
                        test_plan_file = str(plan_file)
                        logging.info(f"Found existing test plan: {test_plan_file}")
                        break

                if not test_plan_file and not self.args.human_feedback:
                    logging.error("No test plan file found and test plan generation is disabled.")
                    logging.error("Either provide --test-plan or enable --generate-plan")
                    return False

        # Step 2: Run test automation (if enabled)
        if self.run_automation:
            logging.info("="*60)
            if self.args.execute_python or self.args.execute_cpp:
                logging.info("STAGE 2: TEST CODE GENERATION & EXECUTION")
            elif self.args.build:
                logging.info("STAGE 2: TEST CODE GENERATION & BUILD")
            else:
                logging.info("STAGE 2: TEST CODE GENERATION")
            logging.info("="*60)
            stage2_start = time.time()

            if not self.run_test_automation(test_plan_file):
                logging.error("FAILED: Failed to run test automation.")
                return False

            stage2_time = time.time() - stage2_start
            logging.info(f"SUCCESS: Stage 2 completed successfully in {stage2_time:.2f} seconds")

            # Step 3: Collect and save results (only if automation ran and tests are executed)
            if self.args.execute_python:
                logging.info("="*60)
                logging.info("STAGE 3: RESULTS COLLECTION & REPORTING")
                logging.info("="*60)
                stage3_start = time.time()

                results = self.collect_test_results()
                self.save_results_to_excel(results)

                stage3_time = time.time() - stage3_start
                logging.info(f"SUCCESS: Stage 3 completed successfully in {stage3_time:.2f} seconds")

                # Print test execution summary
                passed = sum(1 for r in results if r.status == 'Passed')
                failed = sum(1 for r in results if r.status == 'Failed')

                logging.info("="*60)
                logging.info("TEST EXECUTION SUMMARY")
                logging.info("="*60)
                logging.info(f"Total Tests: {len(results)}")
                logging.info(f"Passed: {passed}")
                logging.info(f"Failed: {failed}")
                logging.info(f"Success Rate: {(passed/len(results)*100):.1f}%" if results else "0%")

                workflow_time = time.time() - workflow_start
                logging.info(f"Total Workflow Time: {workflow_time:.2f} seconds")

                return failed == 0
            elif self.args.execute_cpp:
                logging.info("Test execution step completed successfully.")
            elif self.args.build:
                logging.info("Test build step completed successfully.")
            else:
                logging.info("Test execution step skipped.")

        workflow_time = time.time() - workflow_start
        logging.info(f"Total Workflow Time: {workflow_time:.2f} seconds")
        return True

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the complete test workflow using a feature input file and static input_dirs directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run complete workflow (generate plan + run automation) with feature input file
            python test_runner.py --feature-input-file path/to/feature.json --output-dir path/to/output_dir

            # Only generate test plan from feature input file
            python test_runner.py --generate-plan-only --feature-input-file path/to/feature.json --output-dir test_results

            # Only run test automation (requires existing test plan)
            python test_runner.py --test-automation-only --test-plan path/to/plan.json --output-dir test_results

            # Generate tests from feature input file without executing them
            python test_runner.py --feature-input-file path/to/feature.json --output-dir path/to/output_dir --execute-tests=false

            # Generate tests from existing test plan without executing them
            python test_runner.py --test-plan path/to/plan.json --output-dir path/to/output_dir --execute-tests=false
        """
    )
    parser.add_argument('--output_dir', default='test_results', help='Output directory for all artifacts')
    parser.add_argument('--test_plan', help='Path to existing test plan (optional)')
    parser.add_argument('--feature_input_file', help='Path to feature input JSON file containing name and description fields')
    parser.add_argument("--verbose",action="store_true",help="Enable info-level logging instead of error-only.",)
    parser.add_argument('--code_dir', default='./code', help='Path to the code directory for RAG.')
    parser.add_argument('--remove_index_db', action='store_true', help='deletes the already created index db for RAG')
    parser.add_argument('--add_context_dir', help='provide all files as context to the pipeline, and the index db will not be used')
    parser.add_argument("--build", action="store_true", help="Enable build mode. Build the generated code file using user provided command.")
    parser.add_argument("--build_dir", help="build dir path")
    parser.add_argument("--build_cmd", help="Build command that will be used by agent to build the generated code file.")
    parser.add_argument("--execute_cpp",action="store_true", help="Enable execute mode. Executes the generated binary file using user provided arguments if any.")
    parser.add_argument("--execute_dir", help="execute dir path: directory path for execution")
    parser.add_argument("--execute_args", nargs=argparse.REMAINDER, default=[], help="Arguments to be added for execution. ex: ./filename --device gaudi2")
    parser.add_argument('--execute_python',action='store_true',help='Execute generated tests (default: False)')
    parser.add_argument('--human_feedback',action='store_true',help='give human feedback after code has been generated and the review is done.')
    parser.add_argument('--with_test_plan',action='store_true',help='true if in human feedback want to generate test plan')
    parser.add_argument('--index_db_type',type=str,default='property_graph',help='select either "vector_db" or "property_graph" for the corresponding index db creation.')
    # Step control arguments
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument('--generate_plan_only', action='store_true',
                           help='Only generate test plan, skip test automation')
    step_group.add_argument('--test_automation_only', action='store_true',
                           help='Only run test automation, skip test plan generation')

    # Test execution control

    parser.add_argument('--prompt_path', type=str, required=True,
                        help='path to the system prompts directory to use for the test generation workflow.')
    args = parser.parse_args()

    log_level = logger.DEBUG if args.verbose else logger.WARNING
    setup_logging(log_level,project_namespace="VGA")

    if args.build:
        if not args.build_dir or not args.build_cmd:
            parser.error("--build requires --build_dir and --build_cmd")

    if args.execute_cpp:
        if not args.execute_dir or not args.build:
            parser.error("--execute_cpp requires building using --build and also requires execute directrory in --execute_dir")
    try:
        # Dynamically import the module based on the prompt
        if args.prompt_path.endswith('/'): args.prompt_path=args.prompt_path[:-1]
        module_path = args.prompt_path.replace("/", ".")
        code_agent_prompt = importlib.import_module(f'{module_path}.code_agent_system_prompt')
        review_agent_prompt = importlib.import_module(f'{module_path}.review_agent_system_prompt')
        test_coordinator_prompt = importlib.import_module(f'{module_path}.test_coordinator_system_prompt')
        logging.info(f"Running test workflow with prompt: {module_path}")
    except AssertionError as e:
        assert f"Error in dynamic prompt import : {e}"

    # Determine which steps to run
    generate_plan = True
    run_automation = True

    if args.generate_plan_only:
        generate_plan = True
        run_automation = False
        logging.info("Mode: Generate test plan only")
    elif args.test_automation_only:
        generate_plan = False
        run_automation = True
        logging.info("Mode: Test automation only")
    else:
        # Default mode: generate plan (if needed) + test automation
        # The execute_python flag will control whether tests are actually executed
        if args.execute_python:
            logging.info("Mode: Complete workflow (generate plan + test automation + execution)")
        elif args.build:
            logging.info("Mode: Build cpp executables(generate plan + test automation + build)")
        elif args.execute_cpp:
            logging.info("Mode: execute cpp after building cpp executables(generate plan + test automation + build + execute)")
        else:
            logging.info("Mode: Generate tests only (skip execution)")

    index_db_dir='index_db/'
    if args.remove_index_db:
        if os.path.exists(index_db_dir):
            shutil.rmtree(index_db_dir)
            logging.info(f"Deleted the directory: {index_db_dir}")
        else:
            logging.info(f"The directory {index_db_dir} does not exist.")

    runner = TestWorkflowRunner(
        args=args,
        generate_plan=generate_plan,
        run_automation=run_automation,
        code_agent_prompt=code_agent_prompt.CODE_AGENT_SYSTEM_PROMPT,
        review_agent_prompt=review_agent_prompt.REVIEW_AGENT_SYSTEM_PROMPT,
        test_coordinator_prompt=test_coordinator_prompt.TEST_COORDINATOR_AGENT_SYSTEM_PROMPT
    )
    if args.human_feedback:
        while True:
            success = runner.run()
            run_again=input("Do you want to generate tests again? y or n: ")
            if run_again=='n' or run_again == 'no':
                break
    else:
        success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()