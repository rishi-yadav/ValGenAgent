from typing import List, Dict, Any, Optional, Tuple
import os
import json
from docx import Document
import re
import logging

logging = logging.getLogger("VGA") 




class FileIO:
    def __init__ (self, output_dir):
        self.output_dir=output_dir
        
    def _validate_all_files_generated(self, expected_files: List[str], successful_files: List[str], failed_files: List[str]) -> bool:
        """Validate that all expected files were generated successfully and provide detailed reporting"""

        # Check which files actually exist in the output directory
        actually_generated_files = []
        missing_files = []

        try:
            if os.path.exists(self.output_dir):
                existing_files = set(os.listdir(self.output_dir))

                for expected_file in expected_files:
                    if expected_file in existing_files:
                        actually_generated_files.append(expected_file)
                    else:
                        missing_files.append(expected_file)
            else:
                missing_files = expected_files.copy()
        except Exception as e:
            logging.error(f"[Orchestrator]- Error checking output directory: {e}")
            missing_files = expected_files.copy()

        # Print detailed file generation report
        logging.info("[Orchestrator]- ===== FILE GENERATION REPORT ======")
        logging.info(f"[Orchestrator]- Expected files: {len(expected_files)}")
        logging.info(f"[Orchestrator]- Successfully processed: {len(successful_files)}")
        logging.info(f"[Orchestrator]- Actually generated: {len(actually_generated_files)}")

        if actually_generated_files:
            logging.info(f"[Orchestrator]- Generated files: {actually_generated_files}")

        if missing_files:
            logging.info(f"[Orchestrator]- Missing files: {missing_files}")

        if failed_files:
            logging.error(f"[Orchestrator]- Failed to process: {failed_files}")

        # Determine overall success
        all_files_generated = len(missing_files) == 0 and len(actually_generated_files) == len(expected_files)

        if all_files_generated:
            logging.info("[Orchestrator]- SUCCESS: All expected files were generated successfully!")
        else:
            logging.warning(f"[Orchestrator]- PARTIAL SUCCESS: Only {len(actually_generated_files)}/{len(expected_files)} files were generated")
            logging.warning("[Orchestrator]- This is considered a failure as all files must be generated")

        logging.info("[Orchestrator]- ===== END REPORT =====")

        return all_files_generated

    def _verify_file_created(self, expected_file: str) -> bool:
        """Verify that a specific file was actually created in the output directory with meaningful content"""
        try:
            if not os.path.exists(self.output_dir):
                return False

            extensions_and_indicators = {
                '.py': ['def test_', 'import pytest', 'class Test'],
                '.cpp': ['TEST(', 'ASSERT_', 'EXPECT_','TEST','TEST_F','INSTANTIATE_TEST_SUITE_P','cout','iostream'],
                '.c': ['printf'],
                '.asm': ['; Test', '; Assert', '; Expect']  # Assembly comments might indicate test content
            }

            file_path = os.path.join(self.output_dir, expected_file)
            exists = False

            expected_name, expected_ext = os.path.splitext(expected_file)  # name + extension

            for fname in os.listdir(self.output_dir):
                fname_no_ext, fname_ext = os.path.splitext(fname)
                if expected_name in fname_no_ext and expected_ext == fname_ext:  # match both
                    exists = True
                    file_path = os.path.join(self.output_dir, fname)
                    break


            if exists:
                # Check if file has meaningful content (not empty)
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                # Also check if it contains test-like content
                    _, ext = os.path.splitext(expected_file)

                    # Get the test indicators for the file extension
                    indicators = extensions_and_indicators.get(ext, [])

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Look for test indicators
                        has_test_content = any(indicator in content for indicator in indicators)

                        if has_test_content:
                            logging.info(f"[Orchestrator]- File verified with test content: {expected_file} ({file_size} bytes)")
                            return True
                        else:
                            logging.error(f"[Orchestrator]- File exists but lacks test content: {expected_file}")
                            return False

                    except Exception as read_error:
                        logging.error(f"[Orchestrator]- File exists but couldn't read content: {expected_file} - {read_error}")
                        # Still consider it valid if file exists and has size
                        return True
                else:
                    logging.warning(f"[Orchestrator]- File exists but is empty: {expected_file}")
                    return False
            else:
                logging.error(f"[Orchestrator]- File not found: {expected_file}")
                return False

        except Exception as e:
            logging.error(f"[Orchestrator]- Error verifying file {expected_file}: {e}")
            return False


class TestPlanParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.is_json = file_path.endswith('.json')
        if not self.is_json:
            self.document = Document(file_path)

    def extract_test_cases(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        """Extract test cases from the test plan file.

        Returns:
            tuple: (success: bool, test_cases: List[Dict], implementation_files: List[str])
        """
        try:
            if self.is_json:
                return self._extract_from_json()
            else:
                return self._extract_from_docx()
        except Exception as e:
            logging.error(f"Error extracting test cases: {e}")
            return False, [], []

    def _extract_from_json(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        try:
            with open(self.file_path, 'r') as f:
                test_plan = json.load(f)

            test_cases = []
            implementation_files = set()

            # Handle different JSON structures
            if 'tests' in test_plan:
                tests_section = test_plan['tests']

                # Check if tests is a list (array of test categories)
                if isinstance(tests_section, list):
                    for test_category in tests_section:
                        implementation_file = test_category.get('implementation_file')
                        if implementation_file:
                            implementation_files.add(implementation_file)

                        # Extract test cases and add implementation file info to each
                        for test_case in test_category.get('test_cases', []):
                            enhanced_test_case = test_case.copy()
                            enhanced_test_case['implementation_file'] = implementation_file
                            enhanced_test_case['test_category'] = test_category.get('test_category', '')

                            # Map test_id to id for compatibility
                            if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                                enhanced_test_case['id'] = enhanced_test_case['test_id']

                            test_cases.append(enhanced_test_case)

                # Check if it's the new single test category format (tests as object)
                elif isinstance(tests_section, dict) and 'test_category' in tests_section and 'implementation_file' in tests_section:
                    # New format: single test category with implementation file
                    implementation_file = tests_section.get('implementation_file')
                    if implementation_file:
                        implementation_files.add(implementation_file)

                    # Extract test cases and add implementation file info to each
                    for test_case in tests_section.get('test_cases', []):
                        # Create a copy of the test case and add implementation file
                        enhanced_test_case = test_case.copy()
                        enhanced_test_case['implementation_file'] = implementation_file
                        enhanced_test_case['test_category'] = tests_section.get('test_category', '')

                        # Map test_id to id for compatibility
                        if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                            enhanced_test_case['id'] = enhanced_test_case['test_id']

                        test_cases.append(enhanced_test_case)

            # Handle the old format with test_categories directly
            elif 'test_categories' in test_plan:
                for test_category in test_plan['test_categories']:
                    if test_category.get('implementation_file'):
                        implementation_files.add(test_category['implementation_file'])
                        for test_case in test_category.get('test_cases', []):
                            test_cases.append(test_case)

            # Handle simple format with direct test_cases (like the original pytorch_collective_operations format)
            elif 'test_cases' in test_plan:
                for test_case in test_plan['test_cases']:
                    # For this format, we'll create a default implementation file name
                    test_category = test_plan.get('test_category', 'default_tests')
                    impl_file = f"test_{test_category.lower().replace(' ', '_').replace('.', '_')}.py"

                    enhanced_test_case = test_case.copy()
                    enhanced_test_case['implementation_file'] = impl_file
                    enhanced_test_case['test_category'] = test_category

                    # Map test_id to id for compatibility
                    if 'test_id' in enhanced_test_case and 'id' not in enhanced_test_case:
                        enhanced_test_case['id'] = enhanced_test_case['test_id']

                    test_cases.append(enhanced_test_case)
                    implementation_files.add(impl_file)

            return True, test_cases, list(implementation_files)

        except Exception as e:
            logging.error(f"Error reading JSON test plan: {e}")
            import traceback
            traceback.print_exc()
            return False, [], []

    def _extract_from_docx(self) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        try:
            test_cases = []
            implementation_files = set()
            current_test_case = {}
            for paragraph in self.document.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                if 'Implementation file:' in text:
                    file_name = text.split('Implementation file:')[1].strip()
                    if file_name.endswith('.py'):
                        implementation_files.add(file_name)
                        if current_test_case:
                            current_test_case['implementation_file'] = file_name
                if re.match(r'^(Test Case|TC)\s*\d+:', text, re.IGNORECASE):
                    if current_test_case:
                        test_cases.append(current_test_case)
                    current_test_case = self._init_test_case(text)
                elif current_test_case:
                    self._update_test_case(current_test_case, text)
            if current_test_case:
                test_cases.append(current_test_case)
            return True, test_cases, list(implementation_files)
        except Exception as e:
            logging.error(f"Error extracting from DOCX: {e}")
            return False, [], []

    def _init_test_case(self, text: str) -> Dict[str, Any]:
        return {
            'title': text,
            'description': '',
            'steps': [],
            'expected_results': '',
            'implementation_file': None,
            'data_types': []
        }

    def _update_test_case(self, test_case: Dict[str, Any], text: str) -> None:
        if text.lower().startswith('steps:'):
            test_case['steps'] = []
        elif text.lower().startswith('description:'):
            test_case['description'] = text.split(':', 1)[1].strip()
        elif text.lower().startswith('expected result:') or text.lower().startswith('expected_results:'):
            test_case['expected_results'] = text.split(':', 1)[1].strip()
        elif text.lower().startswith(('data types:', 'data_types:')):
            test_case['data_types'] = [dt.strip() for dt in re.split(r'[,;\s]+', text.split(':', 1)[1].strip()) if dt.strip()]
        elif test_case.get('steps') is not None:
            test_case['steps'].append(text)
        else:
            test_case['description'] += text + '\n'
