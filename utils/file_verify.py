from typing import List, Dict, Any, Optional, Tuple
import os

class VerifyFile:
    def __init__ (self, logger, output_dir):
        self.logger=logger
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
            self.logger.log("Orchestrator", f"Error checking output directory: {e}")
            missing_files = expected_files.copy()

        # Print detailed file generation report
        self.logger.log("Orchestrator", "===== FILE GENERATION REPORT ======")
        self.logger.log("Orchestrator", f"Expected files: {len(expected_files)}")
        self.logger.log("Orchestrator", f"Successfully processed: {len(successful_files)}")
        self.logger.log("Orchestrator", f"Actually generated: {len(actually_generated_files)}")

        if actually_generated_files:
            self.logger.log("Orchestrator", f"Generated files: {actually_generated_files}")

        if missing_files:
            self.logger.log("Orchestrator", f"Missing files: {missing_files}")

        if failed_files:
            self.logger.log("Orchestrator", f"Failed to process: {failed_files}")

        # Determine overall success
        all_files_generated = len(missing_files) == 0 and len(actually_generated_files) == len(expected_files)

        if all_files_generated:
            self.logger.log("Orchestrator", "SUCCESS: All expected files were generated successfully!")
        else:
            self.logger.log("Orchestrator", f"PARTIAL SUCCESS: Only {len(actually_generated_files)}/{len(expected_files)} files were generated")
            self.logger.log("Orchestrator", "This is considered a failure as all files must be generated")

        self.logger.log("Orchestrator", "===== END REPORT =====")

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
                            self.logger.log("Orchestrator", f"File verified with test content: {expected_file} ({file_size} bytes)")
                            return True
                        else:
                            self.logger.log("Orchestrator", f"File exists but lacks test content: {expected_file}")
                            return False

                    except Exception as read_error:
                        self.logger.log("Orchestrator", f"File exists but couldn't read content: {expected_file} - {read_error}")
                        # Still consider it valid if file exists and has size
                        return True
                else:
                    self.logger.log("Orchestrator", f"File exists but is empty: {expected_file}")
                    return False
            else:
                self.logger.log("Orchestrator", f"File not found: {expected_file}")
                return False

        except Exception as e:
            self.logger.log("Orchestrator", f"Error verifying file {expected_file}: {e}")
            return False