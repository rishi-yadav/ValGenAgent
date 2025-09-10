import os
import subprocess
from datetime import datetime
from utils.execute import ExeRunner  
import autogen
import logging

logging = logging.getLogger("VGA") 


class ExeBuilder:
    """
        A class that manages the build process for generated C++ code.  
        It saves the generated source file, compiles it using the specified build command, and, if errors occur, summarizes the build log to highlight issues before retrying.  
        If the build succeeds, it reports success; otherwise, it provides detailed error insights.
    """
    def __init__(self, code, filepath, build_dir, log_dir, summarizer_agent, build_cmd):
        self.code = code
        self.filepath = filepath
        self.build_dir = build_dir
        self.log_dir = log_dir
        self.summarizer_agent = summarizer_agent
        self.build_cmd = build_cmd


    def save_file(self) -> str:
        """save file with new code."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(self.code)
            return f"saved file: {self.filepath}"
        except Exception as e:
            return f"Failed to write file {self.filepath}: {e}"

    def execute_build_command(self) -> tuple[bool, list[str]]:
        """Run the build command, save log, return (success, messages)."""
        msgs = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"build_log_{timestamp}.txt")


        logging.debug("Build started, waiting for output...")
        try:
            msgs.append(f"Running build: {self.build_cmd} in {self.build_dir}")
            logging.info(f"[TestBuildAndExecuteProxy] Running build: {self.build_cmd} in {self.build_dir}")


            build_proc = subprocess.run(
                self.build_cmd,
                cwd=self.build_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                shell=True
            )

            # Log stderr as ERROR
            if build_proc.stderr:
                for line in build_proc.stderr.splitlines():
                    line = line.strip()
                    if line:
                        logging.error(f"BUILD ERROR: {line}")


            logging.debug("Build finished, waiting for exit code...")
            
            os.makedirs(self.log_dir, exist_ok=True)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(build_proc.stdout)
                f.write(build_proc.stderr)

            logging.info(f"[TestBuildAndExecuteProxy] Please refer to the full logs here[ Full build log saved at: {log_file} ]")

            if build_proc.returncode != 0:
                msgs.append(f"Build failed with code {build_proc.returncode}")
                logging.info("[TestBuildAndExecuteProxy] Build failed, summarizing build logs")
                logs=build_proc.stderr+build_proc.stdout
                summary = self.summarize_log(logs, "build log")
                msgs.append("Build Log Summary:\n" + summary)
                return False, msgs
            else:
                msgs.append("Build succeeded")
                logging.info("TestBuildAndExecuteProxy Build is successful")
                return True, msgs

        except Exception as e:
            msgs.append(f"Error during build: {e}")
            return False, msgs

    def summarize_log(self, log: str, context: str) -> str:
        """Summarize logs using the summarizer agent."""
        try:
            reply = self.summarizer_agent.generate_reply(
                messages=[{
                    "role": "user",
                    "content": f"Summarize the following {context}:\n\n{log}"
                }]
            )
            return str(reply)
        except Exception as e:
            return f"Failed to summarize {context}: {e}"


def save_build_run(code: str, filename: str, directory: str, build: bool, build_cmd: str, build_dir: str, execute: bool, execute_dir: str, execute_args: list, llm_config=None) -> str:
    """
    Save the test file, optionally build it, and optionally run executables.
    """
    filepath = os.path.join(directory, filename)
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(directory, exist_ok=True)

    msgs = []

    build_runner = ExeBuilder(
        code=code,
        filepath=filepath,
        build_dir=build_dir,
        log_dir=os.path.join(build_dir, "logs"),
        summarizer_agent=None,  # will set below
        build_cmd=build_cmd,
    )

    save_file_logs = build_runner.save_file()
    logging.info(f"[TestBuildAndExecuteProxy] save_file logs: {save_file_logs}")
    msgs.append(save_file_logs)

    # Step 2: Init summarizer agent
    summarizer_agent = autogen.ConversableAgent(
        name="BuildLogSummarizer",
        system_message="""
        You are a **Build Log Summarizer Agent**.
        Your job is to:
        - Summarize the build log concisely.
        - Highlight the **errors, compiler diagnostics, and failed tests** clearly.
        - Give function names and variables related to errors.
        - <IMPORTANT> Provide debug-ready insights, with 100 lines from start, 
          100 lines around errors, 100 lines from end. Don't miss any errors at all.
        - If no errors exist, confirm build success.
        """,
        llm_config=llm_config,
    )
    build_runner.summarizer_agent = summarizer_agent

    # Step 3: Run build
    success, build_msgs = build_runner.execute_build_command()
    msgs.extend(build_msgs)

    if success:
        logging.info("[TestBuildAndExecuteProxy] Build succeeded")
        if execute:
            logging.info("[TestBuildAndExecuteProxy] Proceeding with execution")
            exe_runner = ExeRunner(
                exe_dir=execute_dir,
                log_dir=os.path.join(execute_dir, "logs"),
            )
            executables = exe_runner.find_executables()
            if not executables:
                msgs.append(f"No executables found in {execute_dir}")
            else:
                msgs.append(f"Found executables: {executables}")
                exe_msgs = exe_runner.run_executables(execute_args)
                msgs.extend(exe_msgs)
    else:
        logging.info("[TestBuildAndExecuteProxy] Build failed")

    return "\n".join(msgs)