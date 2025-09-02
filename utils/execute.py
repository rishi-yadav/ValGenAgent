import os
import subprocess
from datetime import datetime


class ExecutableRunner:
    """Class to find and run executables in a directory, storing logs in one file."""

    def __init__(self, exe_dir: str, log_dir: str, logger):
        self.exe_dir = os.path.abspath(exe_dir)
        self.log_dir = os.path.abspath(log_dir)
        self.logger = logger
        os.makedirs(self.log_dir, exist_ok=True)

    def find_executables(self) -> list[str]:
        """Find executables in the given directory."""
        return [
            os.path.join(self.exe_dir, f)
            for f in os.listdir(self.exe_dir)
            if os.access(os.path.join(self.exe_dir, f), os.X_OK)
            and not os.path.isdir(os.path.join(self.exe_dir, f))
        ]

    def run_executables(self, execute_args: list) -> list[str]:
        """Run executables, log output into a single file, return status messages."""
        msgs = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_log_file = os.path.join(self.log_dir, f"executables_log_{timestamp}.txt")

        executables = self.find_executables()

        with open(combined_log_file, "w", encoding="utf-8") as log_f:
            for exe in executables:
                exe_name = os.path.basename(exe)
                msgs.append(f"Running {exe_name}...")
                self.logger.log("TestBuildAndExecuteProxy", f"Running {exe_name}...")

                run_proc = subprocess.run(
                    [exe] + execute_args,
                    cwd=os.path.dirname(exe),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False
                )

                # Write header + output for each exe into one file
                log_f.write(f"\n===== {exe_name} (exit {run_proc.returncode}) =====\n")
                log_f.write(run_proc.stdout)
                log_f.write("\n")

                if run_proc.returncode != 0:
                    msg = f"{exe_name} failed (exit {run_proc.returncode}), log saved at {combined_log_file}"
                    msgs.append(msg)
                    self.logger.log("TestBuildAndExecuteProxy", msg)
                else:
                    msg = f"{exe_name} succeeded, log saved at {combined_log_file}"
                    msgs.append(msg)
                    self.logger.log("TestBuildAndExecuteProxy", msg)

        msgs.append(f"Combined log saved at {combined_log_file}")
        return msgs
