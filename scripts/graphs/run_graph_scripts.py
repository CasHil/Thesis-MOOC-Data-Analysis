# Run all Python scripts in this folder except this one.

import os
import subprocess

for script in os.listdir():
    if script.endswith('.py') and script != 'run_graph_scripts.py':
        print("Running script: ", script)
        subprocess.run(["py", script])