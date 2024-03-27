import os
import subprocess

def main() -> None:
    if not os.path.exists('figures'):
        os.makedirs('figures')

    for script in os.listdir():
        if script.endswith('.py') and script != 'run_graph_scripts.py':
            print("Running script: ", script)
            subprocess.run(["py", script])

if __name__ == '__main__':
    main()