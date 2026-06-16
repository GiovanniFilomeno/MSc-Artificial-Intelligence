import subprocess
from pathlib import Path

ex = "max_assignment"

def run_popper():
    popper_dir = Path(__file__).resolve().parent / "Popper"
    cmd = ["conda", "run", "-n", 
           "knowledge", "python", 
           str(popper_dir / "popper.py"),
           str(popper_dir / "examples" / ex)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(f"popper_results_{ex}.txt", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
    print(f"FINISHED {ex}")

run_popper()
