import subprocess

ex = "sum_assignment"

def run_popper():
    cmd = ["conda", "run", "-n", 
           "knowledge", "python", 
           "/Users/giovannifilomeno/Desktop/Master-Artificial-Intelligence/Knowledge Representation/Second Assignment/Popper/popper.py", 
           f"/Users/giovannifilomeno/Desktop/Master-Artificial-Intelligence/Knowledge Representation/Second Assignment/Popper/examples/{ex}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(f"popper_results_{ex}.txt", "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
    print(f"FINISHED {ex}")

run_popper()
