import subprocess

def run_multiple_times(n=100):
    """Runs the command `python test_model.py 0 --variance` n times."""
    for i in range(n):
        print("QALY Run", i)
        subprocess.run(["python", "test_model.py", "0", "--variance"], check=True)
    for i in range(n):
        print("HSC Run", i)
        subprocess.run(["python", "test_model.py", "2", "--variance"], check=True)

if __name__ == "__main__":
    run_multiple_times(10)
