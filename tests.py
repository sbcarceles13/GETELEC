import os
import sys
import subprocess

# from getelec import band_structure
os.environ["MPLBACKEND"] = "Agg"


def run_check(label, condition_bool):
    status = "PASS" if condition_bool else "FAIL"
    print(f"{status}: {label}")
    return condition_bool

def main():
    root_dir = os.getcwd()
    venv_path = os.path.join(root_dir, "gt_venv")
    examples_dir = os.path.join(root_dir, "examples")
    
    all_passed = True

    all_passed &= run_check("Virtual Environment exists", os.path.exists(venv_path))

    txt_exists = any(f.endswith('.txt') for f in os.listdir(examples_dir)) if os.path.exists(examples_dir) else False
    all_passed &= run_check("Found .txt documents in /examples", txt_exists)

    try:
        sys.path.append(root_dir)
        
        from getelec import potential_barrier, transmission_solver, electron_supply, electron_emitter, band_structure

        my_potential = potential_barrier.SchottkyPotential()
        my_band = band_structure.SmartMetal()
        my_solver = transmission_solver.Noumerov()
        my_supply = electron_supply.LogFermiDirac()

        emitter = electron_emitter.MetalEmitter(my_potential, my_solver, my_supply, my_band)
        
        print("PASS: Modules loaded and GETELEC initialized")
    except Exception as e:
        print(f"FAIL: Module initialization failed: {e}")
        all_passed = False

    python_exe = os.path.join(venv_path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv_path, "bin", "python")
    
    for script in ["calculate_energy_distributions.py", "calculate_iv_curve.py"]:
        script_path = os.path.join(examples_dir, script)
        if os.path.exists(script_path):
            proc = subprocess.run([python_exe, script_path], env=os.environ, capture_output=True)
            all_passed &= run_check(f"Executed {script}", proc.returncode == 0)
        else:
            print(f"SKIP: {script} not found")


    data_path = os.path.join(examples_dir, "test_data.txt")
    if all_passed and os.path.exists(data_path):
        failures = 0
        tolerance = 1e-6
        
        with open(data_path, "r") as f:
            next(f) 
            for line in f:
                v1, v2, v3, v4, expected = [float(x) for x in line.strip().split(",")]
                emitter.update_params(field=v3, work_function=v2, fermi=v1, temp=v4)
                actual = emitter.calculate_current_density()
                
                denom = expected if expected != 0 else 1e-12
                if abs((actual - expected) / denom) > tolerance:
                    failures += 1
        
        all_passed &= run_check("Accuracy within 1e-6 tolerance", failures == 0)
        if failures > 0: print(f"   (Failed samples: {failures})")
    else:
        print("SKIP: Accuracy test skipped due to previous failures or missing data.")

    print("\n" + "="*30)
    if all_passed:
        print("FINAL RESULT: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("FINAL RESULT: TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()