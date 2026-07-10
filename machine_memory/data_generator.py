import numpy as np
import os
import sys
import pandas as pd
import time
import math
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

if '__file__' in locals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import SmartMetal, Metal
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import LogFermiDirac
from getelec.electron_emitter import ThermoFieldEmitter

def generate_custom_array(m, n, resolution_factor=100):
    result = []
    current = m
    epsilon = 1e-12 
    while current <= n + epsilon:

        result.append(round(current, 12))
        
        magnitude = 10**math.floor(math.log10(current + epsilon))

        step = magnitude / resolution_factor
        current += step
    return result

OUTPUT_DIR = 'machine_memory'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

my_band = Metal(lower_energy_limit=5, upper_energy_limit=12, energy_resolution=0.01)
my_solver = Noumerov()
my_potential = SchottkyPotential()
my_supply = LogFermiDirac()
emitter = ThermoFieldEmitter(my_potential, my_solver, my_supply, my_band)

parquet_writer = None

# For W as an example
fermi = np.array([7.5])
workfunction = generate_custom_array(4.4, 4.6, 10)
field = generate_custom_array(3, 5, 10)

total_sims = len(fermi) * len(workfunction) * len(field)
print(f"Starting simulation. Data will be saved to: {OUTPUT_DIR}/")

pbar = tqdm(total=total_sims, unit="sim", desc="Overall Progress")

try:
    for ef in fermi:
        file_name = os.path.join(OUTPUT_DIR, f"ef_{ef:.1f}.parquet")

        if os.path.exists(file_name):
            pbar.update(len(workfunction) * len(field))
            continue

        batch_tables = []

        for wf in workfunction:
            for f in field:
                tic = time.time_ns()

                lower_energy = max(0.001, ef - 3)
                upper_energy = ef + wf + 3

                emitter.update_params(field=f, work_function=wf, fermi=ef, temp=300, 
                                     lower_energy_lim=lower_energy, upper_energy_lim=upper_energy)
                
                ee, ss, tt = emitter._calculate_base_data()

                tt = np.log(tt)

                df_temp = pd.DataFrame({
                    'fermi': np.array([ef] * len(ee), dtype='float16'),
                    'workfunction': np.array([wf] * len(ee), dtype='float16'),
                    'field': np.array([f] * len(ee), dtype='float16'),
                    'ee': np.array(ee, dtype='float16'),
                    'tt': np.array(tt, dtype='float32')
                })

                batch_tables.append(pa.Table.from_pandas(df_temp))
                
                tac = time.time_ns()
                duration = (tac - tic) / 1e9
                
                pbar.set_postfix_str(f"Ef:{ef:.1f} Wf:{wf:.1f} F:{f:.1f} | Last: {duration:.3f}s")
                pbar.update(1)

        # Save checkpoint
        if batch_tables:
            final_ef_table = pa.concat_tables(batch_tables)
            pq.write_table(final_ef_table, file_name, compression='snappy')

except Exception as e:
    print(f"\nAn error occurred: {e}")

finally:
    pbar.close()
    print("Simulation process paused or finished.")