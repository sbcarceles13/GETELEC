## Run this to create the MM
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import joblib

ef_value = 7.5 # select the value of the fermi level for which simulation data exists
input_file = f'machine_memory/ef_{ef_value}.parquet'

output_model = f'machine_memory/ef_{ef_value}.joblib'

print(f"Training {input_file}...")

df = pd.read_parquet(input_file)

features = df[["fermi", "workfunction", "field", "ee"]]
target = df["tt"]

mem_tree = DecisionTreeRegressor(max_depth=None, random_state=42)
mem_tree.fit(features, target)

joblib.dump(mem_tree, output_model)

print(f"Successfully saved: {output_model}")