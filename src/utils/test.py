import os

out_dir = "results"
out_path = os.path.join(out_dir, f"results_.csv")
print(out_path)
print(os.path.isdir(out_path))