# reassemble.py
import os
import zipfile

shard_dir = "animal_best_shards"
output_dir = "models"
output_pt = os.path.join(output_dir, "animal_best.pt")

os.makedirs(output_dir, exist_ok=True)

print("Reassembling model...\n")

with zipfile.ZipFile(output_pt, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(shard_dir):
        for file in files:
            full_path = os.path.join(root, file)

            # keep exact structure INCLUDING .data folder
            relative_path = os.path.relpath(full_path, shard_dir)

            arcname = os.path.join("archive", relative_path)

            zf.write(full_path, arcname)
            print(f"packed: {relative_path}")

print(f"\n Done! Saved to: {output_pt}")