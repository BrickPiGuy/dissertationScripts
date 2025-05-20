from train_model import train_model
from pathlib import Path
import os
import time
import glob

def get_gpu_temp_celsius():
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, "r") as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except Exception:
            continue
    return max(temps) if temps else 0.0

MAX_TEMP = 85.0
completed = 0
total = 150

for tokens in [500_000, 1_000_000, 2_000_000]:
    for trial in range(1, 51):
        output_dir = Path(f"results/{tokens}_tokens/trial_{trial}")
        result_file = output_dir / "results.txt"
        if result_file.exists():
            print(f"✅ Skipping existing trial: {tokens} tokens, Trial {trial}")
            completed += 1
            continue

        while True:
            temp = get_gpu_temp_celsius()
            print(f"🌡️ Current GPU Temp: {temp:.1f}°C")
            if temp < MAX_TEMP:
                break
            print("🛑 GPU temp too high. Cooling down...")
            time.sleep(30)

        print(f"🚀 Running trial {trial} for {tokens} tokens...")
        try:
            train_model(tokens=tokens, trial_number=trial, output_dir=output_dir)
            completed += 1
        except Exception as e:
            print(f"❌ Error during trial {trial} for {tokens} tokens: {e}")

print(f"🎉 All done! Completed {completed} of {total} trials.")
