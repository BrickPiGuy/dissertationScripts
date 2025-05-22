OpenLLaMA Experimental Procedure
1. Environment Preparation

Before running any trials, prepare your Jetson AGX Orin environment.

•	Option A: One-liner package install (if skipping shell script)

pip install torch torchvision transformers datasets tokenizers accelerate \
            onnx onnxruntime numpy pandas matplotlib scikit-learn

•	Option B: Use setup_openllama_env.sh

chmod +x setup_openllama_env.sh
./setup_openllama_env.sh
source ~/openllama_env/bin/activate

2. (Optional) External SD Card Setup

Use the setup_sd_card_mount.sh script to mount and use a microSD card for cache storage. This improves performance and avoids filling internal storage.

2. (Optional) SSD Setup

Use the setup_openllama_ssd.sh script to mount and use a SSD for cache storage. This also improves performance and avoids filling internal storage.

4. Training Script: train_model.py

This script trains the OpenLLaMA model for 3 epochs using TinyStories data. It logs results to both results.txt (per trial) and run_log.csv (cumulative).

4. Run Full Experiment: run_all_trials.py

This script runs 150 trials across 3 token levels. It includes a GPU temperature safeguard and skips completed trials.

5. Data Analysis: analyze_results.py

This script performs repeated-measures ANOVA on the run_log.csv file, checks assumptions (normality, sphericity), and applies Bonferroni-corrected pairwise comparisons.

Final Notes

- Ensure good airflow using a USB fan to reduce trial cooldowns.
- Trials may take 30–45 hours depending on cooling and token size.
- Monitor logs via the terminal or redirect output to a log file.
- Data analysis requires scipy, statsmodels, and pingouin libraries.

[![DOI](https://zenodo.org/badge/986691745.svg)](https://doi.org/10.5281/zenodo.15468658)
