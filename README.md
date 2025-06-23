# DCASE2025 - Task 1 - Inference Package

Contact: **Shuwei Zhang** (eophoeny@gmail.com)

Official Task Description:  
🔗 [DCASE Website](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information) 
📄 [Task Description Paper (arXiv)](https://arxiv.org/pdf/2505.01747) 


## Getting Started

Follow the steps below to prepare and test your DCASE 2025 Task 1 inference submission:

1. Clone this repository:

```bash
git clone https://github.com/CPJKU/dcase2025_task1_inference
cd dcase2025_task1_inference
```

2. Rename the package (`Schmid_CPJKU_task1`) using your official submission label (see [here](https://dcase.community/challenge2024/submission#submission-label) for informations on how to form your submission label).
3. Rename the module (`Schmid_CPJKU_task1_1` inside package) using your submission label + the submission index (`1` in the example). You may submit up to four modules with increasing submission index (`1` to `4`).
4. Create a Conda environment: `conda create -n d25_t1_inference python=3.13`. Activate your conda environment.
5. Install your package locally `pip install -e .`. Don't forget to adapt the `requirements.txt` file later on if you add additional dependencies.
6. Implement your submission module(s) by defining the required API functions (see above). 
7. Verify that your models comply with the complexity limits (MACs, Params):

```python test_complexity.py --submission_name <submission_label> --submission_index <submission_number>```

8. Download the evaluation set (to be released on June 1st). 
9. Evaluate your submissions on the test split and generate evaluation set predictions:
```
python evaluate_submission.py \
    --submission_name <submission_label> \
    --submission_index <submission_number> \
    --dev_set_dir /path/to/TAU-2022-development-set/ \
    --eval_set_dir /path/to/TAU-2025-eval-set/
```

After successfully running the scripts in steps 8. and 9., a folder `predictions` will be generated inside `dcase2025_task1_inference`:

```
predictions/
└── <submission_label>_<submission_index>/
    ├── output.csv             # Evaluation set predictions (submit this file)
    ├── model_state_dict.pt    # Model weights (optional, for reproducibility)
    ├── test_accuracy.json     # Test set accuracy (sanity check only)
    └── complexity.json        # MACs and parameter memoyr per device model
└── <submission_label>_<submission_index>/ # up to four submissions
.
.
.
```
