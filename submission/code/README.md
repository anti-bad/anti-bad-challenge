# Code Submission for Test Phase

## Overview

For the **test phase**, we require participants to submit their defense code along with prediction outputs. This ensures the reliability and reproducibility of submitted results. We will perform sanity checks to verify that your results can be reproduced from the provided models and datasets.

## Directory Structure

Place your defense code in the corresponding track directory:

```
code/
├── classification-track/    # Code for classification track tasks
├── generation-track/        # Code for generation track tasks
├── multilingual-track/      # Code for multilingual track tasks
├── requirements.txt         # (Optional) Python dependencies
└── README.md               # This file - update with your instructions
```

## Submission Requirements

1. **Code Organization**: Place your defense implementation in the appropriate track directory (classification-track, generation-track, or multilingual-track). You may submit Python scripts (`.py` files) or Jupyter notebooks (`.ipynb` files).

2. **Dependencies**:
   - Include a `requirements.txt` file if your code requires specific Python packages
   - Alternatively, if you have specific version requirements or complex dependencies, you may provide a Docker image ID

3. **Documentation**: **Update this README** with clear instructions on how to run your defense method, including:
   - Command-line instructions to reproduce your results (or notebook execution instructions)
   - Any configuration or hyperparameters used
   - Runtime requirements: Your defense must complete within **24 hours on a single A100 80GB GPU**

## Confidentiality

All submitted code will remain **confidential** and will be used solely for verification purposes by the organizers. Your code will not be shared publicly.

For more details, please refer to the FAQ at [anti-bad.github.io/rules/](https://anti-bad.github.io/rules/).
