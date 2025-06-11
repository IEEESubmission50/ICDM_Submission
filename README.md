# SAVL-II Experimentation Pipeline

This project contains the complete Python code and configuration for running SAVL-II.

## 1. Prerequisites

- **Python**: Version 3.12 or newer is required.
- **Database**: A running MySQL server instance.
- **API Key**: An API key for the LLM provider (e.g., TogetherAI) is needed to run the feature generation step.

## 2. Setup Instructions

Follow these steps to configure the project environment.

#### Step A: Clone & Set Up Virtual Environment

First, clone or download the project files. It is highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Navigate to the project root directory
cd /path/to/your/project/

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

#### Step B: Install Required Libraries

Install all necessary Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### Step C: Configure Database Access

The scripts need to connect to a MySQL database.

1.  Navigate to the `ICDM_Submission/src/` directory.
2.  Create a new file named `db_access_config.py`.
3.  Copy the following content into the file and **replace the placeholder values** with your actual database credentials.

    ```python
    # ICDM_Submission/src/db_access_config.py

    DB_CONFIG = {
        'user': 'YOUR_DATABASE_USER',
        'password': 'YOUR_DATABASE_PASSWORD',
        'host': 'YOUR_DATABASE_HOST',  
        'port': 3306,
        'database': 'YOUR_DATABASE_NAME'
    }
    ```

#### Step D: Configure LLM API Key

1.  Open the file `ICDM_Submission/src/llm_exchange_config.json`.
2.  Find the `api_key` field and replace `"ENTER API KEY"` with your actual key.

## 3. Running the Pipeline

The pipeline is designed to be run in sequential order. The following example uses the "shopper analytics" (dunnhumby) dataset.

**Execute these commands from the project's root directory.**

#### Step 1: Data Preparation

In the data subfolder there are two datasets: Dunnhumby and Service Now incident reports. In each folder, there are two files, one that creates the master table and the other that generates the host and remote views. To run both files, cd into the respective dataset name's subfolder in the data folder and run their respective master_table and then create_host_remote_view files.
**Make sure the evaluation_spec.json file is present in the src folder corresponding to the dataset you want to run the prediction tasks.**

```bash

#### Step 2: Run the Pipeline

These scripts must be run from the `src` directory.

```bash
# Change directory to the source folder
cd ICDM_Submission/src

# 3. Run L1: Teacher model training and error segment discovery
python l1.py

# 4. Run LLM Exchange: Generates surrogate feature code from LLM
python llm_exchange.py

# 5. Run L1 R-Hats: Generates p-hats and final surrogates
python l1_r_hats.py

# 6. Run L2: Trains the final student model with all features
python l2.py
```

#### Step 3: (Optional) Run Baselines and Ablation Studies

These scripts evaluate the performance of the final model against various baselines and configurations.

```bash
# Ensure you are still in the ICDM_Submission/src directory

# 7. Run baseline models for comparison
python baseline_group.py

# 8. Run ablation tests to evaluate components of the system
python ablation_tests.py
```

## 4. Configuration

The behavior of each script is controlled by its corresponding `_config.json` file in the `src/` directory and the main `evaluation_spec.json` file located in each data directory (e.g., `data/dunnhumby_weekly_with_analytics/`).

-   **`evaluation_spec.json`**: Defines the dataset schemas, table names, and predictive tasks.
-   **`l1_config.json`**: Controls parameters for the L1 teacher models, segment identification, and remote party analysis.
-   **`l2_config.json`**: Controls parameters for the L2 student model and the final ensemble.
-   **`llm_exchange_config.json`**: Contains the LLM model name, API endpoint, and other related settings.
-   **`baselines_config.json`**: Configures the different baseline models to run for comparison.