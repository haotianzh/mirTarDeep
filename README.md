# MicroRNA Target Prediction using Deep Learning

This project utilizes a deep learning model, specifically a stacked GRU network, to predict microRNA targets. It takes microRNA sequences, target sequences, and Minimum Free Energy (MFE) as input to predict the likelihood of interaction.

## Data Generation

The `data/data_generate.py` script is used to generate pseudo microRNA data. This is often a technique used in machine learning to augment datasets where one class (e.g., non-interacting pairs) might be underrepresented or to create negative samples.

The script reads a file containing tab-separated microRNA and target sequences. For each pair, it generates a pseudo microRNA sequence of the same length as the original microRNA by randomly selecting bases (A, G, C, T). The output is a file where each line contains the original target, the generated pseudo microRNA, and the original microRNA.

**Note:** The current script `data/data_generate.py` has a **hardcoded input file path** (`T:/microRNA_3/tar_mir.txt`) and **output file path** (`T:\microRNA_3\mti_generate3.txt`). You **must** modify these paths in the script to point to your desired input and output locations.

**Input Format (for `data/data_generate.py`):**
A tab-separated file where each line is:
`microrna_sequence	target_sequence`

**Output Format (from `data/data_generate.py`):**
A tab-separated file where each line is:
`target_sequence	pseudo_microrna_sequence	original_microrna_sequence`

## Model Training

The `train.py` script is used to train the microRNA target prediction model.

### Prerequisites

Before running the training script, ensure you have the following dependencies installed:
*   Python 3
*   Keras
*   NumPy
*   Matplotlib (for plotting, though not explicitly used for saving models in the provided script's main flow)
*   TensorFlow (as Keras backend)

You can typically install these using pip:
```bash
pip install keras numpy matplotlib tensorflow
```

### Input Data

The training script expects a specific input file format. Based on the script, the input file (e.g., `T:/microRNA_3/mfe_p_0315.txt` in the original script) should be a tab-separated file where each line contains:
`target_sequence	pseudo_mirna_sequence	real_mirna_sequence	pseudo_mfe	pseudo_p_value	real_mfe	real_p_value`

Note: While `pseudo_p_value` and `real_p_value` are part of this input format, the model architecture described in `train.py` primarily utilizes the MFE (Minimum Free Energy) values (`pseudo_mfe`, `real_mfe`) for training.

**Important:**
*   Target sequences longer than 100 bases will be skipped.
*   The script contains a **hardcoded path to this input file**: `T:/microRNA_3/mfe_p_0315.txt`. You **must** modify this path in `train.py` (inside the `generate_data` function) to point to your actual training data file.
*   The script also contains a **hardcoded path for saving the trained models**: `'your path'` (inside the training loop). You **must** modify this to your desired directory for saving models. The models will be saved in HDF5 format (`.h5`).

### Running Training

Once the prerequisites are met and the input data path is correctly set in the script:

1.  **Modify `train.py`:**
    *   Change the input file path in the `generate_data()` function.
    *   Change the model saving path in the training loop (e.g., `model.save('models/model_fold_{}.h5'.format(i))`).
2.  **Execute the script:**
    ```bash
    python train.py
    ```

The script performs 5-fold cross-validation. For each fold, it will train a model and save it. The training progress, including accuracy and loss, will be printed to the console.

The trained models will be saved in the directory you specified (e.g., `models/`).

## Prediction

The `predict.py` script uses the trained models to predict microRNA-target interactions.

### Prerequisites

Ensure you have the same dependencies installed as for training:
*   Keras
*   NumPy
*   TensorFlow

### Input Data

The prediction script requires an input file passed as a command-line argument. Each line in this file should be tab-separated and contain the following four fields:

`mtiname	mirna_sequence	target_sequence	mfe_value`

*   `mtiname`: A unique identifier for the microRNA-target interaction pair.
*   `mirna_sequence`: The sequence of the microRNA (U will be converted to T).
*   `target_sequence`: The sequence of the target RNA (U will be converted to T).
*   `mfe_value`: The Minimum Free Energy of the interaction.

An example input file (`example/1.example`) is provided:
```
mti-1	TAGGATGCCTGGAACTTGCCGGT	TGTGTATGTGTACCTTTCAGCATCCTAGGAATTT	-23.6
```

### Running Prediction

1.  **Ensure models are present:** The script loads models from the `models/` directory. Make sure your trained `.h5` model files are located there.
2.  **Execute the script:**
    ```bash
    python predict.py path/to/your/input_file.txt
    ```
    Replace `path/to/your/input_file.txt` with the actual path to your input data.

### Output

The script will print the prediction results to the console. For each input line, it will output:
`mtiname : prediction_score`

The `prediction_score` is an average of the scores obtained from all models found in the `models/` directory. A higher score generally indicates a higher likelihood of interaction.

## Model Architecture

The prediction model is a neural network built using Keras. The architecture can be summarized as follows:

*   **Inputs:**
    1.  Target RNA sequence (padded to 100 bases)
    2.  MicroRNA sequence (padded to 25 bases)
    3.  Minimum Free Energy (MFE) of the interaction

*   **Sequence Processing:**
    *   Both target and microRNA sequences are first passed through an Embedding layer to convert their base representation (A, G, C, T) into dense vectors.
    *   Each embedded sequence is then processed by a stack of two Gated Recurrent Unit (GRU) layers. A Dropout layer is applied between the GRU layers for regularization.

*   **Interaction and Prediction:**
    *   The outputs from the final GRU layers for the target and microRNA are combined using a Dot product.
    *   The MFE value is then concatenated with this interaction score.
    *   Finally, a Dense layer with a softmax activation function outputs a 2-dimensional vector, representing the probabilities for the two classes. The model is trained using binary cross-entropy, suitable for distinguishing between two classes (e.g., interacting vs. non-interacting pairs).

This architecture allows the model to learn patterns from the sequences themselves and incorporate the thermodynamic stability (MFE) of the pairing.

## Project Structure

```
.
├── data/
│   └── data_generate.py  # Script to generate pseudo microRNA data
├── example/
│   └── 1.example         # Example input file for predict.py
├── models/
│   ├── model1.h5         # Example trained model (others would be model2.h5, etc.)
│   └── test_acc.txt      # Example file; could be used to store test accuracies. Note: `train.py` prints accuracies to console during cross-validation; this file would be for manual record-keeping or generated by a separate script.
├── predict.py            # Script to make predictions using trained models
├── train.py              # Script to train the prediction models
└── README.md             # This file
```

*   **`data/data_generate.py`**: Generates augmented training data by creating pseudo microRNA sequences.
*   **`example/1.example`**: An example file demonstrating the input format for `predict.py`.
*   **`models/`**: This directory is intended to store the trained model files (in `.h5` format). The `predict.py` script loads models from here. The `train.py` script should be configured to save models into this directory.
*   **`predict.py`**: Takes an input file with miRNA, target, and MFE data, and predicts interaction scores using models from the `models/` directory.
*   **`train.py`**: Trains the deep learning models using provided training data. You **must** modify paths within this script for your data and desired model output location.

## Getting Started / Workflow

Here's a typical workflow for using this project:

1.  **Setup & Dependencies:**
    *   Ensure Python 3 is installed.
    *   Install necessary Python packages:
        ```bash
        pip install keras numpy matplotlib tensorflow
        ```

2.  **Data Preparation (Optional - if you need to generate pseudo-negative samples):**
    *   Prepare your input file of `microrna_sequence	target_sequence` pairs.
    *   Modify the input and output file paths within `data/data_generate.py`. **Remember these are hardcoded.**
    *   Run the script: `python data/data_generate.py`.
    *   This will produce a file with `target_sequence	pseudo_microrna_sequence	original_microrna_sequence`. You might need to further process this to match the format required by `train.py` (which includes MFE values, etc.).

3.  **Model Training:**
    *   Prepare your main training data file in the format: `target	pseudo_mirna	real_mirna	pseudo_mfe	pseudo_p	real_mfe	real_p`.
    *   Modify `train.py` to:
        *   Point to your training data file (in `generate_data` function). **Remember this is hardcoded.**
        *   Specify the directory where trained models should be saved (e.g., `models/model_fold_{}.h5`). It's recommended to save them in the `models/` directory. **Remember this path is also hardcoded.**
    *   Run the training script: `python train.py`.
    *   Models will be saved (e.g., `model_fold_0.h5`, `model_fold_1.h5`, etc.) in the specified directory.

4.  **Prediction:**
    *   Ensure your trained `.h5` models are in the `models/` directory (or that `predict.py` is pointing to the correct directory if you changed it, though `predict.py` currently hardcodes `models/`).
    *   Prepare your input file for prediction with `mtiname	mirna_sequence	target_sequence	mfe_value` on each line.
    *   Run the prediction script:
        ```bash
        python predict.py path/to/your/prediction_input_file.txt
        ```
    *   View the output scores printed to the console.

**Important Considerations:**
*   The script `predict.py` expects models to be in a `models/` subdirectory relative to its location. If you save models elsewhere during training, you'll need to either move them to `models/` or modify the path in `predict.py`.
