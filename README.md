# 🧠 Handwritten Digit Recognition using CNN
⭐ Deep Learning Project | TensorFlow + Keras | Image Classification

## 📊 Project Overview
This project focuses on classifying handwritten digits (0–9) from grayscale images using a **Convolutional Neural Network (CNN)**.

The dataset consists of 28×28 pixel digit images, similar to the MNIST dataset. The CNN model automatically learns hierarchical features from raw pixel data — eliminating the need for manual feature engineering.

The trained model achieves high accuracy on the validation set and provides a solid foundation for real-world image recognition systems.

## 🎯 Objectives
* Load and preprocess digit images for training.
* Build and train a CNN model to classify digits (0–9).
* Evaluate model performance with accuracy and confusion matrix.
* Generate predictions on new unseen data.
* Visualize training progress and results.

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Libraries:**
    * `numpy`, `pandas` → Data manipulation
    * `matplotlib`, `seaborn` → Data visualization
    * `scikit-learn` → Preprocessing, model evaluation
    * `tensorflow`, `keras` → Deep Learning model creation & training

## 📁 Dataset
* **Source:** `Train.csv` (https://drive.google.com/file/d/1fdgfv0IM3nSLYx2BRADQ2QmhagDZRRwv/view?usp=sharing)
* **Description:**
    * The dataset contains 42,000+ grayscale images of handwritten digits.
    * Each row represents one image, with 784 pixel intensity values (0–255) and one `label` column representing the digit.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `label` | Digit (0–9) | Target variable |
| `pixel0`–`pixel783` | Flattened 28×28 | Grayscale pixel intensities |

## 🔬 Methodology & Steps
1️⃣ **Data Loading & Exploration**
* Loaded the dataset from CSV and inspected its structure.
* Visualized sample digits to verify data integrity.
* Normalized pixel values (0–255 → 0–1).

2️⃣ **Data Preprocessing**
* Split data into train (80%) and test (20%) sets.
* Reshaped data to `(28, 28, 1)` format for CNN input.
* Encoded labels using `to_categorical()`.

3️⃣ **Model Architecture (CNN)**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
```
  * **Optimizer:** `Adam`
  * **Loss Function:** `Categorical Crossentropy`
  * **Metrics:** `Accuracy`

4️⃣ **Model Training**

  * Trained the CNN using `model.fit()` for multiple epochs.
  * Monitored training & validation accuracy curves.
  * Used early stopping to prevent overfitting.

5️⃣ **Model Evaluation**

  * Evaluated on internal test split using `model.evaluate()`.
  * Generated predictions and plotted a confusion matrix.
  * Achieved \>98% accuracy on unseen data.

6️⃣ **Visualization**

  * Training vs Validation accuracy/loss curves.
  * Confusion matrix to visualize prediction quality.
  * Random digit predictions displayed with predicted labels.

## 📈 Results

  * ✅ **Final Accuracy:** `~98%`
  * ✅ **Model:** Convolutional Neural Network (CNN)
  * ✅ **Optimizer:** `Adam`
  * ✅ **Loss Function:** `Categorical Crossentropy`

The model effectively classifies handwritten digits with minimal error, demonstrating strong performance on unseen data.

## 🚀 How to Run

1.  **Clone the repository**

    ```bash
    git clone https://github.com/hvlr2111/Handwritten_Digit_Recognition.git
    ```

2.  **Navigate to the project folder**

    ```bash
    cd handwritten-digit-recognition
    ```

3.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset**
    Use the above link and place the `Train.csv` file in the project root directory.

5.  **Run the notebook or script**

    ```bash
    python digit_recognition.py
    ```

    or open and run `Digit_Recognition_CNN.ipynb` cell by cell in Jupyter Notebook.

## 🧠 Future Improvements

  * Implement data augmentation to improve generalization.
  * Experiment with deeper CNN architectures (e.g., ResNet).
  * Convert the model to TensorFlow Lite for mobile inference.
  * Deploy the model using Streamlit or Flask.

## 👤 Author

  * H.V.L. Ranasinghe
  - LinkedIn: https://www.linkedin.com/in/lakshika-ranasinghe-1404ab34a/
