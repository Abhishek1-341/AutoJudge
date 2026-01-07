# AutoJudge: Predicting Programming Problem Difficulty

**Demo Video Link: [https://drive.google.com/file/d/1bmCxKA1RPnC-Gi4-9w06Q_8JCox_Xd5a/view?usp=sharing](https://drive.google.com/file/d/1bmCxKA1RPnC-Gi4-9w06Q_8JCox_Xd5a/view?usp=sharing)** 

**Live Site Link https://autojudge23322002.streamlit.app/**

---

## Project Overview

**AutoJudge** is an intelligent system designed to automatically predict the difficulty of programming problems using machine learning. By analyzing only the textual description of a problem (title, description, and input/output formats), the system performs two main tasks:

1. **Classification:** Predicts the difficulty class as **Easy, Medium, or Hard**.
2. **Regression:** Predicts a continuous **numerical difficulty score**.

## Dataset

The dataset consists of programming problems from platforms like Codeforces and Kattis.

* **Features:** Title, description, input_description, and output_description.


* **Targets:** `problem_class` and `problem_score`.


* **Key Insight:** The data showed a clear class imbalance, with **Hard** problems being the most frequent, necessitating the use of macro F1-score for evaluation.

## Approach & Models

### 1. Feature Engineering (Critical Step)

* **Text Construction:** Combined all text fields into a single unified representation.


* **Statistical Features:** Extracted handcrafted features like word count, mathematical symbol count, and Big-O notation hints to capture structural complexity.


* **Keyword Features:** Binary and count-based features for domain-specific terms (e.g., DP, Graph, Segment Tree).


* **TF-IDF:** Used unigrams and bigrams to capture semantic importance by down-weighting common words.



### 2. Models Used

* **Classification:** **Logistic Regression** was chosen for its effectiveness with sparse text data and stability. It utilized a pipeline of TF-IDF, structured features, and StandardScaler.


* **Regression:** **Ridge Regression** was selected over tree-based models as it handled the high label noise and sparse high-dimensional data more efficiently.


## 📈 Evaluation Metrics

The models were tuned using **Optuna** to achieve the following results:

| Task | Metric | Value |
| --- | --- | --- |
| **Classification** | Macro F1-Score | <br>**~0.45** |
| **Classification** | Accuracy | <br>**~0.50** |
| **Regression** | RMSE | <br>**~2.05** |

## 💻 Web Interface

A simple web UI was built using **Streamlit**.

* **Functionality:** Users can paste the problem description, input format, and output format into text boxes.


* **Output:** Upon clicking "Predict," the interface displays the predicted difficulty class and score instantly.



## ⚙️ Steps to Run Locally

1. **Clone the repository:**
```bash
git clone [your-repo-link]
cd AutoJudge

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the application:**
```bash
streamlit run app.py

```



## 📁 Repository Structure

* `data/`: Dataset used for training.
* `models/`: Saved model files for Classification and Regression.
* `notebooks/`: Data preprocessing, EDA, and model training scripts.
* `app.py`: Streamlit web UI code.
* `infrance,py`: code that runs internally in streamlit.
* `requirements.txt`: List of necessary Python packages.

---

**Developed by:** Abhishek kumar sahu,
**Branch:** Economics,
**enrollment no:** 23322002

