# Customer Churn Prediction & Analytics Dashboard

This project provides an end-to-end solution for predicting customer churn. It involves data processing, model training, customer segmentation, and an interactive web dashboard built with Streamlit to visualize the results. The goal is to empower business stakeholders to identify at-risk customers, understand churn drivers, and analyze customer segments proactively.

 <!-- Replace with a screenshot of your dashboard -->

---

## 🚀 Key Features

*   **Churn Prediction:** A machine learning model predicts the probability of a customer churning.
*   **Customer Segmentation:** Customers are grouped into meaningful segments (e.g., "Champions", "At Risk") using clustering.
*   **Interactive Dashboard:** A user-friendly interface to explore KPIs, sales trends, segment distributions, and model insights.
*   **Model Explainability:** Uses SHAP to identify the key features influencing churn predictions.
*   **High-Risk Customer Identification:** Easily filter and view a list of customers with the highest probability of churning.
*   **Individual Customer Lookup:** Search for a specific customer to see their details and churn score.

---

## 🛠️ Tech Stack

*   **Backend & Machine Learning:** Python, Pandas, Scikit-learn
*   **Dashboard:** Streamlit
*   **Data Visualization:** Plotly Express

---

## 📂 Project Structure

```
churn_project/
├── artifacts/            # Stores output files like models, scores, and datasets
├── data/                 # (Recommended) Raw input data
├── notebooks/            # (Recommended) Jupyter notebooks for exploration
└── src/                  # Source code
    ├── train.py          # (Assumed) Script for data processing and model training
    └── streamlit_app.py  # The Streamlit dashboard application
```

---

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SiddeshDLokhande/churn_project.git
    cd churn_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    *(Note: You will need to create a `requirements.txt` file for this step to work.)*
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

1.  **Generate Artifacts:**
    First, you need to run the training pipeline. This script will process the raw data, train the churn model, perform segmentation, and save the results (e.g., `churn_scores.parquet`, `segment_summary.parquet`) into the `artifacts/` directory.

    ```bash
    # (Assuming your training script is named train.py)
    python src/train.py
    ```

2.  **Launch the Dashboard:**
    Once the artifacts are generated, you can launch the Streamlit dashboard.

    ```bash
    streamlit run project/src/streamlit_app.py
    ```

    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

