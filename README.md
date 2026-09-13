# Medical Anomaly & Model Steering Studio

An interactive, human-in-the-loop machine learning tool for exploring clinical data. Instead of treating a model's output as a fixed black box, this app lets a user visually inspect flagged patients, correct the model's mistakes, and watch its decision logic update in real time.

## The Problem

Machine learning models used in healthcare settings often flag "anomalies" based on statistical patterns alone — without ever seeing whether those patterns actually correspond to real medical risk. When a model gets something wrong, a clinician typically has no way to correct it short of retraining from scratch with new code. This project explores a lightweight interaction pattern for closing that gap: let a human directly relabel misclassified cases on a visualization, and have the model retrain immediately, showing its updated reasoning in plain, readable rules.

## What It Does

1. **Loads and cleans** the UCI Heart Disease dataset (297 patients, 13 clinical features).
2. **Reduces dimensionality** with PCA, projecting the 13 features into 2D for visualization.
3. **Flags anomalies** using an unsupervised IsolationForest, with no access to actual diagnoses.
4. **Trains a supervised DecisionTreeClassifier** on the real diagnosis labels, and extracts its logic as a readable tree diagram.
5. **Lets the user interact**: select any points on the scatter plot (click, box, or lasso), relabel them, and trigger a live retrain — the decision tree visually updates to reflect the correction.

## Screenshots

*(add screenshots here once captured)*

![Patient scatter plot](<img width="855" height="429" alt="ScatterPlot " src="https://github.com/user-attachments/assets/1814e484-32fa-4ebd-b6c4-ee27fc5b7f23" />
)
![Steering panel and decision tree](<img width="1324" height="678" alt="DecisionTree" src="https://github.com/user-attachments/assets/6c005a26-a7e5-4deb-b559-0aa52e493e9c" />
)

## Tech Stack

- **Python** — pandas, numpy
- **scikit-learn** — StandardScaler, PCA, IsolationForest, DecisionTreeClassifier
- **Streamlit** — interactive web app framework
- **Plotly** — interactive scatter plot with point selection
- **Matplotlib** — decision tree visualization

## Running Locally

```bash
git clone https://github.com/Jocelyn18c/Medical-Anomaly-Model-Steering-Studio.git
cd Medical-Anomaly-Model-Steering-Studio
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Key Design Decisions

- **Separating unsupervised and supervised signals.** IsolationForest and the DecisionTreeClassifier answer different questions: one detects statistical outliers with zero label information, the other learns from real diagnoses. Comparing their outputs revealed that only ~60% of flagged anomalies actually had heart disease — a useful, honest finding about the limits of unsupervised flagging alone.
- **Keeping the tree shallow (`max_depth=3`).** This trades some predictive accuracy for interpretability, so the extracted rules stay human-readable rather than an opaque deep tree.
- **Being transparent about information loss.** The 2D PCA projection retains only ~36% of the original variance — a real limitation, documented rather than hidden, since an honest account of a model's constraints matters more than an inflated one.

## Dataset

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), via the [Ruohan-Yang/Heart-Disease-Data-Set](https://github.com/Ruohan-Yang/Heart-Disease-Data-Set) mirror.

## License

MIT
