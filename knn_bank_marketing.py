import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_DIR = Path.cwd() / "data"
SRC_DIR = Path.cwd() / "src"
OUTPUT_DIR = Path.cwd() / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
ZIP_PATH = DATA_DIR / "bank-additional.zip"
CSV_REL_PATH = "bank-additional/bank-additional-full.csv"  # path inside zip

RANDOM_STATE = 42
TEST_SIZE = 0.2
KS = [3, 5, 7]


def download_and_extract():
    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        print(f"Saved {ZIP_PATH}")
    else:
        print("Zip already exists, skipping download.")

    # Extract only if CSV not present
    csv_target = DATA_DIR / "bank-additional-full.csv"
    if not csv_target.exists():
        print("Extracting CSV from zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            # extract the file to data directory
            member = CSV_REL_PATH
            z.extract(member, path=DATA_DIR)
            # Move extracted member to top-level for simpler path
            extracted = DATA_DIR / member
            shutil.move(str(extracted), str(csv_target))
            # remove now-empty extracted folder
            extracted.parent.rmdir()
        print("Extraction done.")
    else:
        print("CSV already extracted.")
    return DATA_DIR / "bank-additional-full.csv"


def load_and_preview(csv_path):
    print("Loading CSV...")
    df = pd.read_csv(csv_path, sep=';')
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Target distribution:")
    print(df['y'].value_counts(dropna=False))
    return df


def preprocess(df):
    # Convert target to 0/1
    df = df.copy()
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    # Drop columns that are identifiers or not useful? (we keep primary features)
    # For this dataset, all available features are potentially useful; keep all except none.
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # But pandas read many numeric as int; some are numeric but may be read as object - handle generically
    # We'll treat these ones as numeric explicitly:
    # common numeric columns in this dataset: age, balance, day, duration, campaign, pdays, previous
    expected_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    numeric_cols = [c for c in expected_numeric if c in df.columns]

    categorical_cols = [c for c in df.columns if c not in numeric_cols + ['y']]

    # One-hot encode categorical
    df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)
    X_num = df[numeric_cols].astype(float)
    X = pd.concat([X_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    y = df['y'].values

    # Scale numeric features (important for KNN)
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print(f"After encoding: X shape = {X.shape}")
    return X, y, numeric_cols, scaler


def run_knn_and_evaluate(X_train, X_test, y_train, y_test, k, out_prefix):
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=['no', 'yes'], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (k={k})')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['no', 'yes'])
    plt.yticks(tick_marks, ['no', 'yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f'confusion_matrix_k{k}.png'
    plt.savefig(cm_path)
    plt.close()

    # Write small result file
    with open(OUTPUT_DIR / f'result_k{k}.txt', 'w') as f:
        f.write(f"k = {k}\n")
        f.write(f"accuracy = {acc:.4f}\n")
        f.write(f"precision = {precision:.4f}\n")
        f.write(f"recall = {recall:.4f}\n")
        f.write(f"f1 = {f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report_text)

    return acc, precision, recall, f1, str(cm_path), report_text


def main():
    csv_path = download_and_extract()
    df = load_and_preview(csv_path)

    X, y, numeric_cols, scaler = preprocess(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    accuracies = []

    print("Training and evaluating KNN for ks:", KS)
    for k in KS:
        acc, prec, rec, f1, cm_path, rep = run_knn_and_evaluate(X_train, X_test, y_train, y_test, k, OUTPUT_DIR)
        print(f"k={k} => acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
        results.append({'k': k, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'cm': cm_path})
        accuracies.append(acc)

    # Plot accuracy vs k
    plt.figure()
    plt.plot(KS, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('KNN accuracy vs k (Bank Marketing)')
    plt.grid(True)
    acc_plot_path = OUTPUT_DIR / 'accuracy_vs_k.png'
    plt.savefig(acc_plot_path)
    plt.close()
    print("Saved accuracy plot to:", acc_plot_path)

    # Summary CSV of results
    pd.DataFrame(results).to_csv(OUTPUT_DIR / 'knn_results_summary.csv', index=False)
    print("Saved summary CSV to:", OUTPUT_DIR / 'knn_results_summary.csv')

    print("All outputs saved in folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
