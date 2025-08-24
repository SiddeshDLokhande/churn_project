import pandas as pd
import numpy as np
import warnings
from sklearn.utils import class_weight
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")

import shap

DATA = Path("E:\churn_project\project\data")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# data loading
def load_data():
  customers=pd.read_csv(DATA/"customers.csv")
  products=pd.read_csv(DATA/"products.csv")
  transactions=pd.read_csv(DATA/"transactions.csv")

  customers["customer_id"]=customers["customer_id"].astype(str)
  transactions["customer_id"]=transactions["customer_id"].astype(str)
  if "product_id" in transactions.columns:
    transactions["product_id"]=transactions["product_id"].astype(str)
  if "product_id" in products.columns:
    products["product_id"]=products["product_id"].astype(str)

  if "date" in transactions.columns:
    transactions["date"] = pd.to_datetime(transactions["date"], errors='coerce')

  return customers,products,transactions

def derive_churn_if_needed(customers, transactions, end_date=None, inactivity_days=90):
    if end_date is None and not transactions.empty:
        end_date = transactions["date"].max()
    if "churned" in customers.columns:
        # Ensure last_activity_date exists for consistency
        customers = customers.copy()
        if "last_activity_date" not in customers.columns:
            last_tx = transactions.groupby("customer_id")["date"].max()
            customers = customers.set_index("customer_id")
            customers["last_activity_date"] = pd.to_datetime(customers.get("last_activity_date", pd.NaT))
            customers.loc[last_tx.index, "last_activity_date"] = last_tx
            customers = customers.reset_index()
        return customers

    # Compute churn from inactivity
    last_tx = transactions.groupby("customer_id")["date"].max()
    customers = customers.set_index("customer_id")
    customers["last_activity_date"] = pd.to_datetime(customers.get("last_activity_date", pd.NaT))
    customers.loc[last_tx.index, "last_activity_date"] = last_tx
    if end_date is None:
        end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date)
    customers["churned"] = ((end_date - customers["last_activity_date"]) > pd.Timedelta(days=inactivity_days)).astype(int)
    # tenure_days from signup_date to end_date
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers["tenure_days"] = (end_date.normalize() - customers["signup_date"].dt.normalize()).dt.days
    customers = customers.reset_index()
    return customers

# feature engineering
def build_features(customers, products, transactions):
  tx=transactions.copy()
  if "product_id" in tx.columns and "product_id" in products.columns:
    tx=tx.merge(products[["product_id","category"]],on="product_id",how="left")

  fav_category=tx.groupby("customer_id")["category"].agg(lambda x:x.mode().get(0,"unknown")).rename("fav_category")

  agg= tx.groupby("customer_id").agg(
      tx_count=("transaction_id","count"),#total number of transaction
      amt_sum=("amount","sum"),
      amt_mean=("amount","mean"),
      amt_std=("amount","std"),
      first_tx=("date","min"),
      last_tx=("date","max"),
      category_nunique=("category","nunique"),#number of unique categories bought
      channel_nunique=("channel","nunique")#number of unique channels used

  )

  if not tx.empty:
    today=pd.to_datetime(tx["date"].max()) # convert to datetime

  else:
    today=pd.Timestamp.today()

  agg["first_tx"] = pd.to_datetime(agg["first_tx"]) # Convert to datetime
  agg["last_tx"] = pd.to_datetime(agg["last_tx"]) # Convert to datetime

  print(f"Data type of today: {type(today)}")
  print(f"Data type of agg['last_tx']: {agg['last_tx'].dtype}")

  agg["recency_days"]=(today-agg["last_tx"]).dt.days #Days since the customer’s last transaction.
  agg["tenure_days_tx"]=(today-agg["first_tx"]).dt.days.clip(lower=0) #Time between first and last transaction (clipped to 0 to avoid negative values if data is messy).
  agg["avg_tx_interval"]=agg["tenure_days_tx"]/agg["tx_count"].replace(0,np.nan) #Average spacing between transactions in days (avoids division by zero by replacing 0 with NaN).

  df=customers.merge(agg,on="customer_id",how="left").merge(fav_category,on="customer_id",how="left")
  fill_0=['tx_count','amt_sum','amt_mean','amt_std','first_tx','last_tx','category_nunique','channel_nunique','recency_days','tenure_days_tx','avg_tx_interval']
  for c in fill_0:
    if c in df.columns:
      df[c]=df[c].fillna(0)

  if "age" in df.columns:
    df["age"]=df["age"].fillna(df["age"].mean())

  for col in ["gender","region","plan","fav_category"]:
    if col in df.columns:
      df[col]=df[col].fillna("unknown")

  # saperating categorical and numerical data
  y=df["churned"].astype(int) if "churned" in df.columns else None
  cat= [c for c in ["gender","region","fav_category","plan"] if c in df.columns]
  num=[c for c in ["age","tenure_days","tx_count","amt_sum","amt_mean","amt_std","recency_days","tenure_days_tx","avg_tx_interval","category_nunique","channel_nunique"] if c in df.columns]

  X=df[cat + num].copy()

  pre = ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat),
        ("num",StandardScaler(),num)
  ])

  return df,X,y,pre,cat,num

#ann model
def build_ann(input_shape):

    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

#rnn model
def build_rnn(input_shape):

    model = keras.Sequential([
        layers.Input(shape=(1, input_shape)),  # Expects data reshaped to (samples, 1, features)
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# training models
def train_and_select(X_train,y_train,preprocessor):
    print("Training models.....")
    # For sklearn models, create a pipeline first
    lr_pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    rf_pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, class_weight="balanced_subsample", random_state=42))])

    # For Keras models, we need the processed data shape
    X_train_processed = preprocessor.fit_transform(X_train)
    n_features = X_train_processed.shape[1]

    models = {
        "logistic": lr_pipe,
        "random_forest": rf_pipe,
        "ann": build_ann(n_features),
        "rnn": build_rnn(n_features)
    }

    best_name, best_auc, best_model = None, -1, None
    class_weights_dict = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    for name, model in models.items():
        print(f"Training {name}.....")

        if name in ["ann", "rnn"]:
            data = np.reshape(X_train_processed, (X_train_processed.shape[0], 1, n_features)) if name == "rnn" else X_train_processed
            model.fit(data, y_train, epochs=20, batch_size=64, class_weight=class_weights_dict, verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor='auc', patience=3, mode='max', restore_best_weights=True)])
            proba_tr = model.predict(data).ravel()
            auc_tr = roc_auc_score(y_train, proba_tr)
        else:  # scikit-learn pipelines
            model.fit(X_train, y_train)
            proba_tr = model.predict_proba(X_train)[:, 1]
            auc_tr = roc_auc_score(y_train, proba_tr)

        print(f"    Training AUC for {name}: {auc_tr:.4f}")
        if auc_tr > best_auc:
            best_auc, best_model, best_name = auc_tr, model, name

    print(f"\nBest model selected: {best_name} with training AUC: {best_auc:.4f}")

    # best_model is already the final, trained model (pipeline or keras model)
    return best_model, best_name

# model evaluation
def evaluate(model, X_test, y_test, model_name, preprocessor):

    print("Evaluating best model...")
    if model_name in ["ann", "rnn"]:
        X_test_processed = preprocessor.transform(X_test)
        if model_name == "rnn": X_test_processed = np.reshape(X_test_processed, (X_test_processed.shape[0], 1, X_test_processed.shape[1]))
        proba = model.predict(X_test_processed).ravel()
    else:
        proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    print(f"  Test Set Performance: AUC={auc:.3f} | AUPRC={auprc:.3f}")
    return proba

# segmentation
def build_segments(features_df, k=5):

    print("Building customer segments...")
    seg_cols = [c for c in ["tx_count", "amt_sum", "recency_days"] if c in features_df.columns and features_df[c].notna().any()]
    if len(seg_cols) < 2:
        print("Not enough segmentation columns with data found. Skipping segmentation.")
        features_df["segment"] = "N/A"
        features_df["segment_label"] = "N/A"
        return features_df, None

    mat = features_df[seg_cols].fillna(0).values
    mat_scaled = StandardScaler().fit_transform(mat)

    if len(mat_scaled) < k:
        print(f"Warning: Number of customers ({len(mat_scaled)}) is less than k ({k}). Adjusting k.")
        k = len(mat_scaled)

    if k < 2:
        print("Not enough customers to form segments. Skipping segmentation.")
        features_df["segment"] = "N/A"
        features_df["segment_label"] = "N/A"
        return features_df, None

    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    features_df["segment"] = km.fit_predict(mat_scaled)

    summary = features_df.groupby("segment")[seg_cols + ["churned"]].mean()
    summary["size"] = features_df.groupby("segment").size()

    # --- Generate human-readable labels based on RFM-like scores ---
    summary['recency_inv'] = summary['recency_days'].max() - summary['recency_days']
    summary['rank_tx'] = summary['tx_count'].rank(method='first')
    summary['rank_amt'] = summary['amt_sum'].rank(method='first')
    summary['rank_recency'] = summary['recency_inv'].rank(method='first')
    summary['total_score'] = summary['rank_tx'] + summary['rank_amt'] + summary['rank_recency']

    label_map_list = ["Champions", "Loyal Customers", "Potential Loyalists", "Needs Attention", "At Risk"]
    sorted_indices = summary.sort_values('total_score', ascending=False).index
    label_mapping = {index: label_map_list[i] if i < len(label_map_list) else f"Segment {index}" for i, index in enumerate(sorted_indices)}

    summary['label'] = summary.index.map(label_mapping)
    features_df['segment_label'] = features_df['segment'].map(label_mapping)
    return features_df, summary.drop(columns=['recency_inv', 'rank_tx', 'rank_amt', 'rank_recency', 'total_score'])

def main():
    customers, products, transactions = load_data()
    customers = derive_churn_if_needed(customers, transactions)
    features_df, X, y, preprocessor, cat_cols, num_cols = build_features(customers, products, transactions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model, name = train_and_select(X_train, y_train, preprocessor)
    proba_test = evaluate(model, X_test, y_test, name, preprocessor)

    # --- Create final artifacts ---
    print("Creating and saving final artifacts...")
    # 1. Churn Scores
    if name in ["ann", "rnn"]:
        X_processed = preprocessor.transform(X)
        if name == "rnn": X_processed = np.reshape(X_processed, (X_processed.shape[0], 1, X_processed.shape[1]))
        churn_proba = model.predict(X_processed).ravel()
    else:
        churn_proba = model.predict_proba(X)[:, 1]
    scores = features_df[["customer_id"]].copy()
    scores["churn_proba"] = churn_proba

    # 2. Segmentation
    features_df, segment_summary = build_segments(features_df)

    # 3. Monthly Sales
    # Ensure 'date' column is datetime
    transactions["date"] = pd.to_datetime(transactions["date"])
    monthly_sales = transactions.groupby(transactions["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly_sales["date"] = monthly_sales["date"].astype(str) # Convert to string for saving

    # 4. SHAP values
    print("Calculating SHAP values...")
    if name in ["ann", "rnn"]:
        X_test_processed = preprocessor.transform(X_test)
        background = preprocessor.transform(X_train.sample(100, random_state=42))
        if name == "rnn":
            X_test_processed = np.reshape(X_test_processed, (X_test_processed.shape[0], 1, X_test_processed.shape[1]))
            background = np.reshape(background, (background.shape[0], 1, background.shape[1]))
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test_processed)[0]
        feature_names = preprocessor.get_feature_names_out()
    else:  # It's a pipeline
        X_test_processed = model.named_steps['pre'].transform(X_test)
        explainer = shap.Explainer(model.named_steps['clf'], X_test_processed)
        shap_values_obj = explainer(X_test_processed)
        shap_values = shap_values_obj.values
        # For binary classification with scikit-learn, shap_values can be 3D (samples, features, classes).
        # We are interested in the explanation for the positive class (churn=1).
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        feature_names = model.named_steps['pre'].get_feature_names_out()

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_top = shap_df.abs().mean().sort_values(ascending=False).reset_index()
    shap_top.columns = ["feature", "importance"]

    # --- Save artifacts ---
    joblib.dump(preprocessor, ART / "preprocessor.joblib")
    if name in ["ann", "rnn"]:
        model.save(ART / "model.h5")
    else:
        joblib.dump(model, ART / "model.joblib")

    features_df.to_parquet(ART / "customer_features_with_segments.parquet", index=False)
    scores.to_parquet(ART / "churn_scores.parquet", index=False)
    if segment_summary is not None:
        segment_summary.to_parquet(ART / "segment_summary.parquet")
    monthly_sales.to_parquet(ART / "monthly_sales.parquet", index=False)
    shap_top.to_parquet(ART / "shap_top_features.parquet", index=False)


    print("\n--- ✅ Process Complete ---")
    print(f"Saved best model ('{name}') and other artifacts to the 'artifacts/' folder.")

if __name__ == "__main__":
    main()