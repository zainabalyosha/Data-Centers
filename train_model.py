import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load merged_training_sample.csv if you want to retrain
df = pd.read_csv("merged_training_sample.csv")
features = ["max_temp_F","rolling_max7","humidity_idx","load_MW","cooling_kW"]
X = df[features]
y = df["extreme_heat_event"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print("AUC:", auc)

joblib.dump(model, "model.pkl")
pd.Series(features).to_csv("feature_order.csv", index=False)
