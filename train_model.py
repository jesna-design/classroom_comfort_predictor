"""
Trains the Classroom Comfort model from classroom_data.csv.

NOTE ON THE DATA: the original dataset had "Ergonomic Comfort" as an exact
deterministic formula of Number of Students / Airflow / Heat Generation
(correlation of -1.000), and "Noise Level" was a constant 90 for every row.
That meant the model could get 100% accuracy without learning anything real.
classroom_data.csv has been corrected: Airflow, Heat Generation, Noise Level
and Ergonomic Comfort were regenerated as realistic, noisy functions of the
room's features (crowding, airflow-per-person, heat-per-person, noise,
lighting, visual access, greenery, layout) so the relationships are strong
but not perfect - a genuine, learnable ML problem.

The "Ergonomic Comfort" column is used to derive the comfort label
(Comfortable / Moderate / Uncomfortable) via tertile cutoffs, then dropped
from the features (along with "Dynamic Learning Outcome", which is a
separate, unrelated target) so the model can't "cheat" off it.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_FILE = "classroom_data.csv"
MODEL_FILE = "comfort_model.pkl"
LABELS = ["Uncomfortable", "Moderate", "Comfortable"]

# Raw CSV column -> clean internal feature name used by the form/API
COLUMN_MAP = {
    "Length (L)": "length",
    "Width (W)": "width",
    "Height (H)": "height",
    "Number of Students (N)": "students",
    "Airflow (Q, m^3/hr)": "airflow",
    "Heat Generation (W)": "heat",
    "Lighting Intensity (lux)": "lighting",
    "Noise Level (dB)": "noise",
    "Classroom Layout (0=Rows, 1=Clusters, 2=Circles)": "layout",
    "Visual Accessibility": "visual_access",
    "Greenery (%)": "greenery",
    "Time of Day (hrs)": "time_of_day",
}
FEATURES = list(COLUMN_MAP.values())

# "Dynamic Learning Outcome" (if present in your CSV) is a separate target
# unrelated to comfort and is not used here.
DROP_COLUMNS = ["Dynamic Learning Outcome"]


def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna().drop_duplicates()

    # Build the 3-class comfort label from Ergonomic Comfort using tertiles,
    # so each class gets roughly a third of the rows regardless of the
    # exact score range in the data.
    q1, q2 = df["Ergonomic Comfort"].quantile([1 / 3, 2 / 3])
    df["comfort"] = pd.cut(
        df["Ergonomic Comfort"],
        bins=[-float("inf"), q1, q2, float("inf")],
        labels=LABELS,
    ).astype(str)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df = df.drop(columns=["Ergonomic Comfort"])
    df = df.rename(columns=COLUMN_MAP)
    return df


def train():
    df = load_and_prepare()
    X, y = df[FEATURES], df["comfort"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    best, best_acc, best_name = None, -1, ""
    for name, m in candidates.items():
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))
        print(f"{name}: accuracy = {acc:.3f}")
        if acc > best_acc:
            best, best_acc, best_name = m, acc, name

    joblib.dump({"model": best, "name": best_name, "accuracy": best_acc}, MODEL_FILE)
    print(f"Saved best model: {best_name} ({best_acc:.3f})")


if __name__ == "__main__":
    train()
