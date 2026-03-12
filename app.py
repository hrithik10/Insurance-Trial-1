import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Policy Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS – light / white theme ─────────────────────────────────────────
st.markdown("""
<style>
/* ---- base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #1a1a2e;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #f0f4ff !important;
    border-right: 1px solid #dde3f5;
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

/* ---- header strip ---- */
.header-strip {
    background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    color: #fff;
}
.header-strip h1 { font-size: 2rem; margin: 0 0 4px; font-weight: 700; }
.header-strip p  { margin: 0; opacity: .85; font-size: .95rem; }

/* ---- step cards ---- */
.step-card {
    background: #f7f9ff;
    border: 1px solid #dde3f5;
    border-left: 5px solid #3949ab;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.step-card h3 { margin: 0 0 6px; color: #1a237e; font-size: 1rem; font-weight: 700; }
.step-card p  { margin: 0; color: #444; font-size: .9rem; }

/* ---- metric boxes ---- */
.metric-row { display: flex; gap: 16px; margin: 12px 0; flex-wrap: wrap; }
.metric-box {
    background: #fff;
    border: 1px solid #dde3f5;
    border-radius: 10px;
    padding: 14px 20px;
    min-width: 130px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(57,73,171,.08);
}
.metric-box .val { font-size: 1.5rem; font-weight: 800; color: #1a237e; }
.metric-box .lbl { font-size: .75rem; color: #777; margin-top: 2px; }

/* ---- info / success / warning boxes ---- */
.info-box    { background:#e8eaf6; border-left:4px solid #3949ab;
               border-radius:8px; padding:12px 16px; margin:8px 0; font-size:.88rem; color:#1a237e; }
.success-box { background:#e8f5e9; border-left:4px solid #43a047;
               border-radius:8px; padding:12px 16px; margin:8px 0; font-size:.88rem; color:#2e7d32; }
.warn-box    { background:#fff8e1; border-left:4px solid #ffa000;
               border-radius:8px; padding:12px 16px; margin:8px 0; font-size:.88rem; color:#e65100; }

/* ---- section header ---- */
.sec-title {
    font-size:1.1rem; font-weight:700; color:#1a237e;
    border-bottom:2px solid #e8eaf6; padding-bottom:6px; margin-bottom:12px;
}

/* ---- accuracy table ---- */
.acc-table { width:100%; border-collapse:collapse; margin:8px 0; font-size:.9rem; }
.acc-table th { background:#1a237e; color:#fff; padding:10px 14px; text-align:left; }
.acc-table td { padding:9px 14px; border-bottom:1px solid #e8eaf6; }
.acc-table tr:nth-child(even) td { background:#f0f4ff; }
.acc-badge-high { background:#c8e6c9; color:#1b5e20; border-radius:20px; padding:3px 10px; font-weight:700; }
.acc-badge-med  { background:#fff9c4; color:#f57f17; border-radius:20px; padding:3px 10px; font-weight:700; }

/* ---- buttons ---- */
.stButton > button {
    background: #1a237e !important;
    color: #fff !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.4rem !important;
    transition: background .2s;
}
.stButton > button:hover { background: #283593 !important; }

/* ---- expander ---- */
details summary { color:#1a237e !important; font-weight:600; }

/* ---- dataframe ---- */
[data-testid="stDataFrame"] { border:1px solid #dde3f5; border-radius:8px; }

/* ---- divider ---- */
hr { border:none; border-top:1px solid #dde3f5; margin:18px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🛡️ Insurance Classifier")
    st.markdown("---")
    st.markdown("**Upload your dataset**")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**Model Hyperparameters**")
    dt_max_depth   = st.slider("Decision Tree – max depth",    2, 20, 5)
    rf_n_estimators = st.slider("Random Forest – n estimators", 50, 300, 100, 50)
    gb_n_estimators = st.slider("Gradient Boost – n estimators",50, 300, 100, 50)
    gb_lr           = st.slider("Gradient Boost – learning rate",0.01, 0.5, 0.1, 0.01)
    test_size       = st.slider("Test split size (%)", 10, 40, 20)
    random_state    = st.number_input("Random state", value=42, step=1)

    st.markdown("---")
    run_btn = st.button("▶  Run Full Pipeline", use_container_width=True)
    st.markdown("---")
    st.caption("Steps executed top-to-bottom. Upload a file and click Run.")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-strip">
  <h1>🛡️ Insurance Policy Status Classifier</h1>
  <p>End-to-end ML pipeline · Decision Tree · Random Forest · Gradient Boosted Tree</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP GUIDE (always visible)
# ══════════════════════════════════════════════════════════════════════════════
steps_meta = [
    ("Step 1", "Import Packages",            "All required libraries loaded."),
    ("Step 2", "Data Check & Column Removal", "Null inspection, shape, dtypes. Drop POLICY_NO & PI_NAME."),
    ("Step 3", "Handle Missing Values",       "Mean imputation for numeric; mode imputation for categorical."),
    ("Step 4", "Label Encoding",              "Encode all object columns; save mapping CSV."),
    ("Step 5", "Data / Label Split",          "Separate features (X) from target (POLICY_STATUS)."),
    ("Step 6", "Train / Test Split",          "80:20 stratified split."),
    ("Step 7", "Train Models",                "Fit DT, RF, GBT classifiers."),
    ("Step 8", "Accuracy Table",              "Training & testing accuracy for all three models."),
    ("Step 9", "Confusion Matrices",          "Per-model confusion matrix with TP/FP/TN/FN labels."),
    ("Step 10","Feature Importance Charts",   "Top-feature bar charts for each model."),
]

with st.expander("📋 Pipeline Overview – click to expand", expanded=False):
    cols = st.columns(2)
    for i, (num, title, desc) in enumerate(steps_meta):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="step-card">
              <h3>{num} · {title}</h3>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
LIGHT_PALETTE = ["#3949ab", "#43a047", "#e53935", "#fb8c00", "#8e24aa",
                 "#00897b", "#1e88e5", "#f4511e"]

def section(icon, title):
    st.markdown(f'<div class="sec-title">{icon} {title}</div>', unsafe_allow_html=True)

def info(msg):    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>',    unsafe_allow_html=True)
def success(msg): st.markdown(f'<div class="success-box">✅ {msg}</div>', unsafe_allow_html=True)
def warn(msg):    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>',   unsafe_allow_html=True)

def badge(val, threshold=90):
    cls = "acc-badge-high" if val >= threshold else "acc-badge-med"
    return f'<span class="{cls}">{val:.2f}%</span>'

def clean_numeric(series):
    """Strip commas/spaces and coerce to float."""
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(),
                         errors="coerce")

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if not run_btn:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#aaa;">
      <div style="font-size:3rem;">📂</div>
      <div style="font-size:1.1rem;margin-top:8px;">Upload a CSV and click <strong>Run Full Pipeline</strong></div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if uploaded is None:
    warn("No file uploaded. Please upload a CSV using the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Imports (already done at top; just display)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("📦", "Step 1 · Importing Packages")

packages = {
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "matplotlib": plt.matplotlib.__version__,
    "seaborn": sns.__version__,
    "scikit-learn": "via sklearn",
}
pkg_df = pd.DataFrame(packages.items(), columns=["Package", "Version"])
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(pkg_df, hide_index=True, use_container_width=True)
with col2:
    st.code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix""", language="python")

success("All packages imported successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Data Check
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🔍", "Step 2 · Basic Data Check & Column Removal")

df_raw = pd.read_csv(uploaded)
info(f"Raw dataset loaded · {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

tab_shape, tab_dtype, tab_null, tab_preview = st.tabs(
    ["Shape", "Data Types", "Null Values", "Preview"])

with tab_shape:
    r, c = df_raw.shape
    st.markdown(f"""<div class="metric-row">
      <div class="metric-box"><div class="val">{r}</div><div class="lbl">Rows</div></div>
      <div class="metric-box"><div class="val">{c}</div><div class="lbl">Columns</div></div>
    </div>""", unsafe_allow_html=True)

with tab_dtype:
    dtype_df = pd.DataFrame({"Column": df_raw.columns,
                              "Dtype": df_raw.dtypes.astype(str).values})
    st.dataframe(dtype_df, hide_index=True, use_container_width=True)

with tab_null:
    null_df = pd.DataFrame({
        "Column": df_raw.columns,
        "Null Count": df_raw.isnull().sum().values,
        "Null %": (df_raw.isnull().mean() * 100).round(2).values,
    })
    st.dataframe(null_df, hide_index=True, use_container_width=True)

with tab_preview:
    st.dataframe(df_raw.head(10), use_container_width=True)

# Drop columns
drop_cols = [c for c in ["POLICY_NO", "PI_NAME"] if c in df_raw.columns]
df = df_raw.drop(columns=drop_cols)
success(f"Dropped columns: {drop_cols}. Remaining shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Handle Missing Values
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🧹", "Step 3 · Handling Missing Values")

# Fix comma-formatted numeric columns
for col in ["SUM_ASSURED", "PI_ANNUAL_INCOME"]:
    if col in df.columns:
        df[col] = clean_numeric(df[col])

numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
# Remove label from imputation target
if "POLICY_STATUS" in categorical_cols:
    categorical_cols.remove("POLICY_STATUS")

impute_log = []
for col in numeric_cols:
    n = df[col].isnull().sum()
    if n > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        impute_log.append({"Column": col, "Type": "Numeric",
                           "Nulls Fixed": n, "Fill Value": f"{mean_val:.4f}"})

for col in categorical_cols:
    n = df[col].isnull().sum()
    if n > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        impute_log.append({"Column": col, "Type": "Categorical",
                           "Nulls Fixed": n, "Fill Value": mode_val})

remaining_nulls = df.isnull().sum().sum()

if impute_log:
    st.dataframe(pd.DataFrame(impute_log), hide_index=True, use_container_width=True)
else:
    info("No null values found after type conversion – nothing to impute.")

if remaining_nulls == 0:
    success(f"Dataset is null-free. Shape after cleaning: {df.shape}")
else:
    warn(f"Still {remaining_nulls} nulls present after imputation.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Label Encoding
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🔢", "Step 4 · Label Encoding")

df_encoded = df.copy()
le_dict    = {}   # col → LabelEncoder
mapping_records = []

all_obj_cols = df_encoded.select_dtypes(include="object").columns.tolist()

for col in all_obj_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le
    for orig, enc in zip(le.classes_, le.transform(le.classes_)):
        mapping_records.append({"Column": col, "Original Value": orig,
                                 "Encoded Value": int(enc)})

mapping_df = pd.DataFrame(mapping_records)

tab_enc, tab_map = st.tabs(["Encoded DataFrame (first 10 rows)", "Encoding Mapping"])
with tab_enc:
    st.dataframe(df_encoded.head(10), use_container_width=True)
with tab_map:
    st.dataframe(mapping_df, hide_index=True, use_container_width=True)

# Provide CSV download
csv_map = mapping_df.to_csv(index=False).encode()
st.download_button("⬇️  Download Encoding Map CSV", csv_map,
                   "encoding_map.csv", "text/csv", use_container_width=False)

success(f"Label encoding applied to {len(all_obj_cols)} columns.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Data / Label Split
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("✂️", "Step 5 · Data & Label Split")

TARGET = "POLICY_STATUS"
X = df_encoded.drop(columns=[TARGET])
y = df_encoded[TARGET]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""<div class="metric-row">
      <div class="metric-box"><div class="val">{X.shape[0]}</div><div class="lbl">Samples</div></div>
      <div class="metric-box"><div class="val">{X.shape[1]}</div><div class="lbl">Features</div></div>
    </div>""", unsafe_allow_html=True)
    info(f"Feature matrix X shape: {X.shape}")
with col_b:
    vc = y.value_counts().reset_index()
    vc.columns = ["Encoded", "Count"]
    # attach original label
    le_target = le_dict[TARGET]
    vc["Label"] = le_target.inverse_transform(vc["Encoded"])
    st.dataframe(vc[["Label", "Encoded", "Count"]], hide_index=True)
    info("Label (y) distribution shown above.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("📊", "Step 6 · Train / Test Split (Stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size / 100,
    random_state=int(random_state),
    stratify=y,
)

col1, col2, col3 = st.columns(3)
col1.metric("Training Samples", len(X_train))
col2.metric("Testing Samples",  len(X_test))
col3.metric("Test Split",        f"{test_size}%")

success(f"Stratified split complete. Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Train Models
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🤖", "Step 7 · Training Classification Models")

models = {
    "Decision Tree":        DecisionTreeClassifier(max_depth=dt_max_depth,
                                                    random_state=int(random_state)),
    "Random Forest":        RandomForestClassifier(n_estimators=rf_n_estimators,
                                                    random_state=int(random_state)),
    "Gradient Boosted Tree": GradientBoostingClassifier(n_estimators=gb_n_estimators,
                                                         learning_rate=gb_lr,
                                                         random_state=int(random_state)),
}

results    = {}
progress   = st.progress(0, text="Training models…")

for i, (name, model) in enumerate(models.items()):
    progress.progress((i) / len(models), text=f"Training {name}…")
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
    results[name] = {
        "model": model,
        "train_acc": train_acc,
        "test_acc":  test_acc,
        "y_pred":    model.predict(X_test),
    }
    progress.progress((i + 1) / len(models), text=f"✔ {name} done")

progress.empty()
success("All three models trained successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – Accuracy Table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("📈", "Step 8 · Model Accuracy Comparison")

rows_html = ""
for name, res in results.items():
    rows_html += f"""<tr>
        <td>{name}</td>
        <td>{badge(res['train_acc'])}</td>
        <td>{badge(res['test_acc'])}</td>
        <td>{res['train_acc'] - res['test_acc']:+.2f}%</td>
    </tr>"""

st.markdown(f"""
<table class="acc-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>Training Accuracy</th>
      <th>Testing Accuracy</th>
      <th>Δ (Train – Test)</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>""", unsafe_allow_html=True)

# Bar chart comparison
fig_acc, ax = plt.subplots(figsize=(8, 3.5))
fig_acc.patch.set_facecolor("white")
ax.set_facecolor("#f7f9ff")

model_names = list(results.keys())
train_accs  = [results[m]["train_acc"] for m in model_names]
test_accs   = [results[m]["test_acc"]  for m in model_names]
x = np.arange(len(model_names))
w = 0.35

bars1 = ax.bar(x - w/2, train_accs, w, label="Train Accuracy",
               color="#1a237e", alpha=0.88, zorder=3)
bars2 = ax.bar(x + w/2, test_accs,  w, label="Test Accuracy",
               color="#3949ab", alpha=0.7, zorder=3)

ax.set_ylim(50, 105)
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.set_title("Training vs Testing Accuracy", fontsize=11, fontweight="bold", color="#1a237e")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3,
                           f"{bar.get_height():.1f}%", ha="center", va="bottom",
                           fontsize=7.5, color="#1a237e")
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3,
                           f"{bar.get_height():.1f}%", ha="center", va="bottom",
                           fontsize=7.5, color="#3949ab")
plt.tight_layout()
st.pyplot(fig_acc)
plt.close(fig_acc)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 – Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🔲", "Step 9 · Confusion Matrices")

class_names = le_dict[TARGET].classes_.tolist()  # original string labels

fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig_cm.patch.set_facecolor("white")
cmap = sns.light_palette("#1a237e", as_cmap=True)

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])

    # annotate with TP/FP/TN/FN for binary case
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        annot_arr = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                               [f"FN\n{fn}", f"TP\n{tp}"]])
    else:
        annot_arr = cm.astype(str)

    sns.heatmap(cm, annot=annot_arr, fmt="", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, linecolor="#dde3f5",
                ax=ax, cbar=False, annot_kws={"size": 9})
    ax.set_title(name, fontsize=10, fontweight="bold", color="#1a237e", pad=8)
    ax.set_xlabel("Predicted Label", fontsize=8)
    ax.set_ylabel("True Label",      fontsize=8)
    ax.tick_params(labelsize=7, rotation=15)

plt.suptitle("Confusion Matrices – All Models", fontsize=12,
             fontweight="bold", color="#1a237e", y=1.01)
plt.tight_layout()
st.pyplot(fig_cm)
plt.close(fig_cm)

info("Rows = True Labels · Columns = Predicted Labels · "
     "TP = correctly predicted positive · TN = correctly predicted negative · "
     "FP = false alarm · FN = missed positive")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 – Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("📊", "Step 10 · Feature Importance Charts")

feat_names = X.columns.tolist()
top_n = min(15, len(feat_names))

colors = ["#1a237e", "#3949ab", "#283593"]
for (name, res), color in zip(results.items(), colors):
    importances = res["model"].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_feats  = [feat_names[i] for i in idx]
    top_imps   = importances[idx]

    fig_fi, ax = plt.subplots(figsize=(10, max(3, top_n * 0.38)))
    fig_fi.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9ff")

    bars = ax.barh(range(top_n), top_imps[::-1], color=color, alpha=0.85, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title(f"Feature Importance – {name}", fontsize=11,
                 fontweight="bold", color="#1a237e")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", ha="left", fontsize=7.5, color="#444")

    plt.tight_layout()
    st.pyplot(fig_fi)
    plt.close(fig_fi)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
section("🏁", "Pipeline Complete")

best_model = max(results, key=lambda m: results[m]["test_acc"])
st.markdown(f"""
<div class="success-box">
  🎉 Pipeline executed successfully!<br>
  Best model by test accuracy: <strong>{best_model}</strong>
  ({results[best_model]['test_acc']:.2f}%)
</div>""", unsafe_allow_html=True)

summary_rows = "".join(
    f"<tr><td>{m}</td><td>{r['train_acc']:.2f}%</td><td>{r['test_acc']:.2f}%</td></tr>"
    for m, r in results.items()
)
st.markdown(f"""
<table class="acc-table">
  <thead><tr><th>Model</th><th>Train Acc</th><th>Test Acc</th></tr></thead>
  <tbody>{summary_rows}</tbody>
</table>""", unsafe_allow_html=True)

st.caption("Built with Streamlit · scikit-learn · matplotlib · seaborn")
