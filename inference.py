import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# ================= Load artifacts =================
tfidf = joblib.load("models/tfidf.joblib")
clf_model = joblib.load("models/clf_model.joblib")
reg_model = joblib.load("models/reg_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")
feature_info = joblib.load("models/feature_info.joblib")

# ================= Text utils =================
def build_full_text(title, desc, inp, out):
    return " ".join([title, desc, inp, out]).lower().strip()

# ================= Feature extraction =================
def extract_stat_features(text):
    data = {
        'char_len': len(text),
        'word_len': len(text.split()),
        'sentence_count': text.count('.') + text.count('?') + text.count('!') + 1,
        'digit_count': sum(c.isdigit() for c in text),
        'math_symbol_count': sum(c in "+-*/=<>" for c in text),
        'log_char_len': np.log1p(len(text)),
        'log_word_len': np.log1p(len(text.split())),
        'constraint_count': text.count('<') + text.count('>'),
        'power_count': text.count('^'),
        'big_o_count': text.count('o('),
        'newline_count': text.count('\n'),
        'colon_count': text.count(':'),
        'avg_word_len': len(text) / (len(text.split()) + 1)
    }
    df = pd.DataFrame([data])
    return df[feature_info["stat_cols"]]

def extract_keyword_features(text):
    feats = {}
    for col in feature_info["kw_cols"]:
        key = col.replace("_present", "").replace("_count", "")
        feats[col] = int(key in text)
    return pd.DataFrame([feats])[feature_info["kw_cols"]]

# ================= Prediction =================
def predict_problem(title, desc, inp, out):
    text = build_full_text(title, desc, inp, out)

    X_tfidf = tfidf.transform([text])
    X_stat = extract_stat_features(text)
    X_kw = extract_keyword_features(text)

    X_final = csr_matrix(hstack([
        X_tfidf,
        X_stat.values,
        X_kw.values
    ]))

    cls_idx = clf_model.predict(X_final)[0]
    cls_label = label_encoder.inverse_transform([cls_idx])[0]

    score = reg_model.predict(X_final)[0]

    return cls_label, round(float(score), 2)
