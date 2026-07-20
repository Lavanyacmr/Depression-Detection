from sklearn.preprocessing import MinMaxScaler
import numpy as np

METADATA_COLS = [
    "Posting_Freq_per_Week", "Avg_Likes", "Avg_Replies", "Avg_Shares",
    "Peak_Activity", "Interaction_Days", "PHQ9_Score", "Sleep_Duration_hrs",
    "Social_Withdrawal", "Speech_wpm", "Speech_Pace", "Facial_Emotion_Recognition"
]
CAT_COLS = ["Peak_Activity", "Social_Withdrawal", "Speech_Pace"]

def preprocess_metadata(df, fit=True, scaler=None):
    df = df.copy()
    for col in CAT_COLS:
        df[col] = df[col].astype('category').cat.codes
    numeric_cols = [c for c in METADATA_COLS if c not in CAT_COLS]
    for col in METADATA_COLS:
        if col in CAT_COLS:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        else:
            df[col] = df[col].fillna(df[col].median())
    if fit:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        return df