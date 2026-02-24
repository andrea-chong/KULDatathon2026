import pandas as pd

from sklearn.model_selection import train_test_split

def read_csv(path: str):
    return pd.read_csv(path)

def split_data(df: pd.DataFrame, test_size: int=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df 

def load_datasetB_txt(path: str) -> pd.DataFrame:
    rows = []
    current_prompt = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # blank line: end of a prompt block
            if not line:
                current_prompt = None
                continue

            # prompt header line
            if line.startswith("prompt_") and "|" in line:
                current_prompt = line.split("|", 1)[0].strip()  # keep prompt_xxx id
                continue

            # translation line
            if current_prompt is None:
                # stray line; skip (or raise)
                continue

            if "|" not in line:
                continue

            trans, prob = line.rsplit("|", 1)
            try:
                p = float(prob)
            except ValueError:
                continue

            rows.append({"prompt_id": current_prompt, "translation": trans, "p": p})

    return pd.DataFrame(rows)

if __name__ == "__main__":
    csv_path = './data/users_fingerprint.csv'
    df = read_csv(csv_path)
    train_df, test_df = train_test_split(df)

    print(f"The size of training set: {train_df.shape}")
    print(f"The size of testing set: {test_df.shape}")