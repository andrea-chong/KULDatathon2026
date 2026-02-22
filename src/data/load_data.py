import pandas as pd

def read_csv(path: str):
    return pd.read_csv(path)

if __name__ == "__main__":
    csv_path = './data/users_fingerprint.csv'
    df = read_csv(csv_path)
    print(df)