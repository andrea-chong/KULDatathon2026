import pandas as pd

from sklearn.model_selection import train_test_split

def read_csv(path: str):
    return pd.read_csv(path)

def split_data(df: pd.DataFrame, test_size: int=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df 


if __name__ == "__main__":
    csv_path = './data/users_fingerprint.csv'
    df = read_csv(csv_path)
    train_df, test_df = train_test_split(df)

    print(f"The size of training set: {train_df.shape}")
    print(f"The size of testing set: {test_df.shape}")