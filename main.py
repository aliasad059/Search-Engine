import pandas as pd


def load_df(file_name):
    """
    Loads a dataframe from a json file.
    """
    return pd.read_json(file_name).transpose()


if __name__ == '__main__':
    # Load dataframe
    df = load_df('data/raw_data.json')
