import pandas as pd
import os
import argparse


def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)

    return files


def get_data_csv(files):
    df = pd.DataFrame()
    for i in files:
        temp = pd.read_csv(i, header=0)
        df = pd.concat([df, temp], axis=0)

    return df


def get_data_parquet(files):
    """Concatenate all parquet files in input channel."""
    df = pd.DataFrame()
    for i in files:
        temp = pd.read_parquet(i)
        df = pd.concat([df, temp], axis=0)

    return df


def processing(df):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=int, default=1, help="0: local test, 1: sagemaker")
    parser.add_argument("--train-test-split-ratio", type=float, default=0.2)

    args = parser.parse_args()
    print(args)

    # Input location for local testing
    if args.option == 0:
        input_channel = "../test-data/input"
        output_channel = "../test-data/output"
    # Sagemaker
    else:
        input_channel = "/opt/ml/processing/input/data"
        # input_channel2 = "/opt/ml/processing/input/data2"
        output_channel = "/opt/ml/processing/output"

    print(f"\nList of files in input channel: {input_channel}")
    files = print_files_in_path(input_channel)
    df = get_data_parquet(files)
    print(df.shape)

    # Processing task
    processing(df)

    # Save files
    filename = os.path.join(output_channel, "processed_data.csv")
    df.to_csv(filename, header=True, index=False)
    df.to_parquet(filename)
