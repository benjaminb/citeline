import pandas as pd


def get_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    empty_mask = df.map(lambda x: x is None or (isinstance(x, str) and x.strip() == ""))

    if empty_mask.any().any():
        empty_positions = list(zip(*empty_mask.stack().loc[empty_mask.stack()].index.tolist()))
        return pd.DataFrame(empty_positions, columns=["row_index", "column_name"])

    return None


def main():
    paths = [
        "acl200_global/context_dataset.csv",
        "arxiv/context_dataset_eval.csv",
        "peerread/context_dataset.csv",
        "refseer/context_dataset_eval.csv",
    ]
    for path in paths:
        print(f"Checking dataset at: \033[1m{path}\033[0m: ", end="")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        invalid_rows = get_invalid_rows(df)
        if invalid_rows is not None:
            print("Invalid rows found:")
            print(invalid_rows)
        else:
            print("All rows are valid.")


if __name__ == "__main__":
    main()
