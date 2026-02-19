"""
Build gold training table: transactions + cards + users + mcc + labels.
One row per transaction; time-ordered; has_label and is_fraud for supervised subset.
"""
from pathlib import Path

import pandas as pd

from src.ingestion.load import (
    load_transactions,
    load_cards,
    load_users,
    load_mcc_codes,
    load_fraud_labels,
)
from src.features.build import (
    prepare_static_transaction,
    prepare_cards,
    prepare_users,
    add_acct_age_days,
)
from src.config import get_paths


def build_gold_table(
    nrows: int | None = None,
    parse_static: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Join all tables and attach labels.
    - nrows: limit transactions (for EDA/speed); None = full.
    - parse_static: run prepare_* for amounts, dates, booleans.
    - save_path: if set, write parquet to this path (and create parent dir).
    """
    paths = get_paths()
    processed_dir = paths.get("data", {}).get("processed", {}).get("dir")
    if save_path is None and processed_dir:
        save_path = Path(processed_dir) / "gold_transactions.parquet"

    transactions = load_transactions(nrows=nrows)
    transactions = transactions.sort_values("date").reset_index(drop=True)
    if parse_static:
        transactions = prepare_static_transaction(transactions)

    cards = load_cards()
    if parse_static:
        cards = prepare_cards(cards)
    transactions = transactions.merge(
        cards,
        left_on="card_id",
        right_on="id",
        how="left",
        suffixes=("", "_card"),
    )
    if parse_static:
        transactions = add_acct_age_days(transactions)

    users = load_users()
    if parse_static:
        users = prepare_users(users)
    transactions = transactions.merge(
        users,
        left_on="client_id",
        right_on="id",
        how="left",
        suffixes=("", "_user"),
    )

    mcc = load_mcc_codes()
    mcc_df = pd.DataFrame(list(mcc.items()), columns=["mcc_str", "mcc_description"])
    mcc_df["mcc_str"] = mcc_df["mcc_str"].astype(str)
    transactions["mcc_str"] = transactions["mcc"].astype(str)
    transactions = transactions.merge(mcc_df, on="mcc_str", how="left")

    labels = load_fraud_labels()
    labels_df = pd.DataFrame(list(labels.items()), columns=["transaction_id_str", "is_fraud"])
    transactions["transaction_id_str"] = transactions["id"].astype(str)
    transactions = transactions.merge(
        labels_df,
        on="transaction_id_str",
        how="left",
    )
    transactions["has_label"] = transactions["is_fraud"].notna()
    # Keep is_fraud 0/1 for labeled rows; NaN for unlabeled (do not use as non-fraud)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        transactions.to_parquet(save_path, index=False)

    return transactions
