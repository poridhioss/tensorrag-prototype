from __future__ import annotations

from sklearn.model_selection import train_test_split

from cards.base import BaseCard


class DataSplitCard(BaseCard):
    card_type = "data_split"
    display_name = "Data Split"
    description = "Split dataset into train and test sets"
    category = "data"
    execution_mode = "local"
    output_view_type = "table"

    config_schema = {
        "type": "object",
        "properties": {
            "train_ratio": {
                "type": "number",
                "default": 0.8,
                "description": "Fraction of data for training",
            },
            "test_ratio": {
                "type": "number",
                "default": 0.2,
                "description": "Fraction of data for testing",
            },
            "random_seed": {
                "type": "integer",
                "default": 42,
                "description": "Random seed for reproducibility",
            },
            "stratify_column": {
                "type": "string",
                "description": "Column to stratify by (optional)",
            },
        },
    }
    input_schema = {"dataset": "dataframe"}
    output_schema = {"train_dataset": "dataframe", "test_dataset": "dataframe"}

    def execute(self, config: dict, inputs: dict, storage) -> dict:
        df = storage.load_dataframe(inputs["dataset"])

        test_size = config.get("test_ratio", 0.2)
        random_seed = config.get("random_seed", 42)
        stratify_col = config.get("stratify_column")

        stratify = df[stratify_col] if stratify_col else None

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_seed, stratify=stratify
        )

        pid = config["_pipeline_id"]
        nid = config["_node_id"]
        train_ref = storage.save_dataframe(pid, nid, "train_dataset", train_df)
        test_ref = storage.save_dataframe(pid, nid, "test_dataset", test_df)

        return {"train_dataset": train_ref, "test_dataset": test_ref}

    def get_output_preview(self, outputs: dict, storage) -> dict:
        train_df = storage.load_dataframe(outputs["train_dataset"])
        test_df = storage.load_dataframe(outputs["test_dataset"])
        return {
            "train": {
                "rows": train_df.head(20).to_dict(orient="records"),
                "row_count": len(train_df),
            },
            "test": {
                "rows": test_df.head(20).to_dict(orient="records"),
                "row_count": len(test_df),
            },
            "split_ratio": {
                "train": round(len(train_df) / (len(train_df) + len(test_df)), 3),
                "test": round(len(test_df) / (len(train_df) + len(test_df)), 3),
            },
        }
