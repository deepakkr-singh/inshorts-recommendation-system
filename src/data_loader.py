"""
Data loading and basic validation.

Loads raw CSVs, performs minimal checks, and saves cleaned data.
"""

from pathlib import Path
import pandas as pd
import glob


class DataLoader:
    def __init__(
        self,
        raw_data_path: str = "test_data",
        processed_data_path: str = "data/processed",
    ):
        project_root = Path(__file__).parent.parent
        self.raw_data_path = project_root / raw_data_path
        self.processed_data_path = project_root / processed_data_path
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: list, name: str):
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    @staticmethod
    def _log_nulls(df: pd.DataFrame, cols: list, name: str):
        nulls = df[cols].isna().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            print(f"[WARN] {name}: nulls detected -> {nulls.to_dict()}")

    # ---------- loaders ----------

    def load_devices(self) -> pd.DataFrame:
        path = (
            self.raw_data_path
            / "devices"
            / "part-00000-cdb2cdd7-9d14-4000-b947-4d0475444217-c000.csv"
        )

        df = pd.read_csv(path, on_bad_lines="skip")
        self._validate_columns(df, ["deviceid", "language_selected"], "devices")

        df["deviceid"] = df["deviceid"].astype(str)
        df["language_selected"] = df["language_selected"].astype(str)

        self._log_nulls(df, ["deviceid", "language_selected"], "devices")
        return df

    def load_events(self) -> pd.DataFrame:
        files = glob.glob(str(self.raw_data_path / "event" / "part-*.csv"))
        if not files:
            raise FileNotFoundError("No event files found")

        dfs = [pd.read_csv(f, on_bad_lines="skip") for f in sorted(files)]
        df = pd.concat(dfs, ignore_index=True)

        self._validate_columns(
            df, ["deviceId", "hashId", "event_type", "eventTimestamp"], "events"
        )

        df["deviceId"] = df["deviceId"].astype(str)
        df["hashId"] = df["hashId"].astype(str)
        df["event_type"] = df["event_type"].astype(str)
        df["eventTimestamp"] = pd.to_datetime(df["eventTimestamp"], errors="coerce")

        self._log_nulls(df, ["deviceId", "hashId", "eventTimestamp"], "events")
        return df

    def load_training_content(self) -> pd.DataFrame:
        path = (
            self.raw_data_path
            / "training_content"
            / "part-00000-a34a1545-5cf1-47b9-93c2-29c1d3f0bfb7-c000.csv"
        )

        df = pd.read_csv(path, on_bad_lines="skip")
        self._validate_columns(
            df, ["hashid", "categories", "newsLanguage", "newsType"],
            "training_content",
        )

        df["hashid"] = df["hashid"].astype(str)
        df["newsLanguage"] = df["newsLanguage"].astype(str)
        df["newsType"] = df["newsType"].astype(str)

        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")

        self._log_nulls(df, ["hashid", "categories"], "training_content")
        return df

    def load_testing_content(self) -> pd.DataFrame:
        path = (
            self.raw_data_path
            / "testing_content"
            / "part-00000-8be13c58-b74d-4e30-8877-c8b5e168035a-c000.csv"
        )

        df = pd.read_csv(path, on_bad_lines="skip")
        self._validate_columns(
            df, ["hashid", "categories", "newsLanguage", "newsType"],
            "testing_content",
        )

        df["hashid"] = df["hashid"].astype(str)
        df["newsLanguage"] = df["newsLanguage"].astype(str)
        df["newsType"] = df["newsType"].astype(str)

        if "createdAt" in df.columns:
            df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")

        self._log_nulls(df, ["hashid", "categories"], "testing_content")
        return df

    # ---------- orchestration ----------

    def load_all(self) -> dict:
        return {
            "devices": self.load_devices(),
            "events": self.load_events(),
            "training_content": self.load_training_content(),
            "testing_content": self.load_testing_content(),
        }

    def save_processed(self, datasets: dict):
        for name, df in datasets.items():
            path = self.processed_data_path / f"{name}.csv"
            df.to_csv(path, index=False)


def main():
    loader = DataLoader()
    datasets = loader.load_all()
    loader.save_processed(datasets)
    print("STEP 1 complete: data loaded and validated.")


if __name__ == "__main__":
    main()
