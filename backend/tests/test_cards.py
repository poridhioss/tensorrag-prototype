import shutil
import tempfile

import pandas as pd
import pytest

from app.services.storage import StorageService

PIPELINE_ID = "test-pipeline"


@pytest.fixture
def storage():
    tmp = tempfile.mkdtemp()
    svc = StorageService(base_dir=tmp)
    yield svc
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
    })


class TestDataLoadCard:
    def test_load_sample(self, storage):
        from cards.data_load import DataLoadCard

        card = DataLoadCard()
        outputs = card.execute(
            {"source": "sample", "sample_name": "california_housing",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {},
            storage,
        )
        assert "dataset" in outputs
        df = storage.load_dataframe(outputs["dataset"])
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_preview(self, storage):
        from cards.data_load import DataLoadCard

        card = DataLoadCard()
        outputs = card.execute(
            {"source": "sample", "sample_name": "california_housing",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {},
            storage,
        )
        preview = card.get_output_preview(outputs, storage)
        assert "rows" in preview
        assert "columns" in preview
        assert "shape" in preview
        assert len(preview["rows"]) <= 100


class TestDataSplitCard:
    def test_split_ratio(self, storage, sample_df):
        from cards.data_split import DataSplitCard

        # Save input data
        ref = storage.save_dataframe(PIPELINE_ID, "n0", "dataset", sample_df)

        card = DataSplitCard()
        outputs = card.execute(
            {"train_ratio": 0.8, "test_ratio": 0.2, "random_seed": 42,
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {"dataset": ref},
            storage,
        )
        train_df = storage.load_dataframe(outputs["train_dataset"])
        test_df = storage.load_dataframe(outputs["test_dataset"])
        assert len(train_df) + len(test_df) == len(sample_df)
        assert len(test_df) == 2  # 20% of 10


class TestModelDefineCard:
    def test_define_linear(self, storage):
        from cards.model_define import ModelDefineCard

        card = ModelDefineCard()
        outputs = card.execute(
            {"model_type": "linear_regression", "hyperparameters": {},
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {},
            storage,
        )
        spec = storage.load_json(outputs["model_spec"])
        assert spec["model_type"] == "linear_regression"

    def test_invalid_model_type(self, storage):
        from cards.model_define import ModelDefineCard

        card = ModelDefineCard()
        with pytest.raises(ValueError, match="Unknown model type"):
            card.execute(
                {"model_type": "random_forest",
                 "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
                {},
                storage,
            )


class TestTrainCard:
    def test_train_linear(self, storage, sample_df):
        from cards.model_define import ModelDefineCard
        from cards.train import TrainCard

        # Prepare inputs
        data_ref = storage.save_dataframe(PIPELINE_ID, "n0", "train", sample_df)
        define_card = ModelDefineCard()
        spec_outputs = define_card.execute(
            {"model_type": "linear_regression",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {},
            storage,
        )

        card = TrainCard()
        outputs = card.execute(
            {"target_column": "target",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n2"},
            {"train_dataset": data_ref, "model_spec": spec_outputs["model_spec"]},
            storage,
        )
        assert "trained_model" in outputs
        assert "train_metrics" in outputs

        metrics = storage.load_json(outputs["train_metrics"])
        assert "train_r2" in metrics
        assert metrics["train_r2"] > 0.9  # linear data should fit well


class TestEvaluateCard:
    def test_evaluate(self, storage, sample_df):
        from cards.evaluate import EvaluateCard
        from cards.model_define import ModelDefineCard
        from cards.train import TrainCard

        # Train first
        data_ref = storage.save_dataframe(PIPELINE_ID, "n0", "data", sample_df)
        define = ModelDefineCard()
        spec = define.execute(
            {"model_type": "linear_regression",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {}, storage,
        )
        train = TrainCard()
        train_out = train.execute(
            {"target_column": "target",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n2"},
            {"train_dataset": data_ref, "model_spec": spec["model_spec"]},
            storage,
        )

        # Evaluate
        card = EvaluateCard()
        outputs = card.execute(
            {"metrics": ["mse", "r2"], "target_column": "target",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n3"},
            {"trained_model": train_out["trained_model"], "test_dataset": data_ref},
            storage,
        )
        report = storage.load_json(outputs["eval_report"])
        assert "R2" in report
        assert "MSE" in report
        assert outputs.get("eval_chart") is not None


class TestInferenceCard:
    def test_inference(self, storage, sample_df):
        from cards.inference import InferenceCard
        from cards.model_define import ModelDefineCard
        from cards.train import TrainCard

        # Train
        data_ref = storage.save_dataframe(PIPELINE_ID, "n0", "data", sample_df)
        define = ModelDefineCard()
        spec = define.execute(
            {"model_type": "linear_regression",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n1"},
            {}, storage,
        )
        train = TrainCard()
        train_out = train.execute(
            {"target_column": "target",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n2"},
            {"train_dataset": data_ref, "model_spec": spec["model_spec"]},
            storage,
        )

        # Inference
        card = InferenceCard()
        outputs = card.execute(
            {"include_features": True, "target_column": "prediction",
             "_pipeline_id": PIPELINE_ID, "_node_id": "n3"},
            {"trained_model": train_out["trained_model"], "dataset": data_ref},
            storage,
        )
        preds = storage.load_dataframe(outputs["predictions"])
        assert "prediction" in preds.columns
        assert len(preds) == len(sample_df)
