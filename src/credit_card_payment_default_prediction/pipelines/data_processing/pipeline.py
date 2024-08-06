from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_dtypes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_dtypes,
                inputs="train_dataset",
                outputs=["preprocessed_train_data"],
                name="preprocess_train_data",
            ),
            node(
                func=preprocess_dtypes,
                inputs="test_dataset",
                outputs=["preprocessed_test_data"],
                name="preprocess_test_data",
            ),
        ]
    )
