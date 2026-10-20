---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: kaggle-airline-satisfaction (3.14.x)
    language: python
    name: python3
---

```python
from dragon.ML_datasetmaster import DragonDataset as ChosenDataset
from dragon.utilities import load_dataframe_with_schema
from dragon.schema import FeatureSchema
from dragon.keys import TaskKeys

from paths import PM
```

```python
SCHEMA_PATH = PM.engineering
TRAIN_DATASET_INPUT_FILE = PM.engineering_train_file
TRAIN_DATASET_OUTPUT_DIR = PM.datasets
```

## 1. Load Schema and Dataframe

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)

df, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_INPUT_FILE, 
                                   schema=schema)
```

## 2. Make Dataset Class

```python
dataset = ChosenDataset(pandas_df=df,
                        schema=schema,
                        kind=TaskKeys.BINARY_CLASSIFICATION,
                        feature_scaler="fit",
                        target_scaler="none", # none for classification/autoencoder tasks
                        validation_size=0.3,
                        test_size=None,
                        random_state=42,
                        class_map={"Not Satisfied": 0, "Satisfied": 1}
                        )
```

```python
# inspect the dataset
dataset
```

```python
# inspect scaler
dataset.feature_scaler
```

## 3. Save Dataset Class

```python
dataset.save_dataset_bundle(directory=TRAIN_DATASET_OUTPUT_DIR)
```
