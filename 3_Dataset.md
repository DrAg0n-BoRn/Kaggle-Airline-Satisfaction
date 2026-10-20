---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
---

```python
from dragon.ML_datasetmaster import DragonDataset as ChosenDataset
from dragon.utilities import load_dataframe_with_schema
from dragon.schema import FeatureSchema
from dragon.IO_tools import load_json
from dragon.keys import TaskKeys

from paths import PM
```

```python
SCHEMA_PATH = PM.engineering
TRAIN_DATASET_INPUT_FILE = PM.imputation_file
TRAIN_DATASET_OUTPUT_DIR = PM.datasets
```

## 1. Load Schema and Dataframe

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)

df, _ = load_dataframe_with_schema(df_path=TRAIN_DATASET_INPUT_FILE, 
                                   schema=schema)
```

```python
# target_class_map = load_json(PM.target_class_map_file)
```

## 2. Make Dataset Class

```python
dataset = ChosenDataset(pandas_df=df,
                        schema=schema,
                        kind=TaskKeys.REGRESSION,
                        feature_scaler="fit",
                        target_scaler="fit", # none for classification/autoencoder tasks
                        validation_size=0.2,
                        test_size=0.1,
                        random_state=42,
                        # class_map=target_class_map # for classification tasks
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
