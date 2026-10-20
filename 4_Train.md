---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: pregnancy-hypothyroidism
    language: python
    name: python3
---

```python
from dragon.ML_datasetmaster import DragonDataset as ChosenDataset
from dragon.ML_trainer import DragonTrainer as ChosenTrainer
from dragon.ML_models import DragonNodeModel as ChosenModel
from dragon.ML_configuration import (
    FormatMultiTargetRegressionMetrics as ChosenMetricsConfig, 
    FinalizeMultiTargetRegression as ChosenFinalizer 
)

from dragon.ML_configuration import DragonTrainingConfig
from dragon.ML_callbacks import DragonPatienceEarlyStopping, DragonPlateauScheduler
from dragon.ML_utilities import build_optimizer_params
from dragon.schema import FeatureSchema
from dragon.keys import TaskKeys
from torch.optim import AdamW

from paths import PM
from helpers.constants import TARGET
```

```python
!yes | plotly_get_chrome
```

```python
SCHEMA_PATH = PM.engineering
TRAIN_DATASET_PATH = PM.datasets
TRAIN_ARTIFACTS_DIR = PM.regression
```

## 1. Load Schema and Dataset

```python
schema = FeatureSchema.from_json(SCHEMA_PATH)
```

```python
dataset = ChosenDataset.from_bundle(TRAIN_DATASET_PATH, verbose=True)
```

## 2. Train Config

```python
train_config = DragonTrainingConfig(
    finalized_filename = "node",
    initial_learning_rate=0.001,
    batch_size=32,
    device = "cuda:0",
    task = TaskKeys.REGRESSION,
    
    target=TARGET,
    # targets=[TARGET_A, TARGET_B] # multitarget tasks
    weight_decay=0.005,
    early_stop_patience=25,
    scheduler_patience=6,
    scheduler_lr_factor=0.8,
    monitor_metric="Validation Loss"
)
```

## 3. Model and Trainer

```python
# Model
model = ChosenModel(schema=schema,
                    out_targets=dataset.number_of_targets, # set to number of classes for classification tasks
                    embedding_dim=16,
                    num_trees=256,
                    num_layers=1,
                    tree_depth=3,
                    additional_tree_output_dim=3,
                    input_dropout=0.0,
                    embedding_dropout=0.0,
                    choice_function='entmax',
                    bin_function='entmoid',
                    batch_norm_continuous=True
                    )

# Initialize decision thresholds before training.
if hasattr(model, "data_aware_initialization"):
    model.data_aware_initialization(train_dataset=dataset.train_dataset, num_samples=1000)

# Optimizer
optim_params = build_optimizer_params(model=model, weight_decay=train_config.weight_decay)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)

# Trainer
trainer = ChosenTrainer(model=model,
                        train_dataset=dataset.train_dataset,
                        validation_dataset=dataset.validation_dataset,
                        save_dir=TRAIN_ARTIFACTS_DIR,
                        kind=train_config.task,
                        optimizer=optimizer,
                        device=train_config.device,
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        checkpoint_config="default")
```

## 4. Training

```python
trainer.fit(epochs=500, 
            batch_size=train_config.batch_size, 
            use_torch_compile=True)
```

## 5. Evaluation

```python
trainer.evaluate(model_checkpoint="best",
                test_data=dataset.test_dataset,
                val_format_configuration=ChosenMetricsConfig(),
                test_format_configuration=ChosenMetricsConfig(),
                # classification_threshold=0.5 # for binary classification tasks
                )
```

## Explanation

```python
trainer.explain_captum(n_samples=500,
                       n_steps=200)
```

```python
trainer.explain_shap(explain_dataset=dataset.test_dataset,
                    n_samples=100,
                    explainer_type="kernel")
```

## 6. Save artifacts

```python
# Dataset artifacts
dataset.save_artifacts(TRAIN_ARTIFACTS_DIR)

# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)

# Train log
trainer.save_training_log(train_config=train_config)
```

## 7. Finalize Deep Learning

```python
trainer.finalize_model_training(finalize_config=ChosenFinalizer(filename=train_config.finalized_filename,
                                                                target_name=train_config.target))
```
