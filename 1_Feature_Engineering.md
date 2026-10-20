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

# Feature Engineering

```python
from dragon.data_exploration import (summarize_dataframe,
                                     drop_macro,
                                     clean_column_names,
                                     split_continuous_categorical_targets,
                                     plot_value_distributions,
                                     plot_numeric_overview_boxplot_macro,
                                     plot_continuous_vs_target,
                                     plot_categorical_vs_target,
                                     plot_correlation_heatmap,
                                     plot_pairgrid_continuous_vs_target,
                                     encode_categorical_features,
                                     check_class_balance,
                                     encode_classification_target,
                                     show_null_columns)
from dragon.outlier_detection import (isolation_forest,
                                      local_outlier_factor,
                                      drop_outliers_mask)
from dragon.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes_horizontal, merge_dataframes_vertical
from dragon.schema import finalize_feature_schema, apply_feature_schema

from paths import PM
from helpers.constants import TARGET, Features, CATEGORICAL_FEATURES
```

## 1. Load Data

```python
df_train_raw, _ = load_dataframe(df_path=PM.base_train_file, kind="pandas")
df_train_raw.set_index(Features.ID, inplace=True, drop=True)
```

```python
summarize_dataframe(df_train_raw)
```

```python
df_test_raw, _ = load_dataframe(df_path=PM.base_test_file, kind="pandas")
df_test_raw.set_index(Features.ID, inplace=True, drop=True)
```

```python
summarize_dataframe(df_test_raw)
```

```python
# check number of missing values in each column
show_null_columns(df_train_raw)
```

```python
show_null_columns(df_test_raw)
```

```python
# fill missing with 0
df_train_raw[Features.ARRIVAL_DELAY_IN_MINUTES] = df_train_raw[Features.ARRIVAL_DELAY_IN_MINUTES].fillna(0)
df_test_raw[Features.ARRIVAL_DELAY_IN_MINUTES] = df_test_raw[Features.ARRIVAL_DELAY_IN_MINUTES].fillna(0)
```

## 2. Data Cleaning

```python
df_cleanI = drop_macro(df=df_train_raw, 
                       log_directory=PM.engineering, 
                       targets=[TARGET],
                       skip_targets=True,
                       threshold=0.7)
```

```python
df_cleanII = clean_column_names(df=df_cleanI)
```

```python
summarize_dataframe(df_cleanII)
```

```python
df_clean = df_cleanII
```

## 3. Outlier Detection

```python
outliers_mask_forest = isolation_forest(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python
outliers_mask_lof = local_outlier_factor(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python
df_clean[outliers_mask_lof & outliers_mask_forest]
```

```python
df_processed = drop_outliers_mask(df=df_clean,
                              outlier_mask=outliers_mask_forest & outliers_mask_lof,
                              reset_index=True)
```

```python
summarize_dataframe(df_processed)
```

## 4. Split Features and Targets

```python
df_continuous, df_categorical, df_targets = split_continuous_categorical_targets(df=df_processed,
                                                                                categorical_cols=CATEGORICAL_FEATURES,
                                                                                target_cols=[TARGET])
```

```python
show_null_columns(df=df_continuous)
```

```python
show_null_columns(df=df_categorical)
```

```python
show_null_columns(df=df_targets)
```

## 5. Plots

```python
plot_value_distributions(df=df_processed, 
                         save_dir=PM.engineering_plots,
                         treat_as_categorical=[Features.INFLIGHT_WIFI_SERVICE,
                                               Features.DEPARTURE_ARRIVAL_TIME_CONVENIENT,
                                               Features.EASE_OF_ONLINE_BOOKING,
                                               Features.GATE_LOCATION,
                                               Features.FOOD_AND_DRINK,
                                               Features.ONLINE_BOARDING,
                                               Features.SEAT_COMFORT,
                                               Features.INFLIGHT_ENTERTAINMENT,
                                               Features.ON_BOARD_SERVICE,
                                               Features.LEG_ROOM_SERVICE,
                                               Features.BAGGAGE_HANDLING,
                                               Features.CHECKIN_SERVICE,
                                               Features.CLEANLINESS])
```

```python
plot_numeric_overview_boxplot_macro(df=df_continuous[[Features.AGE,
                                                      Features.FLIGHT_DISTANCE,
                                                      Features.ARRIVAL_DELAY_IN_MINUTES,
                                                      Features.DEPARTURE_DELAY_IN_MINUTES]],
                                    save_dir=PM.engineering_plots,
                                    plot_title="Continuous Feature Distributions",
                                    handle_zero_variance="constant")
```

```python
# check class balance (classification)
check_class_balance(df=df_targets, target=TARGET, plot_to_dir=PM.engineering_plots)
```

## 6. Encode Categorical Features

```python
# use all categories
_df_all_categorical = merge_dataframes_vertical(dfs=[df_categorical, df_test_raw[CATEGORICAL_FEATURES]], reset_index=True) 
```

```python
ENCODE_NULLS = False

_ , categorical_mapping = encode_categorical_features(df_categorical=_df_all_categorical,
                                                        encode_nulls=ENCODE_NULLS,
                                                        null_label="Other",
                                                        verbose=3)
```

## 7. Create Feature Schema

```python
all_features = df_continuous.columns.tolist() + df_categorical.columns.tolist()

schema = finalize_feature_schema(features=all_features, categorical_mappings=categorical_mapping)
```

## 8. Encode DataFrame

```python
df_train_encoded = apply_feature_schema(df=df_processed, 
                                        schema=schema,
                                        targets=[TARGET],
                                        null_as=0 if ENCODE_NULLS else None,
                                        verbose=3)
df_train_encoded[TARGET] = df_train_encoded[TARGET].astype("int64")
```

```python
df_test_encoded = apply_feature_schema(df=df_test_raw,
                                        schema=schema,
                                        null_as=0 if ENCODE_NULLS else None,
                                        verbose=3)
```

```python
summarize_dataframe(df_train_encoded)
```

```python
summarize_dataframe(df_test_encoded)
```

## 9. Save Artifacts

```python
save_dataframe_with_schema(df=df_train_encoded, full_path=PM.engineering_train_file, schema=schema)
```

```python
save_dataframe_with_schema(df=df_test_encoded, full_path=PM.engineering_test_file, schema=schema)
```

```python
schema.to_json(PM.engineering)
schema.save_artifacts(PM.engineering)
```
