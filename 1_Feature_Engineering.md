---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
---

# Feature Engineering

```python vscode={"languageId": "plaintext"}
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
from dragon.ETL_cleaning import DragonFaultyEntryManager
from dragon.outlier_detection import (isolation_forest,
                                      local_outlier_factor,
                                      drop_outliers_mask)
from dragon.utilities import load_dataframe, save_dataframe_with_schema, merge_dataframes_horizontal
from dragon.schema import finalize_feature_schema

from paths import PM
from helpers.constants import TARGET
```

## 1. Load Data

```python vscode={"languageId": "plaintext"}
df_raw, _ = load_dataframe(df_path=PM.processed_file, kind="pandas")
```

## 2. Data Cleaning

```python vscode={"languageId": "plaintext"}
df_cleanI = drop_macro(df=df_raw, 
                       log_directory=PM.engineering, 
                       targets=[TARGET],
                       skip_targets=True,
                       threshold=0.7)
```

```python vscode={"languageId": "plaintext"}
df_cleanII = clean_column_names(df=df_cleanI)
```

```python vscode={"languageId": "plaintext"}
summarize_dataframe(df_cleanII)
```

```python vscode={"languageId": "plaintext"}
df_clean = df_cleanII
```

### Manual cleaning

```python vscode={"languageId": "plaintext"}
dragon_manager = DragonFaultyEntryManager(df=df_clean)
```

```python vscode={"languageId": "plaintext"}
dragon_manager.find(targets={})
```

```python vscode={"languageId": "plaintext"}
dragon_manager.drop_rows(faulty_ids=[])
```

```python vscode={"languageId": "plaintext"}
dragon_manager.apply_fix(fixes={})
```

```python vscode={"languageId": "plaintext"}
df_clean = dragon_manager.return_df()
```

```python vscode={"languageId": "plaintext"}
summarize_dataframe(df_clean)
```

## 3. Outlier Detection

```python vscode={"languageId": "plaintext"}
outliers_mask_forest = isolation_forest(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python vscode={"languageId": "plaintext"}
outliers_mask_lof = local_outlier_factor(df_features=df_clean,
                                        ignore_columns=[TARGET],
                                        shadow_median_imputation=False)
```

```python vscode={"languageId": "plaintext"}
df_clean[outliers_mask_lof & outliers_mask_forest]
```

```python vscode={"languageId": "plaintext"}
df_processed = drop_outliers_mask(df=df_clean,
                              outlier_mask=outliers_mask_forest & outliers_mask_lof,
                              reset_index=True)
```

```python vscode={"languageId": "plaintext"}
summarize_dataframe(df_processed)
```

## 4. Split Features and Targets

```python vscode={"languageId": "plaintext"}
df_continuous, df_categorical, df_targets = split_continuous_categorical_targets(df=df_processed,
                                                                                categorical_cols=[],
                                                                                target_cols=[TARGET])
```

```python vscode={"languageId": "plaintext"}
show_null_columns(df=df_continuous)
```

```python vscode={"languageId": "plaintext"}
show_null_columns(df=df_categorical)
```

```python vscode={"languageId": "plaintext"}
show_null_columns(df=df_targets)
```

## 5. Plots

```python vscode={"languageId": "plaintext"}
plot_value_distributions(df=df_processed, save_dir=PM.engineering_plots)
```

```python vscode={"languageId": "plaintext"}
plot_correlation_heatmap(df=df_processed, plot_title="All Data", save_dir=PM.engineering_plots)
```

```python vscode={"languageId": "plaintext"}
plot_numeric_overview_boxplot_macro(df=df_continuous,
                                    save_dir=PM.engineering_plots,
                                    plot_title="Continuous Feature Distributions",
                                    handle_zero_variance="constant")
```

```python vscode={"languageId": "plaintext"}
plot_continuous_vs_target(df_continuous=df_continuous, df_targets=df_targets, save_dir=PM.engineering_plots)
```

```python vscode={"languageId": "plaintext"}
plot_categorical_vs_target(df_categorical=df_categorical, df_targets=df_targets, save_dir=PM.engineering_plots)
```

```python vscode={"languageId": "plaintext"}
plot_pairgrid_continuous_vs_target(df_continuous=df_continuous, df_targets=df_targets, save_dir=PM.engineering_plots)
```

```python vscode={"languageId": "plaintext"}
# check class balance (classification)
check_class_balance(df=df_targets, target=TARGET, plot_to_dir=PM.engineering_plots)
```

## 6. Encode Categorical Features

```python vscode={"languageId": "plaintext"}
df_categorical_encoded, categorical_mapping = encode_categorical_features(df_categorical=df_categorical,
                                                                          encode_nulls=True,
                                                                          null_label="Other",
                                                                          verbose=3)
```

## 7. Create Feature Schema

```python vscode={"languageId": "plaintext"}
df_features_final = merge_dataframes_horizontal([df_continuous, df_categorical_encoded])
```

```python vscode={"languageId": "plaintext"}
schema = finalize_feature_schema(features=df_features_final, categorical_mappings=categorical_mapping)
```

## 8. Reconstruct DataFrame

```python vscode={"languageId": "plaintext"}
# encode classification target
df_targets_encoded, target_class_map = encode_classification_target(df=df_targets, target_col=TARGET, save_path=PM.target_class_map_file)
```

```python vscode={"languageId": "plaintext"}
df_final = merge_dataframes_horizontal([df_features_final, df_targets_encoded], reset_index=True)
```

## 9. Save Artifacts

```python vscode={"languageId": "plaintext"}
save_dataframe_with_schema(df=df_final, full_path=PM.engineering_file, schema=schema)
```

```python vscode={"languageId": "plaintext"}
schema.to_json(PM.engineering)
schema.save_artifacts(PM.engineering)
```
