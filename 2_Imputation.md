---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
---

# Data Imputation and Inflation Factor

```python vscode={"languageId": "plaintext"}
import polars as pl
from dragon_imputation import mice_impute

from dragon.imputation import DragonImputationEvaluator
from dragon.VIF import DragonVIF
from dragon.utilities import save_dataframe_with_schema, load_dataframe_with_schema, merge_dataframes_horizontal
from dragon.data_exploration import split_features_targets, summarize_dataframe
from dragon.schema import FeatureSchema

from paths import PM
from helpers.constants import TARGET
```

## 1. Load Feature Schema and Dataset

```python vscode={"languageId": "plaintext"}
schema = FeatureSchema.from_json(PM.engineering)
```

```python vscode={"languageId": "plaintext"}
# load dataframe
df_engineered_raw, _ = load_dataframe_with_schema(df_path=PM.engineering_file, schema=schema)
```

```python vscode={"languageId": "plaintext"}
# split features and targets
df_features, df_targets = split_features_targets(df=df_engineered_raw, targets=[TARGET])
```

```python vscode={"languageId": "plaintext"}
summarize_dataframe(df_features)
```

## 2. Imputation

```python vscode={"languageId": "plaintext"}
# convert features to polars for imputation (Rust backend)
_df_features_pl = pl.from_pandas(df_features)
```

```python vscode={"languageId": "plaintext"}
_df_imputed_pl = mice_impute(df=_df_features_pl,
                            cat_cols=schema.categorical_indices,
                            max_iterations=30,
                            seed=42)
```

```python vscode={"languageId": "plaintext"}
# convert polars dataframe to pandas
df_features_imputed = _df_imputed_pl.to_pandas(use_pyarrow_extension_array=False)
```

```python vscode={"languageId": "plaintext"}
summarize_dataframe(df_features_imputed)
```

## 3. VIF Analysis

```python vscode={"languageId": "plaintext"}
vif_analyzer = DragonVIF(df_or_path=df_features_imputed, 
                         schema=schema,
                         dataset_name="imputed_data")
```

```python vscode={"languageId": "plaintext"}
vif_analyzer.compute_vif()
```

```python vscode={"languageId": "plaintext"}
vif_analyzer.plot_vif(save_dir=PM.imputation)
```

## 4. Imputation Analysis

```python vscode={"languageId": "plaintext"}
evaluator = DragonImputationEvaluator(df_original=df_features,
                                      df_imputed=df_features_imputed,
                                      categorical_schema=schema)
```

```python vscode={"languageId": "plaintext"}
evaluator.plot_missing_overview(save_dir=PM.imputation,
                                plot_filename="Engineered Dataset",
                                use_all_columns=False)
```

```python vscode={"languageId": "plaintext"}
evaluator.plot_overall_distributions(save_dir=PM.imputation / "Overall Distributions",
                                     only_imputed_columns=True,
                                     mode="percentage",
                                     palette="Set2")
```

```python vscode={"languageId": "plaintext"}
evaluator.plot_imputed_vs_observed(save_dir=PM.imputation / "Imputed vs Observed",
                                    mode="percentage",
                                    palette="Accent")
```

```python vscode={"languageId": "plaintext"}
evaluator.evaluate_continuous(save_dir=PM.imputation)
```

```python vscode={"languageId": "plaintext"}
evaluator.evaluate_categorical(save_dir=PM.imputation)
```

## 5. Save Imputed Data

```python vscode={"languageId": "plaintext"}
# merge imputed features with targets
df_imputed_final = merge_dataframes_horizontal([df_features_imputed, df_targets], reset_index=True)
```

```python vscode={"languageId": "plaintext"}
save_dataframe_with_schema(df=df_imputed_final, full_path=PM.imputation_file, schema=schema)
```
