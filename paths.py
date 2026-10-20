from dragon.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results", "backups"]
)

# 2. Define directories and files.
### Base files
PM.base_file = PM.start_data / "clean_data.csv"

### Datasets
PM.datasets = PM.results / "0 Datasets"
PM.processed_file = PM.datasets / "processed_data.csv"

### Feature Engineering
PM.engineering = PM.results / "1 Feature Engineering"
PM.engineering_plots = PM.engineering / "Engineering Plots"
PM.engineering_file = PM.datasets / "engineered_data.csv"
PM.target_class_map_file = PM.engineering / "target_class_map.json"

### Imputation - VIF
PM.imputation = PM.results / "2 Imputation - VIF"
PM.imputation_file = PM.datasets / "imputed_data.csv"

### Classification
PM.classification = PM.results / "4 Classification"
PM.classification_filler = PM.classification / "Filler"
PM.classification_curing = PM.classification / "Curing"

### Regression
PM.regression = PM.results / "4 Regression"
PM.regression_filler = PM.regression / "Filler"

### Regression Chain
PM.chain = PM.results / "5 Regression Chain"
PM.chain_temp = PM.chain / "temp"
# step1
PM.chain_tensile_file = PM.chain_temp / "tensile_strength.csv"
PM.chain_tensile = PM.chain / "Tensile Strength"
# step2
PM.chain_artifacts2 = PM.chain_temp / "artifacts step2"
PM.chain_flexural_file = PM.chain_temp / "flexural_strength.csv"
PM.chain_flexural = PM.chain / "Flexural Strength"

### Optimization
PM.optimization = PM.results / "6 Optimization"
PM.optimization_file = PM.datasets / "predicted_samples.csv"

### Autoencoder
PM.autoencoder = PM.results / "4 Autoencoder"

### Diffusion 
PM.diffusion = PM.results / "5 Diffusion"

### Generation
PM.generation = PM.results / "6 Generation"
PM.generation_file = PM.datasets / "generated_data.csv"

### Analysis
PM.analysis = PM.results / "7 Analysis"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
