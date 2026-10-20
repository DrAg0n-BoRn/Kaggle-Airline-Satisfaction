from dragon.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "start_data", "results"]
)

# 2. Define directories and files.
### Base files
PM.base_train_file = PM.start_data / "train.csv"
PM.base_test_file = PM.start_data / "test.csv"

### Datasets
PM.datasets = PM.results / "0 Datasets"

### Feature Engineering
PM.engineering = PM.results / "1 Feature Engineering"
PM.engineering_plots = PM.engineering / "Engineering Plots"
PM.engineering_train_file = PM.datasets / "engineered_train_data.csv"
PM.engineering_test_file = PM.datasets / "engineered_test_data.csv"

### Classification
PM.classification = PM.results / "4 Classification"



# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
