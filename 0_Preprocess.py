from dragon.ETL_engineering import DragonProcessor

from helpers.function_map import TRANSFORMATION_RECIPE
from paths import PM


if __name__ == "__main__":
    # instantiate processor
    processor = DragonProcessor(recipe=TRANSFORMATION_RECIPE)
    
    # Process df
    processor.load_transform_save(input_path=PM.base_file, output_path=PM.processed_file)
