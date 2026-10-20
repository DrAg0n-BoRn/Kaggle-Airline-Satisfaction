import torch

from dragon.ML_inference import DragonInferenceHandler
from dragon.ML_models import DragonNodeModel
from dragon.ML_utilities import DragonArtifactFinder
from dragon.utilities import load_dataframe, save_dataframe_filename
from dragon.keys import InferenceKeys

from paths import PM
from helpers.constants import Features, TARGET


DEVICE = "cpu"  # Dataset is large for inference


def load_inference_model():
    artifact_finder = DragonArtifactFinder(directory=PM.classification, load_scaler=True)
    
    model = DragonNodeModel.load_architecture(artifact_finder.model_architecture_path)
    
    handler = DragonInferenceHandler(model=model,
                                     state_dict=artifact_finder.weights_path,
                                     device=DEVICE,
                                     scaler=artifact_finder.scaler_path)
    
    return handler


def run_inference(handler: DragonInferenceHandler):
    df_test, _ = load_dataframe(PM.engineering_test_file, kind="pandas")
    
    # transform dataframe to numpy array
    test_data = df_test.to_numpy()
    
    # predict using the inference handler
    results = handler.predict_batch_numpy(features=test_data)
    
    # extract probabilities for positive class
    class_probs = results[InferenceKeys.PROBABILITIES]
    
    return class_probs


def save_predictions(predictions):
    # load original IDs
    df_predictions, _ = load_dataframe(PM.base_test_file, kind="pandas", use_columns=[Features.ID])
    
    df_predictions[TARGET] = predictions
    
    filename = f"airline-satisfaction-predictions.csv"
    # save to CSV
    save_dataframe_filename(df=df_predictions, save_dir=PM.datasets, filename=filename)
    

if __name__ == "__main__":
    
    # Load the chosen model
    inference_handler = load_inference_model()
    
    # Run inference
    predictions = run_inference(inference_handler)
    
    # Save predictions
    save_predictions(predictions)
