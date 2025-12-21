# Prediction is not supported for this training model
# This is a HunyuanVideo 1.5 LoRA training model
# Use the training endpoint to train LoRAs

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """No setup needed for training model"""
        pass

    def predict(
        self,
        prompt: str = Input(description="This is a training model. Use 'train' endpoint instead.", default="")
    ) -> str:
        return "This model is for LoRA training only. Please use the training endpoint with your video dataset."
