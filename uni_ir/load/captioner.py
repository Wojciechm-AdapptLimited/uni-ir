import base64

from io import BytesIO
from PIL import Image as PILImage
from transformers import Blip2ForConditionalGeneration, Blip2Processor, TensorType


class ImageCaptioner:
    def __init__(self, model: Blip2ForConditionalGeneration, processor: Blip2Processor):
        self.model = model
        self.processor = processor

    def caption(
        self,
        payload: str,
    ) -> str:
        if not payload:
            return ""

        content = PILImage.open(BytesIO(base64.b64decode(payload))).convert("RGB")

        inputs = self.processor(
            content, "an image of", return_tensors=TensorType.PYTORCH
        )

        outputs = self.model.generate(
            inputs.pixel_values, inputs.input_ids, inputs.attention_mask
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True).strip()
