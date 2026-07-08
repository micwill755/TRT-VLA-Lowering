from __future__ import annotations

from trt.executor.models.pi05.load.serialize import (
    SerializedPi05Action,
    SerializedPi05Language,
)


class SerializedSmolVLAVision:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, pixel_values):
        input_name = "pixel_values"
        if input_name not in self.engine.config_input_names:
            input_name = self.engine.config_input_names[0]
        return self.engine({input_name: pixel_values})[0]


SerializedSmolVLALanguage = SerializedPi05Language
SerializedSmolVLAAction = SerializedPi05Action
