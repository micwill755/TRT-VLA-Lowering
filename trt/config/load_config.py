from __future__ import annotations

from dataclasses import dataclass

from trt.serialize import SerializedModuleSpec


@dataclass(frozen=True)
class SerializedStageSpec:
    """One serialized TRT stage under ``engine_root/<engine_subdir>/``."""

    key: str
    engine_subdir: str
    wrapper_cls: type
    optional: bool = False

    def to_module_spec(self) -> SerializedModuleSpec:
        return SerializedModuleSpec(self.key, self.engine_subdir, self.wrapper_cls)


@dataclass(frozen=True)
class LoadPipelineConfig:
    stages: tuple[SerializedStageSpec, ...]
