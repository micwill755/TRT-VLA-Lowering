class VLAExportHooks(ABC):
    """Override only the stages that differ between VLAs."""

    tokenizer: Any | None = None

    @abstractmethod
    def preprocess(self, ctx: ExportContext) -> None:
        """Populate ``ctx.pixel_values``, ``ctx.tokenized``, ``ctx.action_side``."""

    @abstractmethod
    def build_vision_spec(self, ctx: ExportContext) -> VisionEngineSpec:
        ...

    def uses_prefix_kv_action(self) -> bool:
        """True when action rollout consumes LM prefix K/V (PI0.5, SmolVLA)."""
        return bool(self.io.lm_to_action_slots) and self.io.action_context is None

    def compute_image_embs(self, ctx: ExportContext, vision_module: nn.Module) -> torch.Tensor | list[torch.Tensor]:
        """Run in-memory vision and return embedding(s) for language packing."""
        from trt.vision import nchw_to_hwc, run_trt_vision_nchw

        images = ctx.action_side.get("images")
        if images is None:
            return run_trt_vision_nchw(vision_module, ctx.pixel_values.to(device=ctx.device))
        return [
            run_trt_vision_nchw(vision_module, image.to(device=ctx.device))
            for image in images
        ]

    def dummy_image_embs(self, ctx: ExportContext) -> torch.Tensor | list[torch.Tensor]:
        spec = ctx.vis_spec
        if spec is None:
            raise RuntimeError("build_vision_spec must run before dummy_image_embs")
        images = ctx.action_side.get("images")
        if images is not None and spec.image_embed_shape:
            return [
                torch.zeros(
                    *spec.image_embed_shape,
                    device=ctx.device,
                    dtype=spec.input_dtype,
                )
                for _ in images
            ]
        if spec.image_embed_shape:
            return torch.zeros(
                *spec.image_embed_shape,
                device=ctx.device,
                dtype=spec.input_dtype,
            )
        return torch.zeros(
            *spec.image_embed_flat_shape,
            device=ctx.device,
            dtype=spec.input_dtype,
        )

    @abstractmethod
    def pack_language_inputs(self, ctx: ExportContext) -> dict:
        """Return packed LM inputs (``inputs_embeds``, masks, ``position_ids``, ...)."""

    @abstractmethod
    def build_language_spec(self, ctx: ExportContext) -> LanguageEngineSpec:
        ...

    @abstractmethod
    def build_chat_template(self, tokenizer: Any) -> dict[str, Any]:
        """Build ``processed_chat_template.json`` for VitRunner."""

    def save_language_artifacts(self, ctx: ExportContext, language_dir: Path) -> None:
        """Write embedding table, tokenizer assets, and chat template into ``language/``."""
        save_embedding_table(ctx.lang_spec.language_model, language_dir)
        save_tokenizer_for_edge_llm(
            language_dir,
            tokenizer=self.tokenizer,
            chat_template=self.build_chat_template(self.tokenizer),
        )

    def has_action_context(self, ctx: ExportContext) -> bool:
        return ctx.io.action_context is not None

    def build_action_context(self, ctx: ExportContext) -> ComponentBuild | None:
        return None

    @abstractmethod
    def build_diffusion_spec(self, ctx: ExportContext) -> DiffusionEngineSpec:
        ...

    def compile_language_in_memory(
        self,
        ctx: ExportContext,
        sink: ExportSink,
    ) -> nn.Module | None:
        return None

    def after_export(self, ctx: ExportContext, sink: ExportSink) -> None:
        """Parity checks, serialized runner smoke, fixture dumps."""
