from exporter import FuseSpec, SCATTER_OP


def fuse_spec() -> FuseSpec:
    return FuseSpec(SCATTER_OP, extra_keys=("image_token_mask",))
