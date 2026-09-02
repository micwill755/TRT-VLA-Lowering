from exporter import FUSE_OP, FuseSpec


def fuse_spec() -> FuseSpec:
    return FuseSpec(FUSE_OP)
