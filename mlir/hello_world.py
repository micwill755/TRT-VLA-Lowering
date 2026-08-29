from graphir import Target, mlir
from graphir.mlir.dialects import nvg
from graphir.mlir.nvg_helpers import row_major_strides, tensor_type

ctx = mlir.Context()
module = mlir.ModuleOp.create_empty(ctx)
builder = mlir.OpBuilder(ctx)
builder.set_insertion_point_to_end(module.get_body())

f32 = ctx.get_f32_type()
tensor_ty = tensor_type(ctx, f32, [1024], row_major_strides([1024]))

graph = nvg.GraphOp.create(builder, "add", [tensor_ty, tensor_ty])
ip = builder.set_insertion_point_to_start(graph.get_body())
a, b = list(graph.get_body().get_arguments())
c = nvg.AddOp.create(builder, a, b).get_result()
nvg.GraphReturnOp.create(builder, [c])
graph.set_function_type([tensor_ty, tensor_ty], [c.get_type()])
builder.restore_insertion_point(ip)

print(module.print())

target = Target.query_local_device()
compiled = module.compile(target=target)