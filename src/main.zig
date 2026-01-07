const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("tensor.zig");
const Op = Tensor.Op;

pub fn main() !void {
    var da = std.heap.DebugAllocator(.{}){};
    const allocator = if (builtin.mode == .Debug)
        da.allocator()
    else
        std.heap.smp_allocator;

    defer _ = da.deinit();
    const Val = Tensor.Tensor(f32);
    const b = Val.init(allocator, 1.0, Op.add, null, null);
    defer allocator.destroy(b);
    std.debug.print("B val: {d}\n", .{b.val});
}
