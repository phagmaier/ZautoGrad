const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("tensor.zig");

pub fn main() !void {
    var da = std.heap.DebugAllocator(.{}){};
    defer _ = da.deinit();
    const allocator = if (builtin.mode == .Debug)
        da.allocator()
    else
        std.heap.smp_allocator;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const Value = Tensor.Tensor(f32);
    const b = try Value.init(arena.allocator(), &[_]u64{1});
    std.debug.print("B val: {d}\n", .{b.vals[0]});
}
