const std = @import("std");

pub fn Tensor(T: type) type {
    if (@typeInfo(T) != .float) @compileError("Scalar requires a floating-point type");

    return struct {
        const Self = @This();
        const BackFn = *const fn (*Self) void;

        vals: []T,
        grads: []T,
        shape: []usize,
        stride: []usize,
        back: ?BackFn,

        pub fn init(allocator: std.mem.Allocator, shape: []const usize) !*Self {
            if (shape.len == 0) {
                std.debug.print("Vector requires a shape greater than 0\n", .{});
                return error.ZeroShape;
            }
            const s = try allocator.create(Self);
            errdefer allocator.destroy(s);
            const strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);
            const shapes = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(shapes);
            var size: usize = 1;
            for (0..shape.len) |i| {
                const item = shape[i];
                if (item == 0) {
                    std.debug.print("0 Dimension in Tensor at index: {d}\n", .{i});
                    return error.ZeroShape;
                }
                shapes[i] = item;
                size *= item;
            }
            strides[shape.len - 1] = 1;
            var i: usize = shape.len - 1;
            while (i > 0) {
                i -= 1;
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            const vals = try allocator.alloc(T, size);
            errdefer allocator.free(vals);
            @memset(vals, 0);
            const grads = try allocator.alloc(T, size);
            @memset(grads, 0);

            s.* = .{
                .vals = vals,
                .grads = grads,
                .shape = shapes,
                .stride = strides,
                .back = null,
            };

            return s;
        }
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.vals);
            allocator.free(self.grads);
            allocator.free(self.shape);
            allocator.free(self.stride);
            allocator.destroy(self);
        }

        pub fn get(self: *Self, index: []const usize) T {
            std.debug.assert(index.len == self.shape.len);
            var idx: usize = 0;
            for (0..self.stride.len) |i| {
                std.debug.assert(index[i] < self.shape[i]);
                idx += index[i] * self.stride[i];
            }
            return idx;
        }
        pub fn set(self: *Self, index: []const usize, val: T) void {
            std.debug.assert(index.len == self.shape.len);
            var idx: usize = 0;
            for (0..self.stride.len) |i| {
                std.debug.assert(index[i] < self.shape[i]);
                idx += index[i] * self.stride[i];
            }
            self.vals[idx] = val;
        }
    };
}
