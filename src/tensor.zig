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
            //@memset(vals, 0);
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

        pub fn get(self: *const Self, index: []const usize) T {
            std.debug.assert(index.len == self.shape.len);
            var idx: usize = 0;
            for (0..self.stride.len) |i| {
                std.debug.assert(index[i] < self.shape[i]);
                idx += index[i] * self.stride[i];
            }
            return self.vals[idx];
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

        pub inline fn fill(self: *Self, num: T) void {
            @memset(self.vals, num);
        }
        pub inline fn fillOnes(self: *Self) void {
            self.fill(1);
        }
        pub inline fn fillZeros(self: *Self) void {
            self.fill(0);
        }
        ///(min,Max] Fills with random numbers exclusive of max
        pub fn fillRandom(self: *Self, min: T, max: T) !void {
            var prng: std.Random.DefaultPrng = .init(blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rand = prng.random();
            for (0..self.vals.len) |i| {
                self.vals[i] = rand.intRangeAtMost(T, min, max);
            }
        }

        ///Fills with Random Normal (Gaussian) numbers in the range of 0-1
        pub fn fillNormal(self: *Self, stdDev: ?T, mean: ?T) !void {
            var prng: std.Random.DefaultPrng = .init(blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rand = prng.random();

            if (stdDev != null and mean == null or mean != null and stdDev == null) {
                std.debug.print("If stdDev is set mean must also be set and vice versa\n");
                return error.InvalidInput;
            }
            if (stdDev) |dev| {
                if (mean) |m| {
                    for (0..self.vals.len) |i| {
                        self.vals[i] = rand.floatNorm(T) * dev + m;
                    }
                    return;
                }
            }
            for (0..self.vals.len) |i| {
                self.vals[i] = rand.floatNorm(T) * stdDev + mean;
            }
        }
    };
}

pub fn tensorFromSlice(T: comptime_float, allocator: std.mem.Allocator, arr: []const T, shape: []const usize) !void {
    const TENSOR = Tensor(T);
    const tensor = try TENSOR.init(allocator, shape);
    for (arr) |i| {
        tensor.vals = i;
    }
}
