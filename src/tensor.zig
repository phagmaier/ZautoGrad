const std = @import("std");

pub fn Tensor(comptime T: type) type {
    if (@typeInfo(T) != .float)
        @compileError("Scalar requires a floating-point type");

    return struct {
        const Self = @This();
        const BackFn = *const fn (*Self) void;
        val: T,
        grad: T = 0,
        lhs: ?*Self = null,
        rhs: ?*Self = null,
        extra: T = 0,
        back: ?BackFn,

        pub fn init(allocator: std.mem.Allocator, val: T, lhs: ?*Self, rhs: ?*Self, extra: T, back: BackFn) !*Self {
            const s = try allocator.create(Self);
            s.* = .{
                .val = val,
                .lhs = lhs,
                .rhs = rhs,
                .extra = extra,
                .back = back,
            };
            return s;
        }

        pub fn constant(allocator: std.mem.Allocator, val: T) !*Self {
            return init(
                allocator,
                val,
                null,
                null,
                null,
                null,
            );
        }

        pub fn backAdd(self: *Self) void {
            if (self.lhs) |lhs| {
                if (self.rhs) |rhs| {
                    lhs.grad += self.grad;
                    rhs.grad += self.grad;
                    return;
                }
            }
            std.debug.assert(false);
        }

        pub fn backMul(self: *Self) void {
            if (self.lhs) |lhs| {
                if (self.rhs) |rhs| {
                    lhs.grad += rhs.val * self.grad;
                    rhs.grad += lhs.val * self.grad;
                    return;
                }
            }
            std.debug.assert(false);
        }

        pub fn backPow(self: *Self) void {
            if (self.lhs) |lhs| {
                if (self.rhs) |rhs| {
                    lhs.grad += (rhs.val * std.math.pow(T, lhs.val, rhs.val - 1)) * self.grad;
                    return;
                }
            }
            std.debug.assert(false);
        }

        pub fn add(a: *Self, b: *Self, allocator: std.mem.Allocator) !*Self {
            return init(allocator, a.val + b.val, a, b, 0, &backAdd);
        }

        pub fn mul(a: *Self, b: *Self, allocator: std.mem.Allocator) !*Self {
            return init(allocator, a.val * b.val, a, b, 0, &backMul);
        }

        pub fn pow(a: *Self, exp: T, allocator: std.mem.Allocator) !*Self {
            return init(
                allocator,
                std.math.pow(T, a.val, exp),
                a,
                null,
                exp,
                &backPow,
            );
        }
    };
}
