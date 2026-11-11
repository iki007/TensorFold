//! Copy drafts for the lone XPU stream (XPU_COPY=1): the tokens that followed the latest earlier occurrence of the
//! context's last GRAM tokens. Agent replies repeat their context (an edit's old text, a rewritten file, a quoted
//! command); a copied run is verified like any draft, so the reply is the same with or without it.
const std = @import("std");

pub const GRAM = 12; // a shorter match proposes runs that mostly fail (simulated on recorded opencode sessions)

pub const Index = struct {
    map: std.AutoHashMapUnmanaged(u64, u32) = .empty, // a gram's hash -> the position after its latest occurrence
    done: usize = 0, // the context's positions indexed so far

    fn key(g: []const u32) u64 {
        return std.hash.Wyhash.hash(0x636f7079, std.mem.sliceAsBytes(g));
    }

    pub fn deinit(x: *Index, gpa: std.mem.Allocator) void {
        x.map.deinit(gpa);
    }

    /// Indexes every gram of `h` that has a token after it (all but the gram ending at h's end).
    pub fn extend(x: *Index, gpa: std.mem.Allocator, h: []const u32) !void {
        var i = @max(x.done, GRAM);
        while (i < h.len) : (i += 1) try x.map.put(gpa, key(h[i - GRAM .. i]), @intCast(i));
        x.done = @max(x.done, h.len);
    }

    /// At most `max` tokens after the latest earlier occurrence of h's last GRAM tokens; empty without one.
    pub fn propose(x: *const Index, h: []const u32, max: usize) []const u32 {
        if (h.len <= GRAM) return &.{};
        const tail = h[h.len - GRAM ..];
        const p = x.map.get(key(tail)) orelse return &.{};
        if (p < GRAM or p >= h.len or !std.mem.eql(u32, h[p - GRAM .. p], tail)) return &.{}; // a hash collision
        return h[p..@min(h.len, p + max)];
    }
};

test "a repeated run is proposed from its earlier occurrence, a new one is not" {
    const gpa = std.testing.allocator;
    var x: Index = .{};
    defer x.deinit(gpa);
    var h: [40]u32 = undefined;
    for (0..20) |i| h[i] = @intCast(100 + i); // a run of 20
    for (20..32) |i| h[i] = @intCast(100 + i - 20); // its first 12 again
    try x.extend(gpa, h[0..32]);
    try std.testing.expectEqualSlices(u32, &.{ 112, 113, 114, 115 }, x.propose(h[0..32], 4));
    h[32] = 7;
    try x.extend(gpa, h[0..33]);
    try std.testing.expectEqual(@as(usize, 0), x.propose(h[0..33], 4).len);
}
