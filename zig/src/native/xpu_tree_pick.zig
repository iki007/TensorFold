//! The lone stream's DFlash2 tree size each round: the most tokens a millisecond at its context (exact at any size).

const std = @import("std");

/// The sizes tried, largest first; a size's tokens are counted from every round that verified a tree at least as big.
pub const sizes = [_]u32{ 15, 7, 3, 1 };
const past = 0.8; // weight of the past in each running average
const warm = 2; // rounds at the biggest size before another is chosen
const first = 4; // rounds a size's tokens are a plain mean of, before the running average
const margin = 1.05; // a smaller size must beat the biggest one's rate by this much: noise does not shrink the tree
const saving = 0.25; // and save this share of its round: a short context, where rows cost little, keeps it
const probe = 20; // a probe at the biggest size once the time saved since the last reaches this many probe costs

pub const Pick = struct {
    tokens: [sizes.len]f64 = @splat(0), // tokens a round a size gives (its kept drafts and the terminal)
    seen: [sizes.len]u32 = @splat(0), // rounds that measured each size
    attn: ?f64 = null, // verify ms per row and context position, above the short-context window cost
    extra: ?f64 = null, // ms of a round outside its verify (drafter, host)
    since: u32 = 0, // rounds since the biggest size was verified

    /// The next tree's nodes, at most `max`; `ms[w - 1]` is the cost of a w-row window at a short context.
    pub fn choose(p: *Pick, ms: []const f64, pos: u64, max: u32) u32 {
        const top = for (sizes, 0..) |n, i| {
            if (n <= max and n + 1 <= ms.len) break i;
        } else return max;
        if (p.seen[top] < warm) return sizes[top];
        var best = top;
        var rate = margin * p.tokens[top] / p.cost(ms, pos, sizes[top]);
        for (sizes[top + 1 ..], top + 1..) |n, i| if (p.seen[i] > 0) {
            const r = p.tokens[i] / p.cost(ms, pos, n);
            if (r > rate) {
                best = i;
                rate = r;
            }
        };
        if (best == top) return sizes[top];
        const saved = p.cost(ms, pos, sizes[top]) - p.cost(ms, pos, sizes[best]); // a probe's extra cost
        if (saved < saving * p.cost(ms, pos, sizes[top])) return sizes[top];
        if (@as(f64, @floatFromInt(p.since)) * p.cost(ms, pos, sizes[best]) >= probe * saved) return sizes[top];
        return sizes[best];
    }

    /// A round of `used` nodes at `pos`: `kept[k]` drafts a k-node tree would have kept, the verify and round seconds.
    pub fn learn(p: *Pick, used: u32, kept: []const u8, verify_s: f64, round_s: f64, ms: []const f64, pos: u64) void {
        if (used + 1 > ms.len) return;
        const rows: f64 = @floatFromInt(used + 1);
        const v = verify_s * 1e3;
        if (pos >= 2048) p.attn = avg(p.attn, @max(0, v - ms[used]) / rows / @as(f64, @floatFromInt(pos)));
        p.extra = avg(p.extra, @max(0, round_s * 1e3 - v));
        for (sizes, 0..) |n, i| if (n <= used) {
            const t: f64 = @floatFromInt(@as(u32, kept[n]) + 1);
            const k: f64 = @floatFromInt(p.seen[i]);
            p.tokens[i] = if (p.seen[i] < first) (p.tokens[i] * k + t) / (k + 1) else avg(p.tokens[i], t);
            p.seen[i] += 1;
        };
        p.since = if (used >= sizes[0]) 0 else p.since + 1;
    }

    fn cost(p: *const Pick, ms: []const f64, pos: u64, n: u32) f64 {
        const rows: f64 = @floatFromInt(n + 1);
        return (p.extra orelse 0) + ms[n] + rows * (p.attn orelse 0) * @as(f64, @floatFromInt(pos));
    }
};

fn avg(old: ?f64, x: f64) f64 {
    return if (old) |o| o * past + x * (1 - past) else x;
}

test "a short context keeps the big tree, a long one with costly rows takes a small one" {
    var ms: [16]f64 = undefined;
    for (&ms, 0..) |*m, i| m.* = 30 + 0.3 * @as(f64, @floatFromInt(i)); // 30 ms a row, 34.5 at 16
    var kept: [16]u8 = .{ 0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5 };
    var short: Pick = .{};
    for (0..warm) |_| {
        try std.testing.expectEqual(@as(u32, 15), short.choose(&ms, 300, 15));
        short.learn(15, &kept, 0.0345, 0.039, &ms, 300);
    }
    try std.testing.expectEqual(@as(u32, 15), short.choose(&ms, 300, 15));
    var long: Pick = .{};
    const pos: u64 = 131072;
    const attn = 0.18e-3; // ms per row and position: 23.6 ms a row at 128k
    const verify = (ms[15] + 16 * attn * @as(f64, @floatFromInt(pos))) / 1e3;
    for (0..warm) |_| long.learn(15, &kept, verify, verify + 0.004, &ms, pos);
    try std.testing.expectEqual(@as(u32, 3), long.choose(&ms, pos, 15));
}

test "one empty first round does not shrink the tree, and the biggest size is probed again" {
    var ms: [16]f64 = undefined;
    for (&ms) |*m| m.* = 30;
    const none: [16]u8 = @splat(0);
    var p: Pick = .{};
    p.learn(15, &none, 0.03, 0.035, &ms, 100);
    try std.testing.expectEqual(@as(u32, 15), p.choose(&ms, 100, 15));
    var q: Pick = .{};
    const pos: u64 = 65536;
    var one: [16]u8 = @splat(1);
    one[0] = 0;
    for (0..warm) |_| q.learn(15, &one, 0.2, 0.204, &ms, pos); // rows cost ~10.6 ms each: one node is best
    try std.testing.expectEqual(@as(u32, 1), q.choose(&ms, pos, 15));
    var rounds: u32 = 0;
    while (q.choose(&ms, pos, 15) == 1) : (rounds += 1) q.learn(1, &one, 0.051, 0.055, &ms, pos);
    try std.testing.expect(rounds > 10 and rounds < 200); // probed, not every round and not never
    try std.testing.expectEqual(@as(u32, 3), q.choose(&ms, pos, 3)); // a smaller maximum is the biggest size then
}
