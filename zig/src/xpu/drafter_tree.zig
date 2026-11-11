//! DFlash2 draft trees: the best-first policy over the drafter's lattice, and one verified tree round.
const std = @import("std");
const dmod = @import("drafter.zig");
const q = @import("qwen27.zig");
const Drafter = dmod.Drafter;
const MAXL = dmod.MAXL;
const TOPK = dmod.TOPK;
const RANK = dmod.RANK;
// the tree policy: edge weight, weight of the target's noise, temperature of the pops, children a node pushes
pub const EDGE = 0.6;
pub const NOISE = 0.7;
pub const SCALE = 1.5;
pub const BRANCH = 4;

/// The tree search's frontier: a few dozen items, popped smallest (score, parent, token, depth) first like heapq.
pub const Heap = struct {
    const Item = struct { score: f64, parent: i32, token: u32, depth: u32 };
    items: [MAXL * BRANCH + BRANCH]Item = undefined,
    n: usize = 0,
    fn push(h: *Heap, it: Item) void {
        if (h.n < h.items.len) {
            h.items[h.n] = it;
            h.n += 1;
        }
    }
    fn before(a: Item, b: Item) bool {
        if (a.score != b.score) return a.score < b.score;
        if (a.parent != b.parent) return a.parent < b.parent;
        if (a.token != b.token) return a.token < b.token;
        return a.depth < b.depth;
    }
    fn pop(h: *Heap) ?Item {
        if (h.n == 0) return null;
        var best: usize = 0;
        for (1..h.n) |i| if (before(h.items[i], h.items[best])) {
            best = i;
        };
        const it = h.items[best];
        h.n -= 1;
        h.items[best] = h.items[h.n];
        return it;
    }
};

/// The chain over the current lattice (propose's choice).
pub fn chain(dr: *const Drafter, pending: u32, count: u32, out: []u32) void {
    var parent = pending;
    for (0..count) |depth| {
        var best: usize = 0;
        var best_score: f64 = -std.math.inf(f64);
        for (0..TOPK) |j| {
            const tok = dr.cand[depth][j];
            const score = scoreOf(dr, parent, depth, j);
            if (score > best_score or (score == best_score and tok < dr.cand[depth][best])) {
                best_score = score;
                best = j;
            }
        }
        out[depth] = dr.cand[depth][best];
        parent = out[depth];
    }
}

/// Candidate j's score at `depth` under `parent`: DFlash2's head logit plus the selector edge; DSpine's refined logit
/// from the table row of the parent's place among the previous depth's candidates (row 0, the anchor's, at depth 0).
pub fn scoreOf(dr: *const Drafter, parent: u32, depth: usize, j: usize) f64 {
    if (dr.kind == .dspine) {
        var jp: usize = 0;
        if (depth > 0) {
            while (jp < TOPK and dr.cand[depth - 1][jp] != parent) jp += 1;
            if (jp == TOPK) return dr.unary[depth][j]; // a parent off the lattice (not drafted by a tree): plain logit
        }
        return dr.tab[depth][jp][j];
    }
    return dr.unary[depth][j] + EDGE * edge(dr, parent, depth, dr.cand[depth][j]);
}

pub fn edge(dr: *const Drafter, parent: u32, depth: usize, tok: u32) f64 {
    var e: f64 = 0;
    for (0..RANK) |rr| e += code(dr.pred, parent, rr) * @as(f64, dr.sp[depth * RANK + rr]) * code(dr.succ, tok, rr);
    return e;
}

/// A best-first tree of ≤ `max_nodes` drafts over the last lattice, `parents[i]` −1 at the root; returns the count.
pub fn tree(dr: *Drafter, pending: u32, max_nodes: u32, noise: ?*const [MAXL][TOPK]f64, temp: f64, tokens: []u32, parents: []i32) u32 {
    var heap: Heap = .{};
    expand(dr, &heap, noise, @max(temp, 1e-6), pending, 0, -1, 0);
    var n: u32 = 0;
    while (n < max_nodes) {
        const it = heap.pop() orelse break;
        tokens[n] = it.token;
        parents[n] = it.parent;
        const me: i32 = @intCast(n);
        n += 1;
        if (it.depth + 1 < dr.depths) expand(dr, &heap, noise, @max(temp, 1e-6), it.token, it.depth + 1, me, it.score);
    }
    preorder(tokens[0..n], parents[0..n], dr.rank[0..n]);
    return n;
}

/// Renumbers a tree depth first, siblings in pop order (`rank`: each node's pop index): fewer serial GDN state saves.
pub fn preorder(tokens: []u32, parents: []i32, rank: []u8) void {
    const n = tokens.len;
    var old_tok: [MAXL]u32 = undefined;
    var old_par: [MAXL]i32 = undefined;
    @memcpy(old_tok[0..n], tokens);
    @memcpy(old_par[0..n], parents);
    var renum: [MAXL]i32 = undefined; // old index -> new index
    var stack: [MAXL]usize = undefined;
    var sp: usize = 0;
    var i = n;
    while (i > 0) { // the roots, the first popped on top
        i -= 1;
        if (old_par[i] < 0) {
            stack[sp] = i;
            sp += 1;
        }
    }
    var k: usize = 0;
    while (sp > 0) {
        sp -= 1;
        const v = stack[sp];
        renum[v] = @intCast(k);
        rank[k] = @intCast(v);
        tokens[k] = old_tok[v];
        parents[k] = if (old_par[v] < 0) -1 else renum[@intCast(old_par[v])];
        k += 1;
        var j = n;
        while (j > v + 1) { // children come after their parent in pop order
            j -= 1;
            if (old_par[j] == @as(i32, @intCast(v))) {
                stack[sp] = j;
                sp += 1;
            }
        }
    }
}

test "preorder puts each first child after its parent and keeps the tree" {
    // pop order: 0 root, 1 root, 2 child of 0, 3 child of 2, 4 child of 1, 5 child of 0
    var tok = [_]u32{ 10, 11, 12, 13, 14, 15 };
    var par = [_]i32{ -1, -1, 0, 2, 1, 0 };
    var rank: [6]u8 = undefined;
    preorder(&tok, &par, &rank);
    try std.testing.expectEqualSlices(u32, &.{ 10, 12, 13, 15, 11, 14 }, &tok);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 0, -1, 4 }, &par);
    try std.testing.expectEqualSlices(u8, &.{ 0, 2, 3, 5, 1, 4 }, &rank);
}

/// Pushes the BRANCH best children of `token` at `depth`, scored by path − log p under the policy.
pub fn expand(dr: *const Drafter, heap: *Heap, noise: ?*const [MAXL][TOPK]f64, temp: f64, token: u32, depth: u32, parent: i32, path: f64) void {
    var v: [TOPK]f64 = undefined;
    for (0..TOPK) |j| {
        v[j] = scoreOf(dr, token, depth, j) / temp;
        if (noise) |nz| v[j] += NOISE * nz[depth][j];
        v[j] /= dr.scale;
    }
    var mx = v[0];
    for (v) |x| mx = @max(mx, x);
    var sum: f64 = 0;
    for (&v) |*x| {
        x.* -= mx;
        sum += @exp(x.*);
    }
    const lse = @log(sum);
    var taken: [TOPK]bool = @splat(false);
    for (0..dr.branch) |_| { // the BRANCH largest, ties to the lower index (numpy's argsort of −logp)
        var j: usize = TOPK;
        for (0..TOPK) |i| if (!taken[i] and (j == TOPK or v[i] > v[j])) {
            j = i;
        };
        taken[j] = true;
        const lp = v[j] - lse;
        if (!std.math.isFinite(lp)) break;
        heap.push(.{ .score = path - lp, .parent = parent, .token = dr.cand[depth][j], .depth = depth });
    }
}

pub fn code(t: []const u8, row: u32, col: usize) f64 {
    const at = (@as(usize, row) * RANK + col) * 2;
    const bits = std.mem.readInt(u16, t[at..][0..2], .little);
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// How a round draws the target's tokens: greedy without one; else row `row` at `position` with keyed Gumbel noise.
pub const Draw = struct {
    ctx: *anyopaque,
    row: *const fn (*anyopaque, row: u32, position: u64) anyerror!u32,
    gumbel: *const fn (*anyopaque, position: u64, id: u64) f64,
    temperature: f64,
};

/// `kept[k]`: the drafts a tree of the first k popped nodes would have kept (k ≤ nodes asked); `verify_s`: its time.
pub const Round = struct { tokens: []const u32, terminal: u32, drafted: u32, accepted: u32, kept: [MAXL]u8, verify_s: f64 };

/// XPU_ROUND_TIMES: seconds per phase (lattice, tree, verify, commit), the queue drained after each phase.
pub var phase_times: ?*[4]f64 = null;

/// One tree round from `pending`: drafts a tree of ≤ `nodes`, verifies it, keeps the path; returns the accepted drafts.
pub fn treeRound(dr: *Drafter, pending: u32, depths: u32, nodes: u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const r = dr.r;
    const p0: u64 = r.pos;
    var t = q.now();
    try dr.lattice(pending, depths);
    if (phase_times) |pt| lap(dr, pt, 0, &t);
    var noise: [MAXL][TOPK]f64 = undefined;
    if (draw) |dw| for (0..depths) |dd| for (0..TOPK) |j| {
        noise[dd][j] = dw.gumbel(dw.ctx, p0 + 1 + dd, dr.cand[dd][j]);
    };
    var ids: [MAXL]u32 = undefined;
    var par: [MAXL]i32 = undefined;
    ids[0] = pending;
    const n = dr.tree(pending, @min(nodes, MAXL - 1), if (draw != null) &noise else null, if (draw) |dw| dw.temperature else 1.0, ids[1..], par[1..]);
    if (phase_times) |pt| lap(dr, pt, 1, &t);
    var rows: [MAXL]i32 = undefined; // the tree in verified rows: row 0 the pending token
    rows[0] = 0;
    for (1..n + 1) |i| rows[i] = if (par[i] < 0) 0 else par[i] + 1;
    const t_verify = q.now();
    const greedy = try r.verifyTree(ids[0 .. n + 1], rows[0 .. n + 1]);
    const verify_s = q.now() - t_verify;
    if (phase_times) |pt| lap(dr, pt, 2, &t);
    var path: [MAXL]u32 = undefined;
    path[0] = 0;
    var len: u32 = 1;
    var terminal = if (draw) |dw| try dw.row(dw.ctx, 0, p0 + 1) else greedy[0];
    walk: while (true) {
        for (1..n + 1) |c| if (rows[c] == path[len - 1] and ids[c] == terminal) {
            path[len] = @intCast(c);
            len += 1;
            terminal = if (draw) |dw| try dw.row(dw.ctx, @intCast(c), p0 + len) else greedy[c];
            continue :walk;
        };
        break;
    }
    try r.commitPath(path[0..len]);
    try dr.addTaps(r.acts.taps, len);
    if (phase_times) |pt| lap(dr, pt, 3, &t);
    for (0..len) |i| out[i] = ids[path[i]];
    var kept: [MAXL]u8 = @splat(0); // a smaller best-first tree is the first nodes popped: its walk stops at a later one
    for (1..@as(usize, @min(nodes, MAXL - 1)) + 1) |k| { // @min narrows to the bound's type: widen before the + 1
        var a: u8 = 0;
        while (a + 1 < len and dr.rank[path[a + 1] - 1] < k) a += 1;
        kept[k] = a;
    }
    return .{ .tokens = out[0..len], .terminal = terminal, .drafted = n, .accepted = len - 1, .kept = kept, .verify_s = verify_s };
}

/// One round on given drafts (a chain after `pending`, e.g. copied from earlier in the context): verified and kept
/// exactly as a tree round's, with no drafter pass; the kept rows' taps still go into the drafter context.
pub fn chainRound(dr: *Drafter, pending: u32, drafts: []const u32, draw: ?Draw, out: *[MAXL]u32) !Round {
    const r = dr.r;
    const p0: u64 = r.pos;
    const n: usize = @min(drafts.len, MAXL - 1); // @min narrows to the bound's type (u4): n + 1 would wrap
    var ids: [MAXL]u32 = undefined;
    var rows: [MAXL]i32 = undefined; // row i's parent row: the row before it
    ids[0] = pending;
    rows[0] = 0;
    for (0..n) |i| {
        ids[i + 1] = drafts[i];
        rows[i + 1] = @intCast(i);
    }
    const t_verify = q.now();
    const greedy = try r.verifyTree(ids[0 .. n + 1], rows[0 .. n + 1]);
    const verify_s = q.now() - t_verify;
    var len: u32 = 1;
    var terminal = if (draw) |dw| try dw.row(dw.ctx, 0, p0 + 1) else greedy[0];
    while (len <= n and ids[len] == terminal) {
        terminal = if (draw) |dw| try dw.row(dw.ctx, len, p0 + len + 1) else greedy[len];
        len += 1;
    }
    var path: [MAXL]u32 = undefined;
    for (0..len) |i| path[i] = @intCast(i);
    try r.commitPath(path[0..len]);
    try dr.addTaps(r.acts.taps, len);
    for (0..len) |i| out[i] = ids[i];
    return .{ .tokens = out[0..len], .terminal = terminal, .drafted = @intCast(n), .accepted = len - 1, .kept = @splat(0), .verify_s = verify_s };
}

fn lap(dr: *Drafter, pt: *[4]f64, phase: usize, t: *f64) void {
    dr.r.d.settle();
    const now = q.now();
    pt[phase] += now - t.*;
    t.* = now;
}
