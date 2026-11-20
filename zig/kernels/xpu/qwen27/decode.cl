// Qwen3.8-27B decode kernels on Intel Xe2: serial GDN (a few tokens) and attention over the cache.
#pragma OPENCL FP_CONTRACT OFF
#include "shape.h"

#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}

// Tree table of a verified block (<= 16 rows): TREE[t*17] = depth | SAVE | LOAD flags, then the ancestor row per depth.
#define DEPTH(t) (TREE[(t) * 17] & 255)
#define SAVE(t) (TREE[(t) * 17] & 256)
#define LOAD(t) (TREE[(t) * 17] & 512)
#define ANC(t, d) TREE[(t) * 17 + 1 + (d)]

// Serial GDN over T tokens: one subgroup per (8 v rows, V head), the lane holds 8 k columns; ROWS, TREE, MT: modes.
KERNEL(16) void gdn_step(global const ushort *Q, global const ushort *K, global const ushort *V, global const float *G,
                         global const float *BETA, global const float *S, global float *SO, global ushort *Y, int T,
                         global const int *ROWS, global const int *TREE, global float *SL, global const ulong *MT) {
  // MT (if non-null): several streams, grid (16, Q27_GV, rows); a window's first row starts from its stream's state.
  const int vb = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), kh = h / Q27_GVK;
  int t0 = 0;
  if (MT) {
    t0 = get_group_id(2);
    if ((int)MT[t0 * 8 + 6] != t0) return;            // not a window's first row
    S = (global const float *)MT[t0 * 8 + 4];
    SO = (global float *)MT[t0 * 8 + 5];
    T = t0 + (int)MT[t0 * 8 + 7];
  }
  float s[8][8];
  for (int r = 0; r < 8; r++)
    for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
  for (int t = t0; t < T; t++) {
    const int row = ROWS ? ROWS[t] : t;
    if (TREE && DEPTH(t) > 0 && LOAD(t)) {              // the parent's state, from its slot
      const int par = ANC(t, DEPTH(t) - 1);
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = SL[(((long)par * Q27_GV + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    } else if (TREE && DEPTH(t) == 0)
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) s[r][i] = S[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane];
    float q[8], k[8];
    for (int i = 0; i < 8; i++) {
      q[i] = bf2f(Q[((long)row * Q27_GK + kh) * 128 + i * 16 + lane]);
      k[i] = bf2f(K[((long)row * Q27_GK + kh) * 128 + i * 16 + lane]);
    }
    const float g = G[(long)row * Q27_GV + h], beta = BETA[(long)row * Q27_GV + h];
    for (int r = 0; r < 8; r++) {
      float mem = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] *= g;
        mem += s[r][i] * k[i];
      }
      mem = sub_group_reduce_add(mem);
      const float u = (bf2f(V[((long)row * Q27_GV + h) * 128 + vb * 8 + r]) - mem) * beta;
      float y = 0.0f;
      for (int i = 0; i < 8; i++) {
        s[r][i] += k[i] * u;
        y += s[r][i] * q[i];
      }
      y = sub_group_reduce_add(y);
      if (lane == 0) Y[((long)t * Q27_GV + h) * 128 + vb * 8 + r] = f2bf(y);
    }
    if (TREE && SAVE(t))
      for (int r = 0; r < 8; r++)
        for (int i = 0; i < 8; i++) SL[(((long)t * Q27_GV + h) * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
  }
  if (!TREE)
    for (int r = 0; r < 8; r++)
      for (int i = 0; i < 8; i++) SO[((long)h * 128 + vb * 8 + r) * 128 + i * 16 + lane] = s[r][i];
}

// Decode attention by key slices: one subgroup per (slice, K/V head, query part, row). Grid y is AKV * QPARTS.
#define SPLIT 32
KERNEL(16) void attn_decode(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,
                            global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,
                            global const ulong *MT, global const int *POS, int cap) {
  // MT (if non-null): several streams, one row each; with TREE, tree windows; cap > 0: int8 cache (256 bytes per row).
#if Q27_QPARTS == 1
  const int sp = get_group_id(0), hk = get_group_id(1), t = get_group_id(2), lane = get_sub_group_local_id();
#else
  const int sp = get_group_id(0), hk = get_group_id(1) / Q27_QPARTS, t = get_group_id(2), lane = get_sub_group_local_id();
  const int qh0 = hk * Q27_GQA + (get_group_id(1) % Q27_QPARTS) * Q27_QPG; // the first query head held
#endif
  int base = p0, wf = 0;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    wf = (int)MT[t * 8 + 6];
    base = TREE ? POS[wf] : 0x7fffffff;
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT, j1 = min(j0 + SPLIT, last + 1);
  float q[Q27_QPG][16], o[Q27_QPG][16], m[Q27_QPG], l[Q27_QPG];
  for (int a = 0; a < Q27_QPG; a++) {
    for (int i = 0; i < 16; i++) {
#if Q27_QPARTS == 1
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + hk * Q27_GQA + a) * 256 + i * 16 + lane]);
#else
      q[a][i] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + i * 16 + lane]);
#endif
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  for (int j = j0; j < j1; j++) {
    const long kr = j < base ? j : base + ANC(t, j - base) - wf; // cache row of key j
    float kv[16];
    if (cap) {
      const global char *k8 = (const global char *)KC + (kr * Q27_AKV + hk) * 256;
      const float ks = ((const global float *)((const global char *)KC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)k8[i * 16 + lane] * ks;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(KC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    float sc[Q27_QPG];
    for (int a = 0; a < Q27_QPG; a++) {
      float d = 0.0f;
      for (int i = 0; i < 16; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    if (cap) {
      const global char *v8 = (const global char *)VC + (kr * Q27_AKV + hk) * 256;
      const float vs = ((const global float *)((const global char *)VC + (long)cap * Q27_KVROW))[kr * Q27_AKV + hk];
      for (int i = 0; i < 16; i++) kv[i] = (float)v8[i * 16 + lane] * vs;
    } else
      for (int i = 0; i < 16; i++) kv[i] = bf2f(VC[(kr * Q27_AKV + hk) * 256 + i * 16 + lane]);
    for (int a = 0; a < Q27_QPG; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 16; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < Q27_QPG; a++) {
#if Q27_QPARTS == 1
    const long base = ((long)t * Q27_AQ + hk * Q27_GQA + a) * NS + sp;
#else
    const long base = ((long)t * Q27_AQ + qh0 + a) * NS + sp;
#endif
    if (lane == 0) {
      PM[base] = m[a];
      PL[base] = l[a];
    }
    for (int i = 0; i < 16; i++) PO[base * 256 + i * 16 + lane] = o[a][i];
  }
}

// XPU_ATTN_LK=1 (local option): decode attention with a key per lane. A subgroup per (row, slice of SPLIT_LK keys,
// K/V head, query part) handles LK_QPG query heads, 16 keys a step: each lane's scores are dots along its own key (the query dims
// broadcast from the lanes that hold them), one max and two exps per head a step instead of a reduction and two exps
// per key; then P·V with a lane holding 16 consecutive dims. A row's arithmetic does not depend on the other rows, so
// drafted == serial and several streams == one hold; the bits differ from attn_decode's (another summation order).
#define SPLIT_LK 128
// XPU_ATTN_SPAN (tgt > 0): a row of n = last + 1 keys reads them in slices of SPLIT_LK * 2^k keys, 2^k the largest
// power of two <= n / (SPLIT_LK * tgt) (at least 1, at most 256), so a row has at most 2 tgt slices whatever its depth:
// fewer partials to write and merge. It depends on the row's own keys only (drafted == serial holds); tgt 0: SPLIT_LK.
static inline int lk_span(int last, int tgt) {
  if (tgt <= 0) return SPLIT_LK;
  const int a = (last + 1) / (SPLIT_LK * tgt);
  return a <= 1 ? SPLIT_LK : SPLIT_LK << min(31 - (int)clz(a), 8);
}
// the query heads a subgroup holds (3 of the 27B's 6: two subgroups per K/V head keep q and o in registers)
#if Q27_GQA % 3 == 0
#define LK_QPG 3
#else
#define LK_QPG 4
#endif
#define LK_PARTS (Q27_GQA / LK_QPG)
// The query parts and rows of one (slice, K/V head) are neighbours in the grid (x = row * LK_PARTS + part, the
// fastest dimension), so they run at the same time and the slice's K/V comes from DRAM once (the rest from L2).
#define LK_IDS                                                                                                        \
  const int t = get_group_id(0) / LK_PARTS, sp = get_group_id(1), hk = get_group_id(2),                              \
            lane = get_sub_group_local_id();                                                                          \
  const int qh0 = hk * Q27_GQA + (get_group_id(0) % LK_PARTS) * LK_QPG;
#define ATTN_LK(NAME, I8)                                                                                                   \
KERNEL(16) void NAME(global const ushort *Q, global const ushort *KC, global const ushort *VC, global float *PM,            \
                               global float *PL, global float *PO, int p0, int NS, float scale, global const int *TREE,     \
                               global const ulong *MT, global const int *POS, int cap) {                                    \
  LK_IDS                                                                                                                    \
  int base = p0, wf = 0;                                                                                                    \
  if (MT) {                                                                                                                 \
    KC = (global const ushort *)MT[t * 8];                                                                                  \
    VC = (global const ushort *)MT[t * 8 + 1];                                                                              \
    wf = (int)MT[t * 8 + 6];                                                                                                \
    base = TREE ? POS[wf] : 0x7fffffff;                                                                                     \
  }                                                                                                                         \
  const int last = MT ? POS[t] : p0 + DEPTH(t), j0 = sp * SPLIT_LK, j1 = min(j0 + SPLIT_LK, last + 1);                      \
  const global float *KS = (const global float *)((const global char *)KC + (long)cap * Q27_KVROW);                         \
  const global float *VS = (const global float *)((const global char *)VC + (long)cap * Q27_KVROW);                         \
  float q[LK_QPG][16], o[LK_QPG][16], m[LK_QPG], lp[LK_QPG];                                                            \
  for (int a = 0; a < LK_QPG; a++) {                                                                                       \
    for (int c = 0; c < 16; c++) q[a][c] = bf2f(Q[((long)t * Q27_AQ + qh0 + a) * 256 + c * 16 + lane]); /* dim c*16+lane */ \
    for (int i = 0; i < 16; i++) o[a][i] = 0.0f;                                                        /* dim lane*16+i */ \
    m[a] = -INFINITY;                                                                                                       \
    lp[a] = 0.0f;                                                                                                           \
  }                                                                                                                         \
  for (int b = j0; b < j1; b += 16) {                                                                                       \
    const int j = min(b + lane, j1 - 1); /* this lane's key (the last one again past the slice's end) */                    \
    const long kr = j < base ? j : base + ANC(t, j - base) - wf;                                                            \
    const long row = kr * Q27_AKV + hk;                                                                                     \
    float s[LK_QPG];                                                                                                       \
    for (int a = 0; a < LK_QPG; a++) s[a] = 0.0f;                                                                          \
_Pragma("unroll")                                                                                                           \
    for (int c = 0; c < 16; c++) {                                                                                          \
      float16 kf;                                                                                                           \
      if (I8)                                                                                                               \
        kf = convert_float16(vload16(c, (const global char *)KC + row * 256));                                              \
      else                                                                                                                  \
        kf = as_float16(convert_uint16(vload16(c, KC + row * 256)) << 16);                                                  \
_Pragma("unroll")                                                                                                           \
      for (int e = 0; e < 16; e++)                                                                                          \
_Pragma("unroll")                                                                                                           \
        for (int a = 0; a < LK_QPG; a++) s[a] = fma(sub_group_broadcast(q[a][c], e), kf[e], s[a]);                         \
    }                                                                                                                       \
    const float ks = I8 ? KS[row] * scale : scale, vs = I8 ? VS[row] : 1.0f;                                                \
    const bool ok = b + lane < j1;                                                                                          \
    float pw[LK_QPG];                                                                                                      \
    for (int a = 0; a < LK_QPG; a++) {                                                                                     \
      const float sc = ok ? s[a] * ks : -INFINITY;                                                                          \
      const float mn = fmax(m[a], sub_group_reduce_max(sc));                                                                \
      const float al = exp(m[a] - mn), p = exp(sc - mn);                                                                    \
      lp[a] = fma(lp[a], al, p);                                                                                            \
      for (int i = 0; i < 16; i++) o[a][i] *= al;                                                                           \
      m[a] = mn;                                                                                                            \
      pw[a] = p * vs;                                                                                                       \
    }                                                                                                                       \
    _Pragma("unroll") /* all 16 keys: their V loads issue together; keys past the slice have pw = 0 */                      \
    for (int k = 0; k < 16; k++) {                                                                                          \
      const long rk = sub_group_broadcast(row, k);                                                                          \
      float16 vf;                                                                                                           \
      if (I8)                                                                                                               \
        vf = convert_float16(vload16(lane, (const global char *)VC + rk * 256));                                            \
      else                                                                                                                  \
        vf = as_float16(convert_uint16(vload16(lane, VC + rk * 256)) << 16);                                                \
      for (int a = 0; a < LK_QPG; a++) {                                                                                   \
        const float pk = sub_group_broadcast(pw[a], k);                                                                     \
        for (int i = 0; i < 16; i++) o[a][i] = fma(pk, vf[i], o[a][i]);                                                     \
      }                                                                                                                     \
    }                                                                                                                       \
  }                                                                                                                         \
  for (int a = 0; a < LK_QPG; a++) {                                                                                       \
    const long base = ((long)t * Q27_AQ + qh0 + a) * NS + sp;                                                               \
    const float l = sub_group_reduce_add(lp[a]);                                                                            \
    if (lane == 0) {                                                                                                        \
      PM[base] = m[a];                                                                                                      \
      PL[base] = l;                                                                                                         \
    }                                                                                                                       \
    float16 ov;                                                                                                             \
    for (int i = 0; i < 16; i++) ov[i] = o[a][i];                                                                           \
    vstore16(ov, lane, PO + base * 256);                                                                                    \
  }                                                                                                                         \
}
ATTN_LK(attn_decode_lk8, 1) // int8 cache (XPU_KV_INT8)
ATTN_LK(attn_decode_lk, 0)  // bf16 cache

// Ordered merge of the slices into O bf16 [T, Q27_AQ, 256]: DPL dims a lane, U slices read at once, the same bits at any.
#define MERGE_TGT , int tgt
#define MERGE_KERNEL(NAME, DPL, U, SPL) MERGE_KERNEL_X(NAME, DPL, U, SPL, )
#define MERGE_KERNEL_X(NAME, DPL, U, SPL, EXTRA)                                                                     \
  KERNEL(16) void NAME(global const float *PM, global const float *PL, global const float *PO, global ushort *O,     \
                       int p0, int NS, global const int *TREE, global const int *POS EXTRA) {                         \
    const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id(), i0 = get_group_id(2) * DPL; \
    const int last = POS ? POS[t] : p0 + DEPTH(t), used = last / (SPL) + 1;                                          \
    const long base = ((long)t * Q27_AQ + hh) * NS;                                                                      \
    float m = -INFINITY;                                                                                             \
    for (int s = lane; s < used; s += 16) m = fmax(m, PM[base + s]);                                                 \
    for (int x = 8; x > 0; x >>= 1) m = fmax(m, sub_group_shuffle_xor(m, x));                                        \
    float l = 0.0f, o[DPL];                                                                                          \
    for (int i = 0; i < DPL; i++) o[i] = 0.0f;                                                                       \
    for (int s0 = 0; s0 < used; s0 += U) {                                                                           \
      const int n = min(U, used - s0);                                                                               \
      const float pm = lane < n ? PM[base + s0 + lane] : 0.0f, pl = lane < n ? PL[base + s0 + lane] : 0.0f;          \
      float po[U][DPL];                                                                                              \
      for (int j = 0; j < U; j++)                                                                                    \
        for (int i = 0; i < DPL; i++) po[j][i] = j < n ? PO[(base + s0 + j) * 256 + (i0 + i) * 16 + lane] : 0.0f;    \
      for (int j = 0; j < n; j++) {                                                                                  \
        const float w = exp(sub_group_broadcast(pm, j) - m);                                                         \
        l += sub_group_broadcast(pl, j) * w;                                                                         \
        for (int i = 0; i < DPL; i++) o[i] += po[j][i] * w;                                                          \
      }                                                                                                              \
    }                                                                                                                \
    for (int i = 0; i < DPL; i++) O[((long)t * Q27_AQ + hh) * 256 + (i0 + i) * 16 + lane] = f2bf(o[i] / l);              \
  }
MERGE_KERNEL(attn_merge, 16, 4, SPLIT) // rows of a window: a slice's 1 KB in one read
MERGE_KERNEL_X(attn_merge_lk, 16, 4, lk_span(last, tgt), MERGE_TGT)
// One row: a subgroup per (head, 16 dims), a lane's dim summed over 16 slices read at once (written out: faster code).
KERNEL(16) void attn_merge1(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                            int p0, int NS, global const int *TREE, global const int *POS) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int d = get_group_id(2) * 16 + lane;                   // the dimension this lane sums
  const int used = (POS ? POS[t] : p0 + DEPTH(t)) / SPLIT + 1; // non-empty slices of row t
  const long base = ((long)t * Q27_AQ + hh) * NS;
  float m = -INFINITY;                                         // the max: the same value in any order
  for (int s = lane; s < used; s += 16) m = fmax(m, PM[base + s]);
  for (int x = 8; x > 0; x >>= 1) m = fmax(m, sub_group_shuffle_xor(m, x));
  float l = 0.0f, o = 0.0f;
  for (int s0 = 0; s0 < used; s0 += 16) {
    const int n = min(16, used - s0);
    const float pm = lane < n ? PM[base + s0 + lane] : 0.0f, pl = lane < n ? PL[base + s0 + lane] : 0.0f;
    float po[16];
    for (int j = 0; j < 16; j++) po[j] = j < n ? PO[(base + s0 + j) * 256 + d] : 0.0f;
    for (int j = 0; j < n; j++) { // the sums in slice order
      const float w = exp(sub_group_broadcast(pm, j) - m);
      l += sub_group_broadcast(pl, j) * w;
      o += po[j] * w;
    }
  }
  O[((long)t * Q27_AQ + hh) * 256 + d] = f2bf(o / l);
}
// attn_merge1 over the XPU_ATTN_LK slices
KERNEL(16) void attn_merge1_lk(global const float *PM, global const float *PL, global const float *PO, global ushort *O,
                            int p0, int NS, global const int *TREE, global const int *POS, int tgt) {
  const int hh = get_group_id(0), t = get_group_id(1), lane = get_sub_group_local_id();
  const int d = get_group_id(2) * 16 + lane;                   // the dimension this lane sums
  const int last = POS ? POS[t] : p0 + DEPTH(t), used = last / lk_span(last, tgt) + 1; // non-empty slices of row t
  const long base = ((long)t * Q27_AQ + hh) * NS;
  float m = -INFINITY;                                         // the max: the same value in any order
  for (int s = lane; s < used; s += 16) m = fmax(m, PM[base + s]);
  for (int x = 8; x > 0; x >>= 1) m = fmax(m, sub_group_shuffle_xor(m, x));
  float l = 0.0f, o = 0.0f;
  for (int s0 = 0; s0 < used; s0 += 16) {
    const int n = min(16, used - s0);
    const float pm = lane < n ? PM[base + s0 + lane] : 0.0f, pl = lane < n ? PL[base + s0 + lane] : 0.0f;
    float po[16];
    for (int j = 0; j < 16; j++) po[j] = j < n ? PO[(base + s0 + j) * 256 + d] : 0.0f;
    for (int j = 0; j < n; j++) { // the sums in slice order
      const float w = exp(sub_group_broadcast(pm, j) - m);
      l += sub_group_broadcast(pl, j) * w;
      o += po[j] * w;
    }
  }
  O[((long)t * Q27_AQ + hh) * 256 + d] = f2bf(o / l);
}

// Commit of a kept path: its K, V rows move to positions p0 + depth in depth order, never overwriting an unread row.
KERNEL(16) void kv_move(global ushort *KC, global ushort *VC, int p0, global const int *PATH, int n, int cap) {
  const int hk = get_group_id(0), lane = get_sub_group_local_id();
  for (int d = 0; d < n; d++) {
    const long src = ((long)(p0 + PATH[d]) * Q27_AKV + hk) * 256, dst = ((long)(p0 + d) * Q27_AKV + hk) * 256;
    if (src == dst) continue;
    if (cap) {                                         // int8: 256 bytes (16 per lane) and the scale
      global uint *k = (global uint *)KC, *v = (global uint *)VC;
      for (int i = 0; i < 4; i++) {
        k[dst / 4 + i * 16 + lane] = k[src / 4 + i * 16 + lane];
        v[dst / 4 + i * 16 + lane] = v[src / 4 + i * 16 + lane];
      }
      if (lane == 0) {
        global float *ks = (global float *)((global char *)KC + (long)cap * Q27_KVROW),
                     *vs = (global float *)((global char *)VC + (long)cap * Q27_KVROW);
        ks[dst / 256] = ks[src / 256];
        vs[dst / 256] = vs[src / 256];
      }
      continue;
    }
    for (int i = 0; i < 16; i++) {
      KC[dst + i * 16 + lane] = KC[src + i * 16 + lane];
      VC[dst + i * 16 + lane] = VC[src + i * 16 + lane];
    }
  }
}

// EXL3 probe (mul1 codebook, docs/recipes/exl3.md): X * W_q over K-tile slices, W_q decoded on the fly from TR.
KERNEL(16) void exl3_gemv(global const ushort *TR, global const half *X, global float *Y, int K, int N, int T, int bits,
                          int kslices) {
  // the tile is read once per subgroup and distributed by shuffle, like the activations of its 16 K rows
  const int nt = get_group_id(0), ks = get_group_id(1), lane = get_sub_group_local_id();
  const int per = K / 16 / kslices, kt0 = ks * per, R = 256 * bits, nw = 8 * bits;
  const float c1 = (float)as_half((ushort)0x1EEE), c0 = (float)as_half((ushort)0xC931);
  float acc0[8], acc1[8];
  for (int t = 0; t < 8; t++) acc0[t] = acc1[t] = 0.0f;
  for (int kt = kt0; kt < kt0 + per; kt++) {
    global const uint *tw = (global const uint *)(TR + ((long)kt * (N / 16) + nt) * 16 * bits);
    const uint r0 = tw[lane], r1 = lane + 16 < nw ? tw[lane + 16] : 0u;
    float xr[8];
    for (int t = 0; t < 8; t++) xr[t] = t < T ? (float)X[(long)t * K + kt * 16 + lane] : 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int v = 0; v < 16; v++) {
      const int p = lane * 16 + v;
      int a = (p + 1) * bits - 16;
      a += a < 0 ? R : 0;
      const int wi = a >> 5, wj = wi + 1 == nw ? 0 : wi + 1, sh = a & 31;
      const uint w0a = sub_group_shuffle(r0, wi & 15), w0b = sub_group_shuffle(r1, wi & 15);
      const uint w1a = sub_group_shuffle(r0, wj & 15), w1b = sub_group_shuffle(r1, wj & 15);
      const uint w0 = wi < 16 ? w0a : w0b, w1 = wj < 16 ? w1a : w1b;
      const uint hi = sh ? (w0 << sh) | (w1 >> (32 - sh)) : w0; // the 32 bits starting at bit a
      const uint s = hi >> 16;
      const uint xx = s * 0x83DCD12Du;
      const uint bs = (xx & 0x00FF00FFu) + ((xx >> 8) & 0x00FF00FFu);
      const float w = (float)(1024u + (bs & 0xFFFFu) + (bs >> 16)) * c1 + c0;
      const int l = p >> 3, j = p & 7, row = 2 * (l & 3) + (j & 1) + 8 * ((j >> 1) & 1);
      for (int t = 0; t < 8; t++) {
        if (t >= T) break;
        const float xv = sub_group_shuffle(xr[t], row);
        if (j >> 2) acc1[t] += xv * w; else acc0[t] += xv * w;
      }
    }
  }
  for (int t = 0; t < T && t < 8; t++) {
    const float a0 = acc0[t] + sub_group_shuffle_xor(acc0[t], 1), a1 = acc1[t] + sub_group_shuffle_xor(acc1[t], 1);
    if ((lane & 1) == 0) {
      const long base = ((long)ks * T + t) * N + nt * 16 + lane / 2;
      Y[base] = a0;
      Y[base + 8] = a1;
    }
  }
}

// Prompt attention holds Q27_GQA query heads of one K/V head. GQA 6 is the 27B image. attn_prompt8 stays at GQA 6.
#if Q27_GQA == 6 || Q27_GQA == 8
// Prompt attention (FlashAttention-2 on bf16 DPAS), causal, 256-wide heads; fixed key order, so a row's bits ignore T.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt(global const ushort *Q, global const ushort *KC, global const ushort *VC, global ushort *O,
                        int p0, int T, float scale) {
  local ushort P[Q27_GQA * 8 * 128];                         // A tiles: (tile of 8 queries, 16 keys), r × 16 + lane
#if Q27_GQA == 6
  local float MX[3][2][16], SM[3][2][16], A[48];
#elif Q27_GQA == 8
  local float MX[Q27_GQA / 2][2][16], SM[Q27_GQA / 2][2][16], A[Q27_GQA * 8];
#endif
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < Q27_GQA;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;      // the lane's query: token t0 + tq, head Q27_GQA hk + hq
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const uint *qrow = (global const uint *)(Q + ((long)(t0 + tq) * Q27_AQ + hk * Q27_GQA + hq) * 256);
  const float sl = scale * M_LOG2E_F;
  float8 o[Q27_GQA][2];
  for (int i = 0; i < Q27_GQA; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;                   // the 64 keys of this subgroup
      float8 s[8];
      for (int b = 0; b < 8; b++) s[b] = 0.0f;
      for (int ds = 0; ds < 16; ds++) {
        const int8 qb = qok ? as_int8(vload8(0, qrow + ds * 8)) : (int8)(0);
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 16 dimensions: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(KC + hk * 256), 256 * 2, nkeys, Q27_AKV * 256 * 2,
                                                     (int2)(ds * 16, kb + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            s[h2 * 4 + b] = intel_sub_group_bf16_bf16_matrix_mad_k16(a, qb, s[h2 * 4 + b]);
          }
        }
      }
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float x = kb + b * 8 + r > qpos ? -INFINITY : s[b][r] * sl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + b * 8 + r;      // within the block
          pt[(key / 16) * 128 + key % 16] = f2bf(p);
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < Q27_GQA; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c = 0; c < 8; c++) {
      if (k0 + c * 16 >= nkeys) break;
      short8 pa[Q27_GQA];
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < Q27_GQA; i++) pa[i] = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c) * 128));
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint8 vw;
        intel_sub_group_2d_block_read_transform_16b_16r16x1c((global void *)(VC + hk * 256), 256 * 2, nkeys,
                                                             Q27_AKV * 256 * 2, (int2)((sg * 2 + dt) * 16, k0 + c * 16),
                                                             (private uint *)&vw);
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < Q27_GQA; i++) o[i][dt] = intel_sub_group_bf16_bf16_matrix_mad_k16(pa[i], as_int8(vw), o[i][dt]);
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < Q27_GQA; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * Q27_AQ + hk * Q27_GQA + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}

#endif // Q27_GQA == 6 || Q27_GQA == 8

#if Q27_GQA == 6
// Prompt attention over the int8 cache: same partitioning as attn_prompt, int8 DPAS for S and fp16 for O += P*V.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(128, 1, 1)))
kernel void attn_prompt8(global const ushort *Q, global const char *KC, global const char *VC, global ushort *O,
                         int p0, int T, float scale, int cap) {
  local ushort P[6 * 8 * 128];
  local uint QI[3 * 8 * 8 * 16];                       // (group, 32 dimensions, word) × lane: Q in int8, 4 per word
  local float MX[3][2][16], SM[3][2][16], A[48], QS[48];
  const int t0 = get_group_id(0) * 8, hk = get_group_id(1), sg = get_sub_group_id(), lane = get_sub_group_local_id();
  const int nkeys = p0 + min(t0 + 7, T - 1) + 1;
  const bool qk = sg < 6;
  const int g = sg / 2, kh = sg % 2;
  const int tq = lane % 8, hq = 2 * g + lane / 8;
  const int qpos = p0 + t0 + tq;
  const bool qok = t0 + tq < T;
  global const float *KS = (global const float *)(KC + (long)cap * Q27_KVROW), *VS = (global const float *)(VC + (long)cap * Q27_KVROW);
  if (qk && kh == 0) {                                 // the lane's query, Q in int8
    global const ushort *qrow = Q + ((long)(t0 + tq) * Q27_AQ + hk * 6 + hq) * 256;
    float mq = 0.0f;
    if (qok)
      for (int e = 0; e < 16; e++) {
        const ushort16 x = vload16(e, qrow);
        for (int b = 0; b < 16; b++) mq = fmax(mq, fabs(bf2f(x[b])));
      }
    const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
    for (int e = 0; e < 16; e++) {                     // 16 dimensions: 4 words
      const ushort16 x = qok ? vload16(e, qrow) : (ushort16)(0);
      for (int j = 0; j < 4; j++) {
        uint u = 0;
        for (int b = 0; b < 4; b++) u |= ((uint)(uchar)(char)rint(bf2f(x[j * 4 + b]) * iq)) << (8 * b);
        const int w = e * 4 + j;
        QI[((g * 8 + w / 8) * 8 + w % 8) * 16 + lane] = u;
      }
    }
    QS[hq * 8 + tq] = qs;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float qsl = qk ? QS[hq * 8 + tq] * scale * M_LOG2E_F : 0.0f;
  float8 o[6][2];
  for (int i = 0; i < 6; i++) o[i][0] = o[i][1] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += 128) {
    if (qk) {
      const int kb = k0 + kh * 64;
      int8 si[8];
      for (int b = 0; b < 8; b++) si[b] = 0;
      for (int ds = 0; ds < 8; ds++) {                 // steps of 32 dimensions
        const int8 qb = as_int8(intel_sub_group_block_read8(QI + (g * 8 + ds) * 128));
        __attribute__((opencl_unroll_hint))
        for (int h2 = 0; h2 < 2; h2++) {
          ushort ka[32];                               // 32 keys × 32 bytes: 4 A tiles of 8 keys
          intel_sub_group_2d_block_read_16b_32r16x1c((global void *)(KC + hk * 256), 256, nkeys, Q27_AKV * 256,
                                                     (int2)(ds * 16, kb + h2 * 32), ka);
          __attribute__((opencl_unroll_hint))
          for (int b = 0; b < 4; b++) {
            short8 a;
            for (int r = 0; r < 8; r++) a[r] = (short)ka[b * 8 + r];
            si[h2 * 4 + b] = intel_sub_group_i8_i8_matrix_mad_k32(a, qb, si[h2 * 4 + b]);
          }
        }
      }
      float ksc[4];                                    // scales of the 64 keys, 4 per lane
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        ksc[c] = key < nkeys ? KS[(long)key * Q27_AKV + hk] : 0.0f;
      }
      float8 s[8];
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float ks = sub_group_broadcast(ksc[kk / 16], kk % 16);
          const float x = kb + kk > qpos ? -INFINITY : (float)si[b][r] * ks * qsl;
          s[b][r] = x;
          mx = fmax(mx, x);
        }
      MX[g][kh][lane] = mx;
      barrier(CLK_LOCAL_MEM_FENCE);
      const float mn = fmax(m, fmax(mx, MX[g][kh ^ 1][lane]));
      const float alpha = mn == -INFINITY ? 1.0f : native_exp2(m - mn);
      float vsc[4];
      for (int c = 0; c < 4; c++) {
        const int key = kb + c * 16 + lane;
        vsc[c] = key < nkeys ? VS[(long)key * Q27_AKV + hk] : 0.0f;
      }
      float sum = 0.0f;
      local ushort *pt = P + (hq * 8) * 128 + tq * 16;
      __attribute__((opencl_unroll_hint))
      for (int b = 0; b < 8; b++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) {
          const int kk = b * 8 + r;
          const float p = mn == -INFINITY ? 0.0f : native_exp2(s[b][r] - mn);
          sum += p;
          const int key = kh * 64 + kk;
          pt[(key / 16) * 128 + key % 16] = as_ushort(convert_half(p * sub_group_broadcast(vsc[kk / 16], kk % 16)));
        }
      SM[g][kh][lane] = sum;
      if (kh == 0) A[hq * 8 + tq] = alpha;
      m = mn;
      barrier(CLK_LOCAL_MEM_FENCE);
      l = l * alpha + SM[g][0][lane] + SM[g][1][lane];
    } else {
      barrier(CLK_LOCAL_MEM_FENCE);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 6; i++)
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 8; r++) {
        const float al = A[i * 8 + r];
        o[i][0][r] *= al;
        o[i][1][r] *= al;
      }
    for (int c2 = 0; c2 < 4; c2++) {                   // 32 keys at a time
      if (k0 + c2 * 32 >= nkeys) break;
      __attribute__((opencl_unroll_hint))
      for (int dt = 0; dt < 2; dt++) {
        uint w8[8];                                    // lane: dimension 16 (2 sg + dt) + lane; word j: keys 4j … 4j + 3
        intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)(VC + hk * 256), 256, nkeys, Q27_AKV * 256,
                                                            (int2)((sg * 2 + dt) * 16, k0 + c2 * 32), w8);
        __attribute__((opencl_unroll_hint))
        for (int hc = 0; hc < 2; hc++) {               // 16 keys: 8 fp16 pairs (keys 2j, 2j + 1), int8 → fp16 exact
          uint8 vw;
          __attribute__((opencl_unroll_hint))
          for (int j = 0; j < 4; j++) {
            const half4 h = convert_half4(as_char4(w8[hc * 4 + j]));
            vw[2 * j] = as_uint(h.s01);
            vw[2 * j + 1] = as_uint(h.s23);
          }
          __attribute__((opencl_unroll_hint))
          for (int i = 0; i < 6; i++) {
            const short8 pa = as_short8(intel_sub_group_block_read_us8(P + (i * 8 + c2 * 2 + hc) * 128));
            o[i][dt] = intel_sub_group_f16_f16_matrix_mad_k16(pa, as_int8(vw), o[i][dt]);
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (qk && kh == 0) A[hq * 8 + tq] = 1.0f / l;
  barrier(CLK_LOCAL_MEM_FENCE);
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 6; i++)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 8; r++) {
      if (t0 + r >= T) break;
      const float inv = A[i * 8 + r];
      global ushort *dst = O + ((long)(t0 + r) * Q27_AQ + hk * 6 + i) * 256 + sg * 32 + lane;
      dst[0] = f2bf(o[i][0][r] * inv);
      dst[16] = f2bf(o[i][1][r] * inv);
    }
}

// XPU_ATTN_DP=1 (local option): attn_decode_lk8's role on DPAS, over the int8 cache. The same slices of SPLIT_LK keys,
// partials and merges, any rows (1 ... 16), trees and several streams. A row's 6 query heads are rows 0-5 of a DPAS tile:
// S = Q8 K8 in int8 (exact; Q in int8 with a scale a head, as attn_prompt8, made once a layer by attn_q8), then
// O += P V in fp16 with fp32 sums, P carrying the V scales and 2^pe (pe from the largest V scale of the row's slice:
// fp16 P stays clear of subnormals). Steps of 32 keys (2 tiles of 16, a key a lane). A row's arithmetic depends only on
// its own keys: drafted == serial and several streams == one hold (the bits differ from attn_decode_lk8's).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The queries in int8: a subgroup per (row, K/V head) writes the 8 A tiles of S (Q8T [row][K/V head][tile ds][row r]
// [lane] shorts: head hk * 6 + r, rows 6 and 7 zero; lane l holds dims 32 ds + 2 l and + 1) and QSL [row][head]: the
// head's int8 scale times the attention scale. Grid (rows, Q27_AKV), groups of 16.
KERNEL(16) void attn_q8(global const ushort *Q, global ushort *Q8T, global float *QSL, float scale) {
  const int t = get_group_id(0), hk = get_group_id(1), lane = get_sub_group_local_id();
  const global ushort *qrow = Q + ((long)t * Q27_AQ + hk * 6) * 256;
  global ushort *dst = Q8T + ((long)t * Q27_AKV + hk) * 1024;
  for (int r = 0; r < 8; r++) {
    float x[16], mq = 0.0f;
    __attribute__((opencl_unroll_hint))
    for (int ds = 0; ds < 8; ds++) {
      const ushort2 u = r < 6 ? vload2(0, qrow + r * 256 + ds * 32 + 2 * lane) : (ushort2)(0);
      x[2 * ds] = bf2f(u.s0);
      x[2 * ds + 1] = bf2f(u.s1);
      mq = fmax(mq, fmax(fabs(x[2 * ds]), fabs(x[2 * ds + 1])));
    }
    mq = sub_group_reduce_max(mq);
    const float qs = mq > 0.0f ? mq / 127.0f : 1.0f, iq = 1.0f / qs;
    __attribute__((opencl_unroll_hint))
    for (int ds = 0; ds < 8; ds++) {
      const uint lo = (uchar)(char)rint(x[2 * ds] * iq), hi = (uchar)(char)rint(x[2 * ds + 1] * iq);
      dst[ds * 128 + r * 16 + lane] = (ushort)(lo | hi << 8);
    }
    if (r < 6 && lane == 0) QSL[t * Q27_AQ + hk * 6 + r] = qs * scale;
  }
}

// The row's slice exponent: P = p vs 2^pe < 2^14 with the largest V scale of the row's keys j0 ... j1 - 1
#define DP_PE(T_, J1)                                                                                                  \
  float vmax = 0.0f;                                                                                                   \
  for (int j = j0 + lane; j < (J1); j += 16) {                                                                         \
    const long kr = j < base ? j : base + ANC(T_, j - base) - wf;                                                      \
    vmax = fmax(vmax, VS[kr * Q27_AKV + hk]);                                                                          \
  }                                                                                                                    \
  vmax = sub_group_reduce_max(vmax);                                                                                   \
  const int ve = vmax > 0.0f ? ((as_int(vmax) >> 23) & 255) - 127 : 0;                                                 \
  const int pe = clamp(13 - ve, -100, 100);

// One step of S and the softmax for row T_ (keys b ... b + 31; m, lp, qa, qsl, vps of the row): pa[2], al[6] out.
#define DP_SOFTMAX(T_, J1, KRS)                                                                                        \
  int8 S[2];                                                                                                           \
  float ks[2], vs[2];                                                                                                  \
  bool ok[2];                                                                                                          \
  __attribute__((opencl_unroll_hint))                                                                                  \
  for (int h = 0; h < 2; h++) {                                                                                        \
    const int j = b + h * 16 + lane, jc = min(j, (J1) - 1); /* past the slice: its last key again, p = 0 */            \
    ok[h] = j < (J1);                                                                                                  \
    const long kr = jc < base ? jc : base + ANC(T_, jc - base) - wf;                                                   \
    const long rowk = kr * Q27_AKV + hk;                                                                               \
    KRS                                                                                                                \
    const global uint *kp = (const global uint *)(K8 + rowk * 256);                                                    \
    int8 acc = 0;                                                                                                      \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (int ds = 0; ds < 8; ds++) acc = intel_sub_group_i8_i8_matrix_mad_k32(qa[ds], as_int8(vload8(ds, kp)), acc);   \
    S[h] = acc;                                                                                                        \
    ks[h] = KS[rowk];                                                                                                  \
    vs[h] = VS[rowk] * vps;                                                                                            \
  }                                                                                                                    \
  float sc[2][6];                                                                                                      \
  __attribute__((opencl_unroll_hint))                                                                                  \
  for (int r = 0; r < 6; r++) {                                                                                        \
    __attribute__((opencl_unroll_hint))                                                                                \
    for (int h = 0; h < 2; h++) sc[h][r] = ok[h] ? (float)S[h][r] * qsl[r] * ks[h] : -INFINITY;                        \
    const float mn = fmax(m[r], sub_group_reduce_max(fmax(sc[0][r], sc[1][r])));                                       \
    al[r] = exp(m[r] - mn);                                                                                            \
    m[r] = mn;                                                                                                         \
  }                                                                                                                    \
  __attribute__((opencl_unroll_hint))                                                                                  \
  for (int r = 0; r < 6; r++) {                                                                                        \
    const float q0 = exp(sc[0][r] - m[r]), q1 = exp(sc[1][r] - m[r]);                                                  \
    lp[r] = fma(lp[r], al[r], q0 + q1);                                                                                \
    pa[0][r] = as_short(convert_half_rte(q0 * vs[0]));                                                                 \
    pa[1][r] = as_short(convert_half_rte(q1 * vs[1]));                                                                 \
  }                                                                                                                    \
  pa[0].s6 = pa[0].s7 = pa[1].s6 = pa[1].s7 = 0;

// One row a group (2 subgroups: dims 0-127 and 128-255; both compute S and the softmax, the same bits), or 8 rows a
// group for several caches (rows of different streams). Grid (ceil(rows / rows a group), slices, Q27_AKV).
__attribute__((intel_reqd_sub_group_size(16))) kernel void attn_decode_dp8(
    global const ushort *Q8T, global const float *QSL, global const ushort *KC, global const ushort *VC,
    global float *PM, global float *PL, global float *PO, int p0, int NS, global const int *TREE,
    global const ulong *MT, global const int *POS, int cap, int T, int tgt) {
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), part = sg & 1;
  const int t = get_group_id(0) * (int)(get_num_sub_groups() / 2) + sg / 2, sp = get_group_id(1), hk = get_group_id(2);
  if (t >= T) return;
  int base = p0, wf = 0;
  if (MT) {
    KC = (global const ushort *)MT[t * 8];
    VC = (global const ushort *)MT[t * 8 + 1];
    wf = (int)MT[t * 8 + 6];
    base = TREE ? POS[wf] : 0x7fffffff;
  }
  const int last = MT ? POS[t] : p0 + DEPTH(t), span = lk_span(last, tgt), j0 = sp * span, j1 = min(j0 + span, last + 1);
  if (j0 >= j1) return; // the merges read slices 0 ... last / span only
  const global char *K8 = (const global char *)KC, *V8 = (const global char *)VC;
  const global float *KS = (const global float *)(K8 + (long)cap * Q27_KVROW);
  const global float *VS = (const global float *)(V8 + (long)cap * Q27_KVROW);
  DP_PE(t, j1)
  const float vps = as_float((pe + 127) << 23), pinv = as_float((127 - pe) << 23);
  short8 qa[8];
  float qsl[6], m[6], lp[6];
  __attribute__((opencl_unroll_hint))
  for (int ds = 0; ds < 8; ds++) qa[ds] = as_short8(intel_sub_group_block_read_us8(Q8T + ((long)t * Q27_AKV + hk) * 1024 + ds * 128));
  __attribute__((opencl_unroll_hint))
  for (int r = 0; r < 6; r++) {
    qsl[r] = QSL[t * Q27_AQ + hk * 6 + r];
    m[r] = -INFINITY;
    lp[r] = 0.0f;
  }
  float8 o[8]; // tile e, lane n: dim 128 part + 4 n + e (e < 4), 128 part + 64 + 4 n + e - 4 (e >= 4); row r: head r
  __attribute__((opencl_unroll_hint))
  for (int e = 0; e < 8; e++) o[e] = 0.0f;
  for (int b = j0; b < j1; b += 32) {
    short8 pa[2];
    float al[6];
    DP_SOFTMAX(t, j1, )
    // the rescale only when a head's max moved (al == 1: o * 1 == o, the same bits); al is the same in every lane
    if (al[0] != 1.0f || al[1] != 1.0f || al[2] != 1.0f || al[3] != 1.0f || al[4] != 1.0f || al[5] != 1.0f)
      __attribute__((opencl_unroll_hint))
      for (int e = 0; e < 8; e++)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 6; r++) o[e][r] *= al[r];
    const long rowk0 = (long)(min(b + lane, j1 - 1) < base ? min(b + lane, j1 - 1) : base + ANC(t, min(b + lane, j1 - 1) - base) - wf) * Q27_AKV + hk;
    const long rowk1 = (long)(min(b + 16 + lane, j1 - 1) < base ? min(b + 16 + lane, j1 - 1) : base + ANC(t, min(b + 16 + lane, j1 - 1) - base) - wf) * Q27_AKV + hk;
    __attribute__((opencl_unroll_hint))
    for (int h = 0; h < 2; h++) {
      uint2 vv[16]; // key k of the tile: the part's 128 bytes, lane n bytes 4 n ... (s0) and 64 + 4 n ... (s1)
      __attribute__((opencl_unroll_hint))
      for (int k = 0; k < 16; k++)
        vv[k] = intel_sub_group_block_read2((const global uint *)(V8 + sub_group_broadcast(h ? rowk1 : rowk0, k) * 256 + part * 128));
      __attribute__((opencl_unroll_hint))
      for (int e = 0; e < 8; e++) {
        int8 bw; // B tile: lane = dim, word i = keys 2 i, 2 i + 1 (int8 -> fp16 exact)
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < 8; i++) {
          const uint w0 = e < 4 ? vv[2 * i].s0 : vv[2 * i].s1, w1 = e < 4 ? vv[2 * i + 1].s0 : vv[2 * i + 1].s1;
          const half a0 = convert_half((char)(w0 >> (8 * (e & 3)))), a1 = convert_half((char)(w1 >> (8 * (e & 3))));
          bw[i] = as_int((half2)(a0, a1));
        }
        o[e] = intel_sub_group_f16_f16_matrix_mad_k16(pa[h], bw, o[e]);
      }
    }
  }
  __attribute__((opencl_unroll_hint))
  for (int r = 0; r < 6; r++) {
    const long pb = ((long)t * Q27_AQ + hk * 6 + r) * NS + sp;
    const float l = sub_group_reduce_add(lp[r]);
    if (part == 0 && lane == 0) {
      PM[pb] = m[r];
      PL[pb] = l;
    }
    global float *dst = PO + pb * 256 + part * 128 + 4 * lane;
    vstore4((float4)(o[0][r], o[1][r], o[2][r], o[3][r]) * pinv, 0, dst);
    vstore4((float4)(o[4][r], o[5][r], o[6][r], o[7][r]) * pinv, 0, dst + 64);
  }
}

// int8 -> fp16 of 2 bytes into one word (exact): byte | 0x6400 is 1024 + byte, so x = (x ^ 0x80) + 1024 - 1152
static inline uint dp_pair(uint lo, uint hi) { // lo, hi: the int8 values in their low bytes
  const uint w = ((lo & 0xffu) | (hi & 0xffu) << 16) ^ 0x64806480u;
  return as_uint(as_half2(w) - (half2)(1152.0h, 1152.0h));
}

// attn_decode_dp8 for up to 8 rows of one cache (one stream, or one window of a shared round): the same arithmetic,
// element for element, so the same bits, with a step's work spread over the 16 subgroups of a group: subgroup
// (row, tile) computes S and the softmax of 16 keys of a row (maxima and tile 1's p through local memory), then
// subgroup s adds P V for dims 16 s ... 16 s + 15 of all 8 rows (V read and turned into fp16 once a group).
// A tree's tail keys (b + 31 >= base with TREE) are read row by row. Grid (ceil(rows / 8), slices, Q27_AKV), groups
// of 256.
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1))) kernel void
attn_decode_dpw(global const ushort *Q8T, global const float *QSL, global const ushort *KC, global const ushort *VC,
                global float *PM, global float *PL, global float *PO, int p0, int NS, global const int *TREE,
                global const ulong *MT, global const int *POS, int cap, int T, int tree, int tgt) {
  local float MX[2][8][16];      // a tile's maxima by step parity, row: tile 0 heads 0-7, tile 1 heads 0-7
  local float QV[2][8][6][16];   // tile 1's p, for the row's sums (kept by its tile 0 subgroup)
  local ushort PA[2][8][2][128]; // A tiles of P (row r = head r, lane = key)
  local float AL[2][64];         // the rows' rescale of the step: row i head r at 8 i + r
  local float VP[8], PI[8];      // the row's 2^pe and 2^-pe
  local int KR[2][8][32];        // a tree's key rows (row * Q27_AKV + head), by row and key of the step
  const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), sp = get_group_id(1), hk = get_group_id(2);
  const int r0 = get_group_id(0) * 8, tr = sg >> 1, hq = sg & 1, tq = r0 + tr;
  if (MT) { // one cache for every row (the host's condition)
    KC = (global const ushort *)MT[r0 * 8];
    VC = (global const ushort *)MT[r0 * 8 + 1];
  }
  const int wf = MT ? (int)MT[r0 * 8 + 6] : 0, base = MT ? (TREE ? POS[wf] : 0x7fffffff) : p0;
  int lmax = 0; // the rows share one span (the host's condition): the deepest row's
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 8; i++) lmax = r0 + i < T ? max(lmax, MT ? POS[r0 + i] : p0 + DEPTH(r0 + i)) : lmax;
  const int span = lk_span(lmax, tgt), j0 = sp * span;
  int jend[8], jmax = j0, j1 = j0; // the rows' key ends in this slice (j0: the row reads none of it)
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 8; i++) {
    const int t = r0 + i;
    const int last = t < T ? (MT ? POS[t] : p0 + DEPTH(t)) : -1;
    jend[i] = max(j0, min(j0 + span, last + 1));
    jmax = max(jmax, jend[i]);
    j1 = i == tr ? jend[i] : j1;
  }
  if (jmax == j0) return; // no row reads this slice (the whole group)
  const global char *K8 = (const global char *)KC, *V8 = (const global char *)VC;
  const global float *KS = (const global float *)(K8 + (long)cap * Q27_KVROW);
  const global float *VS = (const global float *)(V8 + (long)cap * Q27_KVROW);
  const bool act = j1 > j0; // this subgroup's row reads the slice
  short8 qa[8];
  float qsl[6], m[6], lp[6];
  if (act) {
    if (hq == 0) {
      DP_PE(tq, j1)
      if (lane == 0) {
        VP[tr] = as_float((pe + 127) << 23);
        PI[tr] = as_float((127 - pe) << 23);
      }
    }
    __attribute__((opencl_unroll_hint))
    for (int ds = 0; ds < 8; ds++) qa[ds] = as_short8(intel_sub_group_block_read_us8(Q8T + ((long)tq * Q27_AKV + hk) * 1024 + ds * 128));
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 6; r++) {
      qsl[r] = QSL[tq * Q27_AQ + hk * 6 + r];
      m[r] = -INFINITY;
      lp[r] = 0.0f;
    }
  }
  float8 o[8]; // row i: lane n is dim 16 sg + n; row r: head r
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 8; i++) o[i] = 0.0f;
  int buf = 0;
  for (int b = j0; b < jmax; b += 32, buf ^= 1) {
    float sc[6], q[6], al[6], vsr = 0.0f;
    const bool uni = !tree || b + 31 < base; // the same V rows for every row
    uint w8[8];                              // lane n: dim 16 sg + n; word i: keys b + 4 i ... b + 4 i + 3
    if (uni)
      intel_sub_group_2d_block_read_transform_8b_32r16x1c((global void *)(V8 + hk * 256), 256, jmax, Q27_KVROW,
                                                          (int2)(sg * 16, b), w8);
    if (act) {
      const int j = b + hq * 16 + lane, jc = min(j, j1 - 1); // past the slice: its last key again, p = 0
      const bool ok = j < j1;
      const long kr = jc < base ? jc : base + ANC(tq, jc - base) - wf;
      const long rowk = kr * Q27_AKV + hk;
      if (tree) KR[buf][tr][hq * 16 + lane] = (int)rowk;
      const global uint *kp = (const global uint *)(K8 + rowk * 256);
      int8 acc = 0;
      __attribute__((opencl_unroll_hint))
      for (int ds = 0; ds < 8; ds++) acc = intel_sub_group_i8_i8_matrix_mad_k32(qa[ds], as_int8(vload8(ds, kp)), acc);
      const float ks = KS[rowk];
      vsr = VS[rowk];
      float mx = -INFINITY;
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 6; r++) {
        sc[r] = ok ? (float)acc[r] * qsl[r] * ks : -INFINITY;
        const float t = sub_group_reduce_max(sc[r]);
        mx = lane == r ? t : mx;
      }
      if (lane < 8) MX[buf][tr][hq * 8 + lane] = mx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act) {
      const float mxv = as_float(intel_sub_group_block_read((local uint *)MX[buf][tr]));
      const float vs = vsr * VP[tr];
      short8 pa;
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 6; r++) {
        const float mn = fmax(m[r], fmax(sub_group_broadcast(mxv, r), sub_group_broadcast(mxv, 8 + r)));
        al[r] = exp(m[r] - mn);
        m[r] = mn;
        q[r] = exp(sc[r] - mn);
        pa[r] = as_short(convert_half_rte(q[r] * vs));
      }
      pa.s6 = pa.s7 = 0;
      intel_sub_group_block_write_us8(&PA[buf][tr][hq][0], as_ushort8(pa));
      if (hq == 1) {
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 6; r++) intel_sub_group_block_write((local uint *)QV[buf][tr][r], as_uint(q[r]));
      } else if (lane < 8) {
        float a = al[0];
        __attribute__((opencl_unroll_hint))
        for (int r = 1; r < 6; r++) a = lane == r ? al[r] : a;
        AL[buf][tr * 8 + lane] = a;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (act && hq == 0) { // the row's sums: tile 1's p from local memory
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 6; r++) lp[r] = fma(lp[r], al[r], q[r] + as_float(intel_sub_group_block_read((local uint *)QV[buf][tr][r])));
    }
    // P V for this subgroup's 16 dims, keys b ... b + 31
    const float4 alv = as_float4(intel_sub_group_block_read4((local uint *)AL[buf])); // index 8 i + r: element / 16, lane % 16
    int8 bw0, bw1;
    if (uni) {
      __attribute__((opencl_unroll_hint))
      for (int i = 0; i < 4; i++) {
        bw0[2 * i] = dp_pair(w8[i], w8[i] >> 8);
        bw0[2 * i + 1] = dp_pair(w8[i] >> 16, w8[i] >> 24);
        bw1[2 * i] = dp_pair(w8[4 + i], w8[4 + i] >> 8);
        bw1[2 * i + 1] = dp_pair(w8[4 + i] >> 16, w8[4 + i] >> 24);
      }
    }
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 8; i++) {
      if (jend[i] <= j0) continue;
      if (!uni) { // a tree's tail: the row's own key rows (from its S subgroups)
        const int kr0 = KR[buf][i][lane], kr1 = KR[buf][i][16 + lane];
        __attribute__((opencl_unroll_hint))
        for (int k = 0; k < 16; k += 2) {
          bw0[k / 2] = dp_pair(V8[(long)sub_group_broadcast(kr0, k) * 256 + sg * 16 + lane],
                               V8[(long)sub_group_broadcast(kr0, k + 1) * 256 + sg * 16 + lane]);
          bw1[k / 2] = dp_pair(V8[(long)sub_group_broadcast(kr1, k) * 256 + sg * 16 + lane],
                               V8[(long)sub_group_broadcast(kr1, k + 1) * 256 + sg * 16 + lane]);
        }
      }
      __attribute__((opencl_unroll_hint))
      for (int r = 0; r < 6; r++) o[i][r] *= sub_group_broadcast(alv[(8 * i + r) / 16], (8 * i + r) % 16);
      o[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(intel_sub_group_block_read_us8(&PA[buf][i][0][0])), bw0, o[i]);
      o[i] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(intel_sub_group_block_read_us8(&PA[buf][i][1][0])), bw1, o[i]);
    }
  }
  if (act && hq == 0)
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 6; r++) {
      const float l = sub_group_reduce_add(lp[r]);
      if (lane == 0) {
        const long pb = ((long)tq * Q27_AQ + hk * 6 + r) * NS + sp;
        PM[pb] = m[r];
        PL[pb] = l;
      }
    }
  __attribute__((opencl_unroll_hint))
  for (int i = 0; i < 8; i++) {
    if (jend[i] <= j0) continue;
    const float pinv = PI[i];
    __attribute__((opencl_unroll_hint))
    for (int r = 0; r < 6; r++)
      PO[(((long)(r0 + i) * Q27_AQ + hk * 6 + r) * NS + sp) * 256 + sg * 16 + lane] = o[i][r] * pinv;
  }
}
#endif // Q27_GQA == 6
