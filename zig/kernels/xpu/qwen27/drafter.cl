// DFlash2 drafter for Qwen3.8-27B on Intel Xe2: 5 layers, a block of L rows attending its context window and itself.

// group means stored × ASC (and B · group / ASC): the low half stays out of fp16 subnormals
#ifndef ASC
#define ASC 256.0f
#endif
#pragma OPENCL FP_CONTRACT OFF

#define GS 64
#define D 5120
#define HD 128
#define NH 32
#define NKV 8
#define KERNEL(N) __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(N, 1, 1))) kernel

static inline float bf2f(ushort h) { return as_float((uint)h << 16); }
static inline ushort f2bf(float f) {
  return intel_convert_bfloat16_as_ushort(f); // hardware conversion (round to nearest even)
}
static inline float rbf(float f) { return bf2f(f2bf(f)); }

// Row sums of a fp16 row held in local memory (K values): one subgroup per group of 64.
static void rowsums(local const half *hl, int K, global half *AHI, global half *ALO, long row) {
  const int lane = get_sub_group_local_id(), nsg = get_num_sub_groups();
  for (int g = get_sub_group_id(); g < K / GS; g += nsg) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[g * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[row * (K / GS) + g] = hi;
      ALO[row * (K / GS) + g] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// Two-tap grouped dynamic causal convolution in the block, y[r] = bf16(x[r]*k0 + x[r-1]*k1); RES adds to the residual.
KERNEL(512) void dconv(global const half *X, global const half *DYN, int DS, global const ushort *BASE, int branch,
                       global float *RES, int has_res, global half *H, global half *AHI, global half *ALO, int L) {
  local half hl[512];
  const int r = get_group_id(0), blk = get_group_id(1), c = blk * 512 + get_local_id(0), g = c / 16;
  const float x = (float)X[(long)r * D + c], prev = r % L > 0 ? (float)X[(long)(r - 1) * D + c] : 0.0f;
  const float d0 = (float)DYN[(long)r * DS + (branch * 2) * 320 + g], d1 = (float)DYN[(long)r * DS + (branch * 2 + 1) * 320 + g];
  const float k0 = rbf(bf2f(BASE[(branch * 2) * D + c]) + d0), k1 = rbf(bf2f(BASE[(branch * 2 + 1) * D + c]) + d1);
  const float y = rbf(x * k0 + prev * k1);
  if (has_res) {
    RES[(long)r * D + c] = rbf(RES[(long)r * D + c] + y);
    return;
  }
  const half o = convert_half_rte(y);
  H[(long)r * D + c] = o;
  hl[get_local_id(0)] = o;
  barrier(CLK_LOCAL_MEM_FENCE);
  const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
  if (sg < 8) {
    float s = 0.0f;
    for (int k = 0; k < 4; k++) s += (float)hl[sg * GS + lane * 4 + k];
    s = sub_group_reduce_add(s) * (1.0f / GS);
    if (lane == 0) {
      const half hi = convert_half_rte(s * ASC);
      AHI[(long)r * (D / GS) + blk * 8 + sg] = hi;
      ALO[(long)r * (D / GS) + blk * 8 + sg] = convert_half_rte(s * ASC - (float)hi);
    }
  }
}

// q, k: per-head RMS norm with weight, then rotate-half RoPE, all in bf16; v is copied; K, V go to the ring caches.
KERNEL(16) void dprep(global const half *P, int PS, int qoff, global const ushort *QN, global const ushort *KN,
                      global const float *INVF, global ushort *Q, global ushort *KC, global ushort *VC, int CAP, int pos0,
                      float eps, global const int4 *ROWS, global const ulong *PTRS) {
  const int r = get_group_id(0), lane = get_sub_group_local_id();
  const int hh = get_group_id(1) + (qoff < 0 ? NH : 0);
  const int kv0 = qoff < 0 ? 0 : NH * HD;            // column of k in P
  int at = pos0 + r;
  if (ROWS) {                                         // several sequences: the row's position and its caches
    const int4 ri = ROWS[r];
    at = ri.x;
    KC = (global ushort *)PTRS[ri.w * 2];
    VC = (global ushort *)PTRS[ri.w * 2 + 1];
  }
  const float pos = (float)at;
  if (hh >= NH + NKV) {                              // v
    const int h = hh - NH - NKV;
    for (int i = 0; i < 8; i++)
      VC[((long)h * CAP + at % CAP) * HD + i * 16 + lane] = f2bf((float)P[(long)r * PS + kv0 + NKV * HD + h * HD + i * 16 + lane]);
    return;
  }
  const bool isq = hh < NH;
  const long src = (long)r * PS + (isq ? hh * HD : kv0 + (hh - NH) * HD);
  float v[8], ss = 0.0f;
  for (int i = 0; i < 8; i++) {
    v[i] = (float)P[src + i * 16 + lane];
    ss += v[i] * v[i];
  }
  const float inv = rsqrt(sub_group_reduce_add(ss) / HD + eps);
  for (int i = 0; i < 8; i++) v[i] = rbf(v[i] * inv * bf2f((isq ? QN : KN)[i * 16 + lane]));
  for (int i = 0; i < 4; i++) {                      // halves: dims 0-63 (v[0..3]) and 64-127 (v[4..7])
    const float ang = pos * INVF[i * 16 + lane], c = cos(ang), s = sin(ang);
    const float a = v[i], b = v[i + 4];
    v[i] = a * c - b * s;
    v[i + 4] = b * c + a * s;
  }
  global ushort *dst = isq ? Q + ((long)r * NH + hh) * HD : KC + ((long)(hh - NH) * CAP + at % CAP) * HD;
  for (int i = 0; i < 8; i++) dst[i * 16 + lane] = f2bf(v[i]);
}

// Block attention (K, V in a ring, CAP > WIN + L): a row sees its context window and every block row (DFlash2), or
// with CAUSAL only the block rows up to itself (DSpine).
KERNEL(16) void dattn(global const ushort *Q, global const ushort *KC, global const ushort *VC, int CAP, int c0,
                      int pos0, int L, int WIN, float scale, global half *O, global const int4 *ROWS,
                      global const ulong *PTRS, int CAUSAL) {
  const int r = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id();
  int p = pos0 + r, bs = pos0;
  if (ROWS) {                                         // several blocks: the row's position, context start, block start
    const int4 ri = ROWS[r];
    p = ri.x;
    c0 = ri.y;
    bs = ri.z;
    KC = (global const ushort *)PTRS[ri.w * 2];
    VC = (global const ushort *)PTRS[ri.w * 2 + 1];
  }
  float q[4][8], o[4][8], m[4], l[4];
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 8; i++) {
      q[a][i] = bf2f(Q[((long)r * NH + h * 4 + a) * HD + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  const int first = max(c0, p - WIN), last = CAUSAL ? p + 1 : bs + L;
  for (int j = first; j < last; j++) {
    float kv[8];
    const long jr = (long)h * CAP + j % CAP;          // the cache is a ring of CAP positions
    for (int i = 0; i < 8; i++) kv[i] = bf2f(KC[jr * HD + i * 16 + lane]);
    float sc[4];
    for (int a = 0; a < 4; a++) {
      float d = 0.0f;
      for (int i = 0; i < 8; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    for (int i = 0; i < 8; i++) kv[i] = bf2f(VC[jr * HD + i * 16 + lane]);
    for (int a = 0; a < 4; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), p = exp(sc[a] - mn);
      l[a] = l[a] * al + p;
      for (int i = 0; i < 8; i++) o[a][i] = o[a][i] * al + p * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 4; a++)
    for (int i = 0; i < 8; i++) O[(long)r * NH * HD + (h * 4 + a) * HD + i * 16 + lane] = convert_half_rte(rbf(o[a][i] / l[a]));
}

// dattn with the keys split over 8 subgroups (XPU_DATTN_SPLIT=1): subgroup s takes keys first + s, first + s + 8, ...
// with its own running max, sum and output for the 4 query heads; the 8 partials merge in local memory. 8 times the
// parallelism of one subgroup a (row, KV head); same reads. Not bit-exact with dattn (another summation order).
#define DSG 8
KERNEL(128) void dattn_split(global const ushort *Q, global const ushort *KC, global const ushort *VC, int CAP, int c0,
                             int pos0, int L, int WIN, float scale, global half *O, global const int4 *ROWS,
                             global const ulong *PTRS, int CAUSAL) {
  local float sm[DSG][4], sl[DSG][4];
  local float so[DSG][4][HD];
  const int r = get_group_id(0), h = get_group_id(1), lane = get_sub_group_local_id(), sg = get_sub_group_id();
  int p = pos0 + r, bs = pos0;
  if (ROWS) {
    const int4 ri = ROWS[r];
    p = ri.x;
    c0 = ri.y;
    bs = ri.z;
    KC = (global const ushort *)PTRS[ri.w * 2];
    VC = (global const ushort *)PTRS[ri.w * 2 + 1];
  }
  float q[4][8], o[4][8], m[4], l[4];
  for (int a = 0; a < 4; a++) {
    for (int i = 0; i < 8; i++) {
      q[a][i] = bf2f(Q[((long)r * NH + h * 4 + a) * HD + i * 16 + lane]);
      o[a][i] = 0.0f;
    }
    m[a] = -INFINITY;
    l[a] = 0.0f;
  }
  const int first = max(c0, p - WIN), last = CAUSAL ? p + 1 : bs + L;
  for (int j = first + sg; j < last; j += DSG) {
    float kv[8];
    const long jr = (long)h * CAP + j % CAP;
    for (int i = 0; i < 8; i++) kv[i] = bf2f(KC[jr * HD + i * 16 + lane]);
    float sc[4];
    for (int a = 0; a < 4; a++) {
      float d = 0.0f;
      for (int i = 0; i < 8; i++) d += q[a][i] * kv[i];
      sc[a] = sub_group_reduce_add(d) * scale;
    }
    for (int i = 0; i < 8; i++) kv[i] = bf2f(VC[jr * HD + i * 16 + lane]);
    for (int a = 0; a < 4; a++) {
      const float mn = fmax(m[a], sc[a]);
      const float al = exp(m[a] - mn), pe = exp(sc[a] - mn);
      l[a] = l[a] * al + pe;
      for (int i = 0; i < 8; i++) o[a][i] = o[a][i] * al + pe * kv[i];
      m[a] = mn;
    }
  }
  for (int a = 0; a < 4; a++) {
    if (lane == 0) {
      sm[sg][a] = m[a];
      sl[sg][a] = l[a];
    }
    for (int i = 0; i < 8; i++) so[sg][a][i * 16 + lane] = o[a][i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const int t = get_local_id(0), a = t / 32, d0 = (t % 32) * 4; // a work-item: one head, 4 of its 128 values
  float M = -INFINITY;
  for (int s = 0; s < DSG; s++) M = fmax(M, sm[s][a]);
  float Ls = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < DSG; s++) {
    const float w = sm[s][a] == -INFINITY ? 0.0f : exp(sm[s][a] - M);
    Ls += sl[s][a] * w;
    for (int e = 0; e < 4; e++) acc[e] += so[s][a][d0 + e] * w;
  }
  for (int e = 0; e < 4; e++) O[(long)r * NH * HD + (h * 4 + a) * HD + d0 + e] = convert_half_rte(rbf(acc[e] / Ls));
}

// Row sums for a fp16 matrix already written (o_proj's input after dattn): grid L, groups of 256.
KERNEL(256) void rowsums_rows(global const half *H, int K, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < K; e += 256) hl[e] = H[(long)r * K + e];
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, K, AHI, ALO, r);
}

// Target taps bf16 [n, 25 600] to fp16 with row sums (fc's input). Grid n, groups of 256.
KERNEL(256) void taps_in(global const ushort *TAPS, global half *H, global half *AHI, global half *ALO) {
  local half hl[25600];
  const int r = get_group_id(0);
  for (int e = get_local_id(0); e < 25600; e += 256) {
    const half v = convert_half_rte(bf2f(TAPS[(long)r * 25600 + e]));
    H[(long)r * 25600 + e] = v;
    hl[e] = v;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  rowsums(hl, 25600, AHI, ALO, r);
}

// fp16 [n, D] to the fp32 residual stream (rounded to bf16): the drafter's x after fc's norm or the embedding.
KERNEL(256) void to_res(global const half *H, global float *X) {
  const long i = get_global_id(0);
  X[i] = rbf((float)H[i]);
}

// Top 16 of each row of logits in two passes: 256 groups per row keep their top 16, the host merges; ties: lower index.
KERNEL(16) void topk_part(global const half *L, int N, int NS, global float *PV, global int *PI) {
  const int gid = get_group_id(0), row = get_group_id(1), lane = get_sub_group_local_id();
  const int per = (N + 255) / 256, a = gid * per, b = min(N, a + per);
  float bv[16];
  int bi[16];
  for (int i = 0; i < 16; i++) { bv[i] = -INFINITY; bi[i] = 0x7fffffff; }
  if (lane == 0)
    for (int j = a; j < b; j++) {
      const float v = (float)L[(long)row * NS + j];
      if (v <= bv[15]) continue;
      int k = 15;
      while (k > 0 && v > bv[k - 1]) { bv[k] = bv[k - 1]; bi[k] = bi[k - 1]; k--; }
      bv[k] = v;
      bi[k] = j;
    }
  if (lane == 0)
    for (int i = 0; i < 16; i++) {
      PV[((long)row * 256 + gid) * 16 + i] = bv[i];
      PI[((long)row * 256 + gid) * 16 + i] = bi[i];
    }
}

// Affine 4-bit quantization of a bf16 weight in groups of 64 along K (MLX layout), one work-item per (row, group).
kernel void quantize4(global const ushort *W, int N, int K, global uint *MW, global ushort *MS, global ushort *MB) {
  const int n = get_global_id(0), g = get_global_id(1);
  if (n >= N || g >= K / GS) return;
  float lo = INFINITY, hi = -INFINITY;
  for (int i = 0; i < GS; i++) {
    const float v = bf2f(W[(long)n * K + g * GS + i]);
    lo = fmin(lo, v);
    hi = fmax(hi, v);
  }
  const float scale = rbf(fmax((hi - lo) / 15.0f, 1e-8f)), bias = rbf(lo);
  MS[(long)n * (K / GS) + g] = f2bf(scale);
  MB[(long)n * (K / GS) + g] = f2bf(bias);
  for (int w = 0; w < GS / 8; w++) {
    uint word = 0;
    for (int j = 0; j < 8; j++) {
      const float v = bf2f(W[(long)n * K + g * GS + w * 8 + j]);
      const int q = clamp((int)rint((v - bias) / scale), 0, 15);
      word |= (uint)q << (4 * j);
    }
    MW[(long)n * (K / 8) + g * (GS / 8) + w] = word;
  }
}

// ---- DSpine (tencent/Qwen3.8-27B-DSpine-b16): DFlash with causal blocks, a code bridge after every layer and the
// serial refinement of the last bridge write over the top-16 candidates. Bridge weights stay bf16 (as the reference
// GGUF keeps them); activations fp32, rounded to bf16 where the bf16 model rounds.
#define CODE 1024
#define BR 512

// x += r (the residual add the DFlash2 path does inside dconv). Grid rows * D / 256.
KERNEL(256) void dsp_add(global float *X, global const half *R) {
  const long i = get_global_id(0);
  X[i] = rbf(X[i] + (float)R[i]);
}

// Each row's RMS norm without weight (the bridge read's input) and its plain RMS (the write's scale). Grid rows.
KERNEL(256) void dsp_norm(global const float *X, float eps, global float *N, global float *RMS) {
  local float red[16];
  const int r = get_group_id(0), lid = get_local_id(0);
  float ss = 0.0f;
  for (int e = lid; e < D; e += 256) {
    const float x = X[(long)r * D + e];
    ss += x * x;
  }
  ss = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += red[i];
  const float ms = t / D, inv = rsqrt(ms + eps);
  if (lid == 0) RMS[r] = sqrt(ms);
  for (int e = lid; e < D; e += 256) N[(long)r * D + e] = rbf(X[(long)r * D + e] * inv);
}

// OUT[r][n] = sum_k A[r][k] W[n][k] for bf16 W [N][K] (N a multiple of 16, K of 256), fp32 sums rounded to bf16.
// A group: 16 columns (a subgroup each) x 16 rows; each tile of 256 K of the 16 rows is read into local memory once
// and shared by the 16 columns (a column per subgroup re-read all rows from the cache: 0.4 TFLOPS). Lanes take 4
// consecutive K (float4 / ushort4) 64 apart: contiguous across the subgroup, no bank conflicts.
// Grid N / 16 x ceil(R / 16).
#define MT 256
KERNEL(256) void dsp_mm(global const float *A, int lda, global const ushort *W, int K, global float *OUT, int ldo, int R) {
  local float4 at[16][MT / 4];
  const int lid = get_local_id(0), lane = get_sub_group_local_id(), n = get_group_id(0) * 16 + get_sub_group_id();
  const int r0 = get_group_id(1) * 16, nr = min(16, R - r0);
  float acc[16];
  for (int i = 0; i < 16; i++) acc[i] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += MT) {
    for (int i = lid; i < 16 * (MT / 4); i += 256) {
      const int r = i / (MT / 4), c = i % (MT / 4);
      at[r][c] = r < nr ? vload4(0, A + (long)(r0 + r) * lda + k0 + c * 4) : (float4)(0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = 0; j < MT / 64; j++) {
      const int c = j * 16 + lane;                      // float4 column of the tile
      const ushort4 wv = vload4(0, W + (long)n * K + k0 + c * 4);
      const float4 w = (float4)(bf2f(wv.s0), bf2f(wv.s1), bf2f(wv.s2), bf2f(wv.s3));
      for (int r = 0; r < 16; r++) {
        const float4 a = at[r][c];
        acc[r] += a.s0 * w.s0 + a.s1 * w.s1 + a.s2 * w.s2 + a.s3 * w.s3;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int r = 0; r < 16; r++) {
    const float s = sub_group_reduce_add(acc[r]);
    if (lane == 0 && r < nr) OUT[(long)(r0 + r) * ldo + n] = rbf(s);
  }
}

// The rows' unit codes x sqrt(CODE): a block's row 0 (the anchor) takes its token's codebook row, the others the
// L2-normalized read of their hidden state. Grid rows.
KERNEL(256) void dsp_codes(global const float *CR, global const int *IDS, int L, global const ushort *CB, float eps,
                           global float *CODES) {
  local float red[16];
  const int r = get_group_id(0), lid = get_local_id(0);
  if (r % L == 0) {
    const long at = (long)IDS[r] * CODE;
    for (int c = lid; c < CODE; c += 256) CODES[(long)r * CODE + c] = rbf(bf2f(CB[at + c]) * 32.0f);
    return;
  }
  float ss = 0.0f;
  for (int c = lid; c < CODE; c += 256) {
    const float v = CR[(long)r * CODE + c];
    ss += v * v;
  }
  ss = sub_group_reduce_add(ss);
  if (get_sub_group_local_id() == 0) red[get_sub_group_id()] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  float t = 0.0f;
  for (int i = 0; i < 16; i++) t += red[i];
  const float sc = 32.0f / fmax(sqrt(t), eps);
  for (int c = lid; c < CODE; c += 256) CODES[(long)r * CODE + c] = rbf(CR[(long)r * CODE + c] * sc);
}

// message(t) = sigmoid(Wg c_t + Wp c_(t-1) + b) * (Wv c_(t-1)), none for a block's row 0. QSV rows: [q | s | v].
// Grid rows * BR / 256.
KERNEL(256) void dsp_msg(global const float *QSV, global const ushort *B, int L, global float *MSG) {
  const long e = get_global_id(0);
  const int r = (int)(e / BR), i = (int)(e % BR);
  if (r % L == 0) {
    MSG[e] = 0.0f;
    return;
  }
  const float z = QSV[(long)r * 3 * BR + i] + QSV[(long)(r - 1) * 3 * BR + BR + i] + bf2f(B[i]);
  const float g = rbf(1.0f / (1.0f + exp(-rbf(z))));
  MSG[e] = rbf(g * QSV[(long)(r - 1) * 3 * BR + 2 * BR + i]);
}

// x += delta * rms (the bridge write); RAW (when given) keeps x before it, for the refinement. Grid rows * D / 256.
KERNEL(256) void dsp_write(global float *X, global const float *DELTA, global const float *RMS, global float *RAW) {
  const long i = get_global_id(0);
  const float x = X[i];
  if (RAW) RAW[i] = x;
  X[i] = rbf(x + rbf(DELTA[i] * RMS[i / D]));
}

// The refinement's predecessor codes: codebook row of each (depth, candidate) predecessor token x sqrt(CODE).
// Grid rows * CODE / 256.
KERNEL(256) void dsp_pred(global const int *PIDS, global const ushort *CB, global float *PRED) {
  const long e = get_global_id(0);
  const int rr = (int)(e / CODE), c = (int)(e % CODE);
  PRED[e] = rbf(bf2f(CB[(long)PIDS[rr] * CODE + c]) * 32.0f);
}

// The last write again with the exact predecessor: row (d, j) uses block row t = d + 1's own gate projection q_t.
// SV rows: [Wp c | Wv c]. Grid rows * BR / 256.
KERNEL(256) void dsp_rmsg(global const float *SV, global const float *QSV, int row0, global const ushort *B,
                          global float *MSG) {
  const long e = get_global_id(0);
  const int rr = (int)(e / BR), i = (int)(e % BR), t = rr / 16 + 1;
  const float z = SV[(long)rr * 2 * BR + i] + QSV[(long)(row0 + t) * 3 * BR + i] + bf2f(B[i]);
  const float g = rbf(1.0f / (1.0f + exp(-rbf(z))));
  MSG[e] = rbf(g * SV[(long)rr * 2 * BR + BR + i]);
}

// The refined pre-norm state of row (d, j): block row t's state before its last write, plus the new write.
// Grid rows * D / 256.
KERNEL(256) void dsp_refine(global const float *RAW, global const float *RMS, int row0, global const float *DELTA,
                            global float *X2) {
  const long e = get_global_id(0);
  const int rr = (int)(e / D), c = (int)(e % D), t = row0 + rr / 16 + 1;
  X2[e] = rbf(RAW[(long)t * D + c] + rbf(DELTA[e] * RMS[t]));
}

// table[d][j][k]: the refined row (d, j)'s logit of depth d's candidate k, straight from the 4-bit head (the GEMM
// layout `repack` writes: word w of column c at W[w NT + c], even K in the low half, odd in the high; S, BG per 64)
// and the row's normed fp16 values and group means (AHI + ALO = mean x ASC, as the GEMM reads them). Only the 16
// columns a row needs instead of the whole head (a 240-row GEMM: 2.4 ms a round). Grid rows; a subgroup a candidate.
KERNEL(256) void dsp_headdot(global const half *H, global const half *AHI, global const half *ALO, global const uint *W,
                             global const half *S, global const half *BG, int NT, global const int *IDX,
                             global float *TAB) {
  const int rr = get_group_id(0), k = get_sub_group_id(), lane = get_sub_group_local_id(), d = rr / 16;
  const int col = IDX[d * 16 + k], G = D / GS;
  float acc = 0.0f;
  for (int w = lane; w < D / 8; w += 16) {
    const int g = w / (GS / 8);
    const uint o = W[(long)w * NT + col];
    const half8 h = vload8(0, H + (long)rr * D + w * 8);
    float dot = 0.0f;
    dot += (float)((o >> 0) & 0xFu) * (float)h.s0 + (float)((o >> 16) & 0xFu) * (float)h.s1;
    dot += (float)((o >> 4) & 0xFu) * (float)h.s2 + (float)((o >> 20) & 0xFu) * (float)h.s3;
    dot += (float)((o >> 8) & 0xFu) * (float)h.s4 + (float)((o >> 24) & 0xFu) * (float)h.s5;
    dot += (float)((o >> 12) & 0xFu) * (float)h.s6 + (float)((o >> 28) & 0xFu) * (float)h.s7;
    acc += (float)S[(long)g * NT + col] * dot;
    if (w % (GS / 8) == 0) acc += (float)BG[(long)g * NT + col] * ((float)AHI[(long)rr * G + g] + (float)ALO[(long)rr * G + g]);
  }
  const float sum = sub_group_reduce_add(acc);
  if (lane == 0) TAB[rr * 16 + k] = sum;
}
