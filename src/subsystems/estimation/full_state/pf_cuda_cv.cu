// pf_cuda_cv.cu  —  CUDA constant-velocity particle filter for full_state_estimation_sim
//
// Designed to be a drop-in replacement for the CuPy ParticleFilter in pf.py.
//
// State vector (6-D):  [xc, yc, vx, vy, theta, omega]
//
// Motion model: integrated Wiener-process, matching pf.py exactly.
//   For each velocity dim q with noise std Q:
//     d_vel  = sqrt(dt)          * Q * noise_1
//     d_pos  = vel*dt + 0.5*dt*d_vel + sqrt(1/12)*dt^1.5 * Q * noise_0
//   noise_0, noise_1 are independent N(0,1) samples.
//
// Measurement model: back-projection + π/2-periodic yaw, matching pf.py.
//   For each observation (x_obs, y_obs, yaw_obs):
//     cx = x_obs - r*cos(yaw_obs)    (pseudo-centre)
//     cy = y_obs - r*sin(yaw_obs)
//   Particle panels sorted by distance to sensor (0,0), matched rank-for-rank.
//   log_w += -0.5*(dx²+dy²)*R_inv_pos  +  -0.5*yaw_diff²*R_inv_yaw
//   where yaw_diff = (panel_yaw - obs_yaw) mod π/2, wrapped to (-π/4, π/4].
//   Max-shift before exp for numerical stability.
//
// Resampling: systematic resampling (identical to pf.py).
// Stats: host-side single-pass after D2H copy.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      throw std::runtime_error(                                                 \
          std::string("CUDA error at ") + __FILE__ + ":" +                     \
          std::to_string(__LINE__) + " — " + cudaGetErrorString(_e));          \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

// 6-D state.  Layout mirrors Python: [0]=xc [1]=yc [2]=vx [3]=vy [4]=theta [5]=omega
struct State { float xc, yc, vx, vy, theta, omega; };

// One Cartesian panel observation  (world frame)
struct Obs   { float x, y, yaw; };

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr float kHalfPi = 1.5707963267948966f;   // π/2
static constexpr float kQuarPi = 0.7853981633974483f;   // π/4

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Insertion-sort 4 indices by d2[idx]; unchanged for already-sorted input.
__device__ __forceinline__
void sort4_by_dist(float* d2, int* order)
{
    for (int i = 1; i < 4; ++i) {
        int ki = order[i];
        float key = d2[ki];
        int j = i - 1;
        while (j >= 0 && d2[order[j]] > key) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = ki;
    }
}

// Wrap (panel_yaw - obs_yaw) to (−π/4, π/4] using π/2 periodicity.
__device__ __forceinline__
float yaw_residual(float diff)
{
    float r = fmodf(diff, kHalfPi);
    if (r < 0.0f) r += kHalfPi;            // [0, π/2)
    if (r > kQuarPi) r -= kHalfPi;          // (−π/4, π/4]
    return r;
}

// Binary search: smallest i s.t. cdf[i] >= value.
__device__ int find_index(const float* cdf, int n, float value)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < value) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void setup_rng(curandStatePhilox4_32_10_t* rng,
                           unsigned long long seed, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &rng[i]);
}

// Initialise particles as prior + init_std * N(0,1) per component.
__global__ void init_states_kernel(State* states,
                                    curandStatePhilox4_32_10_t* rng,
                                    State prior, State init_std, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t local = rng[i];
    states[i].xc    = prior.xc    + curand_normal(&local) * init_std.xc;
    states[i].yc    = prior.yc    + curand_normal(&local) * init_std.yc;
    states[i].vx    = prior.vx    + curand_normal(&local) * init_std.vx;
    states[i].vy    = prior.vy    + curand_normal(&local) * init_std.vy;
    states[i].theta = prior.theta + curand_normal(&local) * init_std.theta;
    states[i].omega = prior.omega + curand_normal(&local) * init_std.omega;
    rng[i] = local;
}

// Wiener-process constant-velocity motion — matches pf.py _motion_model exactly.
__global__ void motion_update_wiener(State* states,
                                      curandStatePhilox4_32_10_t* rng,
                                      float q_vx, float q_vy, float q_omega,
                                      float dt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t local = rng[i];

    // noise_1 → velocity level:  d_vel = sqrt(dt) * Q * noise_1
    // noise_0 → position level:  d_pos_noise = sqrt(1/12) * dt^1.5 * Q * noise_0
    const float vel_scale = sqrtf(dt);
    const float pos_scale = sqrtf(1.0f / 12.0f) * dt * sqrtf(dt);  // sqrt(1/12)*dt^1.5

    // vx / x
    float n1_vx = curand_normal(&local);
    float n0_vx = curand_normal(&local);
    float d_vx  = vel_scale * q_vx * n1_vx;
    float d_x   = states[i].vx * dt + 0.5f * dt * d_vx + pos_scale * q_vx * n0_vx;

    // vy / y
    float n1_vy = curand_normal(&local);
    float n0_vy = curand_normal(&local);
    float d_vy  = vel_scale * q_vy * n1_vy;
    float d_y   = states[i].vy * dt + 0.5f * dt * d_vy + pos_scale * q_vy * n0_vy;

    // omega / theta
    float n1_om = curand_normal(&local);
    float n0_om = curand_normal(&local);
    float d_om  = vel_scale * q_omega * n1_om;
    float d_th  = states[i].omega * dt + 0.5f * dt * d_om + pos_scale * q_omega * n0_om;

    states[i].xc    += d_x;
    states[i].yc    += d_y;
    states[i].vx    += d_vx;
    states[i].vy    += d_vy;
    states[i].theta += d_th;
    states[i].omega += d_om;

    rng[i] = local;
}

// Back-projection measurement model — matches pf.py _weights exactly.
// Writes LOG-weights (unnormalised) into log_weights[i].
__global__ void measurement_backproj(const State* __restrict__ states,
                                      float* log_weights,
                                      const Obs* __restrict__ obs, int M,
                                      float radius,
                                      float r_inv_pos, float r_inv_yaw,
                                      int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const State& s = states[i];

    // Compute 4 panel angles, positions and distances to sensor (0,0).
    float panel_ang[4], panel_x[4], panel_y[4], dist2[4];
    int   order[4] = {0, 1, 2, 3};
    for (int k = 0; k < 4; ++k) {
        panel_ang[k] = s.theta + k * kHalfPi;
        panel_x[k]   = s.xc + radius * cosf(panel_ang[k]);
        panel_y[k]   = s.yc + radius * sinf(panel_ang[k]);
        dist2[k]     = panel_x[k] * panel_x[k] + panel_y[k] * panel_y[k];
    }
    sort4_by_dist(dist2, order);  // order[0] = nearest panel

    float log_w = 0.0f;
    for (int k = 0; k < M; ++k) {
        // Back-project observation k to pseudo robot-centre
        float cx_obs = obs[k].x - radius * cosf(obs[k].yaw);
        float cy_obs = obs[k].y - radius * sinf(obs[k].yaw);

        // Position term (2-D isotropic Gaussian on pseudo-centre error)
        float dx = s.xc - cx_obs;
        float dy = s.yc - cy_obs;
        log_w += -0.5f * (dx * dx + dy * dy) * r_inv_pos;

        // Yaw term (π/2-periodic, wrapped to (−π/4, π/4])
        float yr = yaw_residual(panel_ang[order[k]] - obs[k].yaw);
        log_w += -0.5f * yr * yr * r_inv_yaw;
    }
    log_weights[i] = log_w;
}

// Numerically-stable conversion: w[i] = exp(log_w[i] - max_log_w)
__global__ void shift_exp_weights(float* weights, float max_log_w, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] = expf(weights[i] - max_log_w);
}

__global__ void normalize_weights(float* weights, float inv_sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    weights[i] *= inv_sum;
}

// Systematic resampling: copies selected parent states into tmp.
__global__ void resample_kernel(const State* __restrict__ states,
                                 State* __restrict__ tmp,
                                 const float* __restrict__ cdf,
                                 float offset, float step, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u   = offset + i * step;
    int   sel = find_index(cdf, n, u);
    tmp[i]    = states[sel];
}

// ---------------------------------------------------------------------------
// Context (global singleton)
// ---------------------------------------------------------------------------

struct PFContext {
    bool initialized = false;
    int  N           = 0;
    int  blockSize   = 256;
    int  blocks      = 0;

    // Noise params
    float q_vx = 0.7f, q_vy = 0.7f, q_omega = 1.2f;
    float r_inv_pos  = 0.0f;    // 1 / r_pos²
    float r_inv_yaw  = 0.0f;    // 1 / r_yaw²
    float radius     = 0.235f;
    float thresh_frac = 1.0f / 3.0f;  // N_eff / N below which we resample

    // Init distribution (kept for reinit)
    State prior_state   = {};
    State init_std_state = {};

    // Device buffers
    State* d_states     = nullptr;
    State* d_states_tmp = nullptr;
    float* d_weights    = nullptr;   // dual-use: log-weights then linear weights
    Obs*   d_obs        = nullptr;   // capacity 4
    curandStatePhilox4_32_10_t* d_rng = nullptr;

    thrust::device_vector<float> cdf;
    std::vector<State> host_states;   // D2H staging buffer

    // Cached outputs (set after every update call)
    float estimate[6]  = {};
    float vx_std       = 0.0f;
    float vy_std       = 0.0f;
    float vel_cov_xy   = 0.0f;
    float speed_std    = 0.0f;
    float omega_std    = 0.0f;
    bool  compute_stats = false;
};

static PFContext g_ctx;

// ---------------------------------------------------------------------------
// Device-side helper: square functor for Thrust transform_reduce
// ---------------------------------------------------------------------------
struct SquareFunctor {
    __host__ __device__ float operator()(float w) const { return w * w; }
};

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Free all GPU/host memory (idempotent).
static void ctx_destroy()
{
    if (!g_ctx.initialized) return;
    cudaFree(g_ctx.d_states);
    cudaFree(g_ctx.d_states_tmp);
    cudaFree(g_ctx.d_weights);
    cudaFree(g_ctx.d_obs);
    cudaFree(g_ctx.d_rng);
    thrust::device_vector<float>().swap(g_ctx.cdf);
    g_ctx.host_states.clear();
    g_ctx.host_states.shrink_to_fit();
    g_ctx.d_states = g_ctx.d_states_tmp = nullptr;
    g_ctx.d_weights = nullptr;
    g_ctx.d_obs = nullptr;
    g_ctx.d_rng = nullptr;
    g_ctx.initialized = false;
}

// D2H copy + compute estimate mean and (optionally) noise stats.
static void compute_estimate_stats()
{
    const int N = g_ctx.N;
    auto& hs = g_ctx.host_states;

    CUDA_CHECK(cudaMemcpy(hs.data(), g_ctx.d_states,
                          N * sizeof(State), cudaMemcpyDeviceToHost));

    // Single-pass accumulate sums and sum-of-squares for all needed quantities.
    double sx = 0, sy = 0, svx = 0, svy = 0, sth = 0, som = 0;
    double svx2 = 0, svy2 = 0, som2 = 0, svxvy = 0, ssp = 0, ssp2 = 0;

    for (int i = 0; i < N; ++i) {
        const State& s = hs[i];
        sx  += s.xc;   sy  += s.yc;
        svx += s.vx;   svy += s.vy;
        sth += s.theta; som += s.omega;
        if (g_ctx.compute_stats) {
            double sp = std::sqrt(s.vx * s.vx + s.vy * s.vy);
            svx2  += s.vx * s.vx;
            svy2  += s.vy * s.vy;
            som2  += s.omega * s.omega;
            svxvy += s.vx * s.vy;
            ssp   += sp;
            ssp2  += sp * sp;
        }
    }

    double inv_N = 1.0 / N;
    g_ctx.estimate[0] = (float)(sx  * inv_N);
    g_ctx.estimate[1] = (float)(sy  * inv_N);
    g_ctx.estimate[2] = (float)(svx * inv_N);
    g_ctx.estimate[3] = (float)(svy * inv_N);
    g_ctx.estimate[4] = (float)(sth * inv_N);
    g_ctx.estimate[5] = (float)(som * inv_N);

    if (g_ctx.compute_stats) {
        double mvx = g_ctx.estimate[2];
        double mvy = g_ctx.estimate[3];
        double mom = g_ctx.estimate[5];
        double msp = ssp * inv_N;
        // Population std (matches CuPy _weighted_std with uniform weights)
        g_ctx.vx_std     = (float)std::sqrt(std::max(svx2 * inv_N - mvx * mvx, 0.0));
        g_ctx.vy_std     = (float)std::sqrt(std::max(svy2 * inv_N - mvy * mvy, 0.0));
        g_ctx.omega_std  = (float)std::sqrt(std::max(som2 * inv_N - mom * mom, 0.0));
        g_ctx.vel_cov_xy = (float)(svxvy * inv_N - mvx * mvy);
        g_ctx.speed_std  = (float)std::sqrt(std::max(ssp2 * inv_N - msp * msp, 0.0));
    }
}

// Normalize weights, compute N_eff/N, optionally resample.
// Expects d_weights to hold LINEAR (not log) weights that sum > 0.
static float resample_step()
{
    const int N = g_ctx.N;
    thrust::device_ptr<float> wp(g_ctx.d_weights);

    // Normalize
    float wsum = thrust::reduce(wp, wp + N, 0.0f, thrust::plus<float>());
    if (wsum <= 0.0f || !std::isfinite(wsum)) {
        thrust::fill(wp, wp + N, 1.0f / N);
        wsum = 1.0f;
    } else {
        normalize_weights<<<g_ctx.blocks, g_ctx.blockSize>>>(
            g_ctx.d_weights, 1.0f / wsum, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // N_eff = 1 / sum(w²)
    float sum_w2 = thrust::transform_reduce(
        wp, wp + N, SquareFunctor{}, 0.0f, thrust::plus<float>());
    float n_eff = (sum_w2 > 0.0f) ? 1.0f / sum_w2 : (float)N;
    float confidence = n_eff / N;

    // Resample if N_eff < thresh
    if (n_eff < g_ctx.thresh_frac * N) {
        // Build CDF
        thrust::inclusive_scan(wp, wp + N, g_ctx.cdf.begin());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Systematic resample
        float step   = 1.0f / N;
        float offset = ((float)rand() / (float)RAND_MAX) * step;
        resample_kernel<<<g_ctx.blocks, g_ctx.blockSize>>>(
            g_ctx.d_states, g_ctx.d_states_tmp,
            thrust::raw_pointer_cast(g_ctx.cdf.data()),
            offset, step, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(g_ctx.d_states, g_ctx.d_states_tmp);

        // Reset weights to uniform after resampling
        thrust::fill(wp, wp + N, 1.0f / N);
    }

    return confidence;
}

// ---------------------------------------------------------------------------
// Public C++ API (called from pybind11 lambdas)
// ---------------------------------------------------------------------------

static void pf_init(int N, float q_vx, float q_vy, float q_omega,
                    float r_pos, float r_yaw, float radius,
                    const std::vector<float>& prior,
                    const std::vector<float>& init_std,
                    bool compute_stats)
{
    ctx_destroy();   // idempotent — frees any existing allocation

    g_ctx.N          = N;
    g_ctx.blockSize  = 256;
    g_ctx.blocks     = (N + 255) / 256;
    g_ctx.q_vx       = q_vx;
    g_ctx.q_vy       = q_vy;
    g_ctx.q_omega    = q_omega;
    g_ctx.r_inv_pos  = 1.0f / (r_pos * r_pos);
    g_ctx.r_inv_yaw  = 1.0f / (r_yaw * r_yaw);
    g_ctx.radius     = radius;
    g_ctx.compute_stats = compute_stats;
    g_ctx.thresh_frac   = 1.0f / 3.0f;

    // Build prior / init_std State structs (pad with zeros if < 6 elements)
    auto to_state = [](const std::vector<float>& v) -> State {
        auto g = [&](int i) { return i < (int)v.size() ? v[i] : 0.0f; };
        return {g(0), g(1), g(2), g(3), g(4), g(5)};
    };
    g_ctx.prior_state    = to_state(prior);
    g_ctx.init_std_state = to_state(init_std);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&g_ctx.d_states,     N * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_states_tmp, N * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_weights,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_obs,        4 * sizeof(Obs)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_rng,        N * sizeof(curandStatePhilox4_32_10_t)));

    g_ctx.cdf.resize(N);
    g_ctx.host_states.resize(N);

    // Seed RNG and initialise particles
    setup_rng<<<g_ctx.blocks, g_ctx.blockSize>>>(g_ctx.d_rng, 1234ULL, N);
    CUDA_CHECK(cudaGetLastError());
    init_states_kernel<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_states, g_ctx.d_rng,
        g_ctx.prior_state, g_ctx.init_std_state, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Uniform initial weights
    thrust::device_ptr<float> wp(g_ctx.d_weights);
    thrust::fill(wp, wp + N, 1.0f / N);

    g_ctx.initialized = true;
}

static void pf_reinit(const std::vector<float>& prior)
{
    if (!g_ctx.initialized)
        throw std::runtime_error("pf_cuda_cv: not initialized");

    auto to_state = [](const std::vector<float>& v) -> State {
        auto g = [&](int i) { return i < (int)v.size() ? v[i] : 0.0f; };
        return {g(0), g(1), g(2), g(3), g(4), g(5)};
    };
    State new_prior = to_state(prior);

    init_states_kernel<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_states, g_ctx.d_rng,
        new_prior, g_ctx.init_std_state, g_ctx.N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<float> wp(g_ctx.d_weights);
    thrust::fill(wp, wp + g_ctx.N, 1.0f / g_ctx.N);
}

// update: motion + measurement + resample; returns (estimate[6], confidence)
static std::pair<std::vector<float>, float>
pf_update(float dt, py::array_t<float, py::array::c_style | py::array::forcecast> obs_arr)
{
    if (!g_ctx.initialized)
        throw std::runtime_error("pf_cuda_cv: not initialized");

    auto buf = obs_arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::invalid_argument("observations must be shape (M, 3)");
    int M = std::min((int)buf.shape[0], 4);

    // Copy observations to device
    std::vector<Obs> h_obs(M);
    float* ptr = static_cast<float*>(buf.ptr);
    for (int k = 0; k < M; ++k)
        h_obs[k] = {ptr[3*k], ptr[3*k+1], ptr[3*k+2]};
    CUDA_CHECK(cudaMemcpy(g_ctx.d_obs, h_obs.data(), M * sizeof(Obs),
                          cudaMemcpyHostToDevice));

    // Motion
    motion_update_wiener<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_states, g_ctx.d_rng,
        g_ctx.q_vx, g_ctx.q_vy, g_ctx.q_omega, dt, g_ctx.N);
    CUDA_CHECK(cudaGetLastError());

    // Measurement → log-weights
    measurement_backproj<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_states, g_ctx.d_weights,
        g_ctx.d_obs, M,
        g_ctx.radius, g_ctx.r_inv_pos, g_ctx.r_inv_yaw, g_ctx.N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Max-shift → exp  (numerical stability, matches CuPy log_w -= max(log_w))
    thrust::device_ptr<float> wp(g_ctx.d_weights);
    float max_log_w = thrust::reduce(
        wp, wp + g_ctx.N, -FLT_MAX, thrust::maximum<float>());
    shift_exp_weights<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_weights, max_log_w, g_ctx.N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float confidence = resample_step();
    compute_estimate_stats();

    return {std::vector<float>(g_ctx.estimate, g_ctx.estimate + 6), confidence};
}

// update_no_obs: motion only, reuse existing weights.
static std::pair<std::vector<float>, float>
pf_update_no_obs(float dt)
{
    if (!g_ctx.initialized)
        throw std::runtime_error("pf_cuda_cv: not initialized");

    motion_update_wiener<<<g_ctx.blocks, g_ctx.blockSize>>>(
        g_ctx.d_states, g_ctx.d_rng,
        g_ctx.q_vx, g_ctx.q_vy, g_ctx.q_omega, dt, g_ctx.N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float confidence = resample_step();
    compute_estimate_stats();

    return {std::vector<float>(g_ctx.estimate, g_ctx.estimate + 6), confidence};
}

// Constant-velocity extrapolation (matches pf.py prediction()).
static std::vector<float> pf_prediction(float dt)
{
    if (!g_ctx.initialized)
        throw std::runtime_error("pf_cuda_cv: not initialized");
    std::vector<float> pred(g_ctx.estimate, g_ctx.estimate + 6);
    pred[0] += pred[2] * dt;
    pred[1] += pred[3] * dt;
    pred[4] += pred[5] * dt;
    return pred;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(pf_cuda_cv, m)
{
    m.doc() = "CUDA constant-velocity particle filter for full_state_estimation_sim (Wiener-process model)";

    m.def("init",
        [](int n_particles,
           float q_vx, float q_vy, float q_omega,
           float r_pos, float r_yaw, float radius,
           std::vector<float> prior,
           std::vector<float> init_std,
           bool compute_stats)
        {
            pf_init(n_particles, q_vx, q_vy, q_omega,
                    r_pos, r_yaw, radius, prior, init_std, compute_stats);
        },
        py::arg("n_particles"),
        py::arg("q_vx"),   py::arg("q_vy"),   py::arg("q_omega"),
        py::arg("r_pos"),  py::arg("r_yaw"),  py::arg("radius"),
        py::arg("prior"),  py::arg("init_std"),
        py::arg("compute_stats") = false,
        "Initialise (or re-initialise) the particle filter.\n"
        "prior / init_std are lists of 6 floats: [xc, yc, vx, vy, theta, omega].");

    m.def("update",
        [](float dt, py::array_t<float, py::array::c_style | py::array::forcecast> obs)
            -> py::tuple
        {
            auto [est, conf] = pf_update(dt, obs);
            return py::make_tuple(est, conf);
        },
        py::arg("dt"), py::arg("observations"),
        "Run motion + measurement + resample.\n"
        "observations: float32 ndarray shape (M, 3) — [[x,y,yaw], ...], M ≤ 4.\n"
        "Returns (estimate[6], confidence) where confidence = N_eff/N.");

    m.def("update_no_obs",
        [](float dt) -> py::tuple
        {
            auto [est, conf] = pf_update_no_obs(dt);
            return py::make_tuple(est, conf);
        },
        py::arg("dt"),
        "Run motion only (no observation this frame).\n"
        "Returns (estimate[6], confidence).");

    m.def("reinit",
        [](std::vector<float> prior) { pf_reinit(prior); },
        py::arg("prior"),
        "Re-seed all particles around prior using the init_std from init().");

    m.def("prediction",
        [](float dt) { return pf_prediction(dt); },
        py::arg("dt"),
        "Constant-velocity extrapolation of the current estimate by dt seconds.\n"
        "Returns estimate[6] (no side-effects).");

    m.def("noise_stats",
        []() -> py::dict
        {
            if (!g_ctx.initialized)
                throw std::runtime_error("pf_cuda_cv: not initialized");
            py::dict d;
            d["vx_std"]     = g_ctx.vx_std;
            d["vy_std"]     = g_ctx.vy_std;
            d["vel_cov_xy"] = g_ctx.vel_cov_xy;
            d["speed_std"]  = g_ctx.speed_std;
            d["omega_std"]  = g_ctx.omega_std;
            return d;
        },
        "Return the last-computed noise statistics dict.\n"
        "Keys: vx_std, vy_std, vel_cov_xy, speed_std, omega_std.\n"
        "Values are 0.0 if compute_stats=False was passed to init().");

    m.def("destroy",
        []() { ctx_destroy(); },
        "Free all GPU memory. Safe to call multiple times.");
}
