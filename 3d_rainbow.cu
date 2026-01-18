#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "opengl32.lib")
#endif

#include <SDL3/SDL.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static const float PI_F = 3.14159265358979323846f;

static const int   NUM_ATTRACTORS = 2200;
static const float STEP = 6.0f;
static const float KILL_DIST = 6.0f;
static const float INFLUENCE_DIST = 90.0f;
static const float WORLD_RADIUS = 330.0f;
static const int   MAX_BRANCHES = 7000;
static const int   MAX_SEGMENTS = 7000;
static const float DEPTH_FADE_RATE = 0.00035f;
static const bool  RAINBOW_MODE       = true;
static const float RAINBOW_DEPTH_SPIN = 0.9f;  
static const float RAINBOW_SAT        = 95.0f;  
static const float RAINBOW_BRI        = 100.0f; 

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(_err));                \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static inline void hsb2rgb(float h, float s, float br, float& r, float& g, float& b) {
    h = std::fmod(h, 360.0f);
    if (h < 0) h += 360.0f;
    s = s / 100.0f;
    br = br / 100.0f;

    float c = br * s;
    float x = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
    float m = br - c;
    float rp, gp, bp;

    if (h < 60) { rp = c; gp = x; bp = 0; }
    else if (h < 120) { rp = x; gp = c; bp = 0; }
    else if (h < 180) { rp = 0; gp = c; bp = x; }
    else if (h < 240) { rp = 0; gp = x; bp = c; }
    else if (h < 300) { rp = x; gp = 0; bp = c; }
    else { rp = c; gp = 0; bp = x; }

    r = rp + m; g = gp + m; b = bp + m;
}

static inline float hashRange(int i, float lo, float hi) {
    float x = std::sin((float)i * 12.9898f) * 43758.5453f;
    float f = x - std::floor(x);
    return lo + f * (hi - lo);
}

struct Segment {
    float ax, ay, az;
    float bx, by, bz;
    int   depth;
    float hue, sat, bri, glow;
};

__global__ void pullKernel(
    const float* __restrict__ ax, const float* __restrict__ ay, const float* __restrict__ az,
    int nAttr, int* __restrict__ alive,
    const float* __restrict__ bx, const float* __restrict__ by, const float* __restrict__ bz,
    int nBranch,
    float* __restrict__ bdx, float* __restrict__ bdy, float* __restrict__ bdz,
    int* __restrict__ bcount,
    float killD2, float infD2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nAttr) return;

    float axx = ax[i], ayy = ay[i], azz = az[i];

    int   nearest = -1;
    float bestD2 = infD2;

    for (int b = 0; b < nBranch; ++b) {
        float dx = axx - bx[b];
        float dy = ayy - by[b];
        float dz = azz - bz[b];
       float d2 = dx * dx + dy * dy + dz * dz;

        if (d2 < killD2) {
            alive[i] = 0;
            return;
        }
        if (d2 < bestD2) {
            bestD2 = d2;
            nearest = b;
        }
    }

    alive[i] = 1;

    if (nearest >= 0) {
        float dx = axx - bx[nearest];
        float dy = ayy - by[nearest];
        float dz = azz - bz[nearest];
        float m = sqrtf(dx * dx + dy * dy + dz * dz);
        if (m > 0.0f) {
            float inv = 1.0f / m;
            atomicAdd(&bdx[nearest], dx * inv);
            atomicAdd(&bdy[nearest], dy * inv);
            atomicAdd(&bdz[nearest], dz * inv);
            atomicAdd(&bcount[nearest], 1);
        }
    }
}

struct BranchPoint {
    float x, y, z;
    int   depth;
};

static std::vector<float> g_ax, g_ay, g_az;
static std::vector<BranchPoint> g_branches;
static std::vector<bool> g_hasKid;
static std::vector<Segment> g_segments;

static float HUE_MIN = 0, HUE_MAX = 0, SAT_BASE = 0, BRI_BASE = 0;

static std::mt19937 g_rng(std::random_device{}());

static float* d_ax = nullptr, * d_ay = nullptr, * d_az = nullptr;
static int* d_alive = nullptr;
static float* d_bx = nullptr, * d_by = nullptr, * d_bz = nullptr;
static float* d_bdx = nullptr, * d_bdy = nullptr, * d_bdz = nullptr;
static int* d_bcount = nullptr;

static bool g_done = false;

static float uniform(float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(g_rng);
}

static void randUnitVec(float& x, float& y, float& z) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    x = nd(g_rng); y = nd(g_rng); z = nd(g_rng);
    float m = std::sqrt(x * x + y * y + z * z);
    if (m < 1e-6f) { x = 1; y = 0; z = 0; return; }
    x /= m; y /= m; z /= m;
}

static void randPointInSphere(float radius, float& x, float& y, float& z) {
    float dx, dy, dz;
    randUnitVec(dx, dy, dz);
    float u = uniform(0.0f, 1.0f);
    float rr = radius * std::cbrt(u);
    x = dx * rr; y = dy * rr; z = dz * rr;
}

static bool insideWorld(float x, float y, float z) {
    return (x * x + y * y + z * z) <= (WORLD_RADIUS * WORLD_RADIUS);
}

static void pickColours() {
    if (RAINBOW_MODE) {
        HUE_MIN = 0.0f;
        HUE_MAX = 360.0f;
        SAT_BASE = 0.0f;
        BRI_BASE = RAINBOW_BRI;
        return;
    }

    static const float ranges[8][2] = {
        {180, 215}, {200, 235}, {240, 275}, {280, 315},
        {330, 360}, {20, 55},   {70, 110},  {120, 160}
    };
    int idx = std::uniform_int_distribution<int>(0, 7)(g_rng);
    HUE_MIN = ranges[idx][0];
    HUE_MAX = ranges[idx][1];
    SAT_BASE = uniform(78.0f, 100.0f);
    BRI_BASE = uniform(85.0f, 100.0f);
}

static float rainbow_hue(float x, float y, float z, int depth) {
    (void)y; 
    float angleDeg = std::atan2(z, x) * 180.0f / PI_F;
    float hue = angleDeg + 180.0f + depth * RAINBOW_DEPTH_SPIN;
    hue = std::fmod(hue, 360.0f);
    if (hue < 0.0f) hue += 360.0f;
    return hue;
}

static Segment makeSegment(float ax_, float ay_, float az_,
    float bx_, float by_, float bz_, int depth) {
    Segment s;
    s.ax = ax_; s.ay = ay_; s.az = az_;
    s.bx = bx_; s.by = by_; s.bz = bz_;
    s.depth = depth;
    if (RAINBOW_MODE) {
        s.hue = rainbow_hue(bx_, by_, bz_, depth);
        s.sat = clampf(RAINBOW_SAT + uniform(-5.0f, 5.0f), 0.0f, 100.0f);
        s.bri = clampf(RAINBOW_BRI + uniform(-5.0f, 0.0f), 0.0f, 100.0f);
    } else {
        s.hue = uniform(HUE_MIN, HUE_MAX);
        s.sat = clampf(SAT_BASE + uniform(-10.0f, 10.0f), 0.0f, 100.0f);
        s.bri = clampf(BRI_BASE + uniform(-8.0f, 8.0f), 0.0f, 100.0f);
    }
    s.glow = uniform(0.9f, 1.25f);
    return s;
}

static void initSystem() {
    g_ax.clear(); g_ay.clear(); g_az.clear();
    g_branches.clear();
    g_hasKid.clear();
    g_segments.clear();
    g_done = false;

    pickColours();

    g_ax.resize(NUM_ATTRACTORS);
    g_ay.resize(NUM_ATTRACTORS);
    g_az.resize(NUM_ATTRACTORS);
    for (int i = 0; i < NUM_ATTRACTORS; ++i) {
        randPointInSphere(WORLD_RADIUS, g_ax[i], g_ay[i], g_az[i]);
    }

    g_branches.push_back({ 0.0f, 0.0f, 0.0f, 0 });
    g_hasKid.push_back(false);

    Segment seed;
    seed.ax = seed.ay = seed.az = 0.0f;
    seed.bx = seed.by = seed.bz = 0.0f;
    seed.depth = 0;
    seed.hue = RAINBOW_MODE ? rainbow_hue(0.0f, 0.0f, 0.0f, 0) : (HUE_MIN + HUE_MAX) * 0.5f;
    seed.sat = SAT_BASE;
    seed.bri = BRI_BASE;
    seed.glow = 1.0f;
    g_segments.push_back(seed);
}

static void allocGpu() {
    CUDA_CHECK(cudaMalloc(&d_ax, sizeof(float) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_ay, sizeof(float) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_az, sizeof(float) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_alive, sizeof(int) * NUM_ATTRACTORS));
    CUDA_CHECK(cudaMalloc(&d_bx, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_by, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bz, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bdx, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bdy, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bdz, sizeof(float) * MAX_BRANCHES));
    CUDA_CHECK(cudaMalloc(&d_bcount, sizeof(int) * MAX_BRANCHES));
}

static void freeGpu() {
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az); cudaFree(d_alive);
    cudaFree(d_bx); cudaFree(d_by); cudaFree(d_bz);
    cudaFree(d_bdx); cudaFree(d_bdy); cudaFree(d_bdz); cudaFree(d_bcount);
}

static void growOneStep() {
    if (g_done) return;
    if (g_ax.empty() || (int)g_branches.size() >= MAX_BRANCHES) {
        g_done = true;
        return;
    }

    int nAttr = (int)g_ax.size();
    int nBranch = (int)g_branches.size();

    CUDA_CHECK(cudaMemcpy(d_ax, g_ax.data(), sizeof(float) * nAttr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ay, g_ay.data(), sizeof(float) * nAttr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az, g_az.data(), sizeof(float) * nAttr, cudaMemcpyHostToDevice));

    {
        std::vector<float> bx(nBranch), by(nBranch), bz(nBranch);
        for (int i = 0; i < nBranch; ++i) {
            bx[i] = g_branches[i].x; by[i] = g_branches[i].y; bz[i] = g_branches[i].z;
        }
        CUDA_CHECK(cudaMemcpy(d_bx, bx.data(), sizeof(float) * nBranch, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_by, by.data(), sizeof(float) * nBranch, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bz, bz.data(), sizeof(float) * nBranch, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemset(d_bdx, 0, sizeof(float) * nBranch));
    CUDA_CHECK(cudaMemset(d_bdy, 0, sizeof(float) * nBranch));
    CUDA_CHECK(cudaMemset(d_bdz, 0, sizeof(float) * nBranch));
    CUDA_CHECK(cudaMemset(d_bcount, 0, sizeof(int) * nBranch));

    const int threads = 256;
    const int blocks = (nAttr + threads - 1) / threads;

    pullKernel << <blocks, threads >> > (
        d_ax, d_ay, d_az, nAttr, d_alive,
        d_bx, d_by, d_bz, nBranch,
        d_bdx, d_bdy, d_bdz, d_bcount,
        KILL_DIST * KILL_DIST, INFLUENCE_DIST * INFLUENCE_DIST);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int>   alive(nAttr);
    std::vector<float> bdx(nBranch), bdy(nBranch), bdz(nBranch);
    std::vector<int>   bcount(nBranch);

    CUDA_CHECK(cudaMemcpy(alive.data(), d_alive, sizeof(int) * nAttr, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bdx.data(), d_bdx, sizeof(float) * nBranch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bdy.data(), d_bdy, sizeof(float) * nBranch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bdz.data(), d_bdz, sizeof(float) * nBranch, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bcount.data(), d_bcount, sizeof(int) * nBranch, cudaMemcpyDeviceToHost));

    {
        std::vector<float> nax, nay, naz;
        nax.reserve(nAttr); nay.reserve(nAttr); naz.reserve(nAttr);
        for (int i = 0; i < nAttr; ++i) {
            if (alive[i]) {
                nax.push_back(g_ax[i]);
                nay.push_back(g_ay[i]);
                naz.push_back(g_az[i]);
            }
        }
        g_ax = std::move(nax);
        g_ay = std::move(nay);
        g_az = std::move(naz);
    }

    int branchTotal = (int)g_branches.size();
    std::vector<BranchPoint> freshBranches;
    std::vector<Segment> freshSegments;

    for (int b = 0; b < nBranch; ++b) {
        if (bcount[b] <= 0) continue;
        if (branchTotal + (int)freshBranches.size() >= MAX_BRANCHES) break;

        float inv = 1.0f / (float)bcount[b];
        float dx = bdx[b] * inv, dy = bdy[b] * inv, dz = bdz[b] * inv;
        float m = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (m == 0.0f) continue;

        dx = (dx / m) * STEP;
        dy = (dy / m) * STEP;
        dz = (dz / m) * STEP;

        const BranchPoint& parent = g_branches[b];
        float nx = parent.x + dx, ny = parent.y + dy, nz = parent.z + dz;
        if (!insideWorld(nx, ny, nz)) continue;

        g_hasKid[b] = true;

        int newDepth = parent.depth + 1;
        freshBranches.push_back({ nx, ny, nz, newDepth });
        freshSegments.push_back(makeSegment(parent.x, parent.y, parent.z, nx, ny, nz, newDepth));
    }

    if (!freshBranches.empty()) {
        g_branches.insert(g_branches.end(), freshBranches.begin(), freshBranches.end());
        g_segments.insert(g_segments.end(), freshSegments.begin(), freshSegments.end());
        g_hasKid.resize(g_branches.size(), false);
    }

    if ((int)g_segments.size() > MAX_SEGMENTS) {
        g_segments.erase(g_segments.begin(), g_segments.begin() + (g_segments.size() - MAX_SEGMENTS));
    }

    if (g_ax.empty() || (int)g_branches.size() >= MAX_BRANCHES) {
        g_done = true;
    }
}

static void drawWireSphre(float radius, int latSegments, int lonSegments,
    float hue, float sat, float bri, float alpha) {
    float r, g, b;
    hsb2rgb(hue, sat, bri, r, g, b);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(r, g, b, alpha);

    for (int i = 1; i < latSegments; ++i) {
        float lat = PI_F * ((float)i / latSegments - 0.5f);
        float ringY = radius * std::sin(lat);
        float ringRadius = radius * std::cos(lat);
        glBegin(GL_LINE_LOOP);
        for (int j = 0; j < lonSegments; ++j) {
            float lon = 2.0f * PI_F * (float)j / lonSegments;
            glVertex3f(ringRadius * std::cos(lon), ringY, ringRadius * std::sin(lon));
        }
        glEnd();
    }
    for (int j = 0; j < lonSegments; ++j) {
        float lon = 2.0f * PI_F * (float)j / lonSegments;
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= latSegments; ++i) {
            float lat = PI_F * ((float)i / latSegments - 0.5f);
            float ringY = radius * std::sin(lat);
            float ringRadius = radius * std::cos(lat);
            glVertex3f(ringRadius * std::cos(lon), ringY, ringRadius * std::sin(lon));
        }
        glEnd();
    }

    glDisable(GL_BLEND);
}

static void drawSegments() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glLineWidth(3.5f);
    glBegin(GL_LINES);
    for (const Segment& s : g_segments) {
        float depthFade = clampf(1.0f - s.depth * DEPTH_FADE_RATE, 0.2f, 1.0f);
        float alphaOuter = 1.0f * depthFade;
        float r, g, b;
        hsb2rgb(s.hue, s.sat, 100.0f, r, g, b);
        glColor4f(r, g, b, alphaOuter);
        glVertex3f(s.ax, s.ay, s.az);
        glVertex3f(s.bx, s.by, s.bz);
    }
    glEnd();

    glLineWidth(1.4f);
    glBegin(GL_LINES);
    for (const Segment& s : g_segments) {
        float depthFade = clampf(1.0f - s.depth * DEPTH_FADE_RATE, 0.2f, 1.0f);
        float alphaCore = (75.0f / 100.0f) * depthFade;
        float r, g, b;
        hsb2rgb(s.hue, std::min(100.0f, s.sat + 12.0f), 100.0f, r, g, b);
        glColor4f(r, g, b, alphaCore);
        glVertex3f(s.ax, s.ay, s.az);
        glVertex3f(s.bx, s.by, s.bz);
    }
    glEnd();

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const Segment& s : g_segments) {
        float depthFade = clampf(1.0f - s.depth * DEPTH_FADE_RATE, 0.2f, 1.0f);
        float r, g, b;
        hsb2rgb(s.hue, std::min(100.0f, s.sat + 20.0f), 100.0f, r, g, b);
        glColor4f(r, g, b, 0.7f * depthFade);
        glVertex3f(s.bx, s.by, s.bz);
    }
    glEnd();

    glDisable(GL_BLEND);
}

static void drawLeaves() {
    int n = (int)g_branches.size();
    if (n == 0) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    struct GlowLayer { float size; float alphaScale; };
    static const GlowLayer layers[3] = {
        { 12.0f, 0.14f },
        { 6.5f,  0.32f },
        { 3.0f,  0.9f  },
    };

    for (const GlowLayer& layer : layers) {
        glBegin(GL_POINTS);
        for (int i = 0; i < n; ++i) {
            if (g_hasKid[i]) continue;

            const BranchPoint& p = g_branches[i];
            float depthFade = clampf(1.0f - p.depth * DEPTH_FADE_RATE, 0.3f, 1.0f);
            float sizeJitter = hashRange(i * 3 + 1, 0.75f, 1.25f);

            float hue, sat;
            if (RAINBOW_MODE) {
                hue = rainbow_hue(p.x, p.y, p.z, p.depth);
                sat = clampf(RAINBOW_SAT + hashRange(i * 7 + 3, -5.0f, 5.0f), 0.0f, 100.0f);
            } else {
                hue = hashRange(i, HUE_MIN, HUE_MAX);
                sat = clampf(hashRange(i * 7 + 3, SAT_BASE - 20.0f, SAT_BASE + 5.0f), 0.0f, 100.0f);
            }
            float r, g, b;
            hsb2rgb(hue, sat, 100.0f, r, g, b);

            glColor4f(r, g, b, layer.alphaScale * depthFade);
            glPointSize(layer.size * sizeJitter);
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();
    }

    glDisable(GL_BLEND);
}

static void draw_attractors() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    float r, g, b;
    hsb2rgb((HUE_MIN + HUE_MAX) * 0.5f, SAT_BASE * 0.28f, 100.0f, r, g, b);
    glColor4f(r, g, b, 1.0f);
    glPointSize(1.6f);

    int n = (int)g_ax.size();
    int skip = std::max(1, n / 1500);

    glBegin(GL_POINTS);
    for (int i = 0; i < n; i += skip) {
        glVertex3f(g_ax[i], g_ay[i], g_az[i]);
    }
    glEnd();

    glDisable(GL_BLEND);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    int winW = 1280, winH = 800;
    SDL_Window* window = SDL_CreateWindow(
        "Space Colonization 3D - Rainbow (CUDA + SDL3)",
        winW, winH,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GLContext glctx = SDL_GL_CreateContext(window);
    if (!glctx) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetSwapInterval(1);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    allocGpu();
    initSystem();

    bool running = true;
    bool dragging = false;
    float yaw = 0.0f, basePitch = 15.0f;
    float camDist = 900.0f;
    Uint64 lastTicks = SDL_GetTicks();
    double timeSeconds = 0.0;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (ev.key.key == SDLK_ESCAPE) running = false;
                if (ev.key.key == SDLK_R) initSystem();
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (dragging) {
                    yaw += ev.motion.xrel * 0.25f;
                    basePitch += ev.motion.yrel * 0.25f;
                    basePitch = clampf(basePitch, -89.0f, 89.0f);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                camDist -= ev.wheel.y * 30.0f;
                camDist = clampf(camDist, 200.0f, 3000.0f);
                break;
            case SDL_EVENT_WINDOW_RESIZED:
                winW = ev.window.data1;
                winH = ev.window.data2;
                break;
            default:
                break;
            }
        }

        Uint64 now = SDL_GetTicks();
        double dt = (now - lastTicks) / 1000.0;
        lastTicks = now;
        timeSeconds += dt;

        if (!dragging) {
            yaw += (float)(dt * 4.13);
        }
        float pitch = basePitch + std::sin(timeSeconds * 0.054) * 3.44f;

        growOneStep();

        SDL_GetWindowSize(window, &winW, &winH);
        glViewport(0, 0, winW, winH);

        glClearColor(0.03f, 0.03f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = winH > 0 ? (float)winW / (float)winH : 1.0f;
        float nearP = 1.0f, farP = 5000.0f;
        float fovRad = 45.0f * PI_F / 180.0f;
        float top = nearP * std::tan(fovRad * 0.5f);
        float right = top * aspect;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(-right, right, -top, top, nearP, farP);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -camDist);
        glRotatef(pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(yaw, 0.0f, 1.0f, 0.0f);

        drawWireSphre(WORLD_RADIUS, 12, 16,
            (HUE_MIN + HUE_MAX) * 0.5f, SAT_BASE * 0.7f, 70.0f, 0.2f);
        drawSegments();
        drawLeaves();
        draw_attractors();

        SDL_GL_SwapWindow(window);
    }

    freeGpu();
    SDL_GL_DestroyContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
