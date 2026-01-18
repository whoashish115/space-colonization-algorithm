
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


static const int   NUM_ATTRACTORS = 2600;
static const float STEP = 6.0f;
static const float KILL_DIST = 6.0f;
static const float INFLUENCE_DIST = 100.0f;
static const int   MAX_BRANCHES = 7000;
static const int   MAX_SEGMENTS = 7000;

static const float GROUND_Y = -250.0f;
static const float WORLD_HALF_X = 520.0f;
static const float WORLD_HALF_Z = 300.0f;
static const float WORLD_MIN_Y = GROUND_Y - 20.0f;
static const float WORLD_MAX_Y = GROUND_Y + 820.0f;

static const int   TRUNK_STEPS = 22;
static const float TRUNK_STEP_HEIGHT = 7.0f;

static const float TRUNK_BASE_RADIUS = 11.0f;
static const float TRUNK_MIN_RADIUS = 0.5f;
static const float TRUNK_TAPER_DEPTH = 90.0f;
static const int   CYLINDER_SIDES = 7;

static const float BARK_HUE_MIN = 10.0f;
static const float BARK_HUE_MAX = 26.0f;

static const float LEAF_HUE_MIN = 330.0f;
static const float LEAF_HUE_MAX = 368.0f;
static const float LEAF_SAT_MIN = 70.0f;
static const float LEAF_SAT_MAX = 95.0f;
static const float LEAF_BRI_MIN = 65.0f;
static const float LEAF_BRI_MAX = 96.0f;

static const int   LEAVES_PER_TIP = 36;
static const float BUSH_RADIUS_MIN = 3.0f;
static const float BUSH_RADIUS_MAX = 24.0f;


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
    float hue, sat, bri;
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

static float SAT_BASE = 0, BRI_BASE = 0;

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

static void randAttractorPt(float& x, float& y, float& z) {
    x = uniform(-WORLD_HALF_X, WORLD_HALF_X);
    z = uniform(-WORLD_HALF_Z, WORLD_HALF_Z);
    y = uniform(GROUND_Y + 160.0f, GROUND_Y + 760.0f);
}

static bool insideWorld(float x, float y, float z) {
    return std::fabs(x) <= WORLD_HALF_X &&
        y >= WORLD_MIN_Y && y <= WORLD_MAX_Y &&
        std::fabs(z) <= WORLD_HALF_Z;
}

static void pickColours() {
    SAT_BASE = uniform(45.0f, 65.0f);
    BRI_BASE = uniform(28.0f, 42.0f);
}

static Segment makeSegment(float ax_, float ay_, float az_,
    float bx_, float by_, float bz_, int depth) {
    Segment s;
    s.ax = ax_; s.ay = ay_; s.az = az_;
    s.bx = bx_; s.by = by_; s.bz = bz_;
    s.depth = depth;
    s.hue = uniform(BARK_HUE_MIN, BARK_HUE_MAX);
    s.sat = clampf(SAT_BASE + uniform(-8.0f, 8.0f), 0.0f, 100.0f);
    s.bri = clampf(BRI_BASE + uniform(-8.0f, 8.0f), 0.0f, 100.0f);
    return s;
}

static void buildTrunk() {
    for (int i = 0; i < TRUNK_STEPS; ++i) {
        int parentIdx = (int)g_branches.size() - 1;
        BranchPoint parent = g_branches[parentIdx];

        float nx = parent.x + uniform(-1.2f, 1.2f);
        float nz = parent.z + uniform(-1.2f, 1.2f);
        float ny = parent.y + TRUNK_STEP_HEIGHT;

        g_hasKid[parentIdx] = true;

        int newDepth = parent.depth + 1;
        g_branches.push_back({ nx, ny, nz, newDepth });
        g_hasKid.push_back(false);

        g_segments.push_back(makeSegment(parent.x, parent.y, parent.z, nx, ny, nz, newDepth));
    }
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
        randAttractorPt(g_ax[i], g_ay[i], g_az[i]);
    }

    g_branches.push_back({ 0.0f, GROUND_Y, 0.0f, 0 });
    g_hasKid.push_back(false);
    buildTrunk();
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


struct NoiseDot {
    float u, v;
    float speed;
    float phase;
};

static std::vector<NoiseDot> g_noiseDots;

static void init_noise_dots() {
    g_noiseDots.resize(2400);
    for (auto& d : g_noiseDots) {
        d.u = uniform(0.0f, 1.0f);
        d.v = uniform(0.0f, 1.0f);
        d.speed = uniform(2.0f, 9.0f);
        d.phase = uniform(0.0f, 2.0f * PI_F);
    }
}

static void drawBgNoise(int winW, int winH, double t) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, winW, winH, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glPointSize(1.2f);
    glBegin(GL_POINTS);
    for (const NoiseDot& d : g_noiseDots) {
        float flick = 0.5f + 0.5f * std::sin((float)t * d.speed + d.phase);
        float a = 0.04f + flick * 0.06f;
        glColor4f(0.62f, 0.66f, 0.75f, a);
        glVertex2f(d.u * (float)winW, d.v * (float)winH);
    }
    glEnd();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}


static void drawGroundGrid(float halfX, float halfZ, float y, int divisions) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.45f, 0.55f, 0.65f, 0.2f);
    glLineWidth(1.0f);

    glBegin(GL_LINES);
    for (int i = 0; i <= divisions; ++i) {
        float t = -halfX + (2.0f * halfX) * ((float)i / divisions);
        glVertex3f(t, y, -halfZ);
        glVertex3f(t, y, halfZ);
    }
    for (int i = 0; i <= divisions; ++i) {
        float t = -halfZ + (2.0f * halfZ) * ((float)i / divisions);
        glVertex3f(-halfX, y, t);
        glVertex3f(halfX, y, t);
    }
    glEnd();

    glDisable(GL_BLEND);
}

static void orthoBasis(float dx, float dy, float dz,
    float& ux, float& uy, float& uz,
    float& vx, float& vy, float& vz) {
    float ax = std::fabs(dx), ay = std::fabs(dy), az = std::fabs(dz);
    float tx, ty, tz;
    if (ax <= ay && ax <= az) { tx = 1; ty = 0; tz = 0; }
    else if (ay <= ax && ay <= az) { tx = 0; ty = 1; tz = 0; }
    else { tx = 0; ty = 0; tz = 1; }

    ux = dy * tz - dz * ty;
    uy = dz * tx - dx * tz;
    uz = dx * ty - dy * tx;
    float um = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (um < 1e-6f) { ux = 1; uy = 0; uz = 0; um = 1; }
    ux /= um; uy /= um; uz /= um;

    vx = dy * uz - dz * uy;
    vy = dz * ux - dx * uz;
    vz = dx * uy - dy * ux;
}

static void drawCylinder(float ax_, float ay_, float az_,
    float bx_, float by_, float bz_,
    float r0, float r1, int sides,
    float cr, float cg, float cb) {
    float dx = bx_ - ax_, dy = by_ - ay_, dz = bz_ - az_;
    float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-6f) return;
    dx /= len; dy /= len; dz /= len;

    float ux, uy, uz, vx, vy, vz;
    orthoBasis(dx, dy, dz, ux, uy, uz, vx, vy, vz);

    glColor3f(cr, cg, cb);
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= sides; ++i) {
        float ang = 2.0f * PI_F * (float)i / (float)sides;
        float ca = std::cos(ang), sa = std::sin(ang);
        float nx = ux * ca + vx * sa;
        float ny = uy * ca + vy * sa;
        float nz = uz * ca + vz * sa;

        glNormal3f(nx, ny, nz);
        glVertex3f(ax_ + nx * r0, ay_ + ny * r0, az_ + nz * r0);
        glVertex3f(bx_ + nx * r1, by_ + ny * r1, bz_ + nz * r1);
    }
    glEnd();
}

static float trunkRadius(int depth) {
    float t = clampf((float)depth / TRUNK_TAPER_DEPTH, 0.0f, 1.0f);
    float eased = 1.0f - (1.0f - t) * (1.0f - t);
    return TRUNK_BASE_RADIUS + (TRUNK_MIN_RADIUS - TRUNK_BASE_RADIUS) * eased;
}

static void drawSegments() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glShadeModel(GL_SMOOTH);

    GLfloat lightPos[] = { 0.35f, 1.0f, 0.55f, 0.0f };
    GLfloat lightAmbient[] = { 0.20f, 0.15f, 0.12f, 1.0f };
    GLfloat lightDiffuse[] = { 1.0f, 0.90f, 0.78f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);

    for (const Segment& s : g_segments) {
        float r0 = trunkRadius(s.depth);
        float r1 = trunkRadius(s.depth + 1);

        float r, g, b;
        hsb2rgb(s.hue, s.sat, s.bri, r, g, b);

        drawCylinder(s.ax, s.ay, s.az, s.bx, s.by, s.bz, r0, r1, CYLINDER_SIDES, r, g, b);
    }

    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);
}

static void drawBuds() {
    int n = (int)g_branches.size();
    if (n <= TRUNK_STEPS + 1) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glPointSize(2.2f);

    glBegin(GL_POINTS);
    for (int i = TRUNK_STEPS + 1; i < n; ++i) {
        if (!g_hasKid[i]) continue;

        const BranchPoint& p = g_branches[i];
        float depthFade = clampf(1.0f - p.depth * 0.00035f, 0.3f, 1.0f);
        float hue = hashRange(i, BARK_HUE_MIN - 4.0f, BARK_HUE_MAX + 8.0f);
        float sat = hashRange(i * 5 + 2, 40.0f, 65.0f);
        float r, g, b;
        hsb2rgb(hue, sat, 70.0f, r, g, b);
        glColor4f(r, g, b, 0.25f * depthFade);
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();

    glDisable(GL_BLEND);
}

static void leafOffset(int seed, float& ox, float& oy, float& oz) {
    float dx = hashRange(seed + 1, -1.0f, 1.0f);
    float dy = hashRange(seed + 2, -1.0f, 1.0f);
    float dz = hashRange(seed + 3, -1.0f, 1.0f);
    float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len < 1e-4f) { dx = 1.0f; dy = 0.0f; dz = 0.0f; len = 1.0f; }
    dx /= len; dy /= len; dz /= len;

    float dist = hashRange(seed + 4, BUSH_RADIUS_MIN, BUSH_RADIUS_MAX);
    ox = dx * dist; oy = dy * dist; oz = dz * dist;
}

static void drawLeaves() {
    int n = (int)g_branches.size();
    if (n <= TRUNK_STEPS + 1) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glBegin(GL_POINTS);
    for (int i = TRUNK_STEPS + 1; i < n; ++i) {
        if (g_hasKid[i]) continue;

        const BranchPoint& p = g_branches[i];
        float depthFade = clampf(1.0f - p.depth * 0.00035f, 0.55f, 1.0f);

        for (int k = 0; k < LEAVES_PER_TIP; ++k) {
            int seed = i * 97 + k * 13;

            float ox, oy, oz;
            leafOffset(seed, ox, oy, oz);

            float hue = hashRange(seed + 5, LEAF_HUE_MIN, LEAF_HUE_MAX);
            float sat = hashRange(seed + 6, LEAF_SAT_MIN, LEAF_SAT_MAX);
            float bri = hashRange(seed + 7, LEAF_BRI_MIN, LEAF_BRI_MAX);

            float r, g, b;
            hsb2rgb(hue, sat, bri, r, g, b);

            float sizeJitter = hashRange(seed + 8, 0.8f, 1.5f);
            glPointSize(7.0f * sizeJitter);
            glColor4f(r, g, b, 0.92f * depthFade);
            glVertex3f(p.x + ox, p.y + oy, p.z + oz);
        }
    }
    glEnd();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    struct GlowLayer { float size; float alphaScale; float briBoost; };
    static const GlowLayer layers[2] = {
        { 18.0f, 0.16f, 0.0f  },
        { 9.0f,  0.35f, 10.0f },
    };

    for (const GlowLayer& layer : layers) {
        glBegin(GL_POINTS);
        for (int i = TRUNK_STEPS + 1; i < n; ++i) {
            if (g_hasKid[i]) continue;

            const BranchPoint& p = g_branches[i];
            float depthFade = clampf(1.0f - p.depth * 0.00035f, 0.55f, 1.0f);

            for (int k = 0; k < LEAVES_PER_TIP; ++k) {
                int seed = i * 97 + k * 13;

                float ox, oy, oz;
                leafOffset(seed, ox, oy, oz);

                float hue = hashRange(seed + 5, LEAF_HUE_MIN, LEAF_HUE_MAX);
                float sat = hashRange(seed + 6, LEAF_SAT_MIN, LEAF_SAT_MAX);
                float bri = clampf(hashRange(seed + 7, LEAF_BRI_MIN, LEAF_BRI_MAX) + layer.briBoost, 0.0f, 100.0f);

                float r, g, b;
                hsb2rgb(hue, sat, bri, r, g, b);

                float sizeJitter = hashRange(seed + 8, 0.8f, 1.5f);
                glColor4f(r, g, b, layer.alphaScale * depthFade);
                glPointSize(layer.size * sizeJitter);
                glVertex3f(p.x + ox, p.y + oy, p.z + oz);
            }
        }
        glEnd();
    }

    glDepthMask(GL_TRUE);
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
        "Procedural Cherry Tree (CUDA + SDL3)",
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
    glEnable(GL_NORMALIZE);

    allocGpu();
    initSystem();
    init_noise_dots();

    bool running = true;
    bool dragging = false;
    bool draggingPan = false;
    float yaw = 0.0f, basePitch = 15.0f;
    float camDist = 1000.0f;
    float panX = 0.0f, panY = 0.0f;
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
                if (ev.key.key == SDLK_R) {
                    initSystem();
                    yaw = 0.0f; basePitch = 15.0f;
                    camDist = 1000.0f;
                    panX = 0.0f; panY = 0.0f;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = true;
                if (ev.button.button == SDL_BUTTON_RIGHT) draggingPan = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (ev.button.button == SDL_BUTTON_LEFT) dragging = false;
                if (ev.button.button == SDL_BUTTON_RIGHT) draggingPan = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (dragging) {
                    yaw += ev.motion.xrel * 0.25f;
                    basePitch += ev.motion.yrel * 0.25f;
                    basePitch = clampf(basePitch, -89.0f, 89.0f);
                }
                if (draggingPan) {
                    float panScale = camDist * 0.0016f;
                    panX += ev.motion.xrel * panScale;
                    panY -= ev.motion.yrel * panScale;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                camDist -= ev.wheel.y * 30.0f;
                camDist = clampf(camDist, 250.0f, 4000.0f);
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

        drawBgNoise(winW, winH, timeSeconds);

        float aspect = winH > 0 ? (float)winW / (float)winH : 1.0f;
        float nearP = 1.0f, farP = 6000.0f;
        float fovRad = 45.0f * PI_F / 180.0f;
        float top = nearP * std::tan(fovRad * 0.5f);
        float right = top * aspect;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(-right, right, -top, top, nearP, farP);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(panX, panY, -camDist);
        glRotatef(pitch, 1.0f, 0.0f, 0.0f);
        glRotatef(yaw, 0.0f, 1.0f, 0.0f);

        drawGroundGrid(WORLD_HALF_X, WORLD_HALF_Z, GROUND_Y, 24);
        drawSegments();
        drawBuds();
        drawLeaves();

        SDL_GL_SwapWindow(window);
    }

    freeGpu();
    SDL_GL_DestroyContext(glctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
