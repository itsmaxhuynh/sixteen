// Sanity tests: bitboard moves vs. a naive 2048 reference, symmetries, checkpoint round trip.
// Build and run with `make test`. Uses small patterns so it needs only a few MB.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "Agent.h"
#include "arena.h"
#include "HardStarts.h"
#include "Merge.h"
#include "StartPositions.h"
#include "Board64.h"
#include "NTupleNetwork.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// naive reference: g[r][c] ranks, direction 0=U 1=R 2=D 3=L
static int NaiveSlide(int g[4][4], int dir) {
    int reward = 0;
    bool moved = false;
    for (int line = 0; line < 4; ++line) {
        int cell[4][2];
        for (int k = 0; k < 4; ++k) {
            int r, c;
            if (dir == 3) { r = line; c = k; }
            else if (dir == 1) { r = line; c = 3 - k; }
            else if (dir == 0) { r = k; c = line; }
            else { r = 3 - k; c = line; }
            cell[k][0] = r;
            cell[k][1] = c;
        }
        int vals[4], n = 0;
        for (int k = 0; k < 4; ++k) if (g[cell[k][0]][cell[k][1]]) vals[n++] = g[cell[k][0]][cell[k][1]];
        int out[4] = {0, 0, 0, 0}, m = 0;
        for (int i = 0; i < n; ++i) {
            if (i + 1 < n && vals[i] == vals[i + 1] && vals[i] < MAX_RANK) {
                out[m++] = vals[i] + 1;
                reward += 1 << (vals[i] + 1);
                ++i;
            } else {
                out[m++] = vals[i];
            }
        }
        for (int k = 0; k < 4; ++k) {
            int &v = g[cell[k][0]][cell[k][1]];
            if (v != out[k]) moved = true;
            v = out[k];
        }
    }
    return moved ? reward : -1;
}

static void TestSlides() {
    std::mt19937_64 rng(42);
    for (int trial = 0; trial < 200000; ++trial) {
        int g[4][4];
        Board64 b;
        for (int i = 0; i < 16; ++i) {
            // bias toward empties and small ranks so merges are common, include rank 15
            int v = (rng() % 3 == 0) ? 0 : int(rng() % (trial % 7 == 0 ? 16 : 5));
            g[i / 4][i % 4] = v;
            b.Set(i, v);
        }
        for (int d = 0; d < 4; ++d) {
            int ng[4][4];
            std::copy(&g[0][0], &g[0][0] + 16, &ng[0][0]);
            int expected = NaiveSlide(ng, d);
            Board64 nb = b;
            reward_t got = nb.Slide(d);
            CHECK(int(got) == expected);
            if (expected != -1) {
                for (int i = 0; i < 16; ++i) CHECK(nb(i) == ng[i / 4][i % 4]);
            } else {
                CHECK(nb == b);
            }
        }
    }
    // the classic examples
    Board64 b;
    b.Set(0, 1); b.Set(1, 1); b.Set(2, 1); b.Set(3, 1); // 2 2 2 2 -> 4 4
    CHECK(b.Slide(3) == 8);
    CHECK(b(0) == 2 && b(1) == 2 && b(2) == 0 && b(3) == 0);
    Board64 full;
    for (int i = 0; i < 16; ++i) full.Set(i, (i % 2) ^ ((i / 4) % 2) ? 1 : 2);
    CHECK(full.IsTerminal());
    CHECK(!b.IsTerminal());
}

static void TestSymmetry() {
    Pattern p({0, 1, 2, 3, 4, 5});
    // all 8 isomorphisms of a board with distinct ranks must give 8 distinct indices
    Board64 b;
    for (int i = 0; i < 16; ++i) b.Set(i, i);
    std::set<size_t> seen;
    for (int iso = 0; iso < 8; ++iso) seen.insert(p.Index(b.GetBoard(), iso));
    CHECK(seen.size() == 8);
    // identity isomorphism reads the cells in order
    CHECK(p.Index(b.GetBoard(), 0) == 0x543210);
}

static Board64 MakeBoard(std::initializer_list<int> ranks) {
    Board64 b;
    int i = 0;
    for (int r : ranks) b.Set(i++, r);
    return b;
}

static void TestSplits() {
    // ranks: 11=2048 12=4096 13=8192 14=16384 15=32768
    Board64 none = MakeBoard({13, 12, 11, 0, 13});
    Board64 t16 = MakeBoard({14, 12, 11});
    Board64 t16_8 = MakeBoard({14, 13, 11});
    Board64 t16_8_4 = MakeBoard({14, 13, 12});
    Board64 t16_8_4_2 = MakeBoard({14, 13, 12, 11});
    CHECK(SplitLevel(none.GetBoard(), 4) == 0);
    CHECK(SplitLevel(t16.GetBoard(), 4) == 1);
    CHECK(SplitLevel(t16_8.GetBoard(), 4) == 2);
    CHECK(SplitLevel(t16_8_4.GetBoard(), 4) == 3);
    CHECK(SplitLevel(t16_8_4_2.GetBoard(), 4) == 4);
    CHECK(SplitLevel(t16_8_4_2.GetBoard(), 2) == 2); // 3-stage strategy has only 2 splits
    CHECK(SplitLevel(MakeBoard({15, 13}).GetBoard(), 4) == 0); // 32768 alone: the paper splits on time, see Agent
}

// one scalar feature at a time: two boards share a weight iff they share the feature value
static void TestScalarFeatures() {
    struct Case { uint32_t flag; Board64 a, same, other; };
    std::vector<Case> cases = {
            // empty tiles: a has 14 empty, same has 14, other 13
            {kFeatEmpty, MakeBoard({1, 2}), MakeBoard({0, 0, 5, 7}), MakeBoard({1, 2, 3})},
            // distinct values: {2,4} vs {8,16} vs {2,4,8}
            {kFeatDistinct, MakeBoard({1, 2, 1}), MakeBoard({3, 4}), MakeBoard({1, 2, 3})},
            // mergeable pairs: 2 2 | 4 4 in row 0 -> 2 pairs; vertical pair of 8s -> 1 pair + one more
            {kFeatMergeable, MakeBoard({1, 1, 2, 2}), MakeBoard({3, 0, 0, 0, 3, 0, 0, 0, 5, 5}), MakeBoard({1, 1})},
            // v/2v neighbours: 2-4 and 4-8 -> 2 pairs; 2 above 4 and 16-32 -> 2; 2-8 -> 0
            {kFeatNeighbor, MakeBoard({1, 2, 3}), MakeBoard({1, 0, 0, 0, 2, 0, 4, 5}), MakeBoard({1, 3})},
            // large tiles: same multiset of >= 2048 tiles, different small tiles
            {kFeatLargeTile, MakeBoard({11, 11, 14, 1}), MakeBoard({14, 3, 11, 0, 11}), MakeBoard({11, 14})},
    };
    for (const Case &c : cases) {
        NTupleNetwork net(1, c.flag, {});
        net.InitStage(0);
        net.UpdateValue(c.a.GetBoard(), 0, 1.0f);
        CHECK(net.GetValue(c.a.GetBoard(), 0) == 1.0f);
        CHECK(net.GetValue(c.same.GetBoard(), 0) == 1.0f);
        CHECK(net.GetValue(c.other.GetBoard(), 0) == 0.0f);
    }
}

// [MS-TD] eqs. (3), (4), (6) written out directly
static float ReferenceReturn(const std::vector<float> &r, const std::vector<float> &v, size_t t, float lambda) {
    size_t T = r.size();
    auto R = [&](int n) {
        float sum = 0;
        for (int k = 1; k <= n; ++k) {
            if (t + k >= T) return sum; // game over before n steps: terminal value 0
            sum += r[t + k];
        }
        return sum + v[t + n];
    };
    float w[5];
    for (int n = 1; n <= 4; ++n) w[n - 1] = (1 - lambda) * std::pow(lambda, n - 1);
    w[4] = std::pow(lambda, 4);
    float g = 0;
    for (int n = 1; n <= 5; ++n) g += w[n - 1] * R(n);
    return g;
}

static void TestLambdaReturn() {
    std::vector<float> r = {4, 8, 0, 16, 4, 32, 8, 0, 4};
    std::vector<float> v = {100, 90, 85, 70, 60, 41, 30, 9, 3};
    for (float lambda : {0.0f, 0.5f, 0.9f}) {
        for (size_t t = 0; t < r.size(); ++t) {
            float got = TdLearner::LambdaReturn(r, v, t, lambda, 5);
            float want = ReferenceReturn(r, v, t, lambda);
            CHECK(std::abs(got - want) < 1e-3f);
        }
    }
    // TD(0): r_{t+1} + V(s'_{t+1}); last afterstate: 0
    CHECK(TdLearner::LambdaReturn(r, v, 2, 0.0f, 5) == 16 + 70);
    CHECK(TdLearner::LambdaReturn(r, v, 8, 0.5f, 5) == 0);
    // eq. (6) weights 0.5 .25 .125 .0625 .0625 with all rewards 0 and V = 1, 2, 4, 8, 16
    std::vector<float> zr(6, 0), pv = {0, 1, 2, 4, 8, 16};
    CHECK(std::abs(TdLearner::LambdaReturn(zr, pv, 0, 0.5f, 5) - (0.5f + 0.5f + 0.5f + 0.5f + 1.0f)) < 1e-6f);
}

static void TestCheckpoint() {
    std::vector<std::vector<int>> small = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    NTupleNetwork net(3, kFeatAll, small);
    net.InitStage(0);
    net.InitStage(1);
    Board64 early = MakeBoard({3, 0, 0, 0, 0, 1});
    Board64 late = MakeBoard({14, 2, 11});
    net.UpdateValue(early.GetBoard(), 0, 1.5f);
    net.UpdateValue(late.GetBoard(), 1, -0.25f);
    float e0 = net.GetValue(early.GetBoard(), 0), l1 = net.GetValue(late.GetBoard(), 1);
    CHECK(net.GetValue(late.GetBoard(), 0) != l1); // stages hold separate weights

    TrainingState s;
    s.alpha = 0.00025f;
    s.lambda = 0.5f;
    s.active = 1;
    s.games = {5000000, 123, 0};
    std::string path = "/tmp/sixteen-test.ckpt";
    if (const char *dir = std::getenv("TMPDIR")) path = std::string(dir) + "/sixteen-test.ckpt";
    net.Save(path, s);

    NTupleNetwork loaded(1, 0, {{0, 1}});
    TrainingState t;
    loaded.Load(path, t);
    CHECK(t.alpha == 0.00025f && t.lambda == 0.5f && t.active == 1);
    CHECK(t.games == s.games);
    CHECK(loaded.NumStages() == 3 && loaded.Features() == kFeatAll);
    CHECK(loaded.Initialized(0) && loaded.Initialized(1) && !loaded.Initialized(2));
    CHECK(loaded.GetValue(early.GetBoard(), 0) == e0);
    CHECK(loaded.GetValue(late.GetBoard(), 1) == l1);
    std::remove(path.c_str());
}

// boards for the fast-path tests: uniform random cells, sparse random boards, and corner cases
static std::vector<board_t> TestBoards() {
    std::mt19937_64 rng(2048);
    std::vector<board_t> boards;
    for (int k = 0; k < 200000; ++k) boards.push_back(rng());
    for (int k = 0; k < 100000; ++k) { // few distinct small ranks: many merges and neighbours
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 4 ? rng() % 4 : 0) << (4 * i);
        boards.push_back(b);
    }
    for (int k = 0; k < 100000; ++k) { // large tiles and empties
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 3 ? 10 + rng() % 6 : 0) << (4 * i);
        boards.push_back(b);
    }
    for (int r = 0; r < 16; ++r) boards.push_back(0x1111111111111111ULL * board_t(r)); // all cells equal
    boards.push_back(0xFEDCBA9876543210ULL);
    boards.push_back(0x0123456789ABCDEFULL);
    boards.push_back(0x1212212112122121ULL); // checkerboard of 2 and 4
    return boards;
}

// mask-based tuple indices == Pattern::Index for every default pattern and symmetry
static void TestFastTupleIndices() {
    std::vector<Pattern> patterns;
    for (const std::vector<int> &p : NTupleNetwork::DefaultPatterns()) patterns.emplace_back(p);
    int bad = 0;
    for (board_t b : TestBoards()) {
        size_t idx[4][8];
        NTupleNetwork::TupleIndices(b, idx);
        for (int p = 0; p < 4; ++p) {
            for (int s = 0; s < 8; ++s) bad += idx[p][s] != patterns[p].Index(b, s);
        }
    }
    CHECK(bad == 0);
}

// row-table scalar features == the cell-by-cell reference, for every combination of features
static void TestFastScalarFeatures() {
    std::vector<board_t> boards = TestBoards();
    for (uint32_t flags = 1; flags <= kFeatAll; ++flags) {
        NTupleNetwork net(1, flags, {});
        int bad = 0;
        for (board_t b : boards) {
            size_t got[5], want[5];
            int n = net.ScalarIndices(b, got), m = net.ScalarIndicesReference(b, want);
            bad += n != m;
            for (int j = 0; j < std::min(n, m); ++j) bad += got[j] != want[j];
        }
        CHECK(bad == 0);
    }
}

// full default network with random weights: fast GetValue is bit-identical to the reference, and
// UpdateValue touches exactly the reference weights (needs about 270 MB)
static void TestFastValue() {
    NTupleNetwork net(1, kFeatAll);
    CHECK(net.FastTuples());
    net.InitStage(0);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (std::vector<float> &table : net.StageWeights(0)) for (float &w : table) w = dist(rng);
    std::vector<board_t> boards = TestBoards();
    int bad = 0;
    for (board_t b : boards) bad += net.GetValue(b, 0) != net.GetValueReference(b, 0);
    CHECK(bad == 0);

    // updates: zero the weights, then fast +delta followed by reference -delta must restore all zeros
    for (std::vector<float> &table : net.StageWeights(0)) std::fill(table.begin(), table.end(), 0.0f);
    bad = 0;
    for (size_t k = 0; k < boards.size(); k += 101) {
        board_t b = boards[k];
        net.UpdateValue(b, 0, 1.0f);
        net.UpdateValueReference(b, 0, -1.0f);
        bad += net.GetValue(b, 0) != 0.0f;
    }
    for (const std::vector<float> &table : net.StageWeights(0)) {
        for (float w : table) bad += w != 0.0f;
    }
    CHECK(bad == 0);
}

// cheap stand-in for TdLearner: a hash of the board as value, splits as in the real network
struct FakeLearner {
    float Evaluate(board_t b, int game_stage) const {
        uint64_t x = (b + uint64_t(game_stage)) * 0x9E3779B97F4A7C15ULL;
        return float((x >> 40) % 100000) * 0.01f;
    }
    int Split(board_t b) const { return SplitLevel(b, 4); }
};

// plain expectimax written out as in TdLearner (no cache, no time limit); with prune > 0, a spawn
// outcome whose path probability is below prune is valued at ply 1, as Search does
template<class L> static float RefMax(const L &l, const Board64 &b, int ply, int stage, int *move,
                                      float prune = 0, float prob = 1);
template<class L> static float RefChance(const L &l, const Board64 &after, int ply, int stage, float prune, float prob) {
    stage = std::max(stage, l.Split(after.GetBoard()));
    int empty = after.CountEmpty();
    float p2 = prob * 0.9f / float(empty), p4 = prob * 0.1f / float(empty);
    float total = 0;
    for (int i = 0; i < 16; ++i) {
        if (after(i)) continue;
        Board64 two = after, four = after;
        two.Set(i, 1);
        four.Set(i, 2);
        total += 0.9f * RefMax(l, two, prune > 0 && p2 < prune ? 1 : ply, stage, nullptr, prune, p2);
        total += 0.1f * RefMax(l, four, prune > 0 && p4 < prune ? 1 : ply, stage, nullptr, prune, p4);
    }
    return empty ? total / float(empty) : 0.0f;
}
template<class L> static float RefMax(const L &l, const Board64 &b, int ply, int stage, int *move, float prune, float prob) {
    int dir = -1;
    float best = -INFINITY;
    for (int d = 0; d < 4; ++d) {
        Board64 c = b;
        reward_t r = c.Slide(d);
        if (r == -1) continue;
        float v = r + (ply <= 1 ? l.Evaluate(c.GetBoard(), stage) : RefChance(l, c, ply - 1, stage, prune, prob));
        if (v > best) { best = v; dir = d; }
    }
    if (move) *move = dir;
    return dir == -1 ? 0.0f : best;
}

// boards from random games, with some large tiles so stages > 0 occur
static std::vector<Board64> SearchBoards(int n) {
    std::mt19937 rng(99);
    std::vector<Board64> boards;
    while (int(boards.size()) < n) {
        Board64 b;
        if (rng() % 2) { b.Set(rng() % 16, 14); b.Set(rng() % 16, 13); } // 16384 + 8192: split 2
        for (int k = 0; k < 2; ++k) { int e; do e = rng() % 16; while (b(e)); b.Set(e, 1); }
        for (int step = 0; step < 300; ++step) {
            int d = rng() % 4;
            Board64 a = b;
            if (a.Slide(d) == -1) { if (b.IsTerminal()) break; continue; }
            b = a;
            int e; do e = rng() % 16; while (b(e)); b.Set(e, rng() % 10 ? 1 : 2);
            if (step % 37 == 0) boards.push_back(b);
        }
    }
    return boards;
}

// cached search == plain expectimax, for every ply, with the cache kept across positions
static void TestSearchExact() {
    FakeLearner l;
    Search<FakeLearner> cached(l, 16), uncached(l, 0);
    int bad = 0, n = 0;
    for (const Board64 &b : SearchBoards(150)) {
        int stage = l.Split(b.GetBoard());
        for (int ply = 1; ply <= 3; ++ply) {
            int want;
            RefMax(l, b, ply, stage, &want);
            bad += cached.Fixed(b, ply, stage).move != want;
            bad += uncached.Fixed(b, ply, stage).move != want;
            ++n;
        }
    }
    CHECK(n > 400 && bad == 0);
    // a 2-entry cache collides all the time: the same board searched at every ply and stage must
    // never reuse a value stored for another ply or stage
    Search<FakeLearner> tiny(l, 1);
    bad = 0;
    for (const Board64 &b : SearchBoards(60)) {
        // both loop orders: ply inside repeats a board at a new ply right away, stage inside at a new stage
        for (int order = 0; order < 2; ++order) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int ply = 1 + (order ? i : j), stage = order ? j : i;
                    int want;
                    RefMax(l, b, ply, stage, &want);
                    bad += tiny.Fixed(b, ply, stage).move != want;
                }
            }
        }
    }
    CHECK(bad == 0);
}

static std::shared_ptr<TdLearner> MakeLearner(const std::string &args);

// Search on the real learner == plain expectimax on its Evaluate, with and without the cache, with and without pruning
static void TestSearchLearner() {
    // 5 trained stages with different random weights, so a leaf valued at the wrong stage shows
    NTupleNetwork net(5, kFeatAll, {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}});
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> w(-50, 50);
    for (int s = 0; s < 5; ++s) {
        net.InitStage(s);
        for (auto &table : net.StageWeights(s)) for (float &x : table) x = w(rng);
    }
    TrainingState st;
    st.active = 4;
    st.games = {1, 1, 1, 1, 1};
    std::string ckpt = "/tmp/sixteen-search.ckpt";
    if (const char *dir = std::getenv("TMPDIR")) ckpt = std::string(dir) + "/sixteen-search.ckpt";
    net.Save(ckpt, st);
    auto l = MakeLearner("load=" + ckpt);
    Search<TdLearner> cached(*l, 16), uncached(*l, 0), pruned(*l, 16);
    const float prune = 0.05f;
    pruned.SetPrune(prune);
    // plus boards where a slide merges two equal big tiles into a new split tile (two 8192s into
    // 16384, two 4096s next to a 16384 into 8192), so a leaf's stage is above its parent's
    std::vector<Board64> boards = SearchBoards(150);
    for (int k = 0; k < 60; ++k) {
        Board64 b;
        for (int i = 0; i < 16; ++i) b.Set(i, rng() % 3 ? 1 + rng() % 6 : 0);
        int big = k % 2 ? 12 : 13, row = rng() % 4;
        b.Set(4 * row, big);
        b.Set(4 * row + 1, big);
        if (k % 2) b.Set(4 * ((row + 2) % 4) + 3, 14);
        if (!b.IsTerminal()) boards.push_back(b);
    }
    int bad = 0, n = 0, pruned_differs = 0;
    for (const Board64 &b : boards) {
        int stage = l->Split(b.GetBoard());
        for (int ply = 1; ply <= 3; ++ply) {
            int want, want_pruned;
            RefMax(*l, b, ply, stage, &want);
            RefMax(*l, b, ply, stage, &want_pruned, prune);
            bad += cached.Fixed(b, ply, stage).move != want;
            bad += uncached.Fixed(b, ply, stage).move != want;
            bad += pruned.Fixed(b, ply, stage).move != want_pruned;
            pruned_differs += want_pruned != want;
            ++n;
        }
    }
    CHECK(n > 400 && bad == 0);
    CHECK(pruned_differs > 0); // the pruned search really searched something else
    std::remove(ckpt.c_str());
}

// a learner whose leaf values are leaf-downgraded: Search with SetLeafDowngrade must equal plain
// expectimax on it (the real board searched, each leaf halved on its own and valued in its stage)
struct LeafDowngradeLearner {
    const TdLearner &l;
    int rank;
    int Split(board_t b) const { return l.Split(b); }
    float Evaluate(board_t b, int stage) const {
        board_t d = DowngradeRoot(b, rank, true);
        return d == b ? l.Evaluate(b, stage) : l.Evaluate(d, l.Split(d));
    }
};

// the tile DowngradeRoot halves above (its largest missing rank), -1 if it leaves b unchanged
static int HalvedAbove(board_t b) {
    if (DowngradeRoot(b, 15) == b) return -1;
    unsigned present = 0;
    for (int i = 0; i < 16; ++i) present |= 1u << ((b >> (4 * i)) & 0xf);
    int r = 31 - __builtin_clz(present) - 1;
    while ((present >> r) & 1) --r;
    return r;
}

static void TestLeafDowngrade() {
    NTupleNetwork net(5, kFeatAll, {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}});
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> w(-50, 50);
    for (int s = 0; s < 5; ++s) {
        net.InitStage(s);
        for (auto &table : net.StageWeights(s)) for (float &x : table) x = w(rng);
    }
    TrainingState st;
    st.active = 4;
    st.games = {1, 1, 1, 1, 1};
    std::string ckpt = "/tmp/sixteen-leafdg.ckpt";
    if (const char *dir = std::getenv("TMPDIR")) ckpt = std::string(dir) + "/sixteen-leafdg.ckpt";
    net.Save(ckpt, st);
    auto l = MakeLearner("load=" + ckpt);
    LeafDowngradeLearner ref{*l, 15};
    Search<TdLearner> cached(*l, 16), uncached(*l, 0), plain(*l, 16);
    cached.SetLeafDowngrade(15);
    uncached.SetLeafDowngrade(15);
    // endgame boards: 32768 16384 8192 4096 along the bottom, 2048 missing, two 1024s that a slide
    // can merge into the missing 2048 (so leaves are halved at different tiles), small tiles elsewhere
    std::vector<Board64> boards;
    while (boards.size() < 120) {
        Board64 b;
        for (int i = 0; i < 12; ++i) b.Set(i, rng() % 4 ? 1 + rng() % 5 : 0);
        for (int i = 0; i < 4; ++i) b.Set(12 + i, 15 - i);
        int row = rng() % 3, col = rng() % 3;
        b.Set(4 * row + col, 10);
        if (rng() % 2) b.Set(4 * row + col + 1, 10);
        else if (row < 2) b.Set(4 * (row + 1) + col, 10);
        if (!b.IsTerminal() && DowngradeRoot(b.GetBoard(), 15) != b.GetBoard()) boards.push_back(b);
    }
    // and boards from late in a game (stage 4 reached) whose 16384 and 8192 have merged away: 32768
    // and 4096 with small tiles, halved to a stage-1 board, so a leaf valued in the game's stage shows
    const size_t endgame = boards.size();
    while (boards.size() < endgame + 60) {
        Board64 b;
        for (int i = 0; i < 16; ++i) b.Set(i, rng() % 4 ? 1 + rng() % 6 : 0);
        b.Set(rng() % 16, 15);
        int c; do c = rng() % 16; while (b(c) == 15); b.Set(c, 12);
        if (!b.IsTerminal() && DowngradeRoot(b.GetBoard(), 15) != b.GetBoard()) boards.push_back(b);
    }
    // and boards just before a second 32768: a 32768 and two 16384s a slide merges, so leaves hold
    // two 32768s (halved there as in a wide game; the root, with one 32768, is not)
    const size_t pairs = boards.size();
    while (boards.size() < pairs + 60) {
        Board64 b;
        for (int i = 0; i < 16; ++i) b.Set(i, rng() % 4 ? 1 + rng() % 6 : 0);
        int row = rng() % 4;
        b.Set(4 * row, 15);
        b.Set(4 * row + 1, 14);
        b.Set(4 * row + 2, 14);
        if (!b.IsTerminal()) boards.push_back(b);
    }
    int bad = 0, n = 0, differs = 0, rehalved = 0, two_leaves = 0;
    for (size_t k = 0; k < boards.size(); ++k) {
        const Board64 &b = boards[k];
        int stage = k < endgame || k >= pairs ? l->Split(b.GetBoard()) : 4;
        for (int ply = 1; ply <= 3; ++ply) {
            int want, root_move;
            RefMax(ref, b, ply, stage, &want);
            RefMax(*l, b, ply, stage, &root_move);
            bad += cached.Fixed(b, ply, stage).move != want;
            bad += uncached.Fixed(b, ply, stage).move != want;
            differs += plain.Fixed(b, ply, stage).move != want;
            ++n;
        }
        // some children fill the missing 2048, so their leaves are halved at a lower tile
        for (int d = 0; d < 4; ++d) {
            Board64 c = b;
            if (c.Slide(d) == -1) continue;
            rehalved += HalvedAbove(c.GetBoard()) != HalvedAbove(b.GetBoard());
            two_leaves += DowngradeRoot(c.GetBoard(), 15) != DowngradeRoot(c.GetBoard(), 15, true);
        }
    }
    CHECK(n >= 720 && bad == 0);
    CHECK(two_leaves > 30); // children with two 32768s, which only the leaf rule halves
    CHECK(differs > 20);   // leaf downgrading changes the search, so the boards exercise it
    CHECK(rehalved > 10);  // and some leaves are halved at another tile than the root would be
    std::remove(ckpt.c_str());
}

// time-limited search: a legal move, at least 2 plies, the deepest finished depth's move, and
// within the hard deadline (with slack for one 16-node check interval)
static void TestSearchTimed() {
    FakeLearner l;
    Search<FakeLearner> s(l, 18);
    int bad_move = 0, shallow = 0, late = 0, n = 0;
    double max_ms = 0;
    for (const Board64 &b : SearchBoards(40)) {
        if (b.IsTerminal()) continue;
        Search<FakeLearner>::Result r = s.Timed(b, 0, 5.0, 8.0);
        Board64 c = b;
        bad_move += r.move < 0 || c.Slide(r.move) == -1;
        shallow += r.ply < 2;
        late += r.ms > 8.0 + 5.0;
        max_ms = std::max(max_ms, r.ms);
        int want;
        RefMax(l, b, r.ply, 0, &want);
        bad_move += r.move != want;
        ++n;
    }
    CHECK(n > 20 && bad_move == 0 && shallow == 0 && late == 0);

    // force aborts: a huge soft budget lets it start depths that the 2 ms hard deadline cuts off;
    // the move must then come from the last finished depth
    Search<FakeLearner> t(l, 1);
    int aborted = 0;
    bad_move = late = 0;
    for (const Board64 &b : SearchBoards(40)) {
        if (b.IsTerminal()) continue;
        Search<FakeLearner>::Result r = t.Timed(b, 0, 1e9, 2.0, 6);
        aborted += r.ply < 6;
        Board64 c = b;
        bad_move += r.move < 0 || c.Slide(r.move) == -1;
        int want;
        RefMax(l, b, r.ply, 0, &want);
        bad_move += r.move != want;
        late += r.ms > 2.0 + 5.0;
    }
    CHECK(aborted > 10 && bad_move == 0 && late == 0);
    // a terminal board: no move
    Board64 full;
    for (int i = 0; i < 16; ++i) full.Set(i, (i + i / 4) % 2 + 1);
    CHECK(s.Timed(full, 0, 5.0).move == -1);
}

// plain expectimax with sampled chance nodes: below the first chance layer (depth >= 2 from the
// root), average SpawnCells(after, k) instead of every empty cell
template<class L> static float RefSampledMax(const L &l, const Board64 &b, int ply, int stage, int k, int depth, int *move);
template<class L> static float RefSampledChance(const L &l, const Board64 &after, int ply, int stage, int k, int depth) {
    stage = std::max(stage, l.Split(after.GetBoard()));
    int cells[16];
    int n = SpawnCells(after, depth >= 2 ? k : 0, cells);
    float total = 0;
    for (int c = 0; c < n; ++c) {
        Board64 two = after, four = after;
        two.Set(cells[c], 1);
        four.Set(cells[c], 2);
        total += 0.9f * RefSampledMax(l, two, ply, stage, k, depth, nullptr);
        total += 0.1f * RefSampledMax(l, four, ply, stage, k, depth, nullptr);
    }
    return n ? total / float(n) : 0.0f;
}
template<class L> static float RefSampledMax(const L &l, const Board64 &b, int ply, int stage, int k, int depth, int *move) {
    int dir = -1;
    float best = -INFINITY;
    for (int d = 0; d < 4; ++d) {
        Board64 c = b;
        reward_t r = c.Slide(d);
        if (r == -1) continue;
        float v = r + (ply <= 1 ? l.Evaluate(c.GetBoard(), stage) : RefSampledChance(l, c, ply - 1, stage, k, depth + 1));
        if (v > best) { best = v; dir = d; }
    }
    if (move) *move = dir;
    return dir == -1 ? 0.0f : best;
}

// root move pruning as specified: value every root move one ply shallower, keep the `keep` best
// minus those more than cut*|best| below the best, and pick the best kept move at the full ply
template<class L> static int RefRootCut(const L &l, const Board64 &b, int ply, int stage, int keep, float cut) {
    float shallow[4], deep[4];
    for (int d = 0; d < 4; ++d) {
        Board64 c = b;
        reward_t r = c.Slide(d);
        shallow[d] = deep[d] = -INFINITY;
        if (r == -1) continue;
        shallow[d] = r + (ply - 1 <= 1 ? l.Evaluate(c.GetBoard(), stage) : RefChance(l, c, ply - 2, stage, 0, 1));
        deep[d] = r + (ply <= 1 ? l.Evaluate(c.GetBoard(), stage) : RefChance(l, c, ply - 1, stage, 0, 1));
    }
    float best = -INFINITY;
    for (float v : shallow) best = std::max(best, v);
    int dir = -1, rank_of[4];
    for (int d = 0; d < 4; ++d) {
        rank_of[d] = 0;
        for (int e = 0; e < 4; ++e) rank_of[d] += shallow[e] > shallow[d] || (shallow[e] == shallow[d] && e < d);
    }
    float top = -INFINITY;
    for (int d = 0; d < 4; ++d) {
        if (shallow[d] == -INFINITY) continue;
        bool kept = rank_of[d] == 0 || (rank_of[d] < keep && !(cut > 0 && shallow[d] < best - cut * std::fabs(best)));
        if (kept && deep[d] > top) { top = deep[d]; dir = d; }
    }
    return dir;
}

// rootk/rootcut and sample= match their references, with the cache, across a game's searches
// (including a tiny colliding cache and sampled/exact searches interleaved on one cache)
static void TestSearchApprox() {
    FakeLearner l;
    std::vector<Board64> boards = SearchBoards(120);
    int bad = 0, n = 0, root_differs = 0;
    Search<FakeLearner> rc(l, 16), rc_tiny(l, 1);
    for (auto keep_cut : std::vector<std::pair<int, float>>{{1, 0}, {2, 0}, {3, 0.05f}, {4, 0.02f}}) {
        rc.SetRootCut(keep_cut.first, keep_cut.second);
        rc_tiny.SetRootCut(keep_cut.first, keep_cut.second);
        for (const Board64 &b : boards) {
            int stage = l.Split(b.GetBoard());
            for (int ply = 2; ply <= 3; ++ply) {
                int want = RefRootCut(l, b, ply, stage, keep_cut.first, keep_cut.second), exact;
                RefMax(l, b, ply, stage, &exact);
                bad += rc.Fixed(b, ply, stage).move != want;
                bad += rc_tiny.Fixed(b, ply, stage).move != want;
                root_differs += want != exact;
                ++n;
            }
        }
    }
    CHECK(n > 800 && bad == 0);
    CHECK(root_differs > 0);
    // timed search with root pruning: a legal move, as deep as with it off is not required
    rc.SetRootCut(2, 0);
    bad = 0;
    for (const Board64 &b : boards) {
        if (b.IsTerminal()) continue;
        Search<FakeLearner>::Result r = rc.Timed(b, 0, 3.0, 6.0);
        Board64 c = b;
        bad += r.move < 0 || c.Slide(r.move) == -1;
    }
    CHECK(bad == 0);

    // sampled chance nodes; one cache shared with exact searches of the same boards
    Search<FakeLearner> mixed(l, 16), tiny(l, 1), sampled(l, 0);
    int sample_differs = 0;
    bad = n = 0;
    for (int k : {1, 3}) {
        sampled.SetSample(k);
        tiny.SetSample(k);
        for (const Board64 &b : boards) {
            int stage = l.Split(b.GetBoard());
            for (int ply = 2; ply <= 4; ++ply) {
                int want, exact;
                RefSampledMax(l, b, ply, stage, k, 0, &want);
                RefMax(l, b, ply, stage, &exact);
                mixed.SetSample(0);
                bad += mixed.Fixed(b, ply, stage).move != exact;
                mixed.SetSample(k);
                bad += mixed.Fixed(b, ply, stage).move != want;
                bad += tiny.Fixed(b, ply, stage).move != want;
                bad += sampled.Fixed(b, ply, stage).move != want;
                sample_differs += want != exact;
                ++n;
            }
        }
    }
    CHECK(n > 600 && bad == 0);
    CHECK(sample_differs > 0);
    // k >= 16 never samples: exact
    Search<FakeLearner> all(l, 16);
    all.SetSample(16);
    bad = 0;
    for (const Board64 &b : boards) {
        int want;
        RefMax(l, b, 3, 0, &want);
        bad += all.Fixed(b, 3, 0).move != want;
    }
    CHECK(bad == 0);
}

// deep=e1,e2,...: one more ply per threshold the empty cells are at or below (inclusive)
static void TestAdaptivePly() {
    std::vector<int> none, deep = {4, 1};
    for (int empty = 0; empty <= 16; ++empty) {
        CHECK(AdaptivePly(3, empty, none) == 3);
        int want = empty <= 1 ? 5 : empty <= 4 ? 4 : 3;
        CHECK(AdaptivePly(3, empty, deep) == want);
    }
}

// tile-downgrading written out on a grid: the rank r above which tiles halve (largest missing rank
// with 1 <= r < top), or -1 when downgrading does not apply (two 32768s skip it unless two_max)
static int NaiveSkippedRank(board_t b, int min_rank, bool two_max = false) {
    int top = 0, n_max = 0;
    bool present[16] = {};
    for (int i = 0; i < 16; ++i) {
        int x = int((b >> (4 * i)) & 0xf);
        present[x] = true;
        top = std::max(top, x);
        n_max += x == 15;
    }
    if (top < min_rank || (n_max >= 2 && !two_max)) return -1;
    for (int k = top - 1; k >= 1; --k) if (!present[k]) return k;
    return -1;
}

static board_t NaiveDowngrade(board_t b, int min_rank, bool two_max = false) {
    int r = NaiveSkippedRank(b, min_rank, two_max);
    int g[16];
    for (int i = 0; i < 16; ++i) g[i] = int((b >> (4 * i)) & 0xf);
    if (r < 0) return b;
    board_t out = 0;
    for (int i = 0; i < 16; ++i) out |= board_t(g[i] > r ? g[i] - 1 : g[i]) << (4 * i);
    return out;
}

static void TestDowngrade() {
    // ranks: 8=256 14=16384 15=32768. Paper's Fig. 10 case: 256 missing, everything above it halves
    Board64 fig = MakeBoard({15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 1, 0, 1});
    CHECK(DowngradeRoot(fig.GetBoard(), 15) == MakeBoard({14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1}).GetBoard());
    CHECK(DowngradeRoot(fig.GetBoard(), 0) == fig.GetBoard());                       // off
    Board64 no32k = MakeBoard({14, 13, 12, 3});
    CHECK(DowngradeRoot(no32k.GetBoard(), 15) == no32k.GetBoard());                  // largest tile too small
    CHECK(DowngradeRoot(no32k.GetBoard(), 14) == MakeBoard({13, 12, 11, 3}).GetBoard()); // lower threshold
    CHECK(DowngradeRoot(MakeBoard({15, 3, 1}).GetBoard(), 15) == MakeBoard({14, 3, 1}).GetBoard()); // r = 14
    Board64 full = MakeBoard({15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
    CHECK(DowngradeRoot(full.GetBoard(), 15) == full.GetBoard());                    // every tile 2..32768: none missing
    // every tile from 8 to 16384 present (just before a second 32768): r = 1 (no 2-tile), all halve
    Board64 snake = MakeBoard({15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 2, 2, 3});
    CHECK(DowngradeRoot(snake.GetBoard(), 15) == MakeBoard({14, 13, 12, 11, 7, 8, 9, 10, 6, 5, 4, 3, 0, 1, 1, 2}).GetBoard());
    Board64 no4 = MakeBoard({15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 1, 0});
    CHECK(DowngradeRoot(no4.GetBoard(), 15) == MakeBoard({14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 0}).GetBoard()); // r = 2
    Board64 two_max = MakeBoard({15, 15, 3});
    CHECK(DowngradeRoot(two_max.GetBoard(), 15) == two_max.GetBoard());              // 32768 pair: skipped
    // leaf valuation (two_max): the pair halves like any tiles, as Fit does for a wide game
    CHECK(DowngradeRoot(two_max.GetBoard(), 15, true) == MakeBoard({14, 14, 3}).GetBoard());
    CHECK(DowngradeRoot(MakeBoard({15, 15, 14, 3}).GetBoard(), 15, true) == MakeBoard({14, 14, 13, 3}).GetBoard());
    Board64 two_full = MakeBoard({15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
    CHECK(DowngradeRoot(two_full.GetBoard(), 15, true) == two_full.GetBoard());     // none missing
    CHECK(DowngradeRoot(fig.GetBoard(), 15, true) == DowngradeRoot(fig.GetBoard(), 15)); // one 32768: as before

    // random boards with large tiles: same as the naive version, and every move is legal on both
    // boards and gives the downgraded result of the real move
    std::mt19937_64 rng(77);
    int applied = 0, two_applied = 0, bad = 0;
    for (int k = 0; k < 200000; ++k) {
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 4 ? rng() % 16 : 0) << (4 * i);
        if (k % 2) b = (b & ~board_t(0xf)) | 15; // half of them hold a 32768 in cell 0
        board_t d = DowngradeRoot(b, 15);
        bad += d != NaiveDowngrade(b, 15);
        // two_max: the naive version, the same as without it below two 32768s, and on two 32768s
        // the board Fit gives the wide game (WideBoard.h), which closes the same missing rank
        board_t d2 = DowngradeRoot(b, 15, true);
        bad += d2 != NaiveDowngrade(b, 15, true);
        int n_max = 0;
        for (int i = 0; i < 16; ++i) n_max += ((b >> (4 * i)) & 0xf) == 15;
        if (n_max < 2) bad += d2 != d;
        if (n_max == 2) {
            bad += d2 != Fit(WideBoard::From(b));
            two_applied += d2 != b;
        }
        if (d == b) continue;
        ++applied;
        int r = NaiveSkippedRank(b, 15);
        for (int dir = 0; dir < 4; ++dir) {
            Board64 rb(b), db(d);
            reward_t rr = rb.Slide(dir), dr = db.Slide(dir);
            bad += (rr == -1) != (dr == -1);
            if (rr == -1) continue;
            board_t want = 0;
            for (int i = 0; i < 16; ++i) {
                int x = rb(i);
                want |= board_t(x > r ? x - 1 : x) << (4 * i);
            }
            bad += db.GetBoard() != want;
        }
    }
    CHECK(applied > 10000 && two_applied > 1000 && bad == 0);
}

// TdPlayer with downgrade=: the move is the plain player's move on the downgraded board, valued
// with the stage of the downgraded board; boards it does not apply to are played as before
// WideBoard: the same slides and rewards as Board64 wherever Board64 makes every merge (at most one
// 32768), and 32768 + 32768 = 65536, which Board64 refuses
static void TestWideBoard() {
    std::mt19937_64 rng(5);
    int bad = 0, compared = 0;
    for (int trial = 0; trial < 100000; ++trial) {
        Board64 b;
        for (int i = 0; i < 16; ++i) b.Set(i, rng() % 3 == 0 ? 0 : int(rng() % (trial % 5 ? 6 : 16)));
        WideBoard w0 = WideBoard::From(b.GetBoard());
        if (!w0.Fits()) continue;
        bad += Fit(w0) != b.GetBoard(); // fits: Fit changes nothing
        for (int d = 0; d < 4; ++d) {
            Board64 nb = b;
            WideBoard w = w0;
            reward_t r = nb.Slide(d);
            long wr = w.Slide(d);
            bad += long(r) != wr;
            if (wr != -1) bad += !(WideBoard::From(nb.GetBoard()) == w);
            compared++;
        }
    }
    CHECK(compared > 300000 && bad == 0);

    WideBoard pair = WideBoard::From(MakeBoard({15, 15, 0, 3}).GetBoard());
    Board64 pair64 = MakeBoard({15, 15, 0, 3});
    CHECK(pair64.Slide(3) == 0);                    // Board64 moves the 8 but cannot merge the pair
    CHECK(pair.Slide(3) == 65536 && pair.cell[0] == 16 && pair.cell[1] == 3 && pair.MaxRank() == 16);
    CHECK(pair.Hex() == "g300000000000000");
    CHECK(pair.Place(1, 1) == -1 && pair.Place(2, 1) == 0 && pair.cell[2] == 1);
}

// Fit: unchanged when the board fits; otherwise the largest missing ranks closed until it does, which
// keeps empties, equalities and order, so every move is legal on both and makes the same slides and
// merges (the same cells end up empty). The boards after it need not match in shape: a merge can
// build the rank that was closed, which then equals a lowered tile (a phantom merge, as inside the
// search), so the game refits the real board after every move.
static void TestFit() {
    auto wide = [](std::initializer_list<int> ranks) {
        WideBoard w;
        int i = 0;
        for (int r : ranks) w.cell[i++] = uint8_t(r);
        return w;
    };
    auto narrow = [](std::initializer_list<int> ranks) { return MakeBoard(ranks).GetBoard(); };
    CHECK(Fit(wide({15, 14, 3, 1})) == narrow({15, 14, 3, 1}));     // fits: unchanged
    CHECK(Fit(wide({15, 15, 13, 1})) == narrow({14, 14, 13, 1}));   // pair of 32768s: 16384 missing
    CHECK(Fit(wide({15, 15, 14, 13})) == narrow({14, 14, 13, 12})); // 8192 present, 4096 missing
    CHECK(Fit(wide({16, 3, 1})) == narrow({15, 3, 1}));             // 65536 alone becomes 32768
    CHECK(Fit(wide({16, 15, 3})) == narrow({15, 14, 3}));           // 65536 + 32768
    CHECK(Fit(wide({17, 16, 16, 2})) == narrow({15, 14, 14, 2}));   // closes two ranks
    // full board of every tile 2..16384 plus two 32768s: nothing to close, capped (and no hang)
    CHECK(Fit(wide({15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1})) ==
          narrow({15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}));

    // same structure: zero, equal and order relations between any two cells
    auto same_shape = [](const WideBoard &w, board_t f) {
        for (int i = 0; i < 16; ++i) {
            int fi = int((f >> (4 * i)) & 0xf);
            if ((w.cell[i] == 0) != (fi == 0)) return false;
            for (int j = 0; j < 16; ++j) {
                int fj = int((f >> (4 * j)) & 0xf);
                if ((w.cell[i] == w.cell[j]) != (fi == fj) || (w.cell[i] < w.cell[j]) != (fi < fj)) return false;
            }
        }
        return true;
    };
    std::mt19937_64 rng(11);
    int fitted = 0, bad = 0;
    for (int trial = 0; trial < 100000; ++trial) {
        WideBoard w;
        for (int i = 0; i < 16; ++i) w.cell[i] = uint8_t(rng() % 3 == 0 ? 0 : 10 + int(rng() % 8)); // 1024..131072
        if (trial % 2) w.cell[rng() % 16] = 15, w.cell[rng() % 16] = 15;                       // 32768 pairs
        board_t f = Fit(w);
        if (WideBoard::From(f) == w) continue;
        fitted++;
        bad += !WideBoard::From(f).Fits() || !same_shape(w, f);
        for (int d = 0; d < 4; ++d) {
            WideBoard ws = w;
            Board64 fs(f);
            long wr = ws.Slide(d);
            reward_t fr = fs.Slide(d);
            bad += (wr == -1) != (fr == -1);
            for (int i = 0; i < 16 && wr != -1; ++i) bad += (ws.cell[i] == 0) != (fs(i) == 0);
        }
    }
    CHECK(fitted > 50000 && bad == 0);
}

// a game kept on a WideBoard: from the full 32768..4 snake, the cascade of 14 slides ends in 65536;
// agents only ever see boards that fit, and the same game on Board64 stops at the last merge
static void TestWideEpisode() {
    board_t snake = MakeBoard({15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 0, 2, 3}).GetBoard();
    const char *line = "RRURRLURRRURRR";
    Episode wide, narrow;
    wide.Start(snake, 0);
    wide.UseWideBoard();
    narrow.Start(snake, 0);
    CHECK(wide.Wide() && !narrow.Wide());
    CHECK(wide.ApplyAction(Action::Place(13, 2)) && narrow.ApplyAction(Action::Place(13, 2)));
    bool fits = true, narrow_ok = true;
    for (const char *c = line; *c; ++c) {
        unsigned d = unsigned(std::string("URDL").find(*c));
        CHECK(wide.ApplyAction(Action::Slide(d)));
        fits = fits && WideBoard::From(wide.state().GetBoard()).Fits();
        fits = fits && Fit(wide.WideState()) == wide.state().GetBoard();
        bool ok = narrow.ApplyAction(Action::Slide(d));
        if (c[1]) narrow_ok = narrow_ok && ok;
        else CHECK(!ok);                                  // Board64: 32768 + 32768 refused
    }
    CHECK(fits && narrow_ok);
    CHECK(wide.WideState().MaxRank() == 16 && wide.WideMoves().size() == 15);
    CHECK(wide.WideMoves().back() == wide.WideState());
    CHECK(wide.score() == 131064 && narrow.score() == 65528); // every merge of the cascade, 4+4 .. 32768+32768
    CHECK(wide.state().GetBoard() == MakeBoard({0, 0, 0, 15}).GetBoard()); // agents see 65536 as 32768
    CHECK(wide.GetMoves().back().board == wide.state().GetBoard());
}

// an arena match keeps the real board on a WideBoard: two 32768s placed by the server merge
static void TestArenaWide() {
    auto dummy = std::make_shared<Agent>("name=x role=dummy");
    arena::match m("M1", dummy, dummy);
    CHECK(m.Wide());
    CHECK(m.ApplyAction(Action::Place(0, 15)) && m.ApplyAction(Action::Place(1, 15)));
    CHECK(m.state().GetBoard() == MakeBoard({14, 14}).GetBoard()); // what the agent searches
    CHECK(m.ApplyAction(Action::Slide(3)) && m.score() == 65536);
    CHECK(m.WideState().cell[0] == 16 && m.state().GetBoard() == MakeBoard({15}).GetBoard());
}

static void TestDowngradePlayer() {
    std::vector<std::vector<int>> small = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}};
    NTupleNetwork net(5, kFeatAll, small);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> w(-1000, 1000);
    for (int s = 0; s < 5; ++s) {
        net.InitStage(s);
        for (auto &table : net.StageWeights(s)) for (float &x : table) x = w(rng);
    }
    TrainingState st;
    st.active = 4;
    st.games = {1, 1, 1, 1, 1};
    std::string path = "/tmp/sixteen-downgrade.ckpt";
    if (const char *dir = std::getenv("TMPDIR")) path = std::string(dir) + "/sixteen-downgrade.ckpt";
    net.Save(path, st);

    for (const char *search : {"ply=1", "ply=2", "ply=3 tt=12", "time=10000 maxply=2 tt=12"}) {
        TdPlayer plain("load=" + path + " " + search);
        TdPlayer down(plain.shared_learner(), std::string("downgrade=32768 ") + search);
        TdPlayer fresh(plain.shared_learner(), search);
        int same = 0, differ_stage = 0, n = 0, bad = 0;
        uint64_t expected_downgrades = 0;
        for (int k = 0; k < 400; ++k) {
            Board64 b;
            for (int i = 0; i < 16; ++i) b.Set(i, rng() % 3 ? rng() % 12 : 0);
            if (k % 2) { b.Set(rng() % 16, 15); b.Set(rng() % 16, 14); b.Set(rng() % 16, 13); }
            if (b.IsTerminal()) continue;
            ++n;
            board_t d = DowngradeRoot(b.GetBoard(), 15);
            down.OpenEpisode();
            int got = down.TakeAction(b).event() & 0b11;
            plain.OpenEpisode();
            int want;
            if (d == b.GetBoard()) {
                want = plain.TakeAction(b).event() & 0b11;
                same++;
            } else {
                expected_downgrades++;
                // the real board's stage differs from the downgraded one's for many of these boards
                differ_stage += SplitLevel(b.GetBoard(), 4) != SplitLevel(d, 4);
                fresh.OpenEpisode();
                want = fresh.TakeAction(Board64(d)).event() & 0b11;
            }
            bad += got != want;
            Board64 c = b;
            bad += c.Slide(got) == -1;
        }
        CHECK(n > 300 && same > 50 && expected_downgrades > 50 && differ_stage > 20 && bad == 0);
        CHECK(down.TakeMoveStats().downgraded == expected_downgrades);
    }
    std::remove(path.c_str());
}

// TdPlayer::Explain (--explain): on the same sequence of boards, the first best move by its search
// values is the move TakeAction plays, with the same game stage carried from board to board; illegal
// moves are -inf; its network values are reward + the learner's afterstate value
static void TestExplain() {
    std::vector<std::vector<int>> small = {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}};
    NTupleNetwork net(5, kFeatAll, small);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> w(-1000, 1000);
    for (int s = 0; s < 5; ++s) {
        net.InitStage(s);
        for (auto &table : net.StageWeights(s)) for (float &x : table) x = w(rng);
    }
    TrainingState st;
    st.active = 4;
    st.games = {1, 1, 1, 1, 1};
    std::string path = "/tmp/sixteen-explain.ckpt";
    if (const char *dir = std::getenv("TMPDIR")) path = std::string(dir) + "/sixteen-explain.ckpt";
    net.Save(path, st);

    for (const char *search : {"ply=1", "ply=2 tt=12", "ply=1 deep=9,5 tt=12", "ply=2 deep=6 downgrade=32768 tt=12",
                               "ply=2 leafdg=32768 tt=12"}) {
        const bool leafdg = std::string(search).find("leafdg") != std::string::npos;
        TdPlayer play("load=" + path + " " + search);
        TdPlayer explain(play.shared_learner(), search);
        int n = 0, bad = 0, deeper = 0, downgraded = 0, illegal = 0;
        for (int game = 0; game < 4; ++game) {
            play.OpenEpisode();
            explain.OpenEpisode();
            // boards of rising tiles, so the game stage only ever grows within a game; some boards
            // early in a game reach a split, and later boards without its tiles keep that stage
            for (int k = 0; k < 120; ++k) {
                Board64 b;
                int top = 4 + k / 10;
                for (int i = 0; i < 16; ++i) b.Set(i, rng() % 10 ? 1 + rng() % (top - 1) : 0);
                if (k % 7 == 3) { b.Set(rng() % 16, 15); b.Set(rng() % 16, 14); b.Set(rng() % 16, 13); }
                if (k % 7 == 5) { b.Set(0, 15); b.Set(1, 14); b.Set(2, 14); } // Left makes two 32768s
                if (b.IsTerminal()) continue;
                ++n;
                int move = play.TakeAction(b).event() & 0b11;
                TdPlayer::Explanation e = explain.Explain(b);
                bad += int(std::max_element(e.search, e.search + 4) - e.search) != move;
                deeper += e.ply > 2;
                downgraded += e.downgraded;
                Board64 root = e.downgraded ? Board64(DowngradeRoot(b.GetBoard(), 15)) : b;
                for (int d = 0; d < 4; ++d) {
                    Board64 c = root;
                    reward_t r = c.Slide(d);
                    illegal += r == -1;
                    if (r == -1) {
                        bad += e.search[d] != -INFINITY || e.network[d] != -INFINITY;
                    } else {
                        float v = leafdg ? LeafDowngradeLearner{*play.shared_learner(), 15}.Evaluate(c.GetBoard(), e.stage)
                                         : play.shared_learner()->Evaluate(c.GetBoard(), e.stage);
                        bad += e.network[d] != r + v;
                        bad += e.search[d] == -INFINITY;
                    }
                }
            }
        }
        CHECK(n > 300 && illegal > 50 && bad == 0);
        if (std::string(search).find("deep") != std::string::npos) CHECK(deeper > 20);
        if (std::string(search).find("downgrade") != std::string::npos) CHECK(downgraded > 20);
    }
    std::remove(path.c_str());
}

// small checkpoint for learner tests: random stage-0 weights, stage 1 not created yet
static std::string SmallCheckpoint(const std::string &name, uint64_t games0) {
    NTupleNetwork net(2, kFeatAll, {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}});
    net.InitStage(0);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> w(-50, 50);
    for (auto &table : net.StageWeights(0)) for (float &x : table) x = w(rng);
    TrainingState st;
    st.active = 0;
    st.games = {games0, 0};
    std::string path = "/tmp/" + name;
    if (const char *dir = std::getenv("TMPDIR")) path = std::string(dir) + "/" + name;
    net.Save(path, st);
    return path;
}

// one game by a greedy player of `learner` against a seeded environment, and its afterstates
static Episode PlayGame(std::shared_ptr<TdLearner> learner, unsigned seed, std::vector<board_t> *boards) {
    TdPlayer player(learner, "ply=1");
    RandomEnvironment evil("seed=" + std::to_string(seed));
    Episode game;
    player.OpenEpisode();
    while (true) {
        Agent &agent = game.TakeTurns(player, evil);
        if (!game.ApplyAction(agent.TakeAction(game.state()))) break;
    }
    for (const Episode::Move &m : game.GetMoves()) boards->push_back(m.board);
    return game;
}

static std::shared_ptr<TdLearner> MakeLearner(const std::string &args) {
    std::map<std::string, std::string> kv;
    std::stringstream ss(args);
    for (std::string pair; ss >> pair;) kv[pair.substr(0, pair.find('='))] = pair.substr(pair.find('=') + 1);
    return std::make_shared<TdLearner>(kv);
}

static void TestCosineAnnealing() {
    std::string at0 = SmallCheckpoint("sixteen-cos0.ckpt", 0);
    std::string at500 = SmallCheckpoint("sixteen-cos500.ckpt", 500);
    std::string at1000 = SmallCheckpoint("sixteen-cos1000.ckpt", 1000);
    const std::string cos = " anneal=cos anneal_games=1000 alpha=0.01 alpha_min=0.001";

    // the schedule itself
    auto c = MakeLearner("load=" + at0 + cos);
    CHECK(c->Alpha(0) == 0.01f);
    CHECK(std::fabs(c->Alpha(500) - 0.0055f) < 1e-7f);
    CHECK(std::fabs(c->Alpha(250) - (0.001f + 0.009f * 0.5f * (1 + std::cos(M_PI / 4)))) < 1e-7f);
    CHECK(c->Alpha(1000) == 0.001f && c->Alpha(5000) == 0.001f);
    bool monotone = true;
    for (uint64_t g = 1; g <= 1100; ++g) monotone &= c->Alpha(g) <= c->Alpha(g - 1);
    CHECK(monotone);
    CHECK(MakeLearner("load=" + at500 + " alpha=0.01")->Alpha(500) == 0.01f); // off: alpha as given
    // anneal_from shifts the same curve: alpha until game F, then the curve at g - F
    auto shifted = MakeLearner("load=" + at0 + cos + " anneal_from=500");
    CHECK(shifted->Alpha(0) == 0.01f && shifted->Alpha(500) == 0.01f);
    CHECK(shifted->Alpha(750) == c->Alpha(250) && shifted->Alpha(1000) == c->Alpha(500));
    CHECK(shifted->Alpha(1500) == 0.001f && shifted->Alpha(9000) == 0.001f);
    CHECK(MakeLearner("load=" + at0 + cos + " anneal_from=0")->Alpha(300) == c->Alpha(300));

    // Learn uses the alpha of the stage's game count (carried by the checkpoint): a cosine learner
    // at game g learns exactly like a plain learner with alpha = Alpha(g)
    for (const std::string &ckpt : {at0, at500, at1000}) {
        auto annealed = MakeLearner("load=" + ckpt + cos);
        char alpha[32];
        std::snprintf(alpha, sizeof alpha, "%.9g", annealed->Alpha(annealed->ActiveGames()));
        auto plain = MakeLearner("load=" + ckpt + " alpha=" + alpha);
        auto other = MakeLearner("load=" + ckpt + " alpha=0.004"); // a different alpha must differ
        std::vector<board_t> boards;
        Episode game = PlayGame(plain, 3, &boards);
        annealed->Learn(game);
        plain->Learn(game);
        other->Learn(game);
        int differ = 0, differ_other = 0;
        for (board_t b : boards) {
            differ += annealed->Evaluate(b, 0) != plain->Evaluate(b, 0);
            differ_other += other->Evaluate(b, 0) != plain->Evaluate(b, 0);
        }
        CHECK(boards.size() > 100 && differ == 0 && differ_other > 0);
        CHECK(annealed->ActiveGames() == plain->ActiveGames());
    }
    // alpha follows the games learned in this run, not only the count loaded from the checkpoint:
    // over 2 games alpha anneals to 0, so a third game changes nothing
    auto quick = MakeLearner("load=" + at0 + " anneal=cos anneal_games=2 alpha=0.01 alpha_min=0");
    std::vector<board_t> boards;
    quick->Learn(PlayGame(quick, 4, &boards));
    quick->Learn(PlayGame(quick, 5, &boards));
    std::vector<float> before;
    for (board_t b : boards) before.push_back(quick->Evaluate(b, 0));
    std::vector<board_t> third;
    quick->Learn(PlayGame(quick, 6, &third));
    int changed = 0;
    for (size_t i = 0; i < boards.size(); ++i) changed += quick->Evaluate(boards[i], 0) != before[i];
    CHECK(quick->ActiveGames() == 3 && changed == 0);
    for (const std::string &p : {at0, at500, at1000}) std::remove(p.c_str());
}

// use=<map>: a stage's region is played by the mapped stage's weights; the identity map changes nothing
static void TestStageMap() {
    std::string ckpt = SmallCheckpoint("sixteen-use.ckpt", 7);
    auto base = MakeLearner("load=" + ckpt);                              // stage 0 plays everywhere
    auto own = MakeLearner("load=" + ckpt + " stage=1 init=500");          // stage 1 values all at 500
    auto same = MakeLearner("load=" + ckpt + " stage=1 init=500 use=0,1"); // identity map
    auto mapped = MakeLearner("load=" + ckpt + " stage=1 init=500 use=0,0");
    std::mt19937_64 rng(5);
    int bad = 0, split1 = 0;
    for (int k = 0; k < 4000; ++k) {
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 3 ? rng() % 15 : 0) << (4 * i);
        if (rng() % 2) b = (b & ~board_t(0xf)) | 14; // a 16384 in the corner: past the split
        split1 += SplitLevel(b, 1) == 1;
        for (int gs = 0; gs < 2; ++gs) {
            bad += mapped->Evaluate(b, gs) != base->Evaluate(b, gs);
            bad += same->Evaluate(b, gs) != own->Evaluate(b, gs);
        }
        if (SplitLevel(b, 1) == 1) bad += std::fabs(own->Evaluate(b, 1) - 500.0f) > 0.01f; // the map matters here
    }
    CHECK(split1 > 1000);
    CHECK(bad == 0);
    bool threw = false;
    try { MakeLearner("load=" + ckpt + " stage=1 use=1,1"); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw); // stage 0 cannot use a later stage
    threw = false;
    try { MakeLearner("load=" + ckpt + " stage=1 use=0,0 save=" + ckpt + ".x"); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw); // evaluation only
    std::remove(ckpt.c_str());
}

static void TestOptimisticInit() {
    std::string ckpt = SmallCheckpoint("sixteen-oi.ckpt", 7);
    auto base = MakeLearner("load=" + ckpt);
    auto zero = MakeLearner("load=" + ckpt + " stage=1");
    auto oi = MakeLearner("load=" + ckpt + " stage=1 init=500");
    std::mt19937_64 rng(3);
    int bad = 0;
    for (int k = 0; k < 2000; ++k) {
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 3 ? rng() % 15 : 0) << (4 * i);
        // the new stage values every board at V; stage 0 is untouched
        bad += std::fabs(oi->Evaluate(b, 1) - 500.0f) > 0.01f;
        bad += zero->Evaluate(b, 1) != 0.0f;
        if (SplitLevel(b, 1) == 0) bad += oi->Evaluate(b, 0) != base->Evaluate(b, 0);
    }
    CHECK(bad == 0);
    // init= does not touch a stage that already exists
    auto again = MakeLearner("load=" + ckpt + " init=500");
    Board64 b = MakeBoard({1, 2, 3, 0, 5});
    CHECK(again->Evaluate(b.GetBoard(), 0) == base->Evaluate(b.GetBoard(), 0));
    std::remove(ckpt.c_str());

    // a fresh network (default patterns, 4 x 8 tuples + 5 scalar features) starts at V too
    auto fresh = MakeLearner("stages=1 init=320000");
    CHECK(std::fabs(fresh->Evaluate(b.GetBoard(), 0) - 320000.0f) < 1.0f);
}

// generic fast path (any patterns) == Pattern::Index reference, for 8x6 and a small network;
// FeatureIndices lists the weights the reference sums, for default patterns too
static void TestGenericValue() {
    std::vector<board_t> boards = TestBoards();
    for (auto patterns : {NTupleNetwork::Patterns8x6(), std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}, {3, 6, 9}},
                          NTupleNetwork::DefaultPatterns()}) {
        NTupleNetwork net(1, kFeatAll, patterns);
        net.InitStage(0);
        uint64_t x = 88172645463325252ULL; // xorshift: fast enough for 134M weights
        for (std::vector<float> &table : net.StageWeights(0)) {
            for (float &w : table) {
                x ^= x << 13, x ^= x >> 7, x ^= x << 17;
                w = float(int64_t(x % 2000001) - 1000000) * 0.001f;
            }
        }
        int bad = 0, bad_list = 0, bad_lookup = 0;
        for (size_t k = 0; k < boards.size(); k += (patterns.size() == 8 ? 3 : 1)) {
            board_t b = boards[k];
            float ref = net.GetValueReference(b, 0);
            bad += net.GetValue(b, 0) != ref;
            uint32_t table[NTupleNetwork::kMaxFeatures];
            size_t index[NTupleNetwork::kMaxFeatures];
            int n = net.FeatureIndices(b, table, index);
            float sum = 0;
            for (int i = 0; i < n; ++i) sum += net.StageWeights(0)[table[i]][index[i]];
            bad_list += n != net.NumActiveFeatures() || sum != ref;
            NTupleNetwork::Lookup l; // the prefetching path: same weights, same order
            net.Prepare(b, l);
            net.Prefetch(l, 0);
            bad_lookup += net.GetValue(l, 0) != ref;
        }
        CHECK(bad == 0 && bad_list == 0 && bad_lookup == 0);
        CHECK(net.FastTuples() == (patterns == NTupleNetwork::DefaultPatterns()));
        CHECK(net.Tuples8x6() == (patterns == NTupleNetwork::Patterns8x6()));

        // updates: from zero weights, fast +1 then reference -1 restores all zeros
        for (std::vector<float> &t : net.StageWeights(0)) std::fill(t.begin(), t.end(), 0.0f);
        int moved = 0;
        for (size_t k = 0; k < boards.size(); k += 997) {
            net.UpdateValue(boards[k], 0, 1.0f);
            moved += net.GetValueReference(boards[k], 0) != 0.0f;
            net.UpdateValueReference(boards[k], 0, -1.0f);
            NTupleNetwork::Lookup l;
            net.Prepare(boards[k], l);
            net.UpdateValue(l, 0, 1.0f);
            moved += net.GetValueReference(boards[k], 0) != 0.0f;
            net.UpdateValueReference(boards[k], 0, -1.0f);
        }
        bad = 0;
        for (const std::vector<float> &t : net.StageWeights(0)) for (float w : t) bad += w != 0.0f;
        CHECK(bad == 0 && moved > 200);
    }
}

// TC update, eqs. 7-9 of arXiv:2111.11090, on a board whose weights are all distinct
static void TestTCUpdate() {
    NTupleNetwork net(1, kFeatAll, {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 4, 5}});
    net.InitStage(0);
    std::vector<std::vector<float>> E(net.NumTables()), A(net.NumTables());
    for (size_t t = 0; t < net.NumTables(); ++t) E[t].assign(net.TableSize(t), 0), A[t].assign(net.TableSize(t), 0);
    std::mt19937_64 rng(9);
    board_t b = 0;
    uint32_t table[NTupleNetwork::kMaxFeatures];
    size_t index[NTupleNetwork::kMaxFeatures];
    int n = 0;
    for (bool distinct = false; !distinct;) {
        b = rng();
        n = net.FeatureIndices(b, table, index);
        std::set<std::pair<uint32_t, size_t>> seen;
        for (int i = 0; i < n; ++i) seen.insert({table[i], index[i]});
        distinct = int(seen.size()) == n;
    }
    // errors 2, -1, 3 with alpha 0.5: beta = 1 (A = 0), then |2|/2 = 1, then |1|/3
    //   w: 0 -> 1 -> 0.5 -> 0.5 + 0.5 * (1/3) * 3 = 1;  E: 2, 1, 4;  A: 2, 3, 6
    net.UpdateValueTC(b, 0, 0.5f, 2.0f, E, A);
    net.UpdateValueTC(b, 0, 0.5f, -1.0f, E, A);
    net.UpdateValueTC(b, 0, 0.5f, 3.0f, E, A);
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += std::fabs(net.StageWeights(0)[table[i]][index[i]] - 1.0f) > 1e-6f;
        bad += E[table[i]][index[i]] != 4.0f || A[table[i]][index[i]] != 6.0f;
    }
    float untouched = 0;
    for (size_t t = 0; t < net.NumTables(); ++t) for (float w : net.StageWeights(0)[t]) untouched += std::fabs(w);
    CHECK(bad == 0 && std::fabs(untouched - float(n)) < 1e-3f);

    // negative E: beta uses |E|. errors -2, -1, 3: beta 1, 1, |-3|/3 = 1
    //   w: 0 -> -1 -> -1.5 -> 0;  E: -2, -3, 0;  A: 2, 3, 6   (beta = E/A would end at w = -2)
    for (size_t t = 0; t < net.NumTables(); ++t) {
        std::fill(net.StageWeights(0)[t].begin(), net.StageWeights(0)[t].end(), 0.0f);
        std::fill(E[t].begin(), E[t].end(), 0.0f);
        std::fill(A[t].begin(), A[t].end(), 0.0f);
    }
    net.UpdateValueTC(b, 0, 0.5f, -2.0f, E, A);
    net.UpdateValueTC(b, 0, 0.5f, -1.0f, E, A);
    net.UpdateValueTC(b, 0, 0.5f, 3.0f, E, A);
    bad = 0;
    for (int i = 0; i < n; ++i) {
        bad += std::fabs(net.StageWeights(0)[table[i]][index[i]]) > 1e-6f;
        bad += E[table[i]][index[i]] != 0.0f || A[table[i]][index[i]] != 6.0f;
    }
    CHECK(bad == 0);
}

// TdLearner tc=1: E and A survive a save and reload (a reloaded learner learns exactly like one
// that kept running), belong to one stage, and a non-TC save drops a stale .tc file
static void TestTCLearner() {
    std::string ckpt = SmallCheckpoint("sixteen-tc.ckpt", 0);
    std::string saved = ckpt + ".saved", resumed = ckpt + ".resumed";
    auto keep = MakeLearner("load=" + ckpt + " tc=1 alpha=0.02 save=" + saved);
    std::vector<board_t> boards;
    keep->Learn(PlayGame(keep, 21, &boards));
    keep->Save();
    std::ifstream probe(saved + ".tc");
    CHECK(bool(probe));
    auto reload = MakeLearner("load=" + saved + " tc=1 alpha=0.02 save=" + resumed);
    auto fresh_tc = MakeLearner("load=" + saved + " tc=1 alpha=0.02 stage=1"); // other stage: E, A at zero
    Episode second = PlayGame(keep, 22, &boards);
    keep->Learn(second);
    reload->Learn(second);
    auto plain = MakeLearner("load=" + saved + " alpha=0.02");
    plain->Learn(second);
    int differ = 0, differ_plain = 0;
    for (board_t b : boards) {
        differ += keep->Evaluate(b, 0) != reload->Evaluate(b, 0);
        differ_plain += keep->Evaluate(b, 0) != plain->Evaluate(b, 0);
    }
    CHECK(differ == 0 && differ_plain > 0);
    CHECK(reload->TCLoaded() && !keep->TCLoaded());
    CHECK(fresh_tc->ActiveStage() == 1 && !fresh_tc->TCLoaded()); // the .tc is for stage 0
    // a learner without tc saving over the same path removes the .tc that no longer matches
    auto no_tc = MakeLearner("load=" + saved + " save=" + saved);
    no_tc->Save();
    std::ifstream gone(saved + ".tc");
    CHECK(!gone);
    for (const std::string &p : {ckpt, saved, resumed, resumed + ".tc"}) std::remove(p.c_str());
}

static void TestUpgradeTile() {
    // stage 0 (next split needs 16384): the largest tile below 8192 is doubled
    CHECK(UpgradeTile(MakeBoard({12, 11, 3}).GetBoard(), 0) == MakeBoard({13, 11, 3}).GetBoard());
    CHECK(UpgradeTile(MakeBoard({13, 12, 5}).GetBoard(), 0) == MakeBoard({13, 13, 5}).GetBoard());
    CHECK(UpgradeTile(MakeBoard({13, 2}).GetBoard(), 0) == MakeBoard({13, 3}).GetBoard()); // never makes 16384
    CHECK(UpgradeTile(MakeBoard({13}).GetBoard(), 0) == MakeBoard({13}).GetBoard());
    // stage 1 (16384 present, next split needs 8192): 16384 stays, tiles up to 2048 are doubled
    CHECK(UpgradeTile(MakeBoard({14, 12, 11}).GetBoard(), 1) == MakeBoard({14, 12, 12}).GetBoard());
    CHECK(UpgradeTile(MakeBoard({14, 12}).GetBoard(), 1) == MakeBoard({14, 12}).GetBoard());
    CHECK(UpgradeTile(0, 0) == 0);
}

static void TestHardStarts() {
    auto learner = MakeLearner("stages=2");
    std::vector<board_t> boards;
    Episode game = PlayGame(learner, 5, &boards); // an untrained player dies long before 16384
    const auto &moves = game.GetMoves();
    size_t first = game.FirstSlide();
    size_t slides = (moves.size() - first + 1) / 2;
    auto split = [&](board_t b) { return learner->Split(b); };
    std::mt19937_64 rng(1);

    // rewind 1: the board right before the last slide, with the score made up to that point
    HardStarts pool(10, 1);
    pool.Offer(game, 0, split, rng);
    StartPosition p;
    CHECK(pool.Offered() == 1 && pool.Sample(rng, p));
    size_t idx = first + 2 * (slides - 1) - 1;
    CHECK(p.board == moves[idx].board);
    CHECK(moves[idx].code.type() != Action::Slide::type_ && moves[idx + 1].code.type() == Action::Slide::type_);
    reward_t score = 0;
    for (size_t i = 0; i <= idx; ++i) score += moves[i].reward;
    CHECK(p.score == float(score));
    Board64 next(p.board);
    CHECK(next.Slide(moves[idx + 1].code.event()) == moves[idx + 1].reward); // the game's own move replays

    // boards outside the active stage's region are not kept, and rewinds past the start are skipped
    HardStarts other(10, 1);
    other.Offer(game, 1, split, rng);
    CHECK(other.Offered() == 0 && !other.Sample(rng, p));
    HardStarts deep(10, unsigned(2 * slides)); // about half the rewinds land before the start
    for (int i = 0; i < 200; ++i) deep.Offer(game, 0, split, rng);
    CHECK(deep.Offered() > 0 && deep.Offered() < 200);

    // a full pool keeps the newest boards
    HardStarts ring(1, 1);
    Episode game2 = PlayGame(learner, 6, &boards);
    ring.Offer(game, 0, split, rng);
    ring.Offer(game2, 0, split, rng);
    CHECK(ring.Sample(rng, p) && p.board != moves[idx].board);
}

static std::string TmpPath(const std::string &name) {
    if (const char *dir = std::getenv("TMPDIR")) return std::string(dir) + "/" + name;
    return "/tmp/" + name;
}

// a 3-stage network with every stage present and random weights, plus its training state
static void RandomNetwork(NTupleNetwork &net, TrainingState &st, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> w(-50, 50);
    for (int s = 0; s < net.NumStages(); ++s) {
        net.InitStage(s);
        for (auto &table : net.StageWeights(s)) for (float &x : table) x = w(rng);
    }
    st.active = net.NumStages() - 1;
    st.games = {100, 200, 300};
}

static void TestMerge() {
    const std::vector<std::vector<int>> layout = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    NTupleNetwork base(3, kFeatAll, layout);
    TrainingState base_state;
    RandomNetwork(base, base_state, 5);
    std::string base_path = TmpPath("sixteen-merge-base.ckpt");
    base.Save(base_path, base_state);

    // two sessions from the base: one trains stage 1, the other stage 2 (as TdLearner would: only
    // the active stage's weights and game count change)
    auto session = [&](int stage, uint64_t extra, unsigned seed) {
        NTupleNetwork net(1, 0, {});
        TrainingState st;
        net.Load(base_path, st);
        std::mt19937_64 rng(seed);
        for (int i = 0; i < 2000; ++i) net.UpdateValue(board_t(rng()), stage, 0.5f);
        st.games[stage] += extra;
        st.active = stage;
        std::string path = TmpPath("sixteen-merge-s" + std::to_string(stage) + ".ckpt");
        net.Save(path, st);
        return path;
    };
    std::string p1 = session(1, 50, 1), p2 = session(2, 70, 2);
    NTupleNetwork s1(1, 0, {}), s2(1, 0, {});
    TrainingState t1, t2;
    s1.Load(p1, t1);
    s2.Load(p2, t2);

    std::string out = TmpPath("sixteen-merge-out.ckpt");
    std::ostringstream log;
    TrainingState returned = MergeFiles(base_path, {{1, p1}, {2, p2}}, out, log);
    NTupleNetwork merged(1, 0, {});
    TrainingState ms;
    merged.Load(out, ms);
    CHECK(merged.SameLayout(base));
    CHECK(merged.SameStage(base, 0) && merged.SameStage(s1, 1) && merged.SameStage(s2, 2));
    CHECK(!merged.SameStage(base, 1) && !merged.SameStage(base, 2));
    CHECK((ms.games == std::vector<uint64_t>{100, 250, 370}) && ms.active == 2 && returned.games == ms.games);
    CHECK(ms.alpha == base_state.alpha && ms.lambda == base_state.lambda);
    // values: a stage-k board is valued exactly as in the session that trained stage k
    std::mt19937_64 rng(9);
    int wrong = 0;
    for (int i = 0; i < 500; ++i) {
        board_t b = rng();
        wrong += merged.GetValue(b, 0) != base.GetValue(b, 0);
        wrong += merged.GetValue(b, 1) != s1.GetValue(b, 1);
        wrong += merged.GetValue(b, 2) != s2.GetValue(b, 2);
    }
    CHECK(wrong == 0);
    std::remove(out.c_str());

    // a bad part: MergeFiles throws and writes nothing, even when an earlier part was fine
    std::string bad = TmpPath("sixteen-merge-bad.ckpt");
    auto rejects = [&](const NTupleNetwork &part, const TrainingState &st, int stage) {
        part.Save(bad, st);
        std::remove(out.c_str());
        std::ostringstream quiet;
        bool threw = false;
        try {
            MergeFiles(base_path, {{3 - stage, stage == 1 ? p2 : p1}, {stage, bad}}, out, quiet);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        return threw && !std::ifstream(out);
    };
    CHECK(!rejects(s1, t1, 1));              // the good part itself is accepted
    CHECK(rejects(s1, t1, 2) && rejects(s2, t2, 1)); // a session given as the wrong stage
    TrainingState no_games = t1;
    no_games.games[1] = base_state.games[1]; // trained nothing (by its count)
    CHECK(rejects(s1, no_games, 1));
    TrainingState other_counts = t1;
    other_counts.games[0] += 1;              // stage 0 count differs: not from this base
    CHECK(rejects(s1, other_counts, 1));
    NTupleNetwork touched(1, 0, {});
    TrainingState tt;
    touched.Load(p1, tt);
    touched.UpdateValue(board_t(12345), 0, 1.0f); // stage 0 weights changed too
    CHECK(rejects(touched, tt, 1));
    NTupleNetwork other_layout(3, kFeatAll, {{0, 1, 2, 3}, {4, 5, 6, 8}});
    TrainingState ol;
    RandomNetwork(other_layout, ol, 5);
    ol.games[1] = 999;
    CHECK(rejects(other_layout, ol, 1));
    NTupleNetwork missing(1, 0, {});         // the part's own stage was never created
    TrainingState mt;
    missing.Load(base_path, mt);
    missing.DropStage(2);
    mt.games[2] = 999;
    CHECK(rejects(missing, mt, 2));
    bool twice = false;
    try { MergeFiles(base_path, {{1, p1}, {1, p1}}, out, log); } catch (const std::invalid_argument &) { twice = true; }
    CHECK(twice && !std::ifstream(out));
    std::remove(bad.c_str());
    for (const std::string &p : {base_path, p1, p2, out}) std::remove(p.c_str());
}

static void TestStartPositionsUnique() {
    std::string path = TmpPath("sixteen-boards-unique.bin");
    std::remove(path.c_str());
    // a file written before the check: board 7 three times (different scores), board 9 twice
    {
        std::ofstream out(path, std::ios::binary);
        out.write("SXBOARDS", 8);
        uint32_t stage = 2;
        out.write(reinterpret_cast<const char *>(&stage), 4);
        std::vector<std::pair<board_t, float>> recs = {{7, 1}, {9, 2}, {7, 3}, {11, 4}, {9, 5}, {7, 6}};
        for (auto &r : recs) {
            out.write(reinterpret_cast<const char *>(&r.first), 8);
            out.write(reinterpret_cast<const char *>(&r.second), 4);
        }
        out.write("xyz", 3); // torn trailing record
    }
    CHECK(StartPositions::Count(path) == 6);
    {
        StartPositions::Writer w(path, 2);
        CHECK(w.count() == 3 && w.dropped() == 3);
        CHECK(!w.Add(9, 10) && !w.Add(11, 11)); // already there
        CHECK(w.Add(13, 12) && w.count() == 4);
        CHECK(!w.Add(13, 14) && w.repeats() == 3 && w.count() == 4);
    }
    int stage = -1;
    std::vector<StartPosition> got = StartPositions::Load(path, stage);
    CHECK(stage == 2 && got.size() == 4 && StartPositions::Count(path) == 4);
    // first copy of each board kept, in file order, with its own score
    CHECK(got.size() == 4 && got[0].board == 7 && got[0].score == 1 && got[1].board == 9 && got[1].score == 2 &&
          got[2].board == 11 && got[2].score == 4 && got[3].board == 13 && got[3].score == 12);
    {
        StartPositions::Writer w(path, 2); // reopening a clean file changes nothing
        CHECK(w.count() == 4 && w.dropped() == 0 && !w.Add(7, 1));
    }
    CHECK(StartPositions::Count(path) == 4);
    bool threw = false;
    try { StartPositions::Writer w(path, 3); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw);
    std::remove(path.c_str());
}

int main() {
    InitLookUpTables();
    TestGenericValue();
    TestTCUpdate();
    TestTCLearner();
    TestCosineAnnealing();
    TestOptimisticInit();
    TestStageMap();
    TestDowngrade();
    TestWideBoard();
    TestFit();
    TestWideEpisode();
    TestArenaWide();
    TestDowngradePlayer();
    TestExplain();
    TestSearchExact();
    TestSearchLearner();
    TestLeafDowngrade();
    TestSearchTimed();
    TestSearchApprox();
    TestAdaptivePly();
    TestFastTupleIndices();
    TestFastScalarFeatures();
    TestFastValue();
    TestSlides();
    TestSymmetry();
    TestSplits();
    TestScalarFeatures();
    TestLambdaReturn();
    TestCheckpoint();
    TestUpgradeTile();
    TestHardStarts();
    TestMerge();
    TestStartPositionsUnique();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
