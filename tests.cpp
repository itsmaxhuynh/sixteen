// Sanity tests: bitboard moves vs. a naive 2048 reference, symmetries, checkpoint round trip.
// Build and run with `make test`. Uses small patterns so it needs only a few MB.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "Agent.h"
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

// plain expectimax written out as in TdLearner (no cache, no time limit)
static float RefMax(const FakeLearner &l, const Board64 &b, int ply, int stage, int *move);
static float RefChance(const FakeLearner &l, const Board64 &after, int ply, int stage) {
    stage = std::max(stage, l.Split(after.GetBoard()));
    float total = 0;
    int empty = 0;
    for (int i = 0; i < 16; ++i) {
        if (after(i)) continue;
        ++empty;
        Board64 two = after, four = after;
        two.Set(i, 1);
        four.Set(i, 2);
        total += 0.9f * RefMax(l, two, ply, stage, nullptr);
        total += 0.1f * RefMax(l, four, ply, stage, nullptr);
    }
    return empty ? total / empty : 0.0f;
}
static float RefMax(const FakeLearner &l, const Board64 &b, int ply, int stage, int *move) {
    int dir = -1;
    float best = -INFINITY;
    for (int d = 0; d < 4; ++d) {
        Board64 c = b;
        reward_t r = c.Slide(d);
        if (r == -1) continue;
        float v = r + (ply <= 1 ? l.Evaluate(c.GetBoard(), stage) : RefChance(l, c, ply - 1, stage));
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

int main() {
    InitLookUpTables();
    TestSearchExact();
    TestSearchTimed();
    TestFastTupleIndices();
    TestFastScalarFeatures();
    TestFastValue();
    TestSlides();
    TestSymmetry();
    TestSplits();
    TestScalarFeatures();
    TestLambdaReturn();
    TestCheckpoint();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
