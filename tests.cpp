// Sanity tests: bitboard moves vs. a naive 2048 reference, symmetries, checkpoint round trip.
// Build and run with `make test`. Uses small patterns so it needs only a few MB.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>

#include "Agent.h"
#include "HardStarts.h"
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
// with 3 <= r < top), or -1 when downgrading does not apply
static int NaiveSkippedRank(board_t b, int min_rank) {
    int top = 0, n_max = 0;
    bool present[16] = {};
    for (int i = 0; i < 16; ++i) {
        int x = int((b >> (4 * i)) & 0xf);
        present[x] = true;
        top = std::max(top, x);
        n_max += x == 15;
    }
    if (top < min_rank || n_max >= 2) return -1;
    for (int k = top - 1; k >= 3; --k) if (!present[k]) return k;
    return -1;
}

static board_t NaiveDowngrade(board_t b, int min_rank) {
    int r = NaiveSkippedRank(b, min_rank);
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
    CHECK(DowngradeRoot(full.GetBoard(), 15) == full.GetBoard());                    // only 2 or 4 missing
    Board64 two_max = MakeBoard({15, 15, 3});
    CHECK(DowngradeRoot(two_max.GetBoard(), 15) == two_max.GetBoard());              // 32768 pair: skipped

    // random boards with large tiles: same as the naive version, and every move is legal on both
    // boards and gives the downgraded result of the real move
    std::mt19937_64 rng(77);
    int applied = 0, bad = 0;
    for (int k = 0; k < 200000; ++k) {
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(rng() % 4 ? rng() % 16 : 0) << (4 * i);
        if (k % 2) b = (b & ~board_t(0xf)) | 15; // half of them hold a 32768 in cell 0
        board_t d = DowngradeRoot(b, 15);
        bad += d != NaiveDowngrade(b, 15);
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
    CHECK(applied > 10000 && bad == 0);
}

// TdPlayer with downgrade=: the move is the plain player's move on the downgraded board, valued
// with the stage of the downgraded board; boards it does not apply to are played as before
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

int main() {
    InitLookUpTables();
    TestGenericValue();
    TestTCUpdate();
    TestTCLearner();
    TestCosineAnnealing();
    TestOptimisticInit();
    TestDowngrade();
    TestDowngradePlayer();
    TestSearchExact();
    TestSearchTimed();
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
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
