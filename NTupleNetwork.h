#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Board64.h"

/**
 * Value function of Yeh et al., "Multi-Stage Temporal Difference Learning for 2048-like Games"
 * (arXiv:1606.07374), referred to below as [MS-TD].
 *
 *   (i)  four 6-tuples of Fig. 4(b), each over all 8 rotations/reflections        (Sec. II.D)
 *   (ii) the large-tile feature (n2048, n4096, n8k, n16k, n32k)                   (Sec. IV.A)
 *   (a)  number of empty tiles, distinct tiles, mergeable pairs, v/2v neighbor pairs (Sec. IV.E)
 *
 * A state is valued as the sum of the weights of all its occurring features.
 */

enum FeatureFlag : uint32_t {
    kFeatLargeTile = 1u << 0,
    kFeatEmpty = 1u << 1,
    kFeatDistinct = 1u << 2,
    kFeatMergeable = 1u << 3,
    kFeatNeighbor = 1u << 4,
    kFeatAll = (1u << 5) - 1,
};

/**
 * Stage splitting of [MS-TD] Sec. IV.D. Split k (1-based) happens the first time the board holds
 * all of 16384, 8192, ... down to kSplitRanks[k-1] at once:
 *   T_16k, T_16+8k, T_16+8+4k, T_16+8+4+2k.
 * 3 stages is the "simple 3-stage strategy", 5 stages is Strategy 3 (the best one in the paper).
 */
static const int kSplitRanks[4] = {14, 13, 12, 11};
static const int kMaxStages = 5;

// Highest split condition the board satisfies right now (0..n_splits). Stages are monotone in
// game time, so callers combine this with the stage the game already reached.
inline int SplitLevel(board_t b, int n_splits) {
    unsigned present = 0;
    for (int i = 0; i < 16; ++i) present |= 1u << ((b >> (4 * i)) & 0xf);
    int k = 0;
    while (k < n_splits && ((present >> kSplitRanks[k]) & 1)) ++k;
    return k;
}

class Pattern {
public:
    explicit Pattern(const std::vector<int> &cells) : cells_(cells) {
        for (int s = 0; s < 8; ++s) {
            std::vector<int> iso;
            for (int cell : cells) iso.push_back(Transform(cell, s));
            isomorphisms_.push_back(iso);
        }
    }

    const std::vector<int> &cells() const { return cells_; }

    size_t TableSize() const { return size_t(1) << (4 * cells_.size()); }

    size_t Index(board_t b, int iso) const {
        const std::vector<int> &cells = isomorphisms_[iso];
        size_t index = 0;
        for (size_t k = 0; k < cells.size(); ++k) {
            index |= size_t((b >> (4 * cells[k])) & 0xf) << (4 * k);
        }
        return index;
    }

    // symmetry s: rotate clockwise (s & 3) times, then mirror left-right if (s & 4)
    static int Transform(int cell, int s) {
        int r = cell / 4, c = cell % 4;
        for (int i = 0; i < (s & 3); ++i) {
            int nr = c, nc = 3 - r;
            r = nr;
            c = nc;
        }
        if (s & 4) c = 3 - c;
        return r * 4 + c;
    }

private:
    std::vector<int> cells_;
    std::vector<std::vector<int>> isomorphisms_;
};

/**
 * Fast evaluation helpers. Instead of reading 6 cells per tuple through a per-symmetry cell list,
 * build the 8 symmetric images of the board with bit operations and read each default tuple with
 * one or two masks. The images are reordered to match Pattern's symmetry order, so the same weights
 * are summed in the same order as the reference code (bit-identical results, see tests.cpp).
 */
namespace fast {

inline board_t MirrorRows(board_t b) { // reverse the 4 cells within each row
    return ((b & 0x000F000F000F000FULL) << 12) | ((b & 0x00F000F000F000F0ULL) << 4) |
           ((b & 0x0F000F000F000F00ULL) >> 4) | ((b & 0xF000F000F000F000ULL) >> 12);
}

inline board_t FlipRows(board_t b) { // reverse the order of the 4 rows
    return (b << 48) | ((b & 0xFFFF0000ULL) << 16) | ((b >> 16) & 0xFFFF0000ULL) | (b >> 48);
}

// the 8 images in a fixed order: id, mirror, flip, mirror+flip, then the transpose of each
inline void Images(board_t b, board_t *img) {
    board_t m = MirrorRows(b), v = FlipRows(b), mv = MirrorRows(v);
    img[0] = b;
    img[1] = m;
    img[2] = v;
    img[3] = mv;
    img[4] = Transpose(b);
    img[5] = Transpose(m);
    img[6] = Transpose(v);
    img[7] = Transpose(mv);
}

// image order -> Pattern symmetry order: Pattern::Index(b, s) reads the cells of ImageOf(b, s)
struct SymmetryOrder {
    int image[8];

    SymmetryOrder() {
        board_t b = 0;
        for (int i = 0; i < 16; ++i) b |= board_t(i) << (4 * i); // distinct ranks identify cells
        board_t img[8];
        Images(b, img);
        for (int s = 0; s < 8; ++s) {
            board_t want = 0; // want(cell) = b(Transform(cell, s))
            for (int cell = 0; cell < 16; ++cell) want |= board_t(Pattern::Transform(cell, s)) << (4 * cell);
            image[s] = -1;
            for (int k = 0; k < 8; ++k) {
                if (img[k] == want) image[s] = k;
            }
            if (image[s] < 0) throw std::logic_error("symmetry images do not match Pattern::Transform");
        }
    }
};

inline const SymmetryOrder &Order() {
    static const SymmetryOrder order;
    return order;
}

// the 4 default 6-tuples (NTupleNetwork::DefaultPatterns) read from one image
inline size_t Tuple0(board_t b) { return b & 0xFFFFFF; }                                // {0,1,2,3,4,5}
inline size_t Tuple1(board_t b) { return (b >> 16) & 0xFFFFFF; }                        // {4,5,6,7,8,9}
inline size_t Tuple2(board_t b) { return (b & 0xFFF) | ((b >> 4) & 0xFFF000); }          // {0,1,2,4,5,6}
inline size_t Tuple3(board_t b) { return ((b >> 16) & 0xFFF) | ((b >> 20) & 0xFFF000); } // {4,5,6,8,9,10}

/**
 * Per-row facts for the scalar features, packed in one u64 per 16-bit row:
 *   bits  0..15  presence mask of ranks (bit r set if some cell holds rank r, including 0)
 *   bits 16..45  counts of ranks 11..15 (2048..32768), 6 bits each
 *   bits 46..50  empty cells
 *   bits 51..55  equal adjacent non-empty pairs within the row
 *   bits 56..60  adjacent non-empty pairs differing by one rank
 * Each count field is wide enough for the sum over the 4 rows plus, for the pair fields, the 4
 * columns (16, 16, 24, 24), so the parts above bit 16 can be added as plain integers. Columns use the table on the transposed board.
 */
struct RowFacts {
    static constexpr int kLarge = 16, kEmpty = 46, kMerge = 51, kNeighbor = 56;
    uint64_t row[65536];

    RowFacts() {
        for (int r = 0; r < 65536; ++r) {
            int c[4];
            for (int k = 0; k < 4; ++k) c[k] = (r >> (4 * k)) & 0xf;
            uint64_t presence = 0, large = 0, empty = 0, merge = 0, neighbor = 0;
            for (int k = 0; k < 4; ++k) {
                presence |= 1u << c[k];
                if (c[k] >= 11) large += uint64_t(1) << (6 * (c[k] - 11));
                empty += c[k] == 0;
                if (k < 3 && c[k] && c[k + 1]) {
                    merge += c[k] == c[k + 1];
                    neighbor += (c[k] == c[k + 1] + 1 || c[k + 1] == c[k] + 1);
                }
            }
            row[r] = presence | (large << kLarge) | (empty << kEmpty) | (merge << kMerge) | (neighbor << kNeighbor);
        }
    }
};

inline const RowFacts &Rows() {
    static const RowFacts *rows = new RowFacts(); // 512 KB, built once, never freed
    return *rows;
}

} // namespace fast

/**
 * Metadata saved with the weights so training resumes across sessions.
 */
struct TrainingState {
    float alpha = 0.0025f;          // per-weight step size, [MS-TD] eq. (5)
    float lambda = 0.5f;            // [MS-TD] Sec. IV.E (c)
    int active = 0;                 // stage being trained / highest stage used for play
    std::vector<uint64_t> games;    // training games per stage
};

class NTupleNetwork {
public:
    static constexpr uint32_t kVersion = 2;

    // [MS-TD] Fig. 4(b): two knife-shaped tuples (a 4-cell line plus 2 cells beside one end)
    // and two 2x3 rectangles. Up to symmetry these equal the Threes AxeTuple and RectangleTuple.
    static std::vector<std::vector<int>> DefaultPatterns() {
        return {
                {0, 1, 2, 3, 4, 5},
                {4, 5, 6, 7, 8, 9},
                {0, 1, 2, 4, 5, 6},
                {4, 5, 6, 8, 9, 10},
        };
    }

    NTupleNetwork(int stages = kMaxStages, uint32_t features = kFeatAll,
                  const std::vector<std::vector<int>> &patterns = DefaultPatterns()) {
        Configure(patterns, stages, features);
    }

    int NumStages() const { return n_stages_; }

    int NumSplits() const { return n_stages_ - 1; }

    uint32_t Features() const { return features_; }

    int NumActiveFeatures() const { return int(patterns_.size()) * 8 + __builtin_popcount(features_); }

    bool Initialized(int stage) const { return initialized_[stage]; }

    // allocates a stage with every weight = value: 0 by default ([MS-TD] Sec. III: "feature weights are
    // all initialized to be zero"), or an optimistic value (arXiv:2111.11090)
    void InitStage(int stage, float value = 0.0f) {
        if (initialized_[stage]) return;
        for (size_t t = 0; t < weights_[stage].size(); ++t) weights_[stage][t].assign(TableSize(t), value);
        initialized_[stage] = true;
    }

    /*
     * Hogwild training: several threads read and update the same weights without locks.
     * Every weight access goes through a relaxed std::atomic_ref, which compiles to plain
     * loads/stores on x86 but keeps concurrent access well defined (no torn values).
     * A concurrent read-modify-write on the same weight can lose one of the updates; that is
     * the accepted Hogwild trade-off. Save() must not run while workers update (see run()).
     */
    static float LoadWeight(const float &w) { return std::atomic_ref<const float>(w).load(std::memory_order_relaxed); }

    static void AddWeight(float &w, float delta) {
        std::atomic_ref<float> ref(w);
        ref.store(ref.load(std::memory_order_relaxed) + delta, std::memory_order_relaxed);
    }

    // raw weight tables of one stage (tests and tools)
    std::vector<std::vector<float>> &StageWeights(int stage) { return weights_[stage]; }

    // true when the default patterns are used, so GetValue/UpdateValue take the mask-based path
    bool FastTuples() const { return fast_tuples_; }

    float GetValue(board_t b, int stage) const {
        if (!fast_tuples_) return GetValueReference(b, stage);
        const std::vector<std::vector<float>> &w = weights_[stage];
        size_t idx[4][8];
        TupleIndices(b, idx);
        float value = 0;
        for (int p = 0; p < 4; ++p) {
            const float *table = w[p].data();
            for (int iso = 0; iso < 8; ++iso) value += LoadWeight(table[idx[p][iso]]);
        }
        if (features_) {
            size_t sidx[5];
            int n = ScalarIndices(b, sidx);
            for (int j = 0; j < n; ++j) value += LoadWeight(w[4 + j][sidx[j]]);
        }
        return value;
    }

    // adds delta to the weight of every occurring feature, i.e. delta = alpha * td_error
    void UpdateValue(board_t b, int stage, float delta) {
        if (!fast_tuples_) return UpdateValueReference(b, stage, delta);
        std::vector<std::vector<float>> &w = weights_[stage];
        size_t idx[4][8];
        TupleIndices(b, idx);
        for (int p = 0; p < 4; ++p) {
            float *table = w[p].data();
            for (int iso = 0; iso < 8; ++iso) AddWeight(table[idx[p][iso]], delta);
        }
        if (features_) {
            size_t sidx[5];
            int n = ScalarIndices(b, sidx);
            for (int j = 0; j < n; ++j) AddWeight(w[4 + j][sidx[j]], delta);
        }
    }

    // Reference versions: read every cell through Pattern::Index. Used for non-default patterns
    // and by tests.cpp to check that the fast path reads exactly the same weights.
    float GetValueReference(board_t b, int stage) const {
        const std::vector<std::vector<float>> &w = weights_[stage];
        float value = 0;
        for (size_t p = 0; p < patterns_.size(); ++p) {
            const float *table = w[p].data();
            for (int iso = 0; iso < 8; ++iso) value += LoadWeight(table[patterns_[p].Index(b, iso)]);
        }
        if (features_) {
            size_t idx[5];
            int n = ScalarIndicesReference(b, idx);
            for (int j = 0; j < n; ++j) value += LoadWeight(w[patterns_.size() + j][idx[j]]);
        }
        return value;
    }

    void UpdateValueReference(board_t b, int stage, float delta) {
        std::vector<std::vector<float>> &w = weights_[stage];
        for (size_t p = 0; p < patterns_.size(); ++p) {
            float *table = w[p].data();
            for (int iso = 0; iso < 8; ++iso) AddWeight(table[patterns_[p].Index(b, iso)], delta);
        }
        if (features_) {
            size_t idx[5];
            int n = ScalarIndicesReference(b, idx);
            for (int j = 0; j < n; ++j) AddWeight(w[patterns_.size() + j][idx[j]], delta);
        }
    }

    // idx[p][s] == patterns_[p].Index(b, s) for the default patterns
    static void TupleIndices(board_t b, size_t idx[4][8]) {
        board_t img[8];
        fast::Images(b, img);
        const int *order = fast::Order().image;
        for (int s = 0; s < 8; ++s) {
            board_t x = img[order[s]];
            idx[0][s] = fast::Tuple0(x);
            idx[1][s] = fast::Tuple1(x);
            idx[2][s] = fast::Tuple2(x);
            idx[3][s] = fast::Tuple3(x);
        }
    }

    // indices of the enabled scalar features, in flag order; returns how many. Same results as
    // ScalarIndicesReference, from per-row tables of the board and of its transpose.
    int ScalarIndices(board_t b, size_t *idx) const {
        using F = fast::RowFacts;
        const uint64_t *rows = fast::Rows().row;
        uint64_t r0 = rows[b & 0xFFFF], r1 = rows[(b >> 16) & 0xFFFF];
        uint64_t r2 = rows[(b >> 32) & 0xFFFF], r3 = rows[b >> 48];
        uint64_t sum = (r0 >> 16) + (r1 >> 16) + (r2 >> 16) + (r3 >> 16); // count fields of all rows
        if (features_ & (kFeatMergeable | kFeatNeighbor)) {                // pairs inside columns
            board_t t = Transpose(b);
            sum += (rows[t & 0xFFFF] >> F::kMerge << (F::kMerge - 16)) +
                   (rows[(t >> 16) & 0xFFFF] >> F::kMerge << (F::kMerge - 16)) +
                   (rows[(t >> 32) & 0xFFFF] >> F::kMerge << (F::kMerge - 16)) +
                   (rows[t >> 48] >> F::kMerge << (F::kMerge - 16)); // row + column pairs <= 24 fits
        }
        auto field = [sum](int shift, int bits) { return size_t((sum >> (shift - 16)) & ((1u << bits) - 1)); };
        int n = 0;
        if (features_ & kFeatLargeTile) {
            size_t index = 0;
            for (int k = 0; k < 5; ++k) index |= std::min<size_t>(field(F::kLarge + 6 * k, 6), 15) << (4 * k);
            idx[n++] = index;
        }
        if (features_ & kFeatEmpty) idx[n++] = field(F::kEmpty, 5);
        if (features_ & kFeatDistinct) {
            uint64_t presence = (r0 | r1 | r2 | r3) & 0xFFFE; // rank 0 (empty) is not a tile
            idx[n++] = size_t(__builtin_popcountll(presence));
        }
        if (features_ & kFeatMergeable) idx[n++] = field(F::kMerge, 5);
        if (features_ & kFeatNeighbor) idx[n++] = field(F::kNeighbor, 5);
        return n;
    }

    /**
     * Checkpoint layout, version 2 (little endian):
     *   char[8] "SIXTEEN\0", u32 version
     *   u32 n_patterns, then per pattern: u32 n_cells, u8 cells[n_cells]
     *   u32 feature flags, u32 n_stages (splits are kSplitRanks[0..n_stages-2]), u32 active stage
     *   per stage: u8 initialized, u64 games
     *   f32 alpha, f32 lambda
     *   f32 weights of each initialized stage: tuple tables, then enabled scalar tables in flag order
     * Written to "<path>.tmp" then renamed, so a killed session never leaves a torn file.
     */
    void Save(const std::string &path, const TrainingState &state) const {
        std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open " + tmp + " for writing");
            out.write("SIXTEEN", 8);
            WritePod(out, kVersion);
            WritePod(out, uint32_t(patterns_.size()));
            for (const Pattern &p : patterns_) {
                WritePod(out, uint32_t(p.cells().size()));
                for (int c : p.cells()) WritePod(out, uint8_t(c));
            }
            WritePod(out, features_);
            WritePod(out, uint32_t(n_stages_));
            WritePod(out, uint32_t(state.active));
            for (int s = 0; s < n_stages_; ++s) {
                WritePod(out, uint8_t(initialized_[s]));
                WritePod(out, uint64_t(s < int(state.games.size()) ? state.games[s] : 0));
            }
            WritePod(out, state.alpha);
            WritePod(out, state.lambda);
            for (int s = 0; s < n_stages_; ++s) {
                if (!initialized_[s]) continue;
                for (const std::vector<float> &table : weights_[s]) {
                    out.write(reinterpret_cast<const char *>(table.data()), table.size() * sizeof(float));
                }
            }
            out.flush();
            if (!out) throw std::runtime_error("write failed for " + tmp);
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("cannot rename " + tmp + " to " + path);
        }
    }

    // Loads a checkpoint; the network is reconfigured to whatever it holds.
    void Load(const std::string &path, TrainingState &state) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) throw std::runtime_error("cannot open checkpoint " + path);
        std::vector<bool> init = ReadHeader(in, path, state);
        for (int s = 0; s < n_stages_; ++s) {
            if (!init[s]) continue;
            InitStage(s);
            for (std::vector<float> &table : weights_[s]) {
                in.read(reinterpret_cast<char *>(table.data()), table.size() * sizeof(float));
            }
        }
        if (!in) throw std::runtime_error("checkpoint " + path + " is truncated");
    }

    // Reads only the header (cheap, no weights); returns which stages the file holds. Used by --info.
    std::vector<bool> LoadHeader(const std::string &path, TrainingState &state) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) throw std::runtime_error("cannot open checkpoint " + path);
        return ReadHeader(in, path, state);
    }

    void PrintSummary(std::ostream &out) const {
        size_t bytes = 0;
        for (size_t t = 0; t < weights_[0].size(); ++t) bytes += TableSize(t) * sizeof(float);
        out << "network: " << patterns_.size() << " 6-tuples x 8 isomorphisms, features=0x" << std::hex << features_
            << std::dec << ", " << n_stages_ << " stage(s), " << (bytes >> 20) << " MB per stage, initialized:";
        for (int s = 0; s < n_stages_; ++s) out << " " << initialized_[s];
        out << std::endl;
    }

private:
    void Configure(const std::vector<std::vector<int>> &patterns, int stages, uint32_t features) {
        if (stages < 1 || stages > kMaxStages) throw std::invalid_argument("stages must be 1 to 5");
        if (features & ~kFeatAll) throw std::invalid_argument("unknown feature flags");
        patterns_.clear();
        for (const std::vector<int> &p : patterns) patterns_.emplace_back(p);
        fast_tuples_ = patterns == DefaultPatterns();
        n_stages_ = stages;
        features_ = features;
        size_t n_tables = patterns_.size() + __builtin_popcount(features_);
        weights_.assign(n_stages_, std::vector<std::vector<float>>(n_tables));
        initialized_.assign(n_stages_, false);
    }

    std::vector<bool> ReadHeader(std::istream &in, const std::string &path, TrainingState &state) {
        char magic[8];
        in.read(magic, 8);
        if (!in || std::memcmp(magic, "SIXTEEN", 8) != 0) throw std::runtime_error(path + " is not a sixteen checkpoint");
        uint32_t version = ReadPod<uint32_t>(in);
        if (version != kVersion) throw std::runtime_error("unsupported checkpoint version " + std::to_string(version));

        std::vector<std::vector<int>> patterns(ReadPod<uint32_t>(in));
        for (std::vector<int> &p : patterns) {
            p.resize(ReadPod<uint32_t>(in));
            for (int &c : p) c = ReadPod<uint8_t>(in);
        }
        uint32_t features = ReadPod<uint32_t>(in);
        int stages = int(ReadPod<uint32_t>(in));
        Configure(patterns, stages, features);
        state.active = int(ReadPod<uint32_t>(in));
        state.games.assign(n_stages_, 0);
        std::vector<bool> init(n_stages_);
        for (int s = 0; s < n_stages_; ++s) {
            init[s] = ReadPod<uint8_t>(in) != 0;
            state.games[s] = ReadPod<uint64_t>(in);
        }
        state.alpha = ReadPod<float>(in);
        state.lambda = ReadPod<float>(in);
        return init;
    }

    size_t TableSize(size_t t) const {
        if (t < patterns_.size()) return patterns_[t].TableSize();
        // the (t - n_patterns)-th enabled scalar feature
        int j = int(t - patterns_.size());
        for (uint32_t bit = 1; bit <= kFeatAll; bit <<= 1) {
            if (!(features_ & bit)) continue;
            if (j-- == 0) return ScalarTableSize(bit);
        }
        throw std::logic_error("bad table index");
    }

    static size_t ScalarTableSize(uint32_t bit) {
        switch (bit) {
            case kFeatLargeTile: return size_t(1) << 20; // 5 counts, 4 bits each
            case kFeatEmpty: return 17;                  // 0..16 empty cells
            case kFeatDistinct: return 16;               // 0..15 distinct tile values
            case kFeatMergeable: return 25;              // 0..24 adjacent pairs
            case kFeatNeighbor: return 25;
        }
        return 0;
    }

public:
    // reference for ScalarIndices: indices of the enabled scalar features, in flag order
    int ScalarIndicesReference(board_t b, size_t *idx) const {
        int count[16] = {0};
        for (int i = 0; i < 16; ++i) count[(b >> (4 * i)) & 0xf]++;
        int mergeable = 0, neighbor = 0;
        if (features_ & (kFeatMergeable | kFeatNeighbor)) {
            for (int i = 0; i < 16; ++i) {
                int a = int((b >> (4 * i)) & 0xf);
                if (!a) continue;
                int adj[2] = {(i % 4 != 3) ? i + 1 : -1, (i < 12) ? i + 4 : -1}; // right, down
                for (int j : adj) {
                    if (j < 0) continue;
                    int c = int((b >> (4 * j)) & 0xf);
                    if (!c) continue;
                    mergeable += (a == c);
                    neighbor += (a == c + 1 || c == a + 1);
                }
            }
        }
        int n = 0;
        if (features_ & kFeatLargeTile) {
            size_t index = 0;
            for (int k = 0; k < 5; ++k) index |= size_t(std::min(count[11 + k], 15)) << (4 * k); // 2048..32768
            idx[n++] = index;
        }
        if (features_ & kFeatEmpty) idx[n++] = size_t(count[0]);
        if (features_ & kFeatDistinct) {
            int distinct = 0;
            for (int r = 1; r < 16; ++r) distinct += count[r] > 0;
            idx[n++] = size_t(distinct);
        }
        if (features_ & kFeatMergeable) idx[n++] = size_t(mergeable);
        if (features_ & kFeatNeighbor) idx[n++] = size_t(neighbor);
        return n;
    }

private:

    template<class T>
    static void WritePod(std::ostream &out, T v) { out.write(reinterpret_cast<const char *>(&v), sizeof(T)); }

    template<class T>
    static T ReadPod(std::istream &in) {
        T v;
        in.read(reinterpret_cast<char *>(&v), sizeof(T));
        if (!in) throw std::runtime_error("checkpoint is truncated");
        return v;
    }

    std::vector<Pattern> patterns_;
    int n_stages_ = 1;
    uint32_t features_ = 0;
    bool fast_tuples_ = false;
    std::vector<std::vector<std::vector<float>>> weights_; // [stage][table][index]
    std::vector<bool> initialized_;
};
