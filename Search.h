#pragma once

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "Board64.h"

/**
 * Tile-downgrading (Guei, Chen, Wu, "Optimistic Temporal Difference Learning for 2048",
 * arXiv:2111.11090, Sec. IV.C.3), applied to the root of a search only. If the largest tile has
 * rank >= min_rank (15 = 32768 in the paper) and some rank r below it is missing, every tile above
 * r is halved. The downgraded board keeps the board's structure (which cells are empty and which
 * neighbours are equal), so a move chosen on it is legal and does the same merges on the real
 * board, while its large tiles are ones the network has seen far more often in training.
 *
 * r is the largest missing rank, as in the paper's example. We also require r >= 3 (8-tile), our
 * choice: then 2- and 4-tiles keep their meaning, so the spawns inside the search stay exact.
 * Also skipped with two or more MAX_RANK tiles: they cannot merge on our board (4-bit cells) but
 * would after halving, so a move could be legal on the downgraded board and illegal on the real one.
 * Returns b unchanged when the conditions do not hold.
 */
inline board_t DowngradeRoot(board_t b, int min_rank) {
    unsigned present = 0;
    int n_max = 0;
    for (int i = 0; i < 16; ++i) {
        int rank = int((b >> (4 * i)) & 0xf);
        present |= 1u << rank;
        n_max += rank == MAX_RANK;
    }
    int top = 31 - __builtin_clz(present | 1);
    if (min_rank <= 0 || top < min_rank || n_max >= 2) return b;
    int r = top - 1;
    while (r >= 3 && ((present >> r) & 1)) --r;
    if (r < 3) return b;
    board_t out = b;
    for (int i = 0; i < 16; ++i) {
        if (int((b >> (4 * i)) & 0xf) > r) out -= board_t(1) << (4 * i); // rank k > r becomes k-1
    }
    return out;
}

/**
 * Depth for one move with deep=e1,e2,...: one extra ply for each threshold the board's empty cells
 * are at or below. Few empty cells make chance nodes cheap (few spawn cells) and the position
 * dangerous, so that is where depth is both affordable and worth most.
 */
inline int AdaptivePly(int ply, int empty, const std::vector<int> &deep) {
    for (int e : deep) ply += empty <= e;
    return ply;
}

/**
 * Expectimax search for match play ([MS-TD] Fig. 2) with
 *   - a position cache: chance-node values keyed by (afterstate, plies left, stage), kept across
 *     the moves of a game. Entries match the plies exactly, so a cached search returns exactly
 *     what the plain search would (tests.cpp checks this against plain expectimax).
 *   - leaf values in the same cache, keyed with plies left = 0: at 3-ply over half the leaves are
 *     afterstates already valued (other spawn and move orders reach them), and a cached one costs
 *     one memory read instead of the network's 64. Network values are exact, so these are stored
 *     with pruning on as well.
 *   - time-limited iterative deepening: search 1, 2, 3, ... plies and keep the move of the deepest
 *     search that finished. A depth is started only if its predicted cost fits the budget, and an
 *     unfinished depth is dropped when the hard deadline passes.
 *   - optional pruning: a chance outcome whose path probability falls below `prune` is valued by
 *     the network instead of being searched further (0 = off, exact expectimax).
 *
 * One Search per thread (the cache is not shared). Learner is anything with
 *   float Evaluate(board_t afterstate, int game_stage) const;  int Split(board_t) const;
 */
template<class Learner>
class Search {
public:
    struct Result {
        int move = -1;      // -1: no legal move
        int ply = 0;        // depth of the search that chose the move
        double ms = 0;      // time used
        uint64_t nodes = 0; // chance nodes expanded
    };

    // log2_entries: cache size (2^22 entries = 64 MB); 0 disables the cache
    Search(const Learner &learner, int log2_entries = 22) : learner_(learner) {
        if (log2_entries > 0) {
            table_.reset(new Entry[size_t(1) << log2_entries]());
            mask_ = (size_t(1) << log2_entries) - 1;
        }
    }

    // forget all cached values, e.g. at the start of a game (O(1): bumps a generation counter)
    void NewGame() {
        if (++generation_ == 0) { // wrapped: really clear so stale entries cannot match
            std::memset(static_cast<void *>(table_.get()), 0, table_ ? (mask_ + 1) * sizeof(Entry) : 0);
            generation_ = 1;
        }
    }

    void SetPrune(float p) { prune_ = p; }

    // fixed depth, no time limit
    Result Fixed(const Board64 &board, int ply, int game_stage) {
        Start(INFINITY);
        Result r;
        r.move = MaxNode(board, ply, game_stage, 1.0f).first;
        r.ply = ply;
        r.ms = Elapsed();
        r.nodes = nodes_;
        return r;
    }

    /**
     * Iterative deepening within budget_ms. Depth max_ply caps the search (deeper rarely fits).
     * hard_ms: abort an unfinished depth after this long; defaults to budget_ms.
     */
    Result Timed(const Board64 &board, int game_stage, double budget_ms, double hard_ms = 0, int max_ply = 8) {
        if (hard_ms <= 0) hard_ms = budget_ms;
        Start(hard_ms);
        Result r;
        double last = 0, before_last = 0; // time of the last two finished depths
        for (int ply = 1; ply <= max_ply; ++ply) {
            if (ply > 2) {
                // cost grows by a roughly constant factor per ply; predict the next depth from it
                double growth = before_last > 0.001 ? last / before_last : 50.0;
                double predicted = last * std::max(growth, 2.0);
                if (Elapsed() + predicted > budget_ms) break;
            }
            double t0 = Elapsed();
            std::pair<int, float> best = MaxNode(board, ply, game_stage, 1.0f);
            if (aborted_) break; // unfinished: keep the previous depth's move
            r.move = best.first;
            r.ply = ply;
            before_last = last;
            last = Elapsed() - t0;
            if (r.move == -1) break; // game over, nothing to search
        }
        r.ms = Elapsed();
        r.nodes = nodes_;
        return r;
    }

private:
    struct Entry {
        board_t board;
        float value;
        uint16_t generation;
        uint8_t ply;
        uint8_t stage;
    };
    static_assert(sizeof(Entry) == 16, "cache entries should be 16 bytes");

    typedef std::chrono::steady_clock Clock;

    void Start(double hard_ms) {
        start_ = Clock::now();
        hard_ms_ = hard_ms;
        aborted_ = false;
        nodes_ = 0;
    }

    double Elapsed() const { return std::chrono::duration<double, std::milli>(Clock::now() - start_).count(); }

    // checked every 16 chance nodes (one near the leaves costs up to ~25 us, so ~0.4 ms between reads)
    bool OutOfTime() {
        if (aborted_) return true;
        if ((nodes_ & 15) == 0 && std::isfinite(hard_ms_) && Elapsed() > hard_ms_) aborted_ = true;
        return aborted_;
    }

    static size_t Hash(board_t b, int ply, int stage) {
        uint64_t x = b ^ (uint64_t(ply) << 56) ^ (uint64_t(stage) << 60);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return size_t(x);
    }

    // max node: the player moves; value = reward + value of the afterstate
    std::pair<int, float> MaxNode(const Board64 &board, int ply, int game_stage, float prob) {
        // slide all four ways first; at ply 1 also start the cache reads of all four leaves, so
        // their misses overlap instead of following one another
        Board64 child[4];
        reward_t reward[4];
        for (int d = 0; d < 4; ++d) {
            child[d] = board;
            reward[d] = child[d].Slide(d);
            if (ply <= 1 && table_ && reward[d] != -1) {
                __builtin_prefetch(&table_[Hash(child[d].GetBoard(), 0, game_stage) & mask_]);
            }
        }
        int direction = -1;
        float best = -INFINITY;
        for (int d = 0; d < 4; ++d) {
            if (reward[d] == -1) continue;
            float value = reward[d] + (ply <= 1 ? Leaf(child[d].GetBoard(), game_stage)
                                                : ChanceNode(child[d], ply - 1, game_stage, prob));
            if (aborted_) return {-1, 0.0f};
            if (value > best) {
                best = value;
                direction = d;
            }
        }
        return {direction, direction == -1 ? 0.0f : best}; // no legal move: game over, value 0
    }

    // network value of a leaf afterstate, through the cache (entries with ply 0)
    float Leaf(board_t b, int game_stage) {
        if (!table_) return learner_.Evaluate(b, game_stage);
        Entry &slot = table_[Hash(b, 0, game_stage) & mask_];
        if (slot.generation == generation_ && slot.board == b && slot.ply == 0 && slot.stage == game_stage) {
            return slot.value;
        }
        float value = learner_.Evaluate(b, game_stage);
        slot = {b, value, generation_, 0, uint8_t(game_stage)};
        return value;
    }

    // chance node (afterstate): a 2 with p=0.9 or a 4 with p=0.1 on a uniformly chosen empty cell
    float ChanceNode(const Board64 &after, int ply, int game_stage, float prob) {
        game_stage = std::max(game_stage, learner_.Split(after.GetBoard()));
        board_t b = after.GetBoard();
        Entry *slot = nullptr;
        if (table_) {
            slot = &table_[Hash(b, ply, game_stage) & mask_];
            if (slot->generation == generation_ && slot->board == b && slot->ply == ply && slot->stage == game_stage) {
                return slot->value;
            }
        }
        ++nodes_;
        if (OutOfTime()) return 0.0f;

        int empty = after.CountEmpty();
        float total = 0;
        float p2 = prob * 0.9f / float(empty), p4 = prob * 0.1f / float(empty);
        for (int i = 0; i < 16; ++i) {
            if (after(i) != 0) continue;
            Board64 two = after, four = after;
            two.Set(i, 1);
            four.Set(i, 2);
            total += 0.9f * Outcome(two, ply, game_stage, p2);
            total += 0.1f * Outcome(four, ply, game_stage, p4);
            if (aborted_) return 0.0f;
        }
        float value = empty ? total / float(empty) : 0.0f;
        if (slot && prune_ == 0) *slot = {b, value, generation_, uint8_t(ply), uint8_t(game_stage)};
        return value;
    }

    // a spawn outcome: searched further, or (pruned) valued as a 1-ply max node
    float Outcome(const Board64 &board, int ply, int game_stage, float prob) {
        if (prune_ > 0 && prob < prune_) ply = 1;
        return MaxNode(board, ply, game_stage, prob).second;
    }

    const Learner &learner_;
    std::unique_ptr<Entry[]> table_;
    size_t mask_ = 0;
    uint16_t generation_ = 1;
    float prune_ = 0;
    Clock::time_point start_;
    double hard_ms_ = INFINITY;
    bool aborted_ = false;
    uint64_t nodes_ = 0;
};
