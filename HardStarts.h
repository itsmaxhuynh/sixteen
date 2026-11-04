#pragma once

#include <mutex>
#include <random>
#include <vector>

#include "Common.h"
#include "Episode.h"
#include "NTupleNetwork.h"
#include "StartPositions.h"

/**
 * Hard starts: replay the places where games go wrong. When a game started the normal way ends,
 * rewind it by 1..rewind slides and keep that board (player to move) if it still lies in the
 * active stage's own region, i.e. before the next split (for stage 0: before T_16k). With
 * probability `hard` a later game starts from a random kept board instead of the normal start.
 * Only normal games feed the pool, so it follows the current policy's failures.
 *
 * The pool lives in memory only: it refills within minutes, so nothing is added to the checkpoint.
 */
class HardStarts {
public:
    HardStarts(size_t capacity, unsigned rewind) : capacity_(capacity), rewind_(rewind) {}

    // Offer a finished normal game; split(b) is the learner's split level, active the active stage.
    template <class Split>
    void Offer(const Episode &game, int active, Split split, std::mt19937_64 &rng) {
        const std::vector<Episode::Move> &moves = game.GetMoves();
        size_t first = game.FirstSlide();
        if (moves.size() <= first) return;
        size_t slides = (moves.size() - first + 1) / 2;
        size_t back = 1 + rng() % rewind_;
        if (back >= slides) return;
        size_t s = slides - back;            // restart right before slide s (s >= 1)
        size_t idx = first + 2 * s - 1;      // the tile placement that precedes it
        board_t b = moves[idx].board;
        if (split(b) != active) return;      // outside this stage's region
        reward_t score = game.score();
        for (size_t i = idx + 1; i < moves.size(); ++i) score -= moves[i].reward;
        std::lock_guard<std::mutex> lock(mutex_);
        StartPosition p{b, float(score)};
        if (pool_.size() < capacity_) pool_.push_back(p);
        else pool_[next_++ % capacity_] = p; // ring: keep the most recent failures
        offered_++;
    }

    bool Sample(std::mt19937_64 &rng, StartPosition &out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) return false;
        out = pool_[rng() % pool_.size()];
        return true;
    }

    size_t Offered() {
        std::lock_guard<std::mutex> lock(mutex_);
        return offered_;
    }

private:
    std::mutex mutex_;
    std::vector<StartPosition> pool_;
    size_t capacity_, next_ = 0, offered_ = 0;
    unsigned rewind_;
};

/**
 * Tile-upgrading of a start board for stage `active` (not the last stage): double the largest tile
 * that stays below the stage's next split tile, e.g. in stage 0 (next split T_16k) a 4096 becomes
 * 8192 and a board whose largest tile is 2048 gets a 4096; in stage 1 (16384 present, next split
 * needs 8192) the largest tile up to 2048 is doubled. The board thereby sits closer to the hard part
 * of its stage without leaving it. Returns b unchanged when no tile qualifies.
 */
inline board_t UpgradeTile(board_t b, int active) {
    int target = kSplitRanks[active]; // rank of the tile the next split needs
    int best = -1, best_rank = 0;
    for (int i = 0; i < 16; ++i) {
        int r = int((b >> (4 * i)) & 0xf);
        // tiles of earlier splits (16384, ...) are >= target and stay as they are
        if (r > best_rank && r + 1 < target) best = i, best_rank = r;
    }
    if (best < 0) return b;
    return b + (board_t(1) << (4 * best));
}
