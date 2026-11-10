#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include "Board64.h"

/**
 * The real board when tiles can pass 32768: TAAI's 2048 merges two 32768s into 65536, which
 * Board64's 4-bit cells cannot hold (they refuse that merge). One byte per cell holding the rank
 * (tile = 2^rank, 0 = empty), cell i = row i / 4, column i % 4, directions 0=U 1=R 2=D 3=L, the same
 * rules as Board64 without its top cap. It keeps the game, never the search: an agent sees Fit(w).
 * Speed does not matter here (one slide per move), so it is written out plainly.
 */
struct WideBoard {
    uint8_t cell[16] = {};

    static WideBoard From(board_t b) {
        WideBoard w;
        for (int i = 0; i < 16; ++i) w.cell[i] = uint8_t((b >> (4 * i)) & 0xf);
        return w;
    }

    int MaxRank() const { return *std::max_element(cell, cell + 16); }

    // place a tile of the given rank: 0, or -1 if the cell is taken
    int Place(unsigned pos, unsigned rank) {
        if (pos >= 16 || cell[pos]) return -1;
        cell[pos] = uint8_t(rank);
        return 0;
    }

    // merge score of the slide, or -1 if nothing moves
    long Slide(unsigned dir) {
        long reward = 0;
        bool moved = false;
        for (int line = 0; line < 4; ++line) {
            int at[4]; // cells of this line, in the order tiles slide towards
            for (int k = 0; k < 4; ++k) {
                switch (dir & 3) {
                    case 0: at[k] = 4 * k + line; break;       // up: top row first
                    case 1: at[k] = 4 * line + 3 - k; break;   // right
                    case 2: at[k] = 4 * (3 - k) + line; break; // down
                    default: at[k] = 4 * line + k; break;      // left
                }
            }
            int v[4], n = 0, out[4] = {0, 0, 0, 0}, m = 0;
            for (int k = 0; k < 4; ++k) if (cell[at[k]]) v[n++] = cell[at[k]];
            for (int i = 0; i < n; ++i) {
                if (i + 1 < n && v[i] == v[i + 1]) {
                    out[m++] = v[i] + 1;
                    reward += 1L << (v[i] + 1);
                    ++i;
                } else {
                    out[m++] = v[i];
                }
            }
            for (int k = 0; k < 4; ++k) {
                if (cell[at[k]] != out[k]) moved = true;
                cell[at[k]] = uint8_t(out[k]);
            }
        }
        return moved ? reward : -1;
    }

    // every rank fits a 4-bit cell and Board64 would make every merge this board makes
    bool Fits() const {
        int n_max = 0;
        for (uint8_t r : cell) {
            if (r > MAX_RANK) return false;
            n_max += r == MAX_RANK;
        }
        return n_max <= 1;
    }

    // one base-32 digit per cell (0-9, a-v), as --record writes boards: 65536 is 'g'
    std::string Hex() const {
        std::string s(16, '0');
        for (int i = 0; i < 16; ++i) s[i] = "0123456789abcdefghijklmnopqrstuv"[std::min<int>(cell[i], 31)];
        return s;
    }

    bool operator==(const WideBoard &o) const { return std::equal(cell, cell + 16, o.cell); }
};

/**
 * The Board64 an agent searches for the real board w. w itself when it fits; otherwise the largest
 * rank r missing below the top tile is closed (every tile above r drops one rank, as in
 * DowngradeRoot) until it fits: two 32768s become two 16384s, which Board64 merges, and a 65536
 * with small tiles becomes 32768 (then DowngradeRoot takes it to 16384 at the root). This is
 * shift80 of arXiv:2111.11090's code (closing missing ranks one by one).
 *
 * Closing a missing rank keeps which cells are empty, which tiles are equal and their order, and the
 * result holds at most one 32768, so every move is legal on Fit(w) exactly when it is legal on w and
 * does the same slides and merges. When no rank below the top is missing (a full board holding every
 * tile from 2 up), ranks are capped at 32768 as a last resort, which can refuse a merge w would make.
 */
inline board_t Fit(const WideBoard &w) {
    WideBoard f = w;
    while (!f.Fits()) {
        unsigned present = 0;
        for (uint8_t r : f.cell) present |= 1u << r;
        int top = f.MaxRank();
        int r = top - 1;
        while (r >= 1 && ((present >> r) & 1)) --r;
        if (r < 1) {
            for (uint8_t &c : f.cell) c = std::min<uint8_t>(c, MAX_RANK);
            break;
        }
        for (uint8_t &c : f.cell) if (c > r) --c;
    }
    board_t b = 0;
    for (int i = 0; i < 16; ++i) b |= board_t(f.cell[i]) << (4 * i);
    return b;
}
