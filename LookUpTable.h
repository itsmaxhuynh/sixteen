#pragma once

#include <algorithm>
#include <cstdint>

#include "Common.h"

// Per-row tables, indexed by a 16-bit row (4 cells, cell 0 in the low nibble).
// Slides are stored as XOR deltas so a whole board moves with 4 lookups.
static cell_t row_max_table[65536];
static row_t row_left_table[65536];
static row_t row_right_table[65536];
static board_t col_up_table[65536];
static board_t col_down_table[65536];
static reward_t reward_left_table[65536];
static reward_t reward_right_table[65536];

static board_t unpack_col(row_t row) {
    board_t tmp = row;
    return (tmp | (tmp << 12ULL) | (tmp << 24ULL) | (tmp << 36ULL)) & COL_MASK;
}

static row_t reverse_row(row_t row) {
    return (row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0x0F00) | (row << 12);
}

// Must run before any Board64 operation.
inline void InitLookUpTables() {
    for (unsigned row = 0; row < 65536; ++row) {
        unsigned line[4] = {
                (row >> 0) & 0xf,
                (row >> 4) & 0xf,
                (row >> 8) & 0xf,
                (row >> 12) & 0xf
        };
        row_max_table[row] = std::max(std::max(line[0], line[1]), std::max(line[2], line[3]));

        // 2048 slide to the left: compact, then merge each equal pair once, left to right.
        unsigned tiles[4];
        int n = 0;
        for (int i = 0; i < 4; ++i) {
            if (line[i]) tiles[n++] = line[i];
        }

        unsigned result_line[4] = {0, 0, 0, 0};
        reward_t reward = 0;
        int k = 0;
        for (int i = 0; i < n; ++i) {
            // Two 32768 tiles cannot merge: rank 16 does not fit in 4 bits.
            if (i + 1 < n && tiles[i] == tiles[i + 1] && tiles[i] < (unsigned) MAX_RANK) {
                result_line[k++] = tiles[i] + 1;
                reward += reward_t(1u << (tiles[i] + 1));
                ++i;
            } else {
                result_line[k++] = tiles[i];
            }
        }

        row_t result = (result_line[0] << 0) |
                       (result_line[1] << 4) |
                       (result_line[2] << 8) |
                       (result_line[3] << 12);
        row_t rev_result = reverse_row(result);
        row_t rev_row = reverse_row(row);

        row_left_table[row] = row ^ result;
        row_right_table[rev_row] = rev_row ^ rev_result;
        col_up_table[row] = unpack_col(row) ^ unpack_col(result);
        col_down_table[rev_row] = unpack_col(rev_row) ^ unpack_col(rev_result);
        reward_left_table[row] = reward;
        reward_right_table[rev_row] = reward;
    }
}
