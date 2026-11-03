#pragma once

#include <cstdint>

typedef float reward_t;
typedef uint16_t row_t;
typedef uint64_t board_t;
typedef uint8_t cell_t;

static const board_t ROW_MASK = 0xFFFFULL;
static const board_t COL_MASK = 0x000F000F000F000FULL;

// A cell stores a rank r: 0 is empty, r >= 1 is the tile 2^r.
// 4 bits per cell caps the board at rank 15 (32768), see LookUpTable.h.
static const int MAX_RANK = 15;
