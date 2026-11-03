#pragma once

#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>

#include "Common.h"
#include "LookUpTable.h"

/**
 * 64-bit board for 2048, 4 bits per cell holding the rank (tile = 2^rank).
 *
 * index (1-d form), row r lives at bits 16r, column c at bits 4c within the row:
 *  (0)  (1)  (2)  (3)
 *  (4)  (5)  (6)  (7)
 *  (8)  (9) (10) (11)
 * (12) (13) (14) (15)
 *
 * Directions are 0=U 1=R 2=D 3=L everywhere.
 */

static inline board_t Transpose(board_t board) {
    board_t a1 = board & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = board & 0x0000F0F00000F0F0ULL;
    board_t a3 = board & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}

class Board64 {
public:
    Board64(board_t board = 0) : board_(board) {}

    board_t GetBoard() const { return board_; }

    int operator()(int i) const { return int((board_ >> (4 * i)) & 0xf); }

    void Set(int i, int rank) {
        board_ = (board_ & ~(board_t(0xf) << (4 * i))) | (board_t(rank & 0xf) << (4 * i));
    }

    bool operator==(const Board64 &b) const { return board_ == b.board_; }

    bool operator!=(const Board64 &b) const { return board_ != b.board_; }

    bool operator<(const Board64 &b) const { return board_ < b.board_; }

    /**
     * place a tile of the given rank, return 0, or -1 if the cell is not empty
     */
    reward_t Place(unsigned position, cell_t rank) {
        if (position >= 16 || (*this)(position) != 0) return -1;
        Set(position, rank);
        return 0;
    }

    /**
     * slide the board, return the merge score, or -1 if the move changes nothing
     */
    reward_t Slide(unsigned direction) {
        board_t after = board_;
        reward_t reward = 0;
        switch (direction & 0b11) {
            case 0: {
                board_t t = ::Transpose(board_);
                for (int i = 0; i < 4; ++i) {
                    row_t r = row_t((t >> (16 * i)) & ROW_MASK);
                    after ^= col_up_table[r] << (4 * i);
                    reward += reward_left_table[r];
                }
                break;
            }
            case 1:
                for (int i = 0; i < 4; ++i) {
                    row_t r = row_t((board_ >> (16 * i)) & ROW_MASK);
                    after ^= board_t(row_right_table[r]) << (16 * i);
                    reward += reward_right_table[r];
                }
                break;
            case 2: {
                board_t t = ::Transpose(board_);
                for (int i = 0; i < 4; ++i) {
                    row_t r = row_t((t >> (16 * i)) & ROW_MASK);
                    after ^= col_down_table[r] << (4 * i);
                    reward += reward_right_table[r];
                }
                break;
            }
            case 3:
                for (int i = 0; i < 4; ++i) {
                    row_t r = row_t((board_ >> (16 * i)) & ROW_MASK);
                    after ^= board_t(row_left_table[r]) << (16 * i);
                    reward += reward_left_table[r];
                }
                break;
        }
        if (after == board_) return -1;
        board_ = after;
        return reward;
    }

    int CountEmpty() const {
        int n = 0;
        for (int i = 0; i < 16; ++i) n += ((board_ >> (4 * i)) & 0xf) == 0;
        return n;
    }

    int GetMaxTile() const {
        return std::max(std::max(row_max_table[(board_ >> 0) & ROW_MASK], row_max_table[(board_ >> 16) & ROW_MASK]),
                        std::max(row_max_table[(board_ >> 32) & ROW_MASK], row_max_table[(board_ >> 48) & ROW_MASK]));
    }

    bool IsTerminal() const {
        for (int d = 0; d < 4; ++d) {
            Board64 b = *this;
            if (b.Slide(d) != -1) return false;
        }
        return true;
    }

    friend std::ostream &operator<<(std::ostream &out, const Board64 &board) {
        out << "+------------------------+" << std::endl;
        for (int i = 0; i < 4; i++) {
            out << "|" << std::dec;
            for (int j = 0; j < 4; j++) {
                int r = board(i * 4 + j);
                out << std::setw(6) << (r ? (1 << r) : 0);
            }
            out << "|" << std::endl;
        }
        out << "+------------------------+" << std::endl;
        return out;
    }

private:
    board_t board_;
};
