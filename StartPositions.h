#pragma once

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "Common.h"

/**
 * Boards collected at a stage split, used as the starting boards to train the next stage
 * ([MS-TD] Sec. III: "collect a number of boards, say 100,000 samples, at T_16k, and their scores").
 *
 * File: char[8] "SXBOARDS", u32 stage, then records of (u64 board, f32 score). Appending is safe
 * across sessions; a torn last record from a killed process is ignored on load.
 */
struct StartPosition {
    board_t board;
    float score;
};

class StartPositions {
public:
    static constexpr size_t kHeader = 12;
    static constexpr size_t kRecord = 12;

    // returns the positions; stage receives the stage they start
    static std::vector<StartPosition> Load(const std::string &path, int &stage) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in) throw std::runtime_error("cannot open start positions " + path);
        stage = ReadHeader(in, path);
        std::vector<StartPosition> positions;
        StartPosition p;
        while (in.read(reinterpret_cast<char *>(&p.board), 8) && in.read(reinterpret_cast<char *>(&p.score), 4)) {
            positions.push_back(p);
        }
        if (positions.empty()) throw std::runtime_error(path + " holds no positions");
        return positions;
    }

    // number of complete records, 0 if the file does not exist
    static size_t Count(const std::string &path) {
        std::ifstream in(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!in) return 0;
        size_t size = size_t(in.tellg());
        return size < kHeader ? 0 : (size - kHeader) / kRecord;
    }

    /**
     * Appends boards, each board at most once: a board already in the file (whatever its score) is
     * not added again. Boards at a split repeat a lot (one board can be reached by many games), and
     * a stage trains from these boards round robin, so a repeat only takes a slot a new board could
     * use. Opening a file that already holds repeats (written before this check) rewrites it with
     * the first copy of each board, in file order; dropped() says how many were removed.
     */
    class Writer {
    public:
        Writer(const std::string &path, int stage) {
            size_t existing = Count(path);
            if (existing > 0) {
                std::vector<StartPosition> kept;
                {
                    std::ifstream in(path, std::ios::in | std::ios::binary);
                    if (ReadHeader(in, path) != stage) throw std::runtime_error(path + " holds boards of another stage");
                    StartPosition p;
                    for (size_t i = 0; i < existing; ++i) { // stops before a torn trailing record
                        in.read(reinterpret_cast<char *>(&p.board), 8);
                        in.read(reinterpret_cast<char *>(&p.score), 4);
                        if (!in) throw std::runtime_error("cannot read " + path);
                        if (seen_.insert(p.board).second) kept.push_back(p);
                    }
                }
                dropped_ = existing - kept.size();
                std::ifstream probe(path, std::ios::in | std::ios::binary | std::ios::ate);
                if (dropped_ > 0 || size_t(probe.tellg()) != kHeader + existing * kRecord) Rewrite(path, stage, kept);
                out_.open(path, std::ios::out | std::ios::binary | std::ios::app);
                count_ = kept.size();
            } else {
                out_.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
                WriteHeader(out_, stage);
            }
            if (!out_) throw std::runtime_error("cannot open " + path + " for writing");
        }

        // false (and nothing written) if the board is already in the file
        bool Add(board_t board, float score) {
            if (!seen_.insert(board).second) {
                ++repeats_;
                return false;
            }
            out_.write(reinterpret_cast<const char *>(&board), 8);
            out_.write(reinterpret_cast<const char *>(&score), 4);
            out_.flush();
            ++count_;
            return true;
        }

        size_t count() const { return count_; }      // distinct boards in the file
        size_t dropped() const { return dropped_; }  // repeats removed from the file when it was opened
        size_t repeats() const { return repeats_; }  // boards Add turned away since

    private:
        static void Rewrite(const std::string &path, int stage, const std::vector<StartPosition> &kept) {
            std::string tmp = path + ".tmp";
            {
                std::ofstream out(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
                WriteHeader(out, stage);
                for (const StartPosition &p : kept) {
                    out.write(reinterpret_cast<const char *>(&p.board), 8);
                    out.write(reinterpret_cast<const char *>(&p.score), 4);
                }
                out.flush();
                if (!out) throw std::runtime_error("write failed for " + tmp);
            }
            if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + tmp + " to " + path);
        }

        static void WriteHeader(std::ofstream &out, int stage) {
            out.write("SXBOARDS", 8);
            uint32_t s = uint32_t(stage);
            out.write(reinterpret_cast<const char *>(&s), 4);
        }

        std::ofstream out_;
        std::unordered_set<board_t> seen_;
        size_t count_ = 0, dropped_ = 0, repeats_ = 0;
    };

private:
    static int ReadHeader(std::istream &in, const std::string &path) {
        char magic[8];
        uint32_t stage;
        in.read(magic, 8);
        in.read(reinterpret_cast<char *>(&stage), 4);
        if (!in || std::memcmp(magic, "SXBOARDS", 8) != 0) throw std::runtime_error(path + " is not a start positions file");
        return int(stage);
    }
};
