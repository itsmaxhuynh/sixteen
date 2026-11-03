#pragma once

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
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

    class Writer {
    public:
        Writer(const std::string &path, int stage) {
            size_t existing = Count(path);
            if (existing > 0) {
                std::ifstream in(path, std::ios::in | std::ios::binary);
                if (ReadHeader(in, path) != stage) throw std::runtime_error(path + " holds boards of another stage");
                // drop a torn trailing record before appending
                std::ifstream probe(path, std::ios::in | std::ios::binary | std::ios::ate);
                if (size_t(probe.tellg()) != kHeader + existing * kRecord) Truncate(path, kHeader + existing * kRecord);
                out_.open(path, std::ios::out | std::ios::binary | std::ios::app);
            } else {
                out_.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
                out_.write("SXBOARDS", 8);
                uint32_t s = uint32_t(stage);
                out_.write(reinterpret_cast<const char *>(&s), 4);
            }
            if (!out_) throw std::runtime_error("cannot open " + path + " for writing");
            count_ = existing;
        }

        void Add(board_t board, float score) {
            out_.write(reinterpret_cast<const char *>(&board), 8);
            out_.write(reinterpret_cast<const char *>(&score), 4);
            out_.flush();
            ++count_;
        }

        size_t count() const { return count_; }

    private:
        std::ofstream out_;
        size_t count_ = 0;
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

    static void Truncate(const std::string &path, size_t size) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        std::vector<char> data(size);
        in.read(data.data(), std::streamsize(size));
        in.close();
        std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
        out.write(data.data(), std::streamsize(size));
    }
};
