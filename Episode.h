#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "Action.h"
#include "Board64.h"

class Agent;

class Episode {
public:
    struct Move {
        Action code;
        reward_t reward;
        time_t time;
        board_t board; // board after the move (the afterstate for a slide)

        Move(board_t board = 0, Action code = {}, reward_t reward = 0, time_t time = 0)
                : code(code), reward(reward), time(time), board(board) {}

        operator Action() const { return code; }

        friend std::ostream &operator<<(std::ostream &out, const Move &m) {
            out << m.code;
            if (m.reward) out << '[' << std::dec << m.reward << ']';
            if (m.time) out << '(' << std::dec << m.time << ')';
            return out;
        }

        friend std::istream &operator>>(std::istream &in, Move &m) {
            in >> m.code;
            m.reward = 0;
            m.time = 0;
            if (in.peek() == '[') {
                in.ignore(1);
                in >> std::dec >> m.reward;
                in.ignore(1);
            }
            if (in.peek() == '(') {
                in.ignore(1);
                in >> std::dec >> m.time;
                in.ignore(1);
            }
            return in;
        }
    };

    Episode() : ep_state(), ep_score(0), ep_time(0) { ep_moves.reserve(1024); }

    Board64 &state() { return ep_state; }

    const Board64 &state() const { return ep_state; }

    reward_t score() const { return ep_score; }

    const std::vector<Move> &GetMoves() const { return ep_moves; }

    void OpenEpisode(const std::string &tag) { ep_open = {tag, millisec()}; }

    void CloseEpisode(const std::string &tag) { ep_close = {tag, millisec()}; }

    bool ApplyAction(Action move) {
        reward_t reward = move.Apply(state());
        if (reward == -1) return false;
        ep_moves.emplace_back(state().GetBoard(), move, reward, millisec() - ep_time);
        ep_score += reward;
        return true;
    }

    // Start from a collected board instead of an empty one ([MS-TD] Sec. III): the player moves first.
    void Start(board_t board, reward_t score) {
        ep_state = Board64(board);
        ep_score = score;
        start_board_ = board;
        first_slide_ = 0;
    }

    board_t StartBoard() const { return start_board_; }

    // index in GetMoves() of the first slide; slides then alternate with placements
    size_t FirstSlide() const { return first_slide_; }

    // From an empty board the environment places the first 2 tiles, then player and environment alternate.
    template<class P, class E>
    Agent &TakeTurns(P &player, E &evil) {
        ep_time = millisec();
        size_t s = step();
        bool player_turn = s >= first_slide_ && (s - first_slide_) % 2 == 0;
        return player_turn ? static_cast<Agent &>(player) : static_cast<Agent &>(evil);
    }

    template<class P, class E>
    Agent &TakeLastTurns(P &player, E &evil) {
        return TakeTurns(evil, player);
    }

    size_t step(unsigned who = -1u) const {
        if (who == -1u) return ep_moves.size();
        size_t n = 0;
        for (const Move &m : ep_moves) n += m.code.type() == who;
        return n;
    }

    time_t time(unsigned who = -1u) const {
        if (who == -1u) return ep_close.when - ep_open.when;
        time_t time = 0;
        for (const Move &m : ep_moves) if (m.code.type() == who) time += m.time;
        return time;
    }

    friend std::ostream &operator<<(std::ostream &out, const Episode &ep) {
        out << ep.ep_open << '|';
        for (const Move &mv : ep.ep_moves) out << mv;
        out << '|' << ep.ep_close;
        return out;
    }

    friend std::istream &operator>>(std::istream &in, Episode &ep) {
        ep = {};
        std::string token;
        std::getline(in, token, '|');
        std::stringstream(token) >> ep.ep_open;
        std::getline(in, token, '|');
        for (std::stringstream moves(token); !moves.eof(); moves.peek()) {
            ep.ep_moves.emplace_back();
            moves >> ep.ep_moves.back();
            ep.ep_score += Action(ep.ep_moves.back()).Apply(ep.ep_state);
            ep.ep_moves.back().board = ep.ep_state.GetBoard();
        }
        std::getline(in, token, '|');
        std::stringstream(token) >> ep.ep_close;
        return in;
    }

protected:
    struct Meta {
        std::string tag;
        time_t when;

        Meta(const std::string &tag = "N/A", time_t when = 0) : tag(tag), when(when) {}

        friend std::ostream &operator<<(std::ostream &out, const Meta &m) {
            return out << m.tag << "@" << std::dec << m.when;
        }

        friend std::istream &operator>>(std::istream &in, Meta &m) {
            return std::getline(in, m.tag, '@') >> std::dec >> m.when;
        }
    };

    static time_t millisec() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    }

private:
    Board64 ep_state;
    reward_t ep_score;
    std::vector<Move> ep_moves;
    time_t ep_time;

    Meta ep_open;
    Meta ep_close;
    board_t start_board_ = 0;
    size_t first_slide_ = 2;
};
