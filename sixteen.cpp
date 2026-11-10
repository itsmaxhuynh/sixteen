/**
 * sixteen: a 2048 agent (n-tuple TD learning + expectimax), ported from the Threes agent.
 *
 * Multi-stage TD training ([MS-TD], see README.md):
 *   ./sixteen --learn --play="save=w.ckpt"                                  train stage 0 from scratch
 *   ./sixteen --collect=b1.bin --play="load=w.ckpt"                         boards at the next split
 *   ./sixteen --learn --start=b1.bin --play="load=w.ckpt save=w.ckpt stage=1"
 *   hard starts (--learn): --hard=P starts a share P of games from boards rewound 1..--hard-rewind=N
 *   (500) slides before the end of earlier games, kept while still in the active stage (pool of
 *   --hard-pool=N, 100000); --upgrade=Q doubles a tile of a start board with probability Q (HardStarts.h)
 * Merge stages trained in separate sessions (Merge.h):
 *   ./sixteen --merge=out.ckpt base.ckpt 1=s1.ckpt 4=s4.ckpt
 *   common: --threads=N --total=N --block=N --save-every=N --time-limit=SEC --collect-count=N --summary
 * Evaluate:  ./sixteen --total=1000 --seed=1 --summary --play="load=w.ckpt ply=3"
 *   --seed=N: game i uses environment seed N+i, so every checkpoint plays the same games whatever
 *   the thread count; --summary then adds an "eval:" line with 95% intervals and move times
 *   (--move-limit-ms=100 sets the threshold counted as over the limit). See eval.sh.
 *   --record=games.txt [--record-tile=32768]: write every game that reaches the tile, move by move
 *   (RecordGame below has the format); not with --learn
 *   --wide: keep each game on a WideBoard, where two 32768s merge into 65536 as in TAAI (agents see
 *   Fit() of it, WideBoard.h); the eval line then adds r65536=, the games that reached 65536
 * Explain:   zcat games.txt.gz | ./sixteen --explain=- --explain-out=x.txt --explain-game=893 --play="<eval args>"
 *   each recorded slide with the value of every move (ExplainGames below)
 * Inspect:   ./sixteen --info=w.ckpt
 * Match:     --play="load=w.ckpt time=50"  iterative deepening within 50 ms per move (see Agent.h)
 * Arena:     ./sixteen --shell --name=<account> --play="load=w.ckpt ply=3"
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <atomic>
#include <csignal>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "Action.h"
#include "Agent.h"
#include "Board64.h"
#include "Episode.h"
#include "HardStarts.h"
#include "Merge.h"
#include "StartPositions.h"
#include "arena.h"
#include "io.h"

static std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler needs a lock-free flag");

static void RequestStop(int) { stop_requested = true; }

// One block per game: a header line, the start board, then one line per move (slides and tile
// placements alternate):
//   game <index> seed <env seed or -1> score <final> max <largest tile> moves <n>
//   start <board>
//   <step> <move> <reward> <score> <board after the move>
// move is U/R/D/L for a slide or 2@<cell>/4@<cell> for a placed tile (cells 0..15 row by row from the
// top left); board is 16 hex digits, one per cell in the same order, digit r = tile 2^r (0 = empty).
static void RecordGame(std::ostream &out, const Episode &game, uint64_t index, long seed) {
    auto hex = [](board_t b) { return WideBoard::From(b).Hex(); };
    const auto &moves = game.GetMoves();
    // a wide game records its real boards (base-32 digits: 65536 is 'g')
    auto board = [&](size_t i) { return game.Wide() ? game.WideMoves()[i].Hex() : hex(moves[i].board); };
    int top = game.Wide() ? game.WideState().MaxRank() : Board64(game.state()).GetMaxTile();
    // scores as integers: a float prints 1149550 as 1.14955e+06 (rewards are whole numbers)
    out << "game " << index << " seed " << seed << " score " << (long long)(game.score()) << " max "
        << (1ul << top) << " moves " << moves.size() << "\n";
    out << "start " << hex(game.StartBoard()) << "\n";
    reward_t score = game.score();
    for (const auto &m : moves) score -= m.reward;
    for (size_t i = 0; i < moves.size(); ++i) {
        const Action &a = moves[i].code;
        score += moves[i].reward;
        out << i << ' ';
        if (a.type() == Action::Slide::type_) out << "URDL"[a.event() & 3];
        else { Action::Place p(a); out << (1u << p.tile()) << '@' << p.position(); }
        out << ' ' << (long long)(moves[i].reward) << ' ' << (long long)(score) << ' ' << board(i) << "\n";
    }
    out << "\n";
    out.flush();
}

int shell(int argc, const char *argv[]) {
    arena host("anonymous");

    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        if (para.find("--name=") == 0 || para.find("--account=") == 0) {
            host.set_account(para.substr(para.find("=") + 1));
        } else if (para.find("--login=") == 0) {
            host.set_login(para.substr(para.find("=") + 1));
        } else if (para.find("--save=") == 0 || para.find("--dump=") == 0) {
            host.set_dump_file(para.substr(para.find("=") + 1));
        } else if (para.find("--play") == 0) {
            std::shared_ptr<Agent> play(new TdPlayer(para.substr(para.find("=") + 1)));
            host.register_agent(play);
        } else if (para.find("--evil") == 0) {
            std::shared_ptr<Agent> evil(new RandomEnvironment(para.substr(para.find("=") + 1)));
            host.register_agent(evil);
        }
    }

    std::regex match_move("^#\\S+ \\S+$"); // e.g. "#M0001 ?", "#M0001 #U"
    std::regex match_ctrl("^#\\S+ \\S+ \\S+$"); // e.g. "#M0001 open Slider:Placer", "#M0001 close score=15424"
    std::regex arena_ctrl("^[@$].+$"); // e.g. "@ login", "@ error the account "Name" has already been taken"
    std::regex arena_info("^[?%].+$"); // e.g. "? message from anonymous: 2048!!!"

    for (std::string command; input() >> command;) {
        try {
            if (std::regex_match(command, match_move)) {
                std::string id, move;
                std::stringstream(command) >> id >> move;
                if (move == "?") {
                    // our agent needs to take an action
                    Action a = host.at(id).take_action();
                    host.at(id).ApplyAction(a);
                    output() << id << ' ' << a << std::endl;
                } else {
                    // perform the opponent's action
                    Action a;
                    std::stringstream(move) >> a;
                    host.at(id).ApplyAction(a);
                }
            } else if (std::regex_match(command, match_ctrl)) {
                std::string id, ctrl, tag;
                std::stringstream(command) >> id >> ctrl >> tag;
                if (ctrl == "open") {
                    if (host.open(id, tag)) {
                        output() << id << " open accept" << std::endl;
                    } else {
                        output() << id << " open reject" << std::endl;
                    }
                } else if (ctrl == "close") {
                    host.close(id, tag);
                }
            } else if (std::regex_match(command, arena_ctrl)) {
                std::string ctrl;
                std::stringstream(command).ignore(1) >> ctrl;
                if (ctrl == "login") {
                    std::stringstream agents;
                    for (auto who : host.list_agents()) {
                        agents << " " << who->name() << "(" << who->role() << ")";
                    }
                    output("@ ") << "login " << host.login() << agents.str() << std::endl;
                } else if (ctrl == "status") {
                    info() << "+++++ status +++++" << std::endl;
                    info() << "login: " << host.account();
                    for (auto who : host.list_agents()) {
                        info() << " " << who->name() << "(" << who->role() << ")";
                    }
                    info() << std::endl;
                    info() << "match: " << host.list_matches().size() << std::endl;
                    for (auto ep : host.list_matches()) {
                        info() << ep->name() << " " << (*ep) << std::endl;
                    }
                    info() << "----- status -----" << std::endl;
                } else if (ctrl == "error" || ctrl == "exit") {
                    std::string message = command.substr(command.find_first_not_of("@$ "));
                    info() << message << std::endl;
                    break;
                }
            } else if (std::regex_match(command, arena_info)) {
                // message from arena server
            }
        } catch (std::exception &ex) {
            std::string message = std::string(typeid(ex).name()) + ": " + ex.what();
            message = message.substr(0, message.find_first_of("\r\n"));
            output("? ") << "exception " << message << " at \"" << command << "\"" << std::endl;
        }
    }
    return 0;
}

/**
 * Thread-safe progress report, printed every `block` normal games in the format of the Threes
 * Statistic: average and max score, throughput, and per tile the reach rate and end rate.
 * Hard and upgraded starts (HardStarts.h) begin from boards the normal game would not give, so their
 * scores and reach rates are kept apart: the main lines count normal games only (comparable across
 * runs and with eval), each special kind gets one extra line, and the speed counts every game played.
 */
class Reporter {
public:
    explicit Reporter(size_t block) : block_(block), start_(Clock::now()), block_start_(start_) {}

    enum Start { kNormal, kHard, kUpgraded };

    void Add(const Episode &game, Start start = kNormal) {
        int split = MaxSplit(game);
        std::lock_guard<std::mutex> lock(mutex_);
        if (game.Wide()) {
            wide_games_++;
            wide_65536_ += game.WideState().MaxRank() >= 16;
        }
        all_games_++;
        all_moves_ += game.step(Action::Slide::type_);
        if (start != kNormal) {
            Accumulate(start == kHard ? block_hard_ : block_upgraded_, game, split);
            return;
        }
        Accumulate(block_acc_, game, split);
        Accumulate(total_acc_, game, split);
        if (block_acc_.games == block_) {
            auto now = Clock::now();
            double seconds = std::chrono::duration<double>(now - block_start_).count();
            bool special = block_hard_.games || block_upgraded_.games;
            Print(block_acc_, total_acc_.games, seconds, all_games_, all_moves_, special ? "normal starts" : nullptr);
            PrintSpecial(block_hard_, "hard starts");
            PrintSpecial(block_upgraded_, "upgraded starts");
            block_acc_ = block_hard_ = block_upgraded_ = {};
            all_games_ = all_moves_ = 0;
            block_start_ = now;
        }
    }

    // player move times of one game, in seconds (only measured when not learning)
    void AddMoveTimes(double total, double max, size_t over_limit, const TdPlayer::MoveStats &stats) {
        std::lock_guard<std::mutex> lock(mutex_);
        ply_sum_ += stats.ply_sum;
        ply_moves_ += stats.ply_moves;
        downgraded_ += stats.downgraded;
        move_time_ += total;
        move_time_max_ = std::max(move_time_max_, max);
        moves_over_limit_ += over_limit;
    }

    void Summary(double move_limit_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "summary of all games:" << std::endl;
        double seconds = std::chrono::duration<double>(Clock::now() - start_).count();
        Print(total_acc_, total_acc_.games, seconds, total_acc_.games, total_acc_.moves, nullptr);
        const Acc &a = total_acc_;
        if (a.games == 0) return;
        // mean score with a 95% normal interval; reach rates with a 95% Wilson interval
        double n = double(a.games), mean = a.score / n;
        double var = n > 1 ? (a.score_sq - n * mean * mean) / (n - 1) : 0;
        double half = 1.96 * std::sqrt(std::max(var, 0.0) / n);
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << "eval: games=" << a.games << " mean=" << mean << " ci95=" << half;
        auto rate = [&out, n](const std::string &key, size_t count) { // " key=p%[lo,hi]"
            double p = count / n, z = 1.96, d = 1 + z * z / n;
            double c = (p + z * z / (2 * n)) / d, h = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
            out << std::setprecision(2) << " " << key << "=" << 100 * p << "%[" << 100 * std::max(0.0, c - h) << ","
                << 100 * std::min(1.0, c + h) << "]";
        };
        size_t reached = a.games;
        for (int t = 0; t < 16; ++t) {
            if (t >= 13) rate("r" + std::to_string(1 << t), reached); // 8192, 16384, 32768
            reached -= a.end_tile[t];
        }
        if (a.moves && move_time_ > 0) {
            out << std::setprecision(1) << " move_us=" << 1e6 * move_time_ / double(a.moves)
                << std::setprecision(2) << " max_move_ms=" << 1e3 * move_time_max_ << " over_" << std::setprecision(0)
                << move_limit_ms << "ms=" << moves_over_limit_;
        }
        if (ply_moves_) out << std::setprecision(2) << " avg_ply=" << double(ply_sum_) / double(ply_moves_);
        if (downgraded_) out << " downgraded_moves=" << downgraded_;
        if (wide_games_) out << " r65536=" << wide_65536_ << "/" << wide_games_;
        // the full breakdown (eval.sh prints it as a table): reach rate of every tile from 2 to 32768
        // (games whose largest tile got at least that far), then of the stage splits
        // ([MS-TD] T_16k, T_16+8k, T_16+8+4k, T_16+8+4+2k: the boards where stages 1..4 take over)
        reached = a.games;
        for (int t = 0; t < 16; ++t) {
            if (t >= 1) rate("tile" + std::to_string(1 << t), reached);
            reached -= a.end_tile[t];
        }
        static const char *splits[5] = {"", "16k", "16+8k", "16+8+4k", "16+8+4+2k"};
        size_t at_least = a.games;
        for (int k = 0; k <= 4; ++k) {
            if (k >= 1) rate(std::string("split") + splits[k], at_least);
            at_least -= a.split[k];
        }
        std::cout << out.str() << std::endl;
    }

private:
    typedef std::chrono::steady_clock Clock;

    struct Acc {
        size_t games = 0;
        double score = 0;
        double score_sq = 0;
        double max = 0;
        size_t moves = 0;
        size_t end_tile[16] = {0};
        size_t split[5] = {0}; // games whose highest split reached ([MS-TD] T_16k, T_16+8k, ...) was k
    };

    // Highest split the game reached at any point: the max tile alone says nothing in later stages,
    // whose games all start from boards that already hold 16384.
    static int MaxSplit(const Episode &game) {
        int k = SplitLevel(game.StartBoard(), 4);
        for (const Episode::Move &m : game.GetMoves()) {
            if (k == 4) break;
            k = std::max(k, SplitLevel(m.board, 4));
        }
        return k;
    }

    static void Accumulate(Acc &acc, const Episode &game, int split) {
        acc.split[split]++;
        acc.games++;
        acc.score += game.score();
        acc.score_sq += double(game.score()) * double(game.score());
        acc.max = std::max(acc.max, double(game.score()));
        acc.moves += game.step(Action::Slide::type_);
        acc.end_tile[game.state().GetMaxTile()]++;
    }

    // games/moves: every game played in the period (for the speed); label: tag the line as normal starts
    static void Print(const Acc &acc, size_t index, double seconds, size_t games, size_t moves, const char *label) {
        if (acc.games == 0) return;
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << index << "\tavg = " << acc.score / acc.games << ", max = " << acc.max
            << ", games/s = " << std::setprecision(1) << games / seconds
            << ", moves/s = " << std::setprecision(0) << moves / seconds;
        if (label) out << "\t(" << label << ": " << acc.games << " games; speed counts all " << games << ")";
        out << std::endl;
        size_t reached = acc.games;
        for (int t = 0; t < 16; ++t) {
            if (acc.end_tile[t] == 0) {
                continue;
            }
            out << std::setprecision(2) << "\t" << (t ? (1 << t) : 0) << "\t" << reached * 100.0 / acc.games << "%\t("
                << acc.end_tile[t] * 100.0 / acc.games << "%)" << std::endl;
            reached -= acc.end_tile[t];
        }
        // reach rates of the stage splits, printed once some game got to 16384
        static const char *names[5] = {"", "16k", "16+8k", "16+8+4k", "16+8+4+2k"};
        if (acc.games > acc.split[0]) {
            out << "\tsplits";
            size_t at_least = acc.games - acc.split[0];
            for (int k = 1; k <= 4; ++k) {
                out << std::setprecision(2) << "\t" << names[k] << " " << at_least * 100.0 / acc.games << "%";
                at_least -= acc.split[k];
            }
            out << std::endl;
        }
        std::cout << out.str() << std::endl;
    }

    // a kind of special start: games, average, max, reach rates of 8192..32768 and (once some game got
    // to 16384) of the splits
    static void PrintSpecial(const Acc &acc, const char *label) {
        if (acc.games == 0) return;
        static const char *names[5] = {"", "16k", "16+8k", "16+8+4k", "16+8+4+2k"};
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << "\t" << label << ": " << acc.games << " games, avg = "
            << acc.score / acc.games << ", max = " << acc.max << "\n\t\treach";
        size_t reached = acc.games;
        for (int t = 0; t < 16; ++t) {
            if (t >= 13) out << std::setprecision(2) << "\t" << (1 << t) << " " << reached * 100.0 / acc.games << "%";
            reached -= acc.end_tile[t];
        }
        if (acc.games > acc.split[0]) {
            out << "\n\t\tsplits";
            size_t at_least = acc.games - acc.split[0];
            for (int k = 1; k <= 4; ++k) {
                out << std::setprecision(2) << "\t" << names[k] << " " << at_least * 100.0 / acc.games << "%";
                at_least -= acc.split[k];
            }
        }
        std::cout << out.str() << std::endl << std::endl;
    }

    std::mutex mutex_;
    size_t block_;
    Acc block_acc_, total_acc_, block_hard_, block_upgraded_;
    size_t all_games_ = 0, all_moves_ = 0;
    double move_time_ = 0, move_time_max_ = 0;
    size_t moves_over_limit_ = 0;
    uint64_t ply_sum_ = 0, ply_moves_ = 0, downgraded_ = 0;
    uint64_t wide_games_ = 0, wide_65536_ = 0; // games kept on a WideBoard (--wide), those reaching 65536
    Clock::time_point start_, block_start_;
};

// prints a checkpoint's metadata as key=value pairs (used by the Kaggle driver)
int PrintCheckpointInfo(const std::string &path) {
    NTupleNetwork network(1, 0, {});
    TrainingState state;
    std::vector<bool> initialized = network.LoadHeader(path, state);
    std::cout << "stages=" << network.NumStages() << " active=" << state.active << " features=" << network.Features()
              << " alpha=" << state.alpha << " lambda=" << state.lambda << " games=";
    for (size_t s = 0; s < state.games.size(); ++s) std::cout << (s ? "," : "") << state.games[s];
    std::cout << " initialized=";
    for (int s = 0; s < network.NumStages(); ++s) std::cout << (s ? "," : "") << initialized[s];
    std::cout << std::endl;
    return 0;
}

// --merge=<out> <base> <k>=<ckpt> ...: base with stage k from each part (Merge.h)
int MergeCheckpoints(int argc, const char *argv[]) {
    std::string out, base_path;
    std::vector<std::pair<int, std::string>> parts;
    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        size_t eq = para.find('=');
        if (para.find("--merge=") == 0) out = para.substr(8);
        else if (para.find("--") == 0) throw std::invalid_argument("--merge takes no other option: " + para);
        else if (eq == std::string::npos) {
            if (!base_path.empty()) throw std::invalid_argument("--merge takes one base checkpoint");
            base_path = para;
        } else {
            parts.emplace_back(std::stoi(para.substr(0, eq)), para.substr(eq + 1));
        }
    }
    if (out.empty() || base_path.empty() || parts.empty())
        throw std::invalid_argument("usage: --merge=<out> <base.ckpt> <stage>=<ckpt> [<stage>=<ckpt> ...]");
    MergeFiles(base_path, parts, out, std::cerr);
    return PrintCheckpointInfo(out);
}

/**
 * --explain=<games.txt or -> --explain-out=<file> --play="<the eval's args>" [--explain-game=<i,j,...>]
 * [--explain-from=<step>]: replay recorded games (RecordGame format) and append to each slide line at
 * or after step the values the player weighed when it chose that slide, each in U,R,D,L order with
 * "-" for an illegal move:
 *   <step> <move> <reward> <score> <board> v=<search values> n=<network values> ply=<depth> [dg]
 * v: reward + expected value by the player's search (what it maximised); n: reward + network value
 * of the afterstate (1-ply); dg: valued on the downgraded board. Other lines are copied unchanged.
 * Warns about any slide that is not the first best move by v (another checkpoint or other args).
 */
int ExplainGames(int argc, const char *argv[]) {
    std::string in_path, out_path, play_args, games_arg;
    long from = 0;
    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        std::string value = para.substr(para.find('=') + 1);
        if (para.find("--explain=") == 0) in_path = value;
        else if (para.find("--explain-out=") == 0) out_path = value;
        else if (para.find("--explain-game=") == 0) games_arg = value;
        else if (para.find("--explain-from=") == 0) from = std::stol(value);
        else if (para.find("--play=") == 0) play_args = value;
        else throw std::invalid_argument("--explain takes no other option: " + para);
    }
    std::vector<long> wanted;
    for (size_t at = 0; at < games_arg.size();) {
        size_t comma = std::min(games_arg.find(',', at), games_arg.size());
        wanted.push_back(std::stol(games_arg.substr(at, comma - at)));
        at = comma + 1;
    }
    std::ifstream file;
    if (in_path != "-") {
        file.open(in_path);
        if (!file) throw std::invalid_argument("cannot read " + in_path);
    }
    std::istream &in = in_path == "-" ? std::cin : file;
    if (out_path.empty()) throw std::invalid_argument("--explain needs --explain-out=<file>");
    // written as <out>.part and renamed when done, so a file with the final name is always complete
    const std::string part_path = out_path + ".part";
    std::ofstream out(part_path);
    if (!out) throw std::invalid_argument("cannot write " + part_path);

    TdPlayer player(play_args);
    // base-32 digits, so --wide records (65536 = 'g') load too; the agent sees Fit() of the board
    auto parse = [](const std::string &hex) {
        if (hex.size() != 16) throw std::invalid_argument("bad board: " + hex);
        WideBoard b;
        for (int i = 0; i < 16; ++i) b.cell[i] = uint8_t(std::stoi(hex.substr(i, 1), nullptr, 32));
        return b;
    };
    auto list = [](const float v[4]) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1);
        for (int d = 0; d < 4; ++d) {
            if (d) s << ',';
            if (v[d] == -INFINITY) s << '-';
            else s << v[d];
        }
        return s.str();
    };

    std::string line;
    bool keep = false;
    long game = -1;
    WideBoard board;
    size_t explained = 0, differ = 0, games = 0, over_100 = 0;
    double slowest_ms = 0;
    long slowest_game = -1, slowest_step = -1;
    auto start = std::chrono::steady_clock::now();
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string first;
        fields >> first;
        if (first == "game") {
            fields >> game;
            keep = wanted.empty() || std::find(wanted.begin(), wanted.end(), game) != wanted.end();
            if (keep) {
                player.OpenEpisode();
                games++;
            }
        } else if (first == "start" && keep) {
            std::string hex;
            fields >> hex;
            board = parse(hex);
        } else if (!first.empty() && keep && std::isdigit(static_cast<unsigned char>(first[0]))) {
            long step = std::stol(first);
            std::string move, reward, score, hex;
            fields >> move >> reward >> score >> hex;
            WideBoard after = parse(hex);
            int slide = std::string("URDL").find(move);
            if (move.size() == 1 && slide >= 0) {
                if (step < from) {
                    player.Observe(Board64(Fit(board)));
                } else {
                    auto t0 = std::chrono::steady_clock::now();
                    TdPlayer::Explanation e = player.Explain(Board64(Fit(board)));
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (ms > slowest_ms) { slowest_ms = ms; slowest_game = game; slowest_step = step; }
                    over_100 += ms > 100;
                    int best = int(std::max_element(e.search, e.search + 4) - e.search);
                    if (best != slide && differ++ < 10) {
                        std::cerr << "game " << game << " step " << step << ": played " << move << ", search prefers "
                                  << "URDL"[best] << " (" << list(e.search) << ")" << std::endl;
                    }
                    line += " v=" + list(e.search) + " n=" + list(e.network) + " ply=" + std::to_string(e.ply);
                    if (e.downgraded) line += " dg";
                    explained++;
                }
            }
            board = after;
        }
        if (keep) out << line << '\n';
    }
    out.close();
    if (!out || std::rename(part_path.c_str(), out_path.c_str()) != 0)
        throw std::runtime_error("cannot finish " + out_path + " (partial output left in " + part_path + ")");
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "explained " << explained << " moves in " << games << " game(s) in " << std::fixed << std::setprecision(1)
              << secs << " s; " << differ << " differ from the search's choice; slowest move " << slowest_ms << " ms (game "
              << slowest_game << " step " << slowest_step << "), " << over_100 << " over 100 ms" << std::endl;
    return differ ? 2 : 0;
}

/**
 * --tree=<board hex> --play="<args>" [--tree-ply=2] [--tree-stage=k]: print the whole expectimax tree
 * the player searches on one board (agent to move): every move with its reward, every spawn, and
 * every leaf with the board the network values (halved under downgrade= or leafdg=) and its value.
 * Mirrors Search's arithmetic without the cache, and checks the root totals against Search itself.
 * --tree-stage: the stage the game had reached (default: the board's own split).
 */
struct TreePrinter {
    const TdLearner &learner;
    int leaf_rank;
    std::ostream &out;

    static std::string Tiles(board_t b) {
        static const char *label[16] = {".", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1k", "2k", "4k", "8k", "16k", "32k"};
        std::string s;
        for (int i = 0; i < 16; ++i) {
            if (i && i % 4 == 0) s += " |";
            std::string t = label[(b >> (4 * i)) & 0xf];
            s += std::string(4 - std::min<size_t>(4, t.size()), ' ') + t;
        }
        return s;
    }
    static std::string Num(float v) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << v;
        return s.str();
    }
    std::string Pad(int depth) const { return std::string(4 * depth, ' '); }

    float Leaf(board_t b, int stage, int depth, int dir, reward_t reward) {
        const board_t real = b;
        std::string note;
        if (leaf_rank > 0) {
            board_t d = DowngradeRoot(b, leaf_rank);
            if (d != b) {
                note = "  halved:" + Tiles(d);
                b = d;
                stage = learner.Split(d);
            }
        }
        float v = learner.Evaluate(b, stage);
        out << Pad(depth) << "URDL"[dir] << " +" << reward << "  leaf:" << Tiles(real) << note
            << "  stage " << stage << "  net " << Num(v) << "  = " << Num(reward + v) << "\n";
        return v;
    }
    float Max(const Board64 &board, int ply, int stage, int depth) {
        float best = -INFINITY;
        int best_d = -1;
        for (int d = 0; d < 4; ++d) {
            Board64 child = board;
            reward_t reward = child.Slide(d);
            if (reward == -1) continue;
            float value;
            if (ply <= 1) {
                value = reward + Leaf(child.GetBoard(), stage, depth, d, reward);
            } else {
                out << Pad(depth) << "URDL"[d] << " +" << reward << "  after:" << Tiles(child.GetBoard()) << "\n";
                value = reward + Chance(child, ply - 1, stage, depth + 1);
                out << Pad(depth) << "= " << "URDL"[d] << " " << Num(value) << "\n";
            }
            if (value > best) { best = value; best_d = d; }
        }
        if (best_d == -1) {
            out << Pad(depth) << "no legal move: game over, value 0\n";
            return 0.0f;
        }
        return best;
    }
    float Chance(const Board64 &after, int ply, int stage, int depth) {
        stage = std::max(stage, learner.Split(after.GetBoard()));
        int n = after.CountEmpty();
        float total = 0;
        for (int i = 0; i < 16; ++i) {
            if (after(i) != 0) continue;
            for (int tile = 1; tile <= 2; ++tile) {
                Board64 b = after;
                b.Set(i, tile);
                out << Pad(depth) << (tile == 1 ? "2" : "4") << " at cell " << i << " (p " << (tile == 1 ? "0.9" : "0.1")
                    << "/" << n << ")\n";
                float v = Max(b, ply, stage, depth + 1);
                out << Pad(depth) << "  best " << Num(v) << "\n";
                total += (tile == 1 ? 0.9f : 0.1f) * v;
            }
        }
        return n ? total / float(n) : 0.0f;
    }
};

int PrintTree(int argc, const char *argv[]) {
    std::string hex, play_args;
    int ply = 2, stage_arg = -1;
    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        std::string value = para.substr(para.find('=') + 1);
        if (para.find("--tree=") == 0) hex = value;
        else if (para.find("--tree-ply=") == 0) ply = std::stoi(value);
        else if (para.find("--tree-stage=") == 0) stage_arg = std::stoi(value);
        else if (para.find("--play=") == 0) play_args = value;
        else throw std::invalid_argument("--tree takes no other option: " + para);
    }
    if (hex.size() != 16 || ply < 1 || ply > 3) throw std::invalid_argument("usage: --tree=<16 hex digits> --tree-ply=1..3");
    board_t real = 0;
    for (int i = 0; i < 16; ++i) real |= board_t(std::stoi(hex.substr(i, 1), nullptr, 16)) << (4 * i);
    TdPlayer player(play_args + " ply=" + std::to_string(ply));
    auto learner = player.shared_learner();
    if (stage_arg > 0) { // a board at that split, so the game stage starts there
        board_t s = 0;
        for (int k = 0; k < stage_arg; ++k) s |= board_t(14 - k) << (4 * k);
        player.Observe(Board64(s));
    }
    player.Observe(Board64(real));
    int stage = std::max(stage_arg, learner->Split(real));
    board_t root = real;
    std::ostream &out = std::cout;
    out << "real root:" << TreePrinter::Tiles(real) << "\n";
    if (player.DowngradeRank() > 0) {
        board_t d = DowngradeRoot(real, player.DowngradeRank());
        if (d != real) {
            root = d;
            stage = learner->Split(d);
            out << "root downgrade, the whole tree plays the halved board:" << TreePrinter::Tiles(d) << "\n";
        }
    } else if (player.LeafDowngradeRank() > 0) {
        out << "leaf downgrade: the tree plays the real board, each leaf is halved on its own\n";
    }
    out << "stage " << stage << ", " << ply << "-ply\n\n";
    TreePrinter printer{*learner, player.LeafDowngradeRank(), out};
    float printed[4];
    for (int d = 0; d < 4; ++d) {
        Board64 child(root);
        reward_t reward = child.Slide(d);
        printed[d] = -INFINITY;
        if (reward == -1) continue;
        out << "URDL"[d] << " +" << reward << "  after:" << TreePrinter::Tiles(child.GetBoard()) << "\n";
        printed[d] = reward + (ply <= 1 ? printer.Leaf(child.GetBoard(), stage, 1, d, reward)
                                        : printer.Chance(child, ply - 1, stage, 1));
        out << "== " << "URDL"[d] << " " << TreePrinter::Num(printed[d]) << "\n\n";
    }
    TdPlayer::Explanation e = player.Explain(Board64(real));
    bool same = true;
    out << "search values:";
    for (int d = 0; d < 4; ++d) {
        out << ' ' << "URDL"[d] << '=' << (e.search[d] == -INFINITY ? std::string("-") : TreePrinter::Num(e.search[d]));
        same = same && (e.search[d] == printed[d] || std::fabs(e.search[d] - printed[d]) < 1e-3f * std::fabs(e.search[d]) + 0.05f);
    }
    out << (same ? "  (match the tree)" : "  (DIFFER from the tree)") << "\n";
    return same ? 0 : 2;
}

int run(int argc, const char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]).find("--merge=") == 0) return MergeCheckpoints(argc, argv);
        if (std::string(argv[i]).find("--explain=") == 0) return ExplainGames(argc, argv);
        if (std::string(argv[i]).find("--tree=") == 0) return PrintTree(argc, argv);
    }
    size_t total = 1000;
    size_t block = 0;
    size_t save_every = 0;
    size_t collect_count = 100000;
    unsigned threads = 1;
    double time_limit = 0;
    bool learning = false;
    bool summary = false;
    bool wide = false;
    bool seeded = false;
    unsigned seed = 0;
    double move_limit_ms = 100;
    double hard = 0, upgrade = 0;
    unsigned hard_rewind = 500;
    size_t hard_pool = 100000;

    std::string play_args;
    std::string evil_args;
    std::string start_path;
    std::string collect_path;
    std::string record_path;
    unsigned record_tile = 32768;

    for (int i = 1; i < argc; i++) {
        std::string para(argv[i]);
        std::string value = para.substr(para.find("=") + 1);
        if (para.find("--total=") == 0) {
            total = std::stoull(value);
        } else if (para.find("--block=") == 0) {
            block = std::stoull(value);
        } else if (para.find("--play=") == 0) {
            play_args = value;
        } else if (para.find("--evil=") == 0) {
            evil_args = value;
        } else if (para.find("--learn") == 0) {
            learning = true;
        } else if (para.find("--save-every=") == 0) {
            save_every = std::stoull(value);
        } else if (para.find("--time-limit=") == 0) {
            time_limit = std::stod(value);
        } else if (para.find("--threads=") == 0) {
            threads = unsigned(std::stoul(value));
        } else if (para.find("--start=") == 0) {
            start_path = value;
        } else if (para.find("--collect=") == 0) {
            collect_path = value;
        } else if (para.find("--collect-count=") == 0) {
            collect_count = std::stoull(value);
        } else if (para.find("--info=") == 0) {
            return PrintCheckpointInfo(value);
        } else if (para.find("--seed=") == 0) {
            seeded = true;
            seed = unsigned(std::stoul(value));
        } else if (para.find("--move-limit-ms=") == 0) {
            move_limit_ms = std::stod(value);
        } else if (para.find("--hard=") == 0) {
            hard = std::stod(value);
        } else if (para.find("--hard-rewind=") == 0) {
            hard_rewind = unsigned(std::stoul(value));
        } else if (para.find("--hard-pool=") == 0) {
            hard_pool = std::stoull(value);
        } else if (para.find("--upgrade=") == 0) {
            upgrade = std::stod(value);
        } else if (para.find("--record=") == 0) {
            record_path = value;
        } else if (para.find("--record-tile=") == 0) {
            record_tile = unsigned(std::stoul(value));
        } else if (para.find("--summary") == 0) {
            summary = true;
        } else if (para == "--wide") {
            wide = true;
        } else if (para.find("--shell") == 0) {
            return shell(argc, argv);
        } else {
            throw std::invalid_argument("unknown option " + para);
        }
    }
    if (learning && collect_path.size()) throw std::invalid_argument("--learn and --collect are separate phases");
    if (threads < 1) throw std::invalid_argument("--threads must be >= 1");
    if ((hard > 0 || upgrade > 0) && !learning) throw std::invalid_argument("--hard and --upgrade are for --learn");
    if (hard < 0 || hard > 1 || upgrade < 0 || upgrade > 1) throw std::invalid_argument("--hard and --upgrade are probabilities");
    if (hard > 0 && (hard_rewind < 1 || hard_pool < 1)) throw std::invalid_argument("--hard-rewind and --hard-pool must be >= 1");
    if (record_path.size() && learning) throw std::invalid_argument("--record is for evaluation, not --learn");
    if (wide && learning) throw std::invalid_argument("--wide is for evaluation, not --learn");
    if (!block) block = total;

    // one shared learner (Hogwild), one player and environment per thread
    TdPlayer prototype(play_args);
    std::shared_ptr<TdLearner> learner = prototype.shared_learner();
    int active = learner->ActiveStage();

    if (learning && !learner->CanSave()) {
        std::cerr << "warning: --learn without save=... in --play, the trained weights will be lost" << std::endl;
    }

    // Stage k > 0 is trained from boards collected at its split ([MS-TD] Sec. III), round robin.
    std::vector<StartPosition> starts;
    if (start_path.size()) {
        int start_stage;
        starts = StartPositions::Load(start_path, start_stage);
        if (start_stage != active) {
            throw std::invalid_argument(start_path + " starts stage " + std::to_string(start_stage) +
                                        " but the active stage is " + std::to_string(active));
        }
        std::cerr << "start positions: " << starts.size() << " boards for stage " << start_stage << std::endl;
    } else if (active > 0 && (learning || collect_path.size())) {
        throw std::invalid_argument("stage " + std::to_string(active) + " needs --start=<boards collected at its split>");
    }

    // Collection: play with the current weights and record the board (player to move) the first
    // time a game reaches the next split, e.g. T_16k when the active stage is 0.
    std::unique_ptr<StartPositions::Writer> collector;
    std::mutex collector_mutex;
    if (collect_path.size()) {
        if (active + 1 >= learner->NumStages()) throw std::invalid_argument("the active stage is the last one, nothing to collect");
        collector.reset(new StartPositions::Writer(collect_path, active + 1));
        std::cerr << "collecting boards for stage " << active + 1 << " into " << collect_path << " ("
                  << collector->count() << "/" << collect_count << " distinct boards present";
        if (collector->dropped()) std::cerr << ", " << collector->dropped() << " repeated boards removed from the file";
        std::cerr << ")" << std::endl;
    }
    std::ofstream record;
    std::mutex record_mutex;
    if (record_path.size()) {
        record.open(record_path);
        if (!record) throw std::invalid_argument("cannot write " + record_path);
        std::cerr << "recording games that reach " << record_tile << " into " << record_path << std::endl;
    }

    auto collection_done = [&]() {
        if (!collector) return false;
        std::lock_guard<std::mutex> lock(collector_mutex);
        return collector->count() >= collect_count;
    };

    // Kaggle (and most schedulers) send SIGTERM before killing; finish the games, save, exit.
    std::signal(SIGTERM, RequestStop);
    std::signal(SIGINT, RequestStop);
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<bool> out_of_time{false};
    auto should_stop = [&]() { return stop_requested || out_of_time.load() || collection_done(); };

    if (upgrade > 0 && active + 1 >= learner->NumStages())
        throw std::invalid_argument("--upgrade needs a next split: the active stage is the last one");
    HardStarts hard_starts(hard_pool, hard_rewind);
    if (hard > 0 || upgrade > 0)
        std::cerr << "hard starts: " << hard << " of games, rewind 1.." << hard_rewind << ", pool " << hard_pool
                  << ", upgrade " << upgrade << " (progress lines: normal starts, then one line per special kind)" << std::endl;

    Reporter reporter(block);
    const uint64_t round_robin_base = learner->ActiveGames();
    std::atomic<uint64_t> claimed{0}; // games handed out in this run

    std::cerr << "running " << total << " games on " << threads << " thread(s)" << std::endl;

    // claim the next game index if it is below limit
    auto claim = [&](uint64_t limit, uint64_t &n) {
        n = claimed.load();
        while (n < limit) {
            if (claimed.compare_exchange_weak(n, n + 1)) return true;
        }
        return false;
    };

    auto worker = [&](unsigned id, uint64_t limit) {
        TdPlayer player(learner, play_args);
        std::string evil_seeded = evil_args;
        {
            RandomEnvironment probe(evil_args);
            if (evil_args.find("seed=") != std::string::npos) {
                long seed = std::stol(probe.property("seed"));
                evil_seeded += " seed=" + std::to_string(seed + long(id) * 1000003L);
            }
        }
        RandomEnvironment evil(evil_seeded);
        std::mt19937_64 start_rng(std::random_device{}() + id);
        std::mt19937_64 hard_rng(std::random_device{}() + 7919u * id); // own stream: hard=0 changes nothing
        std::uniform_real_distribution<double> coin(0, 1);
        auto split = [&](board_t b) { return learner->Split(b); };

        uint64_t n;
        while (!should_stop() && claim(limit, n)) {
            Episode game;
            if (seeded) { // the same games for every run: seed by game index, not by thread
                evil.Seed(seed + unsigned(n));
                start_rng.seed(seed + n);
            }
            Reporter::Start kind = Reporter::kNormal; // hard or upgraded: own progress lines, not in the pool
            StartPosition hp;
            if (hard > 0 && coin(hard_rng) < hard && hard_starts.Sample(hard_rng, hp)) {
                game.Start(hp.board, hp.score);
                kind = Reporter::kHard;
            } else if (!starts.empty()) {
                size_t i = learning ? size_t((round_robin_base + n) % starts.size()) : size_t(start_rng() % starts.size());
                game.Start(starts[i].board, starts[i].score);
            }
            if (upgrade > 0 && game.FirstSlide() == 0 && coin(hard_rng) < upgrade) {
                board_t up = UpgradeTile(game.StartBoard(), active);
                if (up != game.StartBoard()) {
                    game.Start(up, game.score());
                    kind = Reporter::kUpgraded;
                }
            }
            if (wide) game.UseWideBoard();
            player.OpenEpisode();
            evil.OpenEpisode();

            double move_time = 0, move_time_max = 0;
            size_t over_limit = 0;
            while (true) {
                Agent &agent = game.TakeTurns(player, evil);
                Action move;
                if (!learning && &agent == &player) {
                    auto t0 = std::chrono::steady_clock::now();
                    move = agent.TakeAction(game.state());
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    move_time += dt;
                    move_time_max = std::max(move_time_max, dt);
                    over_limit += dt * 1e3 > move_limit_ms;
                } else {
                    move = agent.TakeAction(game.state());
                }
                if (!game.ApplyAction(move)) break;
                if (collector && &agent == &evil && learner->Split(game.state().GetBoard()) > active) {
                    std::lock_guard<std::mutex> lock(collector_mutex);
                    if (collector->count() < collect_count) collector->Add(game.state().GetBoard(), game.score());
                    break;
                }
            }

            if (learning) learner->Learn(game);
            if (!learning) reporter.AddMoveTimes(move_time, move_time_max, over_limit, player.TakeMoveStats());
            reporter.Add(game, kind);
            int top = wide ? game.WideState().MaxRank() : game.state().GetMaxTile();
            if (record.is_open() && (1ul << top) >= record_tile) {
                std::lock_guard<std::mutex> lock(record_mutex);
                RecordGame(record, game, n, seeded ? long(seed + unsigned(n)) : -1L);
            }
            if (hard > 0 && kind == Reporter::kNormal) hard_starts.Offer(game, active, split, hard_rng);

            if (time_limit > 0) {
                std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed.count() >= time_limit) out_of_time = true;
            }
        }
    };

    // Rounds end at each save point so the checkpoint is written while no thread updates weights.
    while (claimed.load() < total && !should_stop()) {
        uint64_t limit = total;
        if (learning && save_every) limit = std::min<uint64_t>(total, claimed.load() + save_every);

        std::vector<std::thread> pool;
        for (unsigned id = 0; id < threads; ++id) pool.emplace_back(worker, id, limit);
        for (std::thread &t : pool) t.join();

        if (learning && claimed.load() < total && !should_stop()) learner->Save();
    }

    if (stop_requested) std::cerr << "stop requested after " << claimed.load() << " games" << std::endl;
    if (out_of_time) std::cerr << "time limit reached after " << claimed.load() << " games" << std::endl;
    if (collector) std::cerr << "collected " << collector->count() << " distinct boards (" << collector->repeats()
                             << " repeats of boards already collected skipped)" << std::endl;
    if (hard > 0) std::cerr << "hard starts: " << hard_starts.Offered() << " boards offered to the pool" << std::endl;

    if (learning) learner->Save();
    if (summary) reporter.Summary(move_limit_ms);
    return 0;
}

int main(int argc, const char *argv[]) {
    InitLookUpTables();

    std::cout << "sixteen: ";
    std::copy(argv, argv + argc, std::ostream_iterator<const char *>(std::cout, " "));
    std::cout << std::endl << std::endl;

    try {
        return run(argc, argv);
    } catch (std::exception &ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
}
