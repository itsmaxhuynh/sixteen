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
 *   common: --threads=N --total=N --block=N --save-every=N --time-limit=SEC --collect-count=N --summary
 * Evaluate:  ./sixteen --total=1000 --seed=1 --summary --play="load=w.ckpt ply=3"
 *   --seed=N: game i uses environment seed N+i, so every checkpoint plays the same games whatever
 *   the thread count; --summary then adds an "eval:" line with 95% intervals and move times
 *   (--move-limit-ms=100 sets the threshold counted as over the limit). See eval.sh.
 * Inspect:   ./sixteen --info=w.ckpt
 * Match:     --play="load=w.ckpt time=50"  iterative deepening within 50 ms per move (see Agent.h)
 * Arena:     ./sixteen --shell --name=<account> --play="load=w.ckpt ply=3"
 */

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
#include "StartPositions.h"
#include "arena.h"
#include "io.h"

static std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler needs a lock-free flag");

static void RequestStop(int) { stop_requested = true; }

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
 * Thread-safe progress report, printed every `block` games in the format of the Threes
 * Statistic: average and max score, throughput, and per tile the reach rate and end rate.
 */
class Reporter {
public:
    explicit Reporter(size_t block) : block_(block), start_(Clock::now()), block_start_(start_) {}

    void Add(const Episode &game) {
        int split = MaxSplit(game);
        std::lock_guard<std::mutex> lock(mutex_);
        Accumulate(block_acc_, game, split);
        Accumulate(total_acc_, game, split);
        if (block_acc_.games == block_) {
            auto now = Clock::now();
            Print(block_acc_, total_acc_.games, std::chrono::duration<double>(now - block_start_).count());
            block_acc_ = {};
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
        Print(total_acc_, total_acc_.games, std::chrono::duration<double>(Clock::now() - start_).count());
        const Acc &a = total_acc_;
        if (a.games == 0) return;
        // mean score with a 95% normal interval; reach rates with a 95% Wilson interval
        double n = double(a.games), mean = a.score / n;
        double var = n > 1 ? (a.score_sq - n * mean * mean) / (n - 1) : 0;
        double half = 1.96 * std::sqrt(std::max(var, 0.0) / n);
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << "eval: games=" << a.games << " mean=" << mean << " ci95=" << half;
        size_t reached = a.games;
        for (int t = 0; t < 16; ++t) {
            if (t >= 13) { // 8192, 16384, 32768
                double p = reached / n, z = 1.96, d = 1 + z * z / n;
                double c = (p + z * z / (2 * n)) / d, h = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
                out << std::setprecision(2) << " r" << (1 << t) << "=" << 100 * p << "%[" << 100 * std::max(0.0, c - h) << ","
                    << 100 * std::min(1.0, c + h) << "]";
            }
            reached -= a.end_tile[t];
        }
        if (a.moves && move_time_ > 0) {
            out << std::setprecision(1) << " move_us=" << 1e6 * move_time_ / double(a.moves)
                << std::setprecision(2) << " max_move_ms=" << 1e3 * move_time_max_ << " over_" << std::setprecision(0)
                << move_limit_ms << "ms=" << moves_over_limit_;
        }
        if (ply_moves_) out << std::setprecision(2) << " avg_ply=" << double(ply_sum_) / double(ply_moves_);
        if (downgraded_) out << " downgraded_moves=" << downgraded_;
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

    static void Print(const Acc &acc, size_t index, double seconds) {
        if (acc.games == 0) return;
        std::ostringstream out;
        out << std::fixed << std::setprecision(0) << index << "\tavg = " << acc.score / acc.games << ", max = " << acc.max
            << ", games/s = " << std::setprecision(1) << acc.games / seconds
            << ", moves/s = " << std::setprecision(0) << acc.moves / seconds << std::endl;
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

    std::mutex mutex_;
    size_t block_;
    Acc block_acc_, total_acc_;
    double move_time_ = 0, move_time_max_ = 0;
    size_t moves_over_limit_ = 0;
    uint64_t ply_sum_ = 0, ply_moves_ = 0, downgraded_ = 0;
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

int run(int argc, const char *argv[]) {
    size_t total = 1000;
    size_t block = 0;
    size_t save_every = 0;
    size_t collect_count = 100000;
    unsigned threads = 1;
    double time_limit = 0;
    bool learning = false;
    bool summary = false;
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
        } else if (para.find("--summary") == 0) {
            summary = true;
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
                  << collector->count() << "/" << collect_count << " present)" << std::endl;
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
                  << ", upgrade " << upgrade << " (progress lines count normal starts only)" << std::endl;

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
            bool special = false; // hard or upgraded start: kept out of the progress lines and the pool
            StartPosition hp;
            if (hard > 0 && coin(hard_rng) < hard && hard_starts.Sample(hard_rng, hp)) {
                game.Start(hp.board, hp.score);
                special = true;
            } else if (!starts.empty()) {
                size_t i = learning ? size_t((round_robin_base + n) % starts.size()) : size_t(start_rng() % starts.size());
                game.Start(starts[i].board, starts[i].score);
            }
            if (upgrade > 0 && game.FirstSlide() == 0 && coin(hard_rng) < upgrade) {
                board_t up = UpgradeTile(game.StartBoard(), active);
                if (up != game.StartBoard()) {
                    game.Start(up, game.score());
                    special = true;
                }
            }
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
            if (!special) reporter.Add(game);
            if (hard > 0 && !special) hard_starts.Offer(game, active, split, hard_rng);

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
    if (collector) std::cerr << "collected " << collector->count() << " boards" << std::endl;
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
