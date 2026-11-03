#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <cmath>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "Action.h"
#include "Board64.h"
#include "Common.h"
#include "Episode.h"
#include "NTupleNetwork.h"
#include "Search.h"

class Agent {
public:
    virtual ~Agent() {}

    virtual void OpenEpisode(const std::string &flag = "") {}

    virtual void CloseEpisode(const std::string &flag = "") {}

    virtual Action TakeAction(const Board64 &b) { return Action(); }

    virtual bool CheckForWin(const Board64 &b) { return false; }

    Agent(const std::string &args = "") {
        std::stringstream ss("name=unknown role=unknown " + args);
        for (std::string pair; ss >> pair;) {
            std::string key = pair.substr(0, pair.find('='));
            std::string value = pair.substr(pair.find('=') + 1);
            meta_[key] = {value};
        }
    }

    virtual std::string property(const std::string &key) const { return meta_.at(key); }

    virtual void notify(const std::string &msg) {
        meta_[msg.substr(0, msg.find('='))] = {msg.substr(msg.find('=') + 1)};
    }

    virtual std::string name() const { return property("name"); }

    virtual std::string role() const { return property("role"); }

protected:
    bool Has(const std::string &key) const { return meta_.find(key) != meta_.end(); }

    typedef std::string key;

    struct value {
        std::string value;

        operator std::string() const { return value; }

        template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
        operator numeric() const { return numeric(std::stod(value)); }
    };

    std::map<key, value> meta_;
};

class RandomAgent : public Agent {
public:
    RandomAgent(const std::string &args = "") : Agent(args) {
        if (Has("seed")) engine_.seed(unsigned(int(meta_["seed"])));
        else engine_.seed(std::random_device()());
    }

    // reseed, e.g. per game so evaluation games are the same whatever thread plays them
    void Seed(unsigned seed) { engine_.seed(seed); }

protected:
    std::mt19937 engine_;
};

/**
 * Standard 2048 environment: a 2 (90%) or a 4 (10%) on a uniformly random empty cell.
 */
class RandomEnvironment : public RandomAgent {
public:
    RandomEnvironment(const std::string &args = "") : RandomAgent("name=random role=environment " + args) {}

    Action TakeAction(const Board64 &board) override {
        int empty[16];
        int n = 0;
        for (int i = 0; i < 16; ++i) {
            if (board(i) == 0) empty[n++] = i;
        }
        if (n == 0) return Action();
        int position = empty[engine_() % n];
        int tile = (engine_() % 10) ? 1 : 2;
        return Action::Place(position, tile);
    }
};

/**
 * The shared, trainable part of the player: network, training state and the search, following
 * [MS-TD] (see NTupleNetwork.h). One learner is shared by all worker threads (Hogwild): Learn()
 * and the search may run concurrently; Save() and SetActiveStage() must not.
 *
 * Args (key=value):
 *   load=<ckpt> save=<ckpt>
 *   alpha=<f>     per-weight step size, paper: 0.0025, then 0.00025 once saturated
 *   anneal=cos anneal_games=<G> alpha_min=<f>
 *                 cosine annealing instead of a fixed alpha: the active stage's game g (counted
 *                 across sessions) trains with alpha_min + (alpha - alpha_min)(1 + cos(pi g / G)) / 2,
 *                 then alpha_min from game G on. Off by default: alpha is used as given.
 *   init=<V>      optimistic initialization (arXiv:2111.11090 eq. 12): a stage created by this run
 *                 starts with every weight at V / (features per state), so every board is valued V
 *                 (default 0, as in [MS-TD]). Stages that already exist are not touched.
 *   lambda=<f>    paper's final version: 0.5 with returns truncated at 5 steps; 0 = TD(0)
 *   stage=<k>     0-based stage to train (and highest stage used to play); a new stage starts at zero
 *   stages=<n> features=<all|none|mask>   only used when starting fresh (default 5 / all)
 */
class TdLearner {
public:
    static constexpr int kReturns = 5; // [MS-TD] eq. (6): n-step returns R^1..R^5

    // args: the key=value pairs of --play (ply is read by TdPlayer)
    explicit TdLearner(const std::map<std::string, std::string> &args)
            : network_(Get(args, "stages") ? std::stoi(*Get(args, "stages")) : kMaxStages,
                       Get(args, "features") ? ParseFeatures(*Get(args, "features")) : kFeatAll) {
        if (const std::string *v = Get(args, "init")) init_value_ = std::stof(*v);
        if (const std::string *file = Get(args, "load")) {
            network_.Load(*file, state_);
            std::cerr << "loaded " << *file << std::endl;
        } else {
            state_.games.assign(network_.NumStages(), 0);
            InitStage(0);
        }
        if (const std::string *v = Get(args, "stage")) SetActiveStage(std::stoi(*v));
        // command line overrides what the checkpoint carried
        if (const std::string *v = Get(args, "alpha")) state_.alpha = std::stof(*v);
        if (const std::string *v = Get(args, "lambda")) state_.lambda = std::stof(*v);
        if (const std::string *v = Get(args, "save")) save_path_ = *v;
        if (const std::string *v = Get(args, "anneal")) {
            if (*v == "cos") cosine_ = true;
            else if (*v != "none") throw std::invalid_argument("anneal must be cos or none");
        }
        if (const std::string *v = Get(args, "anneal_games")) anneal_games_ = std::stod(*v);
        if (const std::string *v = Get(args, "alpha_min")) alpha_min_ = std::stof(*v);
        if (cosine_ && !(anneal_games_ > 0)) throw std::invalid_argument("anneal=cos needs anneal_games > 0");
        active_games_ = state_.games[state_.active];

        network_.PrintSummary(std::cerr);
        std::cerr << "learner: active stage " << state_.active << ", games/stage " << GamesString()
                  << ", alpha=" << state_.alpha << ", lambda=" << state_.lambda;
        if (cosine_) std::cerr << ", cosine annealing to " << alpha_min_ << " over " << anneal_games_
                               << " games (now " << Alpha(active_games_.load()) << ")";
        std::cerr << std::endl;
    }

    // step size for a game that starts when the active stage has `games` games behind it
    float Alpha(uint64_t games) const {
        if (!cosine_) return state_.alpha;
        double progress = std::min(1.0, double(games) / anneal_games_);
        return float(alpha_min_ + (state_.alpha - alpha_min_) * 0.5 * (1.0 + std::cos(M_PI * progress)));
    }

    /**
     * Learning after the game ends ([MS-TD] Sec. IV.E: "recorded all the state sequences for whole
     * games, and performed updates after the games finished"), backward over the afterstates.
     * Only weights of the active stage change; earlier stages stay frozen (Sec. III).
     */
    void Learn(const Episode &episode) {
        const std::vector<Episode::Move> &moves = episode.GetMoves();
        std::vector<board_t> after;
        std::vector<float> rewards;
        for (size_t i = episode.FirstSlide(); i < moves.size(); i += 2) {
            after.push_back(moves[i].board);
            rewards.push_back(moves[i].reward);
        }
        size_t T = after.size();
        if (T == 0) return;

        // stage of each afterstate: monotone in game time, capped at the active stage
        std::vector<int> stage(T);
        int gs = Split(episode.StartBoard());
        for (size_t t = 0; t < T; ++t) {
            gs = std::max(gs, Split(after[t]));
            stage[t] = Resolve(gs);
        }

        const float alpha = Alpha(active_games_.load(std::memory_order_relaxed));
        std::vector<float> value(T); // V(s'_t) after its own update
        for (size_t t = T; t-- > 0;) {
            float target = LambdaReturn(rewards, value, t, state_.lambda, kReturns);
            if (stage[t] == state_.active) {
                float v = network_.GetValue(after[t], stage[t]);
                network_.UpdateValue(after[t], stage[t], alpha * (target - v));
            }
            value[t] = network_.GetValue(after[t], stage[t]);
        }
        active_games_.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * [MS-TD] eqs. (3), (4), (6) for afterstates. rewards[t] is the score of the move that produced
     * s'_t, value[t] = V(s'_t). With R^n_t = r_{t+1} + ... + r_{t+n} + V(s'_{t+n}),
     *   R^lambda_t = sum_{n<N} (1-lambda) lambda^(n-1) R^n_t + lambda^(N-1) R^N_t
     * (N=5, lambda=0.5 gives 0.5, 0.25, 0.125, 0.0625, 0.0625). Past the end of the game the
     * return is just the remaining rewards (terminal value 0), and it takes all remaining weight.
     */
    static float LambdaReturn(const std::vector<float> &rewards, const std::vector<float> &value,
                              size_t t, float lambda, int n_returns) {
        size_t T = rewards.size();
        float target = 0, reward_sum = 0, lambda_pow = 1; // lambda_pow = lambda^(n-1)
        for (int n = 1; n <= n_returns; ++n) {
            size_t k = t + n;
            if (k >= T) {
                target += lambda_pow * reward_sum;
                break;
            }
            reward_sum += rewards[k];
            float weight = (n < n_returns) ? (1 - lambda) * lambda_pow : lambda_pow;
            target += weight * (reward_sum + value[k]);
            lambda_pow *= lambda;
        }
        return target;
    }

    // best move by n-ply expectimax ([MS-TD] Fig. 2); -1 if there is none
    int BestMove(const Board64 &board, int ply, int game_stage) const {
        return MaxNode(board, ply, game_stage).first;
    }

    // not thread-safe: call only while no worker is learning
    void Save() {
        if (save_path_.empty()) return;
        state_.games[state_.active] = active_games_.load();
        network_.Save(save_path_, state_);
        std::cerr << "saved " << save_path_ << " (games/stage " << GamesString();
        if (cosine_) std::cerr << ", alpha now " << Alpha(active_games_.load());
        std::cerr << ")" << std::endl;
    }

    bool CanSave() const { return !save_path_.empty(); }

    int ActiveStage() const { return state_.active; }

    uint64_t ActiveGames() const { return active_games_.load(std::memory_order_relaxed); }

    int NumStages() const { return network_.NumStages(); }

    int Split(board_t b) const { return SplitLevel(b, network_.NumSplits()); }

    // value of an afterstate reached in a game that is already in game_stage (used by Search.h)
    float Evaluate(board_t b, int game_stage) const { return Value(b, game_stage); }

private:
    static const std::string *Get(const std::map<std::string, std::string> &args, const std::string &key) {
        auto it = args.find(key);
        return it == args.end() ? nullptr : &it->second;
    }

    static uint32_t ParseFeatures(const std::string &v) {
        if (v == "all") return kFeatAll;
        if (v == "none") return 0;
        return uint32_t(std::stoul(v, nullptr, 0));
    }

    void SetActiveStage(int stage) {
        if (stage < 0 || stage >= network_.NumStages()) throw std::invalid_argument("stage out of range");
        for (int s = 0; s < stage; ++s) {
            if (!network_.Initialized(s)) throw std::invalid_argument("stage " + std::to_string(s) + " is not trained yet");
        }
        if (!network_.Initialized(stage)) InitStage(stage);
        state_.active = stage;
    }

    std::string GamesString() const {
        std::string s;
        for (size_t i = 0; i < state_.games.size(); ++i) s += (i ? "," : "") + std::to_string(state_.games[i]);
        return s;
    }

    // a new stage: zero ([MS-TD]) or optimistic, so that every board is valued init_value_
    void InitStage(int stage) {
        float w = init_value_ / float(network_.NumActiveFeatures());
        network_.InitStage(stage, w);
        std::cerr << "stage " << stage << " initialized to " << w << " per weight (boards valued "
                  << init_value_ << ")" << std::endl;
    }

    int Resolve(int game_stage) const { return std::min(game_stage, state_.active); }

    float Value(board_t b, int game_stage) const {
        return network_.GetValue(b, Resolve(std::max(game_stage, Split(b))));
    }

    // max node: the player moves; value = reward + value of the afterstate
    std::pair<int, float> MaxNode(const Board64 &board, int ply, int game_stage) const {
        int direction = -1;
        float best = -INFINITY;
        for (int d = 0; d < 4; ++d) {
            Board64 child = board;
            reward_t reward = child.Slide(d);
            if (reward == -1) continue;
            float value = reward + (ply <= 1 ? Value(child.GetBoard(), game_stage)
                                             : ChanceNode(child, ply - 1, game_stage));
            if (value > best) {
                best = value;
                direction = d;
            }
        }
        return {direction, direction == -1 ? 0.0f : best}; // no legal move: game over, value 0
    }

    // chance node (afterstate): a 2 with p=0.9 or a 4 with p=0.1 on a uniformly chosen empty cell
    float ChanceNode(const Board64 &after, int ply, int game_stage) const {
        game_stage = std::max(game_stage, Split(after.GetBoard()));
        float total = 0;
        int empty = 0;
        for (int i = 0; i < 16; ++i) {
            if (after(i) != 0) continue;
            ++empty;
            Board64 two = after, four = after;
            two.Set(i, 1);
            four.Set(i, 2);
            total += 0.9f * MaxNode(two, ply, game_stage).second;
            total += 0.1f * MaxNode(four, ply, game_stage).second;
        }
        return empty ? total / empty : 0.0f;
    }

    std::string save_path_;
    float init_value_ = 0;
    bool cosine_ = false;
    double anneal_games_ = 0;
    float alpha_min_ = 0;
    TrainingState state_;
    std::atomic<uint64_t> active_games_{0};
    NTupleNetwork network_;
};

/**
 * The agent that plays one game at a time; cheap, one per thread, sharing a TdLearner.
 * Extra args:
 *   ply=<n>      expectimax plies as in the paper: 1 = greedy over afterstates (training),
 *                2 = max, chance, max (Fig. 2), 3 = the paper's main search setting
 *   time=<ms>    iterative deepening within this budget per move instead of a fixed ply
 *                (hard=<ms> aborts an unfinished depth, default = time; maxply=<n> caps it, default 8)
 *   tt=<k>       position cache of 2^k entries x 16 bytes (default 22 = 64 MB, 0 = off)
 *   prune=<p>    do not search spawn outcomes whose path probability is below p (default 0 = off)
 *   downgrade=<tile>  tile-downgrading at the root once the largest tile is >= tile, e.g. 32768
 *                (see DowngradeRoot in Search.h; default 0 = off)
 * Search (ply > 1 or time=) goes through Search.h; ply=1 stays on the cheap training path.
 */
class TdPlayer : public Agent {
public:
    // standalone player (arena shell): builds its own learner from args
    TdPlayer(const std::string &args = "") : Agent("name=sixteen role=player " + args) {
        learner_ = std::make_shared<TdLearner>(Args());
        ReadPly();
    }

    // worker player sharing a learner
    TdPlayer(std::shared_ptr<TdLearner> learner, const std::string &args)
            : Agent("name=sixteen role=player " + args), learner_(std::move(learner)) {
        ReadPly();
    }

    // the key=value args, for building a TdLearner
    std::map<std::string, std::string> Args() const {
        std::map<std::string, std::string> args;
        for (const auto &kv : meta_) args[kv.first] = kv.second.value;
        return args;
    }

    void OpenEpisode(const std::string &flag = "") override {
        game_stage_ = 0;
        if (search_) search_->NewGame();
    }

    Action TakeAction(const Board64 &board) override {
        game_stage_ = std::max(game_stage_, learner_->Split(board.GetBoard()));
        // the search runs on the downgraded board, valued as a board of its own (stage from its tiles)
        Board64 root = board;
        int stage = game_stage_;
        if (downgrade_rank_ > 0) {
            board_t d = DowngradeRoot(board.GetBoard(), downgrade_rank_);
            if (d != board.GetBoard()) {
                root = Board64(d);
                stage = learner_->Split(d);
                downgraded_++;
            }
        }
        int move;
        if (!search_) {
            move = learner_->BestMove(root, ply_, stage);
        } else {
            Search<TdLearner>::Result r = time_ms_ > 0 ? search_->Timed(root, stage, time_ms_, hard_ms_, max_ply_)
                                                       : search_->Fixed(root, ply_, stage);
            move = r.move;
            ply_sum_ += r.ply;
            ply_moves_++;
        }
        return move == -1 ? Action() : Action(Action::Slide(move));
    }

    std::shared_ptr<TdLearner> shared_learner() const { return learner_; }

    struct MoveStats {
        uint64_t ply_sum = 0, ply_moves = 0; // depth of the searched moves
        uint64_t downgraded = 0;             // moves chosen on a downgraded board
    };

    // statistics of the moves since the last call
    MoveStats TakeMoveStats() {
        MoveStats s{ply_sum_, ply_moves_, downgraded_};
        ply_sum_ = ply_moves_ = downgraded_ = 0;
        return s;
    }

private:
    void ReadPly() {
        ply_ = Has("ply") ? int(meta_["ply"]) : 1;
        if (ply_ < 1) throw std::invalid_argument("ply must be >= 1");
        time_ms_ = Has("time") ? double(meta_["time"]) : 0;
        hard_ms_ = Has("hard") ? double(meta_["hard"]) : 0;
        max_ply_ = Has("maxply") ? int(meta_["maxply"]) : 8;
        if (max_ply_ < 1) throw std::invalid_argument("maxply must be >= 1");
        downgrade_rank_ = 0;
        if (Has("downgrade")) {
            unsigned tile = unsigned(meta_["downgrade"]);
            if (tile != 0 && (tile < 16 || (tile & (tile - 1)) || tile > 32768))
                throw std::invalid_argument("downgrade must be 0 or a tile 16..32768");
            if (tile) downgrade_rank_ = __builtin_ctz(tile);
        }
        if (ply_ > 1 || time_ms_ > 0) {
            search_.reset(new Search<TdLearner>(*learner_, Has("tt") ? int(meta_["tt"]) : 22));
            if (Has("prune")) search_->SetPrune(float(meta_["prune"]));
        }
    }

    std::shared_ptr<TdLearner> learner_;
    std::unique_ptr<Search<TdLearner>> search_;
    int ply_ = 1;
    double time_ms_ = 0, hard_ms_ = 0;
    int max_ply_ = 8;
    int game_stage_ = 0;
    int downgrade_rank_ = 0;
    uint64_t ply_sum_ = 0, ply_moves_ = 0, downgraded_ = 0;
};
