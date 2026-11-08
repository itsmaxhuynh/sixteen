#pragma once

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "NTupleNetwork.h"

/**
 * Merging checkpoints whose stages were trained in separate sessions.
 *
 * Once every stage exists, a stage k > 0 trains on its own: its games start from the boards
 * collected at its split, and every move and value inside them uses stage k's weights (TdLearner
 * caps the stage at the active one), so the other stages are never read. Sessions can then train
 * different stages of the same base checkpoint at the same time, and MergeFiles puts the results
 * back together: base stage k is replaced by the part's stage k, and so is its game count.
 *
 * A part must come from this exact base: every other stage must be bit-identical to the base's,
 * with the same game count, and its own stage must have more games than the base's. Anything else
 * (another run's checkpoint, the wrong stage, a session that trained nothing) throws.
 */
// throws unless part is a session of `stage` trained from base (see above)
inline void CheckPart(const NTupleNetwork &base, const TrainingState &base_state, const NTupleNetwork &part,
                      const TrainingState &part_state, int stage, const std::string &name) {
    if (stage < 0 || stage >= base.NumStages()) throw std::invalid_argument(name + ": stage " + std::to_string(stage) + " out of range");
    if (!base.SameLayout(part)) throw std::invalid_argument(name + ": tuples, features or stages differ from the base");
    if (!part.Initialized(stage)) throw std::invalid_argument(name + ": stage " + std::to_string(stage) + " is not trained");
    for (int s = 0; s < base.NumStages(); ++s) {
        if (s == stage) continue;
        if (!base.SameStage(part, s) || base_state.games[s] != part_state.games[s]) {
            throw std::invalid_argument(name + ": stage " + std::to_string(s) + " differs from the base, so it is not a "
                                        "session of stage " + std::to_string(stage) + " trained from this base");
        }
    }
    if (part_state.games[stage] <= base_state.games[stage]) {
        throw std::invalid_argument(name + ": stage " + std::to_string(stage) + " has " +
                                    std::to_string(part_state.games[stage]) + " games, no more than the base's " +
                                    std::to_string(base_state.games[stage]));
    }
}

/**
 * Writes `out`: the checkpoint at base_path with stage k (weights and game count) taken from each
 * (k, path) part. Every part is checked against the unchanged base before anything is replaced, so
 * an error leaves `out` unwritten. Holds the base plus one stage per part in memory, not every part.
 * Returns the merged training state.
 */
inline TrainingState MergeFiles(const std::string &base_path, const std::vector<std::pair<int, std::string>> &parts,
                                const std::string &out, std::ostream &log) {
    for (size_t i = 0; i < parts.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (parts[i].first == parts[j].first) throw std::invalid_argument("stage " + std::to_string(parts[i].first) + " given twice");
        }
    }
    NTupleNetwork base(1, 0, {});
    TrainingState state;
    base.Load(base_path, state);
    log << "base " << base_path << std::endl;
    std::vector<NTupleNetwork> taken; // per part: a network holding only its stage
    std::vector<uint64_t> games;
    for (const auto &part : parts) {
        NTupleNetwork net(1, 0, {});
        TrainingState part_state;
        net.Load(part.second, part_state);
        CheckPart(base, state, net, part_state, part.first, part.second);
        for (int s = 0; s < net.NumStages(); ++s) {
            if (s != part.first) net.DropStage(s);
        }
        taken.push_back(std::move(net));
        games.push_back(part_state.games[part.first]);
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        int k = parts[i].first;
        log << "stage " << k << " from " << parts[i].second << ": " << state.games[k] << " -> " << games[i] << " games" << std::endl;
        base.CopyStage(taken[i], k);
        state.games[k] = games[i];
    }
    base.Save(out, state);
    log << "saved " << out << std::endl;
    return state;
}
