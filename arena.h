#pragma once
#include <fstream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include "Agent.h"
#include "Episode.h"

class arena {
public:
    class match : public Episode {
    public:
        match(const std::string& id, std::shared_ptr<Agent> play, std::shared_ptr<Agent> evil) : id(id), play(play), evil(evil) {}
        std::string name() const { return id; }

        Action take_action() {
            Agent& who = TakeTurns(*play, *evil);
            return who.TakeAction(state());
        }
        void open_episode(const std::string& tag) {
            play->OpenEpisode(tag);
            evil->OpenEpisode(tag);
            Episode::OpenEpisode(tag);
        }
        void close_episode(const std::string& tag) {
            Episode::CloseEpisode(tag);
            play->CloseEpisode(tag);
            evil->CloseEpisode(tag);
        }

    private:
        std::string id;
        std::shared_ptr<Agent> play;
        std::shared_ptr<Agent> evil;
    };

public:
    arena(const std::string& name = "anonymous", const std::string& path = "") : name(name), auth(name) {
        if (path.size()) set_dump_file(path);
    }

    arena::match& at(const std::string& id) {
        return *(ongoing.at(id));
    }
    bool open(const std::string& id, const std::string& tag) {
        if (ongoing.size()) return false;

        if (ongoing.find(id) != ongoing.end()) return false;

        auto play = find_agent(tag.substr(0, tag.find(':')), "play");
        auto evil = find_agent(tag.substr(tag.find(':') + 1), "evil");
        if (play->role() == "dummy" && evil->role() == "dummy") return false;

        std::shared_ptr<match> m(new match(id, play, evil));
        m->open_episode(tag);
        ongoing[id] = m;
        return true;
    }
    bool close(const std::string& id, const std::string& tag) {
        auto it = ongoing.find(id);
        if (it != ongoing.end()) {
            auto m = it->second;
            m->close_episode(tag);
            dump << (*m) << std::endl << std::flush;
            ongoing.erase(it);
            return true;
        }
        return false;
    }

public:
    bool register_agent(std::shared_ptr<Agent> a) {
        if (lounge.find(a->name()) != lounge.end()) return false;
        lounge[a->name()] = a;
        return true;
    }
    bool remove_agent(std::shared_ptr<Agent> a) {
        return lounge.erase(a->name());
    }

private:
    std::shared_ptr<Agent> find_agent(const std::string& name, const std::string& role) {
        if (name[0] == '$' && name.substr(1) == account()) {
            for (auto who : list_agents()) {
                if (who->role()[0] == role[0]) return who;
            }
        }
        auto it = lounge.find(name);
        if (it != lounge.end() && it->second->role()[0] == role[0]) return it->second;
        return std::shared_ptr<Agent>(new Agent("name=" + name + " role=dummy"));
    }

public:
    std::vector<std::shared_ptr<match>> list_matches() const {
        std::vector<std::shared_ptr<match>> res;
        for (auto ep : ongoing) res.push_back(ep.second);
        return res;
    }
    std::vector<std::shared_ptr<Agent>> list_agents() const {
        std::vector<std::shared_ptr<Agent>> res;
        for (auto who : lounge) res.push_back(who.second);
        return res;
    }
    std::string account() const {
        return name;
    }
    std::string login() const {
        return auth;
    }

public:
    void set_account(const std::string& name) {
        this->name = this->auth = name;
    }
    void set_login(const std::string& res) {
        name = res.substr(0, res.find('|'));
        auth = res;
    }
    void set_dump_file(const std::string& path) {
        if (dump.is_open()) dump.close();
        dump.clear();
        dump.open(path, std::ios::out | std::ios::app);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Agent>> lounge;
    std::unordered_map<std::string, std::shared_ptr<match>> ongoing;
    std::string name, auth;
    std::ofstream dump;
};