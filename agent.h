/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent("name=weight_agent role=environment " + args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		reward_history.clear();
		board_history.clear();
	}
	
	virtual void close_episode(const std::string& flag = "") {
		if(board_history.empty()) return;
		if(alpha==0) return;
		adjust_table(board_history[board_history.size()-1],0);
		for(int t=board_history.size()-2;t>=0;t--){
			adjust_table(board_history[t],reward_history[t+1]+v_value(board_history[t+1]));
		}
	}

	float v_value(const board& after) const{
		float val=0;
		val+=net[0][after(0)*25*25*25+after(1)*25*25+after(2)*25+after(3)];
		val+=net[1][after(4)*25*25*25+after(5)*25*25+after(6)*25+after(7)];
		val+=net[2][after(8)*25*25*25+after(9)*25*25+after(10)*25+after(11)];
		val+=net[3][after(12)*25*25*25+after(13)*25*25+after(14)*25+after(15)];
		val+=net[4][after(0)*25*25*25+after(4)*25*25+after(8)*25+after(12)];
		val+=net[5][after(1)*25*25*25+after(5)*25*25+after(9)*25+after(13)];
		val+=net[6][after(2)*25*25*25+after(6)*25*25+after(10)*25+after(14)];
		val+=net[7][after(3)*25*25*25+after(7)*25*25+after(11)*25+after(15)];
		return val;
	}

	void adjust_table(const board& after, float target){
		float current=v_value(after);
		float error=target-current;
		float adjust=alpha*error;
		net[0][after(0)*25*25*25+after(1)*25*25+after(2)*25+after(3)]+=adjust;
		net[1][after(4)*25*25*25+after(5)*25*25+after(6)*25+after(7)]+=adjust;
		net[2][after(8)*25*25*25+after(9)*25*25+after(10)*25+after(11)]+=adjust;
		net[3][after(12)*25*25*25+after(13)*25*25+after(14)*25+after(15)]+=adjust;
		net[4][after(0)*25*25*25+after(4)*25*25+after(8)*25+after(12)]+=adjust;
		net[5][after(1)*25*25*25+after(5)*25*25+after(9)*25+after(13)]+=adjust;
		net[6][after(2)*25*25*25+after(6)*25*25+after(10)*25+after(14)]+=adjust;
		net[7][after(3)*25*25*25+after(7)*25*25+after(11)*25+after(15)]+=adjust;
	}

protected:
	virtual void init_weights(const std::string& info) {
//		net.emplace_back(65536); // create an empty weight table with size 65536
//		net.emplace_back(65536); // create an empty weight table with size 65536
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
		net.emplace_back(25*25*25*25);
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}
	
	virtual action take_action(const board& before) {
		int best_move=-1;
		int max_reward=-1;
		float max_val=-std::numeric_limits<float>::max();
		board next_board;
		for (int op=0;op<4;op++) {
			board after = before;
			board::reward r = after.slide(op);
			if(r==-1) continue;
			float val=v_value(after);
			if (r+val>max_reward+max_val){
				best_move=op;
				max_reward=r;
				max_val=val;
				next_board=after;
			}
		}
		if(best_move!=-1){
			reward_history.push_back(max_reward);
			board_history.push_back(next_board);
		}
		return action::slide(best_move);
	}

protected:
	std::vector<weight> net;
	float alpha;
	std::vector<int> reward_history;
	std::vector<board> board_history;
};


/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

	virtual action take_action(const board& after) {
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			board::cell tile = popup(engine) ? 1 : 2;
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 3, 1, 2 }) {}
/* random version
	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}
*/
/* greedy version
	virtual action take_action(const board& before) {
		int max_reward=-1;
		int move=-1;
		for (int op : opcode) {
			board::reward r = board(before).slide(op);
			if (r>max_reward){
				move=op;
				max_reward=r;
			}
		}
		if (max_reward != -1) return action::slide(move);
		return action();
	}
*/
// some heuristics + greedy
	virtual action take_action(const board& before) {
		//std::cout << before << '\n';
		int max_reward=-1;
		int move=-1;
		for (int op : opcode) {
			board one_step=before;
			board::reward r = board(one_step).slide(op);
			if(r==-1) continue;
			if(op==0) r*=6;
			else if(op==1) r*=7;
			else r*=3;
			if (r>max_reward){
				move=op;
				max_reward=r;
			}
		}
		if (max_reward != -1) return action::slide(move);
		return action();
	}
	
private:
	std::array<int, 4> opcode;
};
