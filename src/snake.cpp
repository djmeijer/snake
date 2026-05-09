#include "util.hpp"
#include "game.hpp"

#include "zig_zag_agent.hpp"
#include "cell_tree_agent.hpp"
#include "hamiltonian_cycle.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <functional>
#include <optional>
#include <random>
#include <thread>
#include <mutex>

//------------------------------------------------------------------------------
// Logging games
//------------------------------------------------------------------------------

int dir_to_notebook_action(Dir d) {
  switch (d) {
    case Dir::up:    return 0;
    case Dir::right: return 1;
    case Dir::down:  return 2;
    case Dir::left:  return 3;
  }
  throw std::logic_error("bad Dir");
}

void write_coord_json(std::ostream& out, Coord c) {
  out << "[" << c.x << "," << c.y << "]";
}

void write_policy_json(std::ostream& out, Dir action) {
  int a = dir_to_notebook_action(action);
  out << "[";
  for (int i = 0; i < 4; ++i) {
    if (i) out << ",";
    out << (i == a ? "1.0" : "0.0");
  }
  out << "]";
}

struct ExpertTransition {
  int w = 0;
  int h = 0;
  int turn = 0;
  std::vector<Coord> snake;
  Coord apple = INVALID;
  Dir prev_action = Dir::right;
  Dir action = Dir::right;
  double value = 0.0;
};

ExpertTransition capture_expert_transition(
    Game const& game,
    Dir prev_dir,
    Dir action
) {
  ExpertTransition sample;
  sample.w = game.grid.w;
  sample.h = game.grid.h;
  sample.turn = game.turn;
  sample.apple = game.apple_pos;
  sample.prev_action = prev_dir;
  sample.action = action;
  for (Coord c : game.snake) {
    sample.snake.push_back(c);
  }
  return sample;
}

void write_expert_row(
    std::ostream& out,
    ExpertTransition const& sample,
    bool include_policy_value
) {
  out << "{";

  out << "\"w\":" << sample.w << ",";
  out << "\"h\":" << sample.h << ",";
  out << "\"turn\":" << sample.turn << ",";

  out << "\"snake\":[";
  bool first = true;
  for (Coord c : sample.snake) {
    if (!first) out << ",";
    first = false;
    write_coord_json(out, c);
  }
  out << "],";

  out << "\"apple\":";
  write_coord_json(out, sample.apple);
  out << ",";

  out << "\"prev_action\":" << dir_to_notebook_action(sample.prev_action) << ",";
  out << "\"action\":" << dir_to_notebook_action(sample.action);

  if (include_policy_value) {
    out << ",\"policy\":";
    write_policy_json(out, sample.action);
    out << ",\"value\":" << std::setprecision(9) << sample.value;
  }

  out << "}\n";
}

struct Log {
  std::vector<Coord> snake_pos;
  std::vector<int>   snake_size;
  std::vector<Coord> apple_pos;
  std::vector<int>   eat_turns;

  void log(Game const& game, Game::Event event) {
    snake_pos.push_back(game.snake_pos());
    snake_size.push_back(game.snake.size());
    if (event == Game::Event::eat) {
      eat_turns.push_back(game.turn);
    }
    if (apple_pos.empty() || event == Game::Event::eat) {
      apple_pos.push_back(game.apple_pos);
    }
  }
};

class LoggedGame : public Game {
public:
  Log log;
  
  LoggedGame(CoordRange dims, RNG const& rng) : Game(dims, rng) {
    log.log(*this, Game::Event::none);
  }
  Event move(Dir d) {
    Event e = Game::move(d);
    log.log(*this, e);
    return e;
  }
};

//------------------------------------------------------------------------------
// Stats of multiple games
//------------------------------------------------------------------------------

struct Stats {
  std::vector<int> turns;
  std::vector<bool> wins;
  
  void add(Game const& game);
};

void Stats::add(Game const& game) {
  wins.push_back(game.win());
  if (game.win()) {
    turns.push_back(game.turn);
  }
}

namespace {

void print_turn_stats(std::ostream& out, std::vector<int> const& turns) {
  out << "mean " << mean(turns);
  out << ", stddev " << stddev(turns);
  if (turns.empty()) {
    out << ", min -, median -, max -";
    return;
  }

  auto q = quantiles(turns);
  out << ", min " << q.front();
  out << ", median " << q[2];
  out << ", max " << q.back();
  out << ", quantiles " << q;
}

}

std::ostream& operator << (std::ostream& out, Stats const& stats) {
  out << "turns: ";
  print_turn_stats(out, stats.turns);
  if (mean(stats.wins) < 1) {
    out << "  LOST: " << (1-mean(stats.wins))*100 << "%";
  }
  return out;
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------

enum class Trace {
  no, eat, all
};
struct Config {
  int num_rounds = 100;
  CoordRange board_size = {30,30};
  Trace trace = Trace::no;
  bool quiet = false;
  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  std::string json_file;
  bool json_compact = true;
  RNG rng = global_rng;
  std::optional<std::array<int, 9>> cell_variant_penalties;
  
  void parse_optional_args(int argc, const char** argv);

  std::string expert_out = "-";
  int max_samples = 0; // 0 means unlimited
  bool skip_length_one = true;
  bool expert_policy_value = false;
  double expert_gamma = 0.98;
  double expert_reward_apple = 0.10;
  double expert_reward_win = 1.00;
  double expert_reward_loss = -1.00;
  double expert_step_penalty = -0.002;
};

std::array<int, 9> const default_cell_variant_penalties = {
  63270,
  91375,
  11566,
  37666,
  16482,
  94303,
  38397,
  52842,
  52566
};

constexpr int min_cell_tree_penalty = 0;
constexpr int max_cell_tree_penalty = 100000;

int normalize_cell_tree_penalty(int value) {
  if (value < min_cell_tree_penalty || value > max_cell_tree_penalty) {
    throw std::invalid_argument(
      "Cell-tree penalties must be between "
      + std::to_string(min_cell_tree_penalty)
      + " and "
      + std::to_string(max_cell_tree_penalty)
      + "."
    );
  }
  return value;
}

std::array<int, 9> normalize_cell_tree_penalties(std::array<int, 9> penalties) {
  for (int& penalty : penalties) {
    penalty = normalize_cell_tree_penalty(penalty);
  }
  return penalties;
}

std::vector<int> normalize_cell_tree_penalties(std::vector<int> const& penalties) {
  std::vector<int> normalized;
  normalized.reserve(penalties.size());
  for (int penalty : penalties) {
    normalized.push_back(normalize_cell_tree_penalty(penalty));
  }
  return normalized;
}

RNG make_random_rng() {
  std::random_device device;
  uint64_t seed0 = (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device());
  uint64_t seed1 = (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device());
  if (seed0 == 0 && seed1 == 0) {
    seed1 = 1;
  }
  uint64_t state[] = {seed0, seed1};
  return RNG(state);
}

void apply_cell_tree_penalties(CellTreeAgent& agent, std::array<int, 9> const& penalties) {
  auto param = penalties.begin();
  agent.same_cell_penalty = *param++;
  agent.new_cell_penalty = *param++;
  agent.parent_cell_penalty = *param++;
  agent.edge_penalty_in = *param++;
  agent.wall_penalty_in = *param++;
  agent.open_penalty_in = *param++;
  agent.edge_penalty_out = *param++;
  agent.wall_penalty_out = *param++;
  agent.open_penalty_out = *param++;
}

//------------------------------------------------------------------------------
// Agents
//------------------------------------------------------------------------------

struct AgentFactory {
  std::string name;
  std::string description;
  std::function<std::unique_ptr<Agent>(Config&)> make;
};
AgentFactory agents[] = {
  {"zig-zag", "Follows a fixed zig-zag cycle", [](Config&) {
    return std::make_unique<FixedZigZagAgent>();
  }},
  {"fixed", "Follows a fixed but random cycle", [](Config& config) {
    return std::make_unique<FixedCycleAgent>(random_hamiltonian_cycle(config.board_size, config.rng));
  }},
  {"zig-zag-cut", "Follows a zig-zag cycle, but can take shortcuts", [](Config& config) {
    return std::make_unique<CutAgent>();
  }},
  {"cell", "Limit movement to a tree of 2x2 cells", [](Config&) {
    return std::make_unique<CellTreeAgent>();
  }},
  {"cell1", "Cell tree agent with limited lookahead", [](Config&) {
    auto agent = std::make_unique<CellTreeAgent>();
    agent->lookahead = Lookahead::one;
    return agent;
  }},
  {"cell-keep", "Cell tree agent which doesn't move snake in lookahead", [](Config&) {
    auto agent = std::make_unique<CellTreeAgent>();
    agent->lookahead = Lookahead::many_keep_tail;
    return agent;
  }},
  {"cell-fixed", "Cell agent that doesn't recalculate paths", [](Config&) {
    auto agent = std::make_unique<CellTreeAgent>();
    agent->recalculate_path = false;
    return agent;
  }},
  {"cell-variant", "Cell tree agent with penalties on moving in the tree", [](Config& config) {
    auto agent = std::make_unique<CellTreeAgent>();
    apply_cell_tree_penalties(*agent, config.cell_variant_penalties.value_or(default_cell_variant_penalties));
    return agent;
  }},
  {"phc", "Perturbed Hamiltonian cycle (zig-zag cycle)", [](Config& config) {
    auto agent = std::make_unique<PerturbedHamiltonianCycle>(make_zig_zag_path(config.board_size));
    return agent;
  }},
  {"dhcr", "Dynamic Hamiltonian Cycle Repair", [](Config& config) {
    auto agent = std::make_unique<DynamicHamiltonianCycleRepair>(make_zig_zag_path(config.board_size));
    return agent;
  }},
  {"dhcr-nascar", "Dynamic Hamiltonian Cycle Repair with Nascar mode", [](Config& config) {
    auto agent = std::make_unique<DynamicHamiltonianCycleRepair>(make_zig_zag_path(config.board_size));
    agent->wall_follow_overshoot = 1;
    return agent;
  }},
};

void list_agents(std::ostream& out = std::cout) {
  out << "Available agents:" << std::endl;
  for (auto const& a : agents) {
    out << "  " << std::left << std::setw(20) << a.name;
    out << a.description << std::endl;
  }
}

AgentFactory const& find_agent(std::string const& name) {
  for (auto const& a : agents) {
    if (a.name == name) return a;
  }
  throw std::invalid_argument("Unknown agent: " + name + "\nUse `list` command to list available agents.");
}

//------------------------------------------------------------------------------
// Argument handling
//------------------------------------------------------------------------------

void print_help(const char* name, std::ostream& out = std::cout) {
  Config def;
  using namespace std;
  out << "Usage: " << name << " <mode> <args>" << endl;
  out << endl;
  out << "These modes are available:" << endl;
  out << "  help                Show this message." << endl;
  out << "  list                List available agents." << endl;
  out << "  all                 Play all agents against each other, output csv summary." << endl;
  out << "  optimize-cell       Tune cell-tree penalties with coarse-to-fine search." << endl;
  out << "  export-data <agent> Export JSONL expert policy/value training data." << endl;
  out << "  <agent>             Play with the given agent." << endl;
  out << endl;
  out << "Optional arguments:" << endl;
  out << "  -n, --n <rounds>    Run the given number of rounds (default: " << def.num_rounds << ")." << endl;
  out << "  -s, --size <size>   Size of the (square) board (default: " << def.board_size.w << ")." << endl;
  out << "      --seed <n>      Random seed." << endl;
  out << "  -T, --trace-all     Print the game state after each move." << endl;
  out << "  -t, --trace         Print the game state each time the snake eats an apple." << endl;
  out << "      --no-color      Don't use ANSI color codes in trace output" << endl;
  out << "  -q, --quiet         Don't print extra output." << endl;
  out << "      --json <file>   Write log of one run a json file." << endl;
  out << "      --json-full     Don't encode json file to save size." << endl;
  out << "  -j, --threads <n>   Specify the maximum number of threads (default: " << def.num_threads << ")." << endl;
  out << "      --penalties <same> <new> <parent> <edge-in> <wall-in> <open-in> <edge-out> <wall-out> <open-out>" << endl;
  out << "                      Override the 9 cell-tree penalties used by cell-variant (range: 0-100000)." << endl;
  out << "      --out <file>    export-data output JSONL file, or '-' for stdout." << endl;
  out << "      --max-samples <n>" << endl;
  out << "                      Stop export-data after writing n samples. 0 means unlimited." << endl;
  out << "      --include-length-one" << endl;
  out << "                      Include the initial length-1 snake states in export-data." << endl;
  out << "      --policy-value  With export-data, include one-hot policy and discounted value target." << endl;
  out << "      --expert-gamma <x>" << endl;
  out << "                      Discount used for export-data value targets (default: " << def.expert_gamma << ")." << endl;
  out << "      --expert-reward-apple <x>" << endl;
  out << "      --expert-reward-win <x>" << endl;
  out << "      --expert-reward-loss <x>" << endl;
  out << "      --expert-step-penalty <x>" << endl;
  out << "                      Rewards used when computing export-data value targets." << endl;
  out << endl;
  list_agents(out);
}

void Config::parse_optional_args(int argc, const char** argv) {
  for (int i=0; i<argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-n" || arg == "--n") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      num_rounds = std::stoi(argv[++i]);
    } else if (arg == "-s" || arg == "--size") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      int size = std::stoi(argv[++i]);
      board_size = {size,size};
    } else if (arg == "-w" || arg == "--width") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      board_size.w = std::stoi(argv[++i]);
    } else if (arg == "-h" || arg == "--height") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      board_size.h = std::stoi(argv[++i]);
    } else if (arg == "--seed") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      int seed = std::stoi(argv[++i]);
      uint64_t s[] = {1234567891234567890u,9876543210987654321u+seed};
      rng = RNG(s);
    } else if (arg == "--json") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      json_file = argv[++i];
    } else if (arg == "--penalties") {
      if (i+9 >= argc) throw std::invalid_argument("Missing arguments to " + arg + ": expected 9 penalty values");
      std::array<int, 9> penalties;
      for (int j = 0; j < static_cast<int>(penalties.size()); ++j) {
        penalties[j] = std::stoi(argv[++i]);
      }
      cell_variant_penalties = normalize_cell_tree_penalties(penalties);
    } else if (arg == "-t" || arg == "--trace") {
      trace = Trace::eat;
      num_rounds = 1;
    } else if (arg == "-T" || arg == "--trace-all") {
      trace = Trace::all;
      num_rounds = 1;
    } else if (arg == "-q" || arg == "--quiet") {
      quiet = true;
    } else if (arg == "-j" || arg == "--threads" || arg == "--num-threads") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      num_threads = std::stoi(argv[++i]);
    } else if (arg == "--no-color") {
      use_color = false;
    } else if (arg == "--json-full") {
      json_compact = false;
    } else if (arg == "--out") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_out = argv[++i];
    } else if (arg == "--max-samples") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      max_samples = std::stoi(argv[++i]);
    } else if (arg == "--include-length-one") {
      skip_length_one = false;
    } else if (arg == "--policy-value" || arg == "--expert-policy-value") {
      expert_policy_value = true;
    } else if (arg == "--expert-gamma") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_gamma = std::stod(argv[++i]);
    } else if (arg == "--expert-reward-apple") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_reward_apple = std::stod(argv[++i]);
    } else if (arg == "--expert-reward-win") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_reward_win = std::stod(argv[++i]);
    } else if (arg == "--expert-reward-loss") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_reward_loss = std::stod(argv[++i]);
    } else if (arg == "--expert-step-penalty") {
      if (i+1 >= argc) throw std::invalid_argument("Missing argument to " + arg);
      expert_step_penalty = std::stod(argv[++i]);
    } else{
      throw std::invalid_argument("Unknown argument: " + arg);
    }
    
  }
}

double expert_reward_for_event(Game const& game_after_move, Game::Event event, Config const& config) {
  double reward = config.expert_step_penalty;
  if (event == Game::Event::eat) {
    reward += config.expert_reward_apple;
  }
  if (event == Game::Event::lose) {
    reward += config.expert_reward_loss;
  }
  if (game_after_move.win()) {
    reward += config.expert_reward_win;
  }
  return reward;
}

void fill_discounted_values(
    std::vector<ExpertTransition>& samples,
    std::vector<double> const& rewards,
    double gamma
) {
  double value = 0.0;
  for (int i = static_cast<int>(samples.size()) - 1; i >= 0; --i) {
    value = rewards[i] + gamma * value;
    samples[i].value = value;
  }
}

void export_expert_data(AgentFactory const& agent_factory, Config& config) {
  config.quiet = true;
  config.num_threads = 1;

  std::ofstream file;
  std::ostream* out = &std::cout;

  if (config.expert_out != "-") {
    file.open(config.expert_out);
    if (!file) {
      throw std::runtime_error("Could not open output file: " + config.expert_out);
    }
    out = &file;
  }

  int written = 0;

  while (config.max_samples == 0 || written < config.max_samples) {
    Game game(config.board_size, config.rng.next_rng());
    auto agent = agent_factory.make(config);

    Dir prev_dir = Dir::right;
    std::vector<ExpertTransition> episode_samples;
    std::vector<double> episode_rewards;

    while (!game.done()) {
      Dir action = (*agent)(game, nullptr);

      // The very first C++ state has snake length 1 and no real previous dir.
      // You can skip it, or use action as prev_dir.
      if (!config.skip_length_one || game.snake.size() >= 2) {
        Dir prev_for_state = game.turn == 0 ? action : prev_dir;
        episode_samples.push_back(capture_expert_transition(game, prev_for_state, action));
      }

      auto event = game.move(action);
      if (!episode_samples.empty() && episode_samples.size() > episode_rewards.size()) {
        episode_rewards.push_back(expert_reward_for_event(game, event, config));
      }
      prev_dir = action;
    }

    if (config.expert_policy_value) {
      fill_discounted_values(episode_samples, episode_rewards, config.expert_gamma);
    }

    for (auto const& sample : episode_samples) {
      write_expert_row(*out, sample, config.expert_policy_value);
      written++;
      if (config.max_samples > 0 && written >= config.max_samples) {
        break;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Json output
//------------------------------------------------------------------------------

void write_json(std::ostream& out, int x) {
  out << x;
}
void write_json(std::ostream& out, Coord c) {
  out << "[" << c.x << "," << c.y << "]";
}
void write_json(std::ostream& out, CoordRange c) {
  out << "[" << c.w << "," << c.h << "]";
}

template <typename T>
void write_json(std::ostream& out, std::vector<T> const& xs) {
  out << "[";
  bool first = true;
  for (auto const& x : xs) {
    if (!first) {
      out << ",";
    }
    first = false;
    write_json(out, x);
  }
  out << "]";
}

// encode paths as strings to save bandwidth
//  * coordinates encoded as  '#' + x, '#' + y, since chars after '#' don't need escapes (except for \)
//  * subsequent coordinates encoded by movement direction (0...3 = Dir::up...Dir::right)
//  * three dirs fit into a char, '\0' + first + 4*(1 + second + 4*(1 + third))
//  * this is a hackish variant of base64 encoding
bool can_encode_path(std::vector<Coord> const& xs) {
  if (xs.empty() || xs[0].x > 85 || xs[0].y > 85) return false;
  for (size_t i=1; i<xs.size(); ++i) {
    if (!is_neighbor(xs[i],xs[i-1])) return false;
  }
  return true;
}
void encode_char(std::ostream& out, int c) {
  if (c == '\\') out << "\\\\";
  else if (c == '\"') out << "\\\"";
  else out << (char)c;
}
void write_json_path(std::ostream& out, std::vector<Coord> const& xs, bool compact) {
  if (!compact || !can_encode_path(xs)) {
    write_json(out,xs);
    return;
  }
  out << "\"";
  encode_char(out, xs[0].x + 35);
  encode_char(out, xs[0].y + 35);
  for (size_t i=1; i<xs.size(); i+=3) {
    int d = 0;
    if (i+2 < xs.size()) {
      d = 1 + (int)(xs[i+2] - xs[i+1]) + 4*d;
    }
    if (i+1 < xs.size()) {
      d = 1 + (int)(xs[i+1] - xs[i]) + 4*d;
    }
    d = (int)(xs[i] - xs[i-1]) + 4*d;
    encode_char(out, d + 35);
  }
  out << "\"";
}
// Encode grids as strings
// 6 bits fit into a char (base64)
void write_json_grid(std::ostream& out, Grid<bool> const& xs, bool compact) {
  out << "\"";
  for (auto it = xs.begin(); it != xs.end();) {
    int d = 0;
    for (size_t j=0; j<6 && it != xs.end() ; ++j, ++it) {
      if (*it) d |= (1<<j);
    }
    encode_char(out, d + 35);
  }
  out << "\"";
}

void write_json_log(std::ostream& out, AgentLog::LogEntry const* prev, AgentLog::NoEntry const& e, bool compact) {
  out << 0;
}
void write_json_log(std::ostream& out, AgentLog::LogEntry const* prev, AgentLog::CopyEntry const& e, bool compact) {
  out << 1;
}
void write_json_log(std::ostream& out, AgentLog::LogEntry const* prev, std::vector<Coord> const& path, bool compact) {
  // compare to previous path
  if (compact) {
    auto prev_path = std::get_if<std::vector<Coord>>(prev);
    if (prev_path && prev_path->size() >= path.size()) {
      if (std::equal(path.begin(), path.end(), prev_path->begin())) {
        // path is a prefix of previous path, encode more efficiently
        out << (1 + prev_path->size() - path.size());
        return;
      }
    }
  }
  write_json_path(out, path, compact);
}
void write_json_log(std::ostream& out, AgentLog::LogEntry const* prev, Grid<bool> const& grid, bool compact) {
  if (compact) {
    auto prev_grid = std::get_if<Grid<bool>>(prev);
    if (prev_grid && std::equal(grid.begin(), grid.end(), prev_grid->begin())) {
      out << 1;
      return;
    }
  }
  write_json_grid(out, grid, compact);
}
void write_json(std::ostream& out, std::vector<AgentLog::LogEntry> const& xs, bool compact) {
  out << "[";
  AgentLog::LogEntry const* prev = nullptr;
  for (auto const& x : xs) {
    if (prev) {
      out << ",";
    }
    std::visit([&out,prev,compact](auto const& x){write_json_log(out,prev,x,compact);}, x);
    prev = &x;
  }
  out << "]";
}

void write_json(std::ostream& out, AgentFactory const& agent, LoggedGame const& game, AgentLog const& agent_log, bool compact = true) {
  out << "{" << std::endl;
  out << "  \"agent\": \"" << agent.name << "\"," << std::endl;
  out << "  \"agent_description\": \"" << agent.description << "\"," << std::endl;
  out << "  \"size\": "; write_json(out, game.dimensions()); out << "," << std::endl;
  out << "  \"snake_pos\": "; write_json_path(out, game.log.snake_pos, compact); out << "," << std::endl;
  if (!compact) {
    out << "  \"snake_size\": "; write_json(out, game.log.snake_size); out << "," << std::endl;
  }
  out << "  \"apple_pos\": "; write_json(out, game.log.apple_pos); out << "," << std::endl;
  out << "  \"eat_turns\": "; write_json(out, game.log.eat_turns);
  for (int i = 0; i < AgentLog::MAX_KEY; ++i) {
    if (!agent_log.logs[i].empty()) {
      out << "," << std::endl;
      out << "  \"" << AgentLog::key_name((AgentLog::Key)i) << "\": ";
      write_json(out, agent_log.logs[i], compact);
    }
  }
   out << std::endl << "}" << std::endl;
}

void write_json(std::string const& filename, AgentFactory const& agent, LoggedGame const& game, AgentLog const& agent_log, bool compact = true) {
  std::ofstream out(filename);
  write_json(out, agent, game, agent_log, compact);
}

//------------------------------------------------------------------------------
// Playing full games
//------------------------------------------------------------------------------

enum class Visualize {
  no, eat, all
};

template <typename Game>
void play(Game& game, Agent& agent, Config const& config, AgentLog* log = nullptr) {
  while (!game.done()) {
    if (config.trace == Trace::all) std::cout << game;
    auto event = game.move(agent(game,log));
    if (event == Game::Event::eat && config.trace == Trace::eat) std::cout << game;
  }
  if (config.trace == Trace::all) std::cout << game;
}

template <typename AgentGen>
Stats play_multiple_threaded(AgentGen make_agent, Config& config) {
  std::mutex mutex;
  std::vector<std::thread> threads;
  int remaining = config.num_rounds;
  Stats stats;
  for (int thread = 0; thread < config.num_threads; ++thread) {
    threads.push_back(std::thread([&,thread](){
      while (true) {
        std::unique_ptr<Agent> agent;
        RNG rng;
        {
          std::lock_guard<std::mutex> guard(mutex);
          if (remaining <= 0) return;
          remaining--;
          agent = make_agent(config); // potentially uses rng
          rng = config.rng.next_rng();
        }
        Game game(config.board_size, rng);
        play(game, *agent, config);
        {
          std::lock_guard<std::mutex> guard(mutex);
          stats.add(game);
          if (!config.quiet) {
            std::cout << stats.wins.size() << "/" << config.num_rounds << "  " << stats << "\033[K\r" << std::flush;
          }
        }
      }
    }));
  }
  // wait
  for (auto& t : threads) {
    t.join();
  }
  // done
  if (!config.quiet) std::cout << std::endl;
  return stats;
}

template <typename AgentGen>
Stats play_multiple(AgentGen make_agent, Config& config) {
  if (config.num_threads > 1) return play_multiple_threaded(make_agent, config);
  Stats stats;
  for (int i = 0; i < config.num_rounds; ++i) {
    Game game(config.board_size, config.rng.next_rng());
    auto agent = make_agent(config);
    play(game, *agent, config);
    stats.add(game);
    if (!config.quiet) {
      if (!game.win()) std::cout << game;
      std::cout << (i+1) << "/" << config.num_rounds << "  " << stats << "\033[K\r" << std::flush;
    }
  }
  if (!config.quiet) std::cout << std::endl;
  return stats;
}


void play_all_agents(Config& config, std::ostream& out = std::cout) {
  using namespace std;
  out << "agent, mean, stddev, min, q.25, median, q.75, max, lost" << endl;
  for (auto const& agent : agents) {
    out << left << setw(15) << agent.name << ", " << flush;
    auto stats = play_multiple(agent.make, config);
    out << right << fixed << setprecision(1) << setw(10);
    out << setw(8) << mean(stats.turns) << ", ";
    out << setw(8) << stddev(stats.turns) << ", ";
    out << setprecision(0);
    if (stats.turns.empty()) {
      for (int i = 0; i < 5; ++i) {
        out << setw(8) << "-" << ", ";
      }
    } else {
      for (auto q : quantiles(stats.turns)) {
        out << setw(8) << q << ", ";
      }
    }
    out << setprecision(1);
    out << setw(8) << ((1-mean(stats.wins))*100) << "%" << endl;
  }
}

//------------------------------------------------------------------------------
// Optimization
// Tune agent parameters with a simple genetic algorithm.
//------------------------------------------------------------------------------

struct ParameterizedAgentFactory {
  std::vector<int> min_param_value;
  std::vector<int> max_param_value;
  ParameterizedAgentFactory(size_t num_params, int min_value, int max_value)
    : min_param_value(num_params, min_value)
    , max_param_value(num_params, max_value)
  {}
  size_t num_params() const { return min_param_value.size(); }
  virtual std::unique_ptr<Agent> make(std::vector<int> params, Config&) const = 0;
};

struct ParameterizedCellTreeAgent : ParameterizedAgentFactory {
  ParameterizedCellTreeAgent() : ParameterizedAgentFactory(9, min_cell_tree_penalty, max_cell_tree_penalty) {}
  
  std::unique_ptr<Agent> make(std::vector<int> params, Config&) const override {
    auto agent = std::make_unique<CellTreeAgent>();
    std::array<int, 9> penalties;
    auto normalized_params = normalize_cell_tree_penalties(params);
    std::copy(normalized_params.begin(), normalized_params.end(), penalties.begin());
    apply_cell_tree_penalties(*agent, penalties);
    return agent;
  }
};

double score(Stats const& stats) {
  return mean(stats.turns) + 1e10 * (1 - mean(stats.wins));
}


void write_params(std::ostream& out, std::vector<int> const& params) {
  bool first = true;
  for (int param : params) {
    if (!first) {
      out << ' ';
    }
    first = false;
    out << param;
  }
}

int quantize_parameter(int value, int lower_bound, int upper_bound, int step) {
  value = std::max(lower_bound, std::min(upper_bound, value));
  if (step <= 1) {
    return value;
  }

  int first_quantized = ((lower_bound + step - 1) / step) * step;
  int last_quantized = (upper_bound / step) * step;
  if (first_quantized > last_quantized) {
    return value;
  }

  int quantized = static_cast<int>(std::llround(static_cast<double>(value) / static_cast<double>(step))) * step;
  return std::max(first_quantized, std::min(last_quantized, quantized));
}

std::vector<int> quantize_parameters(
  pagmo::vector_double const& values,
  std::vector<int> const& lower_bounds,
  std::vector<int> const& upper_bounds,
  int step
) {
  std::vector<int> params;
  params.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    params.push_back(quantize_parameter(static_cast<int>(std::llround(values[i])), lower_bounds[i], upper_bounds[i], step));
  }
  return params;
}

struct ParameterOptimizationProblem {
  ParameterizedAgentFactory const* agent = nullptr;
  Config config;
  std::vector<int> min_param_value;
  std::vector<int> max_param_value;
  int quantization_step = 1;

  pagmo::vector_double fitness(pagmo::vector_double const& x) const {
    auto params = quantize_parameters(x, min_param_value, max_param_value, quantization_step);
    params = normalize_cell_tree_penalties(params);

    Config eval_config = config;
    eval_config.rng = make_random_rng();
    auto stats = play_multiple([this, &params](Config& config) {
      return agent->make(params, config);
    }, eval_config);
    return {::score(stats)};
  }

  std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
    pagmo::vector_double lower;
    pagmo::vector_double upper;
    lower.reserve(agent->num_params());
    upper.reserve(agent->num_params());
    for (auto value : min_param_value) lower.push_back(static_cast<double>(value));
    for (auto value : max_param_value) upper.push_back(static_cast<double>(value));
    return {lower, upper};
  }

  pagmo::vector_double::size_type get_nix() const {
    return agent->num_params();
  }

  std::string get_name() const {
    return "snake parameter optimization";
  }
};

struct OptimizationStageResult {
  std::vector<int> best_params;
  double best_score = 0;
};

OptimizationStageResult optimize_agent_stage(
  ParameterizedAgentFactory& agent,
  Config const& config,
  std::vector<int> const& min_param_value,
  std::vector<int> const& max_param_value,
  int quantization_step,
  unsigned generations,
  unsigned random_seed,
  std::string const& stage_name,
  std::ostream& out
) {
  using namespace std;

  ParameterOptimizationProblem udp;
  udp.agent = &agent;
  udp.config = config;
  udp.config.quiet = true;
  udp.min_param_value = min_param_value;
  udp.max_param_value = max_param_value;
  udp.quantization_step = quantization_step;
  pagmo::problem problem{udp};

  unsigned population_size = static_cast<unsigned>(max<size_t>(20, agent.num_params() * 4));
  if (population_size % 2 != 0) ++population_size;
  double mutation_probability = 1.0 / static_cast<double>(agent.num_params());

  pagmo::population population{problem, population_size, random_seed};
  for (unsigned generation = 1; generation <= generations; ++generation) {
    pagmo::algorithm algorithm{
      pagmo::sga(1u, 0.9, 1.0, mutation_probability, 1.0, 2u, "single", "uniform", "tournament", random_seed + generation)
    };
    population = algorithm.evolve(population);

    auto current_best_params = quantize_parameters(population.champion_x(), min_param_value, max_param_value, quantization_step);

    out << stage_name << " generation " << generation << "/" << generations;
    out << " best score: " << population.champion_f()[0];
    out << " params: ";
    write_params(out, current_best_params);
    out << endl;
  }

  OptimizationStageResult result;
  result.best_params = quantize_parameters(population.champion_x(), min_param_value, max_param_value, quantization_step);
  result.best_score = population.champion_f()[0];
  return result;
}

void optimize_agent(ParameterizedAgentFactory& agent, Config& config, std::ostream& out = std::cout) {
  using namespace std;

  auto random_seed = static_cast<unsigned>(config.rng.next());
  constexpr int coarse_quantization_step = 500;
  constexpr int refine_quantization_step = 50;
  constexpr int refine_radius = 500;
  constexpr unsigned coarse_generations = 120u;
  constexpr unsigned refine_generations = 130u;

  auto coarse_result = optimize_agent_stage(
    agent,
    config,
    agent.min_param_value,
    agent.max_param_value,
    coarse_quantization_step,
    coarse_generations,
    random_seed,
    "coarse",
    out
  );

  vector<int> refine_min_param_value;
  vector<int> refine_max_param_value;
  refine_min_param_value.reserve(agent.num_params());
  refine_max_param_value.reserve(agent.num_params());
  for (size_t i = 0; i < agent.num_params(); ++i) {
    refine_min_param_value.push_back(max(agent.min_param_value[i], coarse_result.best_params[i] - refine_radius));
    refine_max_param_value.push_back(min(agent.max_param_value[i], coarse_result.best_params[i] + refine_radius));
  }

  auto refined_result = optimize_agent_stage(
    agent,
    config,
    refine_min_param_value,
    refine_max_param_value,
    refine_quantization_step,
    refine_generations,
    random_seed + coarse_generations + 1u,
    "refine",
    out
  );

  out << "coarse best score: " << coarse_result.best_score << endl;
  out << "coarse best params: " << coarse_result.best_params << endl;
  out << "best score: " << refined_result.best_score << endl;
  out << "best params: " << refined_result.best_params << endl;
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

int main(int argc, const char** argv) {
  std::string mode = argc >= 2 ? argv[1] : "help";
  
  try {
    if (mode == "help" || mode == "--help" || mode == "-h") {
      print_help(argv[0]);
    } else if (mode == "list") {
      list_agents();
    } else if (mode == "all") {
      Config config;
      config.quiet = true;
      config.parse_optional_args(argc-2, argv+2);
      play_all_agents(config);
    } else if (mode == "optimize-cell") {
      Config config;
      config.parse_optional_args(argc-2, argv+2);
      ParameterizedCellTreeAgent agent;
      optimize_agent(agent, config);
    } else if (mode == "export-data") {
      if (argc < 3) {
        throw std::invalid_argument("Usage: snake export-data <agent> [args]");
      }

      auto agent = find_agent(argv[2]);
      Config config;
      config.parse_optional_args(argc - 3, argv + 3);

      export_expert_data(agent, config);
    } else {
      auto agent = find_agent(mode);
      Config config;
      config.parse_optional_args(argc-2, argv+2);
      if (!config.json_file.empty()) {
        LoggedGame game(config.board_size, config.rng.next_rng());
        AgentLog agent_log;
        auto a = agent.make(config);
        play(game, *a, config, &agent_log);
        if (!config.json_file.empty()) {
          write_json(config.json_file, agent, game, agent_log, config.json_compact);
        }
      } else {
        auto stats = play_multiple(agent.make, config);
        std::cout << stats << std::endl;
      }
    }
  } catch (std::exception const& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
