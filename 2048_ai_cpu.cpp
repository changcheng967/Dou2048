#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <limits>
#include <future>

#define BOARD_SIZE 4
#define TARGET_TILE 65536  // 2^16

class Game2048 {
private:
    std::vector<std::vector<int>> board;
    int score;
    int moves;
    std::mt19937 rng;
    
public:
    Game2048() : score(0), moves(0) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        board = std::vector<std::vector<int>>(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));
        add_random_tile();
        add_random_tile();
    }
    
    void display() {
        std::cout << "\n╔══════╦══════╦══════╦══════╗\n";
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::cout << "║";
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    std::cout << "      ║";
                } else {
                    std::cout << std::setw(6) << board[i][j] << "║";
                }
            }
            if (i < BOARD_SIZE - 1) {
                std::cout << "\n╠══════╬══════╬══════╬══════╣\n";
            } else {
                std::cout << "\n╚══════╩══════╩══════╩══════╝\n";
            }
        }
        std::cout << "Score: " << score << " | Moves: " << moves << "\n";
    }
    
    bool add_random_tile() {
        std::vector<std::pair<int, int>> empty_cells;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    empty_cells.push_back({i, j});
                }
            }
        }
        
        if (empty_cells.empty()) return false;
        
        auto [x, y] = empty_cells[rng() % empty_cells.size()];
        board[x][y] = (rng() % 10 == 0) ? 4 : 2; // 10% chance for 4
        return true;
    }
    
    bool move_left(bool actual_move = true) {
        bool moved = false;
        for (int i = 0; i < BOARD_SIZE; i++) {
            // Remove zeros and combine tiles
            std::vector<int> new_row;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != 0) {
                    new_row.push_back(board[i][j]);
                }
            }
            
            // Combine adjacent equal tiles
            for (size_t j = 0; j < new_row.size(); j++) {
                if (j + 1 < new_row.size() && new_row[j] == new_row[j + 1]) {
                    if (actual_move) {
                        score += new_row[j] * 2;
                    }
                    new_row[j] *= 2;
                    new_row.erase(new_row.begin() + j + 1);
                }
            }
            
            // Pad with zeros
            while (new_row.size() < BOARD_SIZE) {
                new_row.push_back(0);
            }
            
            // Check if row changed
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != new_row[j]) {
                    moved = true;
                }
                if (actual_move) {
                    board[i][j] = new_row[j];
                }
            }
        }
        return moved;
    }
    
    void rotate_board() {
        std::vector<std::vector<int>> temp(BOARD_SIZE, std::vector<int>(BOARD_SIZE));
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                temp[i][j] = board[BOARD_SIZE - j - 1][i];
            }
        }
        board = temp;
    }
    
    bool move(int direction, bool actual_move = true) {
        // 0: up, 1: right, 2: down, 3: left
        auto temp_board = board;
        auto temp_score = score;
        
        for (int i = 0; i < direction; i++) {
            rotate_board();
        }
        
        bool moved = move_left(actual_move);
        
        for (int i = 0; i < (4 - direction) % 4; i++) {
            rotate_board();
        }
        
        if (!actual_move) {
            board = temp_board;
            score = temp_score;
        }
        
        return moved;
    }
    
    bool is_game_over() {
        // Check for empty cells
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) return false;
            }
        }
        
        // Check for possible merges
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int current = board[i][j];
                if ((i < BOARD_SIZE - 1 && current == board[i + 1][j]) ||
                    (j < BOARD_SIZE - 1 && current == board[i][j + 1])) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    bool has_won() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] >= TARGET_TILE) {
                    return true;
                }
            }
        }
        return false;
    }
    
    int get_score() { return score; }
    int get_moves() { return moves; }
    auto get_board() { return board; }
    
    void make_move(int direction) {
        if (move(direction, true)) {
            moves++;
            add_random_tile();
        }
    }
};

class ExpectimaxAI {
private:
    int max_depth;
    int num_threads;
    
    double evaluate_board(const std::vector<std::vector<int>>& board) {
        double score = 0.0;
        
        // Weight factors (optimized for 65536 goal)
        double empty_weight = 100000.0;
        double monotonicity_weight = 10.0;
        double smoothness_weight = 5.0;
        double corner_weight = 1000.0;
        double max_tile_weight = 100.0;
        
        int empty_count = 0;
        int max_tile = 0;
        double monotonicity = 0.0;
        double smoothness = 0.0;
        
        // Count empty cells and find max tile
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    empty_count++;
                } else {
                    if (board[i][j] > max_tile) {
                        max_tile = board[i][j];
                    }
                }
            }
        }
        
        // Monotonicity - prefer increasing/decreasing sequences
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j + 1] != 0) {
                    monotonicity += std::log2(std::abs(board[i][j] - board[i][j + 1]) + 1);
                }
            }
        }
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (board[i][j] != 0 && board[i + 1][j] != 0) {
                    monotonicity += std::log2(std::abs(board[i][j] - board[i + 1][j]) + 1);
                }
            }
        }
        
        // Smoothness - prefer similar adjacent tiles
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j + 1] != 0) {
                    smoothness -= std::abs(std::log2(board[i][j]) - std::log2(board[i][j + 1]));
                }
            }
        }
        
        // Corner preference - prefer high values in corners
        double corner_value = 0.0;
        int corner_tile = board[0][0];
        if (corner_tile > 0) {
            corner_value = std::log2(corner_tile) * 10.0;
        }
        
        // Weighted sum
        score = empty_count * empty_weight +
                monotonicity * monotonicity_weight +
                smoothness * smoothness_weight +
                corner_value * corner_weight +
                max_tile * max_tile_weight;
        
        return score;
    }
    
    double expectimax(Game2048 state, int depth, bool is_maximizing) {
        if (depth == 0 || state.is_game_over()) {
            return evaluate_board(state.get_board());
        }
        
        if (is_maximizing) {
            double best_value = -std::numeric_limits<double>::infinity();
            
            for (int move_dir = 0; move_dir < 4; move_dir++) {
                Game2048 new_state = state;
                if (new_state.move(move_dir, false)) {
                    double value = expectimax(new_state, depth - 1, false);
                    best_value = std::max(best_value, value);
                }
            }
            
            return (best_value == -std::numeric_limits<double>::infinity()) ? 
                   evaluate_board(state.get_board()) : best_value;
        } else {
            // Chance nodes (tile placements)
            double expected_value = 0.0;
            int empty_count = 0;
            
            // Count empty cells
            auto board = state.get_board();
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (board[i][j] == 0) empty_count++;
                }
            }
            
            if (empty_count == 0) {
                return evaluate_board(state.get_board());
            }
            
            // Evaluate all possible tile placements
            int evaluations = 0;
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (board[i][j] == 0) {
                        // 90% chance for 2, 10% for 4
                        for (int tile_value : {2, 4}) {
                            Game2048 new_state = state;
                            auto new_board = new_state.get_board();
                            new_board[i][j] = tile_value;
                            double probability = (tile_value == 2) ? 0.9 : 0.1;
                            
                            double value = expectimax(new_state, depth - 1, true);
                            expected_value += probability * value;
                            evaluations++;
                        }
                    }
                }
            }
            
            return (evaluations > 0) ? expected_value / evaluations : evaluate_board(state.get_board());
        }
    }
    
public:
    ExpectimaxAI(int depth = 5, int threads = 4) : max_depth(depth), num_threads(threads) {}
    
    int get_best_move(Game2048 state) {
        double best_value = -std::numeric_limits<double>::infinity();
        int best_move = 0;
        
        std::vector<std::future<std::pair<int, double>>> futures;
        
        // Evaluate each move in parallel
        for (int move_dir = 0; move_dir < 4; move_dir++) {
            futures.push_back(std::async(std::launch::async, [this, state, move_dir]() {
                Game2048 new_state = state;
                if (new_state.move(move_dir, false)) {
                    double value = expectimax(new_state, max_depth - 1, false);
                    return std::make_pair(move_dir, value);
                }
                return std::make_pair(move_dir, -std::numeric_limits<double>::infinity());
            }));
        }
        
        // Collect results
        for (auto& future : futures) {
            auto [move, value] = future.get();
            if (value > best_value) {
                best_value = value;
                best_move = move;
            }
        }
        
        return best_move;
    }
};

void play_ai_game() {
    Game2048 game;
    ExpectimaxAI ai(5, std::thread::hardware_concurrency());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int display_counter = 0;
    
    std::cout << "🚀 Starting 2048 AI with Expectimax Algorithm\n";
    std::cout << "🎯 Target: Reach " << TARGET_TILE << " tile\n";
    std::cout << "⚡ AI running at full speed (no delays)\n\n";
    
    while (!game.is_game_over() && !game.has_won()) {
        if (display_counter % 10 == 0) {
            game.display();
        }
        
        int best_move = ai.get_best_move(game);
        game.make_move(best_move);
        display_counter++;
        
        if (game.get_moves() > 10000) {
            std::cout << "⚠️  Move limit reached (safety stop)\n";
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // Final display
    game.display();
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "🎮 GAME FINISHED!\n";
    std::cout << "⏱️  Time: " << duration.count() << " seconds\n";
    std::cout << "🔄 Moves: " << game.get_moves() << "\n";
    std::cout << "🏆 Score: " << game.get_score() << "\n";
    
    if (game.has_won()) {
        std::cout << "🎉 SUCCESS: Reached target tile " << TARGET_TILE << "!\n";
    } else {
        std::cout << "💥 Game Over - No more moves possible\n";
    }
    std::cout << std::string(60, '=') << "\n";
}

int main() {
    std::cout << "2048 AI with Expectimax Algorithm - CPU Optimized\n";
    std::cout << "=================================================\n";
    
    try {
        play_ai_game();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
