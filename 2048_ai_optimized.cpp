#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <thread>
#include <future>
#include <limits>
#include <iomanip>

#define BOARD_SIZE 4
#define TARGET_TILE 16  // 2^16 = 65536

class Ultimate2048AI {
private:
    std::vector<std::vector<int>> board;
    int score;
    int moves;
    int max_tile;
    std::mt19937 rng;
    bool fast_mode = true; // set true for CI/automation

    // Tuned heuristic weights for better performance
    static constexpr double EMPTY_WEIGHT = 300000.0;
    static constexpr double MONOTONICITY_WEIGHT = 60.0;
    static constexpr double SMOOTHNESS_WEIGHT = 30.0;
    static constexpr double CORNER_WEIGHT = 200000.0;
    static constexpr double MAX_TILE_WEIGHT = 1000.0;
    static constexpr double MERGE_POTENTIAL_WEIGHT = 30.0;

public:
    Ultimate2048AI() : score(0), moves(0), max_tile(0) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        initialize();
    }

    void initialize() {
        board = std::vector<std::vector<int>>(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));
        add_random_tile();
        add_random_tile();
        update_max_tile();
    }

    void update_max_tile() {
        max_tile = 0;
        for (const auto& row : board) {
            for (int val : row) {
                if (val > max_tile) max_tile = val;
            }
        }
    }

    // Enhanced evaluation function
    double evaluate_state() {
        if (is_game_over()) return -1e9;

        double total_score = 0.0;
        int empty_count = 0;
        double monotonicity = 0.0;
        double smoothness = 0.0;
        double corner_value = 0.0;
        double merge_potential = 0.0;

        // Empty cells
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) empty_count++;
            }
        }

        // Horizontal monotonicity
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0) {
                    monotonicity -= std::abs(board[i][j] - board[i][j+1]);
                }
            }
        }
        // Vertical monotonicity
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (board[i][j] != 0 && board[i+1][j] != 0) {
                    monotonicity -= std::abs(board[i][j] - board[i+1][j]);
                }
            }
        }

        // Smoothness (difference in actual tile values)
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0) {
                    smoothness -= std::abs((1 << board[i][j]) - (1 << board[i][j+1]));
                }
            }
        }
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (board[i][j] != 0 && board[i+1][j] != 0) {
                    smoothness -= std::abs((1 << board[i][j]) - (1 << board[i+1][j]));
                }
            }
        }

        // Merge potential
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j] == board[i][j+1]) {
                    merge_potential += (1 << board[i][j]) * 3.0;
                }
            }
        }
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (board[i][j] != 0 && board[i][j] == board[i+1][j]) {
                    merge_potential += (1 << board[i][j]) * 3.0;
                }
            }
        }

        // Corner preference (max tile in any corner)
        if (board[0][0] == max_tile || board[0][BOARD_SIZE-1] == max_tile ||
            board[BOARD_SIZE-1][0] == max_tile || board[BOARD_SIZE-1][BOARD_SIZE-1] == max_tile) {
            corner_value += 1.0;
        }

        total_score = empty_count * EMPTY_WEIGHT +
                      monotonicity * MONOTONICITY_WEIGHT +
                      smoothness * SMOOTHNESS_WEIGHT +
                      corner_value * CORNER_WEIGHT +
                      max_tile * MAX_TILE_WEIGHT +
                      merge_potential * MERGE_POTENTIAL_WEIGHT;

        return total_score;
    }

    // Dynamic search depth: deeper in late game
    int get_dynamic_depth() {
        int empty_cells = 0;
        int large_tiles = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) empty_cells++;
                if (board[i][j] >= 10) large_tiles++; // 1024 or more
            }
        }
        int base_depth = 3;
        if (empty_cells >= 10) base_depth = 3;
        else if (empty_cells >= 6) base_depth = 4;
        else if (empty_cells >= 3) base_depth = 5;
        else base_depth = 6;
        if (large_tiles >= 2) base_depth += 1;
        if (max_tile >= 12) base_depth += 1; // 4096 or more
        return std::min(base_depth, 7); // allow up to 7
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
        board[x][y] = (rng() % 10 < 9) ? 1 : 2; // 90% 2, 10% 4
        update_max_tile();
        return true;
    }

    void display() {
        if (fast_mode) return;
        std::cout << "\n";
        std::cout << "╔════════╦════════╦════════╦════════╗\n";
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::cout << "║";
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    std::cout << "        ║";
                } else {
                    int value = 1 << board[i][j];
                    if (value < 10) std::cout << "   " << value << "   ║";
                    else if (value < 100) std::cout << "  " << value << "   ║";
                    else if (value < 1000) std::cout << "  " << value << "  ║";
                    else std::cout << " " << value << "  ║";
                }
            }
            if (i < BOARD_SIZE - 1) {
                std::cout << "\n╠════════╬════════╬════════╬════════╣\n";
            } else {
                std::cout << "\n╚════════╩════════╩════════╩════════╝\n";
            }
        }
        std::cout << "Score: " << score << " | Moves: " << moves 
                  << " | Max Tile: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
    }

    // Improved move logic
    bool move_left(bool actual_move = true) {
        std::vector<std::vector<int>> old_board = board;
        int old_score = score;
        bool moved = false;
        int move_score = 0;

        for (int i = 0; i < BOARD_SIZE; i++) {
            std::vector<int> new_row;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != 0) {
                    new_row.push_back(board[i][j]);
                }
            }
            for (size_t j = 0; j + 1 < new_row.size(); ) {
                if (new_row[j] == new_row[j+1]) {
                    new_row[j++];
                    move_score += 1 << new_row[j];
                    new_row.erase(new_row.begin() + j + 1);
                    moved = true;
                } else {
                    j++;
                }
            }
            while (new_row.size() < BOARD_SIZE) {
                new_row.push_back(0);
            }
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != new_row[j]) {
                    moved = true;
                }
                if (actual_move) {
                    board[i][j] = new_row[j];
                }
            }
        }
        if (moved && actual_move) {
            score += move_score;
        } else if (!actual_move) {
            board = old_board;
            score = old_score;
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
        auto old_board = board;
        auto old_score = score;
        bool moved = false;
        for (int i = 0; i < direction; i++) {
            rotate_board();
        }
        moved = move_left(actual_move);
        for (int i = 0; i < (4 - direction) % 4; i++) {
            rotate_board();
        }
        if (!actual_move && !moved) {
            board = old_board;
            score = old_score;
        }
        return moved;
    }

    bool is_game_over() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) return false;
            }
        }
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int current = board[i][j];
                if ((j < BOARD_SIZE - 1 && current == board[i][j+1]) ||
                    (i < BOARD_SIZE - 1 && current == board[i+1][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool has_won() {
        return max_tile >= TARGET_TILE;
    }

    // Expectimax with pruning for performance
    double expectimax_search(int depth, bool is_maximizing, double probability = 1.0) {
        if (depth == 0 || is_game_over()) {
            return evaluate_state();
        }
        if (probability < 0.0001) {
            return evaluate_state();
        }
        if (is_maximizing) {
            double best_value = -1e9;
            bool found_valid = false;
            for (int move_dir = 0; move_dir < 4; move_dir++) {
                auto old_board = board;
                auto old_score = score;
                if (move(move_dir, false)) {
                    double value = expectimax_search(depth - 1, false);
                    if (value > best_value) {
                        best_value = value;
                    }
                    found_valid = true;
                }
                board = old_board;
                score = old_score;
            }
            return found_valid ? best_value : evaluate_state();
        } else {
            double expected_value = 0.0;
            int empty_count = 0;
            std::vector<std::pair<int, int>> empty_cells;
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (board[i][j] == 0) {
                        empty_cells.push_back({i, j});
                        empty_count++;
                    }
                }
            }
            if (empty_count == 0) {
                return expectimax_search(depth - 1, true, probability);
            }
            for (auto [x, y] : empty_cells) {
                auto old_board = board;
                auto old_score = score;
                board[x][y] = 1;
                double value_2 = expectimax_search(depth - 1, true, probability * 0.9 / empty_count);
                board = old_board;
                score = old_score;
                board[x][y] = 2;
                double value_4 = expectimax_search(depth - 1, true, probability * 0.1 / empty_count);
                board = old_board;
                score = old_score;
                expected_value += 0.9 * value_2 + 0.1 * value_4;
            }
            return expected_value / empty_count;
        }
    }

    int find_best_move() {
        int depth = get_dynamic_depth();
        double best_value = -1e9;
        int best_move = 0;
        // Parallel search for each move
        std::vector<std::future<std::pair<int, double>>> futures;
        for (int move_dir = 0; move_dir < 4; move_dir++) {
            futures.push_back(std::async(std::launch::async,
                [this, move_dir, depth]() {
                    auto board_backup = this->board;
                    auto score_backup = this->score;
                    double value = -1e9;
                    if (this->move(move_dir, false)) {
                        value = this->expectimax_search(depth - 1, false);
                    }
                    this->board = board_backup;
                    this->score = score_backup;
                    return std::make_pair(move_dir, value);
                }
            ));
        }
        for (auto& future : futures) {
            auto [move, value] = future.get();
            if (value > best_value) {
                best_value = value;
                best_move = move;
            }
        }
        return best_value > -1e9 ? best_move : 0;
    }

    void play_game() {
        auto start_time = std::chrono::high_resolution_clock::now();
        int display_counter = 0;
        if (!fast_mode) {
            std::cout << "🚀 终极版2048 AI启动 - 基于nneonneo算法深度优化\n";
            std::cout << "🎯 目标: 10万分 + 65536方块\n";
            std::cout << "⚡ 使用动态深度调整和并行Expectimax搜索\n\n";
        }
        while (!is_game_over() && moves < 20000) {
            if (!fast_mode && display_counter % 20 == 0) {
                display();
            }
            int best_move = find_best_move();
            if (move(best_move, true)) {
                moves++;
                add_random_tile();
                update_max_tile();
            }
            display_counter++;
            if (!fast_mode && moves % 100 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);
                std::cout << "📊 进度: " << moves << " 步 | 时间: "
                          << duration.count() << "秒 | 分数: " << score
                          << " | 最大方块: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
            }
            if (has_won() && !fast_mode) {
                std::cout << "🎉 达成65536目标！继续向更高分前进...\n";
            }
            if (score >= 100000 && max_tile >= TARGET_TILE) {
                if (!fast_mode)
                    std::cout << "🎉 目标达成！分数超过10万，最大方块达到65536+\n";
                break;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time);
        display();
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "🎮 游戏结束！\n";
        std::cout << "⏱️  时间: " << duration.count() << " 秒\n";
        std::cout << "🔄 移动次数: " << moves << "\n";
        std::cout << "🏆 最终分数: " << score << "\n";
        std::cout << "💎 最大方块: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
        if (has_won()) {
            std::cout << "🎉 成功达到65536方块目标！\n";
        }
        if (score >= 100000) {
            std::cout << "🎉 达成10万分目标！\n";
        } else if (score >= 50000) {
            std::cout << "✅ 表现良好，接近10万分目标！\n";
        } else {
            std::cout << "💡 建议进一步调整搜索参数以提升性能\n";
        }
        std::cout << std::string(60, '=') << "\n";
    }
};

int main() {
    std::cout << "2048 AI 终极优化版 - 基于nneonneo高性能算法\n";
    std::cout << "==========================================\n";
    try {
        Ultimate2048AI game;
        game.play_game();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
