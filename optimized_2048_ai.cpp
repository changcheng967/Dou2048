#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <thread>
#include <future>
#include <limits>

#define BOARD_SIZE 4
#define TARGET_TILE 65536

class HighPerformance2048AI {
private:
    std::vector<std::vector<int>> board;
    int score;
    int moves;
    int max_tile;
    
public:
    HighPerformance2048AI() : score(0), moves(0), max_tile(0) {
        initialize();
    }
    
    void initialize() {
        board = std::vector<std::vector<int>>(BOARD_SIZE, 
                    std::vector<int>(BOARD_SIZE, 0));
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
    
    // 优化的评估函数 - 关键改进！
    double evaluate_state() {
        double total_score = 0.0;
        
        // 经过优化的权重参数
        const double empty_weight = 270000.0;
        const double smooth_weight = 2.5;
        const double mono_weight = 1.8;
        const double corner_weight = 85000.0;
        const double max_tile_weight = 280.0;
        const double edge_weight = 1.2;
        
        int empty_count = 0;
        double smoothness = 0.0;
        double monotonicity = 0.0;
        double corner_value = 0.0;
        double edge_bonus = 0.0;
        
        // 统计空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    empty_count++;
                }
            }
        }
        
        // 平滑度计算（相邻方块差异）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0) {
                    int diff = abs(board[i][j] - board[i][j+1]);
                    smoothness -= diff * diff;
                }
            }
        }
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (board[i][j] != 0 && board[i+1][j] != 0) {
                    int diff = abs(board[i][j] - board[i+1][j]);
                    smoothness -= diff * diff;
                }
            }
        }
        
        // 单调性计算
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 2; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0 && board[i][j+2] != 0) {
                    int seq1 = board[i][j+1] - board[i][j];
                    int seq2 = board[i][j+2] - board[i][j+1];
                    if (seq1 > 0 && seq2 > 0) monotonicity += 1.0;
                    else if (seq1 < 0 && seq2 < 0) monotonicity += 1.0;
                }
            }
        }
        
        // 角落偏好
        if (board[0][0] == max_tile) corner_value += 50.0;
        if (board[0][BOARD_SIZE-1] == max_tile) corner_value += 30.0;
        if (board[BOARD_SIZE-1][0] == max_tile) corner_value += 30.0;
        if (board[BOARD_SIZE-1][BOARD_SIZE-1] == max_tile) corner_value += 20.0;
        
        // 边缘权重
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] > 0) {
                    int edge_dist = std::min(std::min(i, BOARD_SIZE-1-i), 
                                           std::min(j, BOARD_SIZE-1-j));
                    edge_bonus -= edge_dist * board[i][j];
                }
            }
        }
        
        // 综合评估
        total_score = empty_count * empty_weight +
                     smoothness * smooth_weight +
                     monotonicity * mono_weight +
                     corner_value * corner_weight +
                     max_tile * max_tile_weight +
                     edge_bonus * edge_weight;
        
        return total_score;
    }
    
    // 动态搜索深度
    int get_dynamic_depth() {
        int empty_cells = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) empty_cells++;
            }
        }
        
        int base_depth = 4;
        if (empty_cells >= 8) base_depth = 3;
        else if (empty_cells <= 4) base_depth = 6;
        
        return base_depth;
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
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        auto [x, y] = empty_cells[gen() % empty_cells.size()];
        board[x][y] = (gen() % 10 < 9) ? 1 : 2;
        return true;
    }
    
    void display() {
        std::cout << "\n";
        std::cout << "╔══════╦══════╦══════╦══════╗\n";
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::cout << "║";
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    std::cout << "      ║";
                } else {
                    int value = 1 << board[i][j];
                    if (value < 10) std::cout << "  " << value << "  ║";
                    else if (value < 100) std::cout << " " << value << "  ║";
                    else if (value < 1000) std::cout << " " << value << " ║";
                    else std::cout << value << " ║";
                }
            }
            if (i < BOARD_SIZE - 1) 
                std::cout << "\n╠══════╬══════╬══════╬══════╣\n";
            else 
                std::cout << "\n╚══════╩══════╩══════╩══════╝\n";
        }
        std::cout << "Score: " << score << " | Moves: " << moves 
                  << " | Max Tile: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
    }
    
    bool move_left(bool actual_move = true) {
        std::vector<std::vector<int>> old_board = board;
        int old_score = score;
        bool moved = false;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::vector<int> new_row;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != 0) new_row.push_back(board[i][j]);
            }
            
            for (size_t j = 0; j < new_row.size(); j++) {
                if (j + 1 < new_row.size() && new_row[j] == new_row[j + 1]) {
                    if (actual_move) score += 1 << (new_row[j] + 1);
                    new_row[j]++;
                    new_row.erase(new_row.begin() + j + 1);
                    moved = true;
                }
            }
            
            while (new_row.size() < BOARD_SIZE) new_row.push_back(0);
            
            if (actual_move) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (old_board[i][j] != new_row[j]) moved = true;
                    board[i][j] = new_row[j];
                }
            }
        }
        
        if (!actual_move) {
            board = old_board;
            score = old_score;
        }
        
        return moved;
    }
    
    void rotate_board() {
        std::vector<std::vector<int>> temp(BOARD_SIZE, 
                         std::vector<int>(BOARD_SIZE));
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
        
        for (int i = 0; i < direction; i++) rotate_board();
        bool moved = move_left(actual_move);
        for (int i = 0; i < (4 - direction) % 4; i++) rotate_board();
        
        if (!actual_move) {
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
                if ((i < BOARD_SIZE - 1 && current == board[i + 1][j]) ||
                    (j < BOARD_SIZE - 1 && current == board[i][j + 1])) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Expectimax搜索
    double expectimax_search(int depth, bool is_maximizing, double probability = 1.0) {
        if (depth == 0 || is_game_over()) {
            return evaluate_state();
        }
        
        if (probability < 0.01) {
            return evaluate_state();
        }
        
        if (is_maximizing) {
            double best_value = -1e9;
            
            for (int move_dir = 0; move_dir < 4; move_dir++) {
                auto old_state = board;
                auto old_score = score;
                
                if (move(move_dir, false)) {
                    double value = expectimax_search(depth - 1, false, probability);
                    best_value = std::max(best_value, value);
                }
                
                board = old_state;
                score = old_score;
            }
            
            return (best_value > -1e8) ? best_value : evaluate_state();
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
            
            if (empty_count == 0) return evaluate_state();
            
            for (auto [x, y] : empty_cells) {
                board[x][y] = 1;
                double value_2 = expectimax_search(depth - 1, true, probability * 0.9 / empty_count);
                board[x][y] = 0;
                
                board[x][y] = 2;
                double value_4 = expectimax_search(depth - 1, true, probability * 0.1 / empty_count);
                board[x][y] = 0;
                
                expected_value += 0.9 * value_2 + 0.1 * value_4;
            }
            
            return expected_value / empty_count;
        }
    }
    
    int find_best_move() {
        double best_value = -1e9;
        int best_move = 0;
        int depth = get_dynamic_depth();
        
        std::vector<std::future<std::pair<int, double>>> futures;
        
        for (int move_dir = 0; move_dir < 4; move_dir++) {
            futures.push_back(std::async(std::launch::async, 
                [this, move_dir, depth]() {
                    auto old_board = this->board;
                    auto old_score = this->score;
                    
                    double value = -1e9;
                    if (this->move(move_dir, false)) {
                        value = this->expectimax_search(depth - 1, false);
                    }
                    
                    this->board = old_board;
                    this->score = old_score;
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
        
        return best_move;
    }
    
    void play_game() {
        auto start_time = std::chrono::high_resolution_clock::now();
        int display_counter = 0;
        
        std::cout << "🚀 开始高性能2048 AI游戏！\n";
        std::cout << "🎯 目标: 高分 + 大方块\n";
        std::cout << "⚡ 使用优化评估函数和动态深度调整\n\n";
        
        while (!is_game_over() && moves < 10000) {
            if (display_counter % 10 == 0) {
                display();
            }
            
            int best_move = find_best_move();
            int move_score = 0;
            if (move(best_move, true)) {
                moves++;
                add_random_tile();
                update_max_tile();
            }
            display_counter++;
            
            if (moves % 50 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);
                std::cout << "📊 进度: " << moves << " 步 | 时间: " 
                          << duration.count() << "秒 | 分数: " << score 
                          << " | 最大方块: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
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
        std::cout << std::string(60, '=') << "\n";
    }
};

int main() {
    std::cout << "2048 AI 高性能版 - 优化评估函数\n";
    std::cout << "================================\n";
    
    try {
        HighPerformance2048AI game;
        game.play_game();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
