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
#include <unordered_map>

#define BOARD_SIZE 4
#define TARGET_TILE 16  // 2^16 = 65536

class Ultimate2048AI {
private:
    std::vector<std::vector<int>> board;
    int score;
    int moves;
    int max_tile;
    std::mt19937 rng;
    
    // 基于nneonneo AI优化的启发式权重
    static constexpr double EMPTY_WEIGHT = 270000.0;
    static constexpr double MONOTONICITY_WEIGHT = 35.0;
    static constexpr double SMOOTHNESS_WEIGHT = 25.0;
    static constexpr double CORNER_WEIGHT = 50000.0;
    static constexpr double MAX_TILE_WEIGHT = 400.0;
    static constexpr double MERGE_POTENTIAL_WEIGHT = 15.0;
    
public:
    Ultimate2048AI() : score(0), moves(0), max_tile(0) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
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
    
    // 高性能评估函数
    double evaluate_state() {
        if (is_game_over()) return -1000000.0;
        
        double total_score = 0.0;
        int empty_count = 0;
        double monotonicity = 0.0;
        double smoothness = 0.0;
        double corner_value = 0.0;
        double merge_potential = 0.0;
        
        // 1. 空格子统计（最重要的启发式）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) empty_count++;
            }
        }
        
        // 2. 单调性计算（鼓励有序排列）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0) {
                    double current = std::log2(board[i][j]);
                    double next = std::log2(board[i][j+1]);
                    monotonicity -= std::abs(current - next);
                }
            }
        }
        
        // 3. 平滑度计算（相邻方块差异）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0) {
                    smoothness -= std::abs(board[i][j] - board[i][j+1]);
                }
            }
        }
        
        // 4. 合并潜力评估
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (board[i][j] != 0 && board[i][j+1] != 0 && 
                    board[i][j] == board[i][j+1]) {
                    merge_potential += (1 << board[i][j]) * 3.0;
                }
            }
        }
        
        // 5. 角落偏好（高价值方块在角落）
        if (board[0][0] == max_tile) corner_value += 100.0;
        
        // 综合评估
        total_score = empty_count * EMPTY_WEIGHT +
                     monotonicity * MONOTONICITY_WEIGHT +
                     smoothness * SMOOTHNESS_WEIGHT +
                     corner_value * CORNER_WEIGHT +
                     max_tile * MAX_TILE_WEIGHT +
                     merge_potential * MERGE_POTENTIAL_WEIGHT;
        
        return total_score;
    }
    
    // 动态搜索深度调整
    int get_dynamic_depth() {
        int empty_cells = 0;
        int large_tiles = 0;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) empty_cells++;
                if (board[i][j] >= 8) large_tiles++; // 256或更大
            }
        }
        
        // 基于空格数量的基础深度
        int base_depth = 3;
        if (empty_cells >= 10) base_depth = 3;      // 简单局面
        else if (empty_cells >= 6) base_depth = 4;  // 中等局面  
        else if (empty_cells >= 3) base_depth = 5;  // 复杂局面
        else base_depth = 6;                        // 极复杂局面
        
        // 有大方块时增加搜索深度
        if (large_tiles >= 2) base_depth += 1;
        if (max_tile >= 10) base_depth += 1; // 1024或更大
        
        return std::min(base_depth, 6);
    }
    
    bool add_random_tile() {
        std::vector<std::pair<int, int>> empty_cells;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][极速版j] == 0) {
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
    
    // 修复的移动逻辑
    bool move_left(bool actual_move = true) {
        std::vector<std::vector<int>> old_board = board;
        int old_score = score;
        bool moved = false;
        int move_score = 0;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            // 压缩非零元素
            std::vector<int> new_row;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] != 0) {
                    new_row.push_back(board[i][j]);
                }
            }
            
            // 合并相同元素
            for (size_t j = 0; j < new_row.size(); ) {
                if (j + 1 < new_row.size() && new_row[j] == new_row[j+1]) {
                    new_row[j]++;
                    move_score += 1 << new_row[j];
                    new_row.erase(new_row.begin() + j + 1);
                    moved = true;
                    j++;
                } else {
                    j++;
                }
            }
            
            // 填充零值
            while (new极速版_row.size() < BOARD_SIZE) {
                new_row.push_back(0);
            }
            
            // 更新棋盘
            for (int j = 0; j < BOARD_SIZE; j++) {
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
        // 检查空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) return false;
            }
        }
        
        // 检查可合并的相邻方块
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
    
    // Expectimax搜索算法
    double expectimax_search(int depth, bool is_maximizing, double probability = 1.0) {
        if (depth == 0 || is_game_over()) {
            return evaluate_state();
        }
        
        if (probability < 0.001) {
            return evaluate_state();
        }
        
        if (is_maximizing) {
            double best_value = -1e9;
            bool found_valid = false;
            
            for (int move_dir = 0; move_dir < 4; move_dir++) {
                auto old_board = board;
                auto old_score = score;
                
                if (move(move_dir, false)) {
                    double value = expectimax_search(depth - 1, false, probability);
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
            
            int evaluations = 0;
            for (auto [x, y] : empty_cells) {
                // 90%概率生成2
                auto old_board = board;
                auto old_score = score;
                board[x][y] = 1;
                double value_2 = expectimax_search(depth - 1, true, probability * 0.9 / empty_count);
                board = old_board;
                score = old_score;
                
                // 10%概率生成4
                board[x][y] = 2;
                double value_4 = expectimax_search(depth - 1, true, probability * 0.1 / empty_count);
                board = old_board;
                score = old_score;
                
                expected_value += 0.9 * value_2 + 0.1 * value_4;
                evaluations++;
            }
            
            return evaluations > 0 ? expected_value / empty_count : evaluate_state();
        }
    }
    
    int find_best_move() {
        int depth = get_dynamic_depth();
        double best_value = -1e9;
        int best_move = 0;
        
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
        
        std::cout << "🚀 终极版2048 AI启动 - 基于nneonneo算法深度优化\n";
        std::cout << "🎯 目标: 10万分 + 65536方块\n";
        std::cout << "⚡ 使用动态深度调整和并行Expectimax搜索\n\n";
        
        while (!is_game_over() && moves < 10000) {
            if (display_counter % 5 == 0) {
                display();
            }
            
            int best_move = find_best_move();
            if (move(best_move, true)) {
                moves++;
                add_random_tile();
                update_max_tile();
            }
            
            display_counter++;
            
            if (moves % 20 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);
                std::cout << "📊 进度: " << moves << " 步 | 时间: " 
                          << duration.count() << "秒 | 分数: " << score 
                          << " | 最大方块: " << (max_tile > 0 ? (1 << max_tile) : 0) << "\n";
            }
            
            if (has_won()) {
                std::cout << "🎉 达成65536目标！继续向更高分前进...\n";
            }
            
            if (score >= 100000 && max_tile >= TARGET_TILE) {
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
