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
#include <memory>

#define BOARD_SIZE 4
#define TARGET_4096 12  // 2^12 = 4096

// 游戏状态类（可安全拷贝）
class GameState {
public:
    std::vector<std::vector<int>> board;
    int score;
    int max_tile;
    
    GameState() : score(0), max_tile(0) {
        board.resize(BOARD_SIZE, std::vector<int>(BOARD_SIZE, 0));
    }
    
    GameState(const GameState& other) {
        board = other.board;
        score = other.score;
        max_tile = other.max_tile;
    }
    
    GameState& operator=(const GameState& other) {
        if (this != &other) {
            board = other.board;
            score = other.score;
            max_tile = other.max_tile;
        }
        return *this;
    }
    
    void update_max_tile() {
        max_tile = 0;
        for (const auto& row : board) {
            for (int val : row) {
                if (val > max_tile) max_tile = val;
            }
        }
    }
    
    int count_empty_cells() const {
        int count = 0;
        for (const auto& row : board) {
            for (int val : row) {
                if (val == 0) count++;
            }
        }
        return count;
    }
    
    // 检查游戏是否结束
    bool is_game_over() const {
        // 检查是否有空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] == 0) {
                    return false;
                }
            }
        }
        
        // 检查是否还有可合并的相邻方块
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
    
    bool has_won() const {
        return max_tile >= TARGET_4096;
    }
};

class HighPerformance2048AI {
private:
    GameState current_state;
    int moves;
    std::mt19937 rng;
    
    // 优化后的启发式权重（基于大量测试）
    const double EMPTY_WEIGHT = 15000.0;      // 降低空格子权重，避免过于保守
    const double MONOTONICITY_WEIGHT = 25.0;  // 提高单调性权重
    const double SMOOTHNESS_WEIGHT = 15.0;    // 提高平滑度权重
    const double CORNER_WEIGHT = 5000.0;      // 降低角落权重
    const double MAX_TILE_WEIGHT = 500.0;     // 提高最大方块权重
    const double MERGE_POTENTIAL_WEIGHT = 8.0; // 提高合并潜力权重
    
public:
    HighPerformance2048AI() : moves(0) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        initialize();
    }
    
    void initialize() {
        current_state = GameState();
        add_random_tile(current_state);
        add_random_tile(current_state);
        current_state.update_max_tile();
        moves = 0;
    }
    
    // 修复：安全的随机方块添加
    bool add_random_tile(GameState& state) {
        std::vector<std::pair<int, int>> empty_cells;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (state.board[i][j] == 0) {
                    empty_cells.push_back({i, j});
                }
            }
        }
        
        if (empty_cells.empty()) return false;
        
        auto [x, y] = empty_cells[rng() % empty_cells.size()];
        state.board[x][y] = (rng() % 10 < 9) ? 1 : 2; // 90% 2, 10% 4
        state.update_max_tile();
        return true;
    }
    
    void display() {
        std::cout << "\n";
        std::cout << "╔════════╦════════╦════════╦════════╗\n";
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::cout << "║";
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] == 0) {
                    std::cout << "        ║";
                } else {
                    int value = 1 << current_state.board[i][j];
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
        std::cout << "Score: " << current_state.score << " | Moves: " << moves 
                  << " | Max Tile: " << (current_state.max_tile > 0 ? 
                     (1 << current_state.max_tile) : 0) << "\n";
    }
    
    // 修复：完全重写移动逻辑，避免状态污染
    bool move_left(GameState& state, bool actual_move = true) {
        GameState old_state = state;
        bool moved = false;
        int move_score = 0;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            // 压缩非零元素
            std::vector<int> new_row;
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (state.board[i][j] != 0) {
                    new_row.push_back(state.board[i][j]);
                }
            }
            
            // 合并相同元素（修复合并逻辑）
            for (size_t j = 0; j < new_row.size(); ) {
                if (j + 1 < new_row.size() && new_row[j] == new_row[j+1]) {
                    new_row[j]++; // 值加倍
                    move_score += 1 << new_row[j];
                    new_row.erase(new_row.begin() + j + 1);
                    moved = true;
                    j++; // 跳过下一个元素，防止重复合并
                } else {
                    j++;
                }
            }
            
            // 填充零值
            while (new_row.size() < BOARD_SIZE) {
                new_row.push_back(0);
            }
            
            // 检查是否移动并更新
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (state.board[i][j] != new_row[j]) {
                    moved = true;
                }
                state.board[i][j] = new_row[j];
            }
        }
        
        if (moved && actual_move) {
            state.score += move_score;
        }
        
        if (!actual_move) {
            state = old_state; // 恢复状态
        } else {
            state.update_max_tile();
        }
        
        return moved;
    }
    
    void rotate_board(GameState& state) {
        std::vector<std::vector<int>> temp(BOARD_SIZE, 
                     std::vector<int>(BOARD_SIZE));
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                temp[i][j] = state.board[BOARD_SIZE - j - 1][i];
            }
        }
        state.board = temp;
        state.update_max_tile();
    }
    
    bool move(GameState& state, int direction, bool actual_move = true) {
        GameState old_state = state;
        bool moved = false;
        
        // 通过旋转统一处理方向
        for (int i = 0; i < direction; i++) {
            rotate_board(state);
        }
        
        moved = move_left(state, actual_move);
        
        for (int i = 0; i < (4 - direction) % 4; i++) {
            rotate_board(state);
        }
        
        if (!actual_move && !moved) {
            state = old_state;
        }
        
        return moved;
    }
    
    // 优化后的评估函数
    double evaluate_state(const GameState& state) {
        if (state.is_game_over()) return -1000000.0;
        
        double total_score = 0.0;
        int empty_count = state.count_empty_cells();
        double monotonicity = 0.0;
        double smoothness = 0.0;
        double corner_value = 0.0;
        double merge_potential = 0.0;
        
        // 1. 单调性计算（鼓励有序排列）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (state.board[i][j] != 0 && state.board[i][j+1] != 0) {
                    double diff = std::log2(state.board[i][j]) - std::log2(state.board[i][j+1]);
                    monotonicity -= std::abs(diff); // 差异越小越好
                }
            }
        }
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (state.board[i][j] != 0 && state.board[i+1][j] != 0) {
                    double diff = std::log2(state.board[i][j]) - std::log2(state.board[i+1][j]);
                    monotonicity -= std::abs(diff);
                }
            }
        }
        
        // 2. 平滑度计算（相邻方块差异）
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (state.board[i][j] != 0 && state.board[i][j+1] != 0) {
                    int diff = std::abs(state.board[i][j] - state.board[i][j+1]);
                    smoothness -= diff * 0.1; // 差异惩罚
                }
            }
        }
        
        // 3. 合并潜力评估
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                if (state.board[i][j] != 0 && state.board[i][j] == state.board[i][j+1]) {
                    merge_potential += (1 << state.board[i][j]) * 2.0;
                }
            }
        }
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            for (int i = 0; i < BOARD_SIZE - 1; i++) {
                if (state.board[i][j] != 0 && state.board[i][j] == state.board[i+1][j]) {
                    merge_potential += (1 << state.board[i][j]) * 2.0;
                }
            }
        }
        
        // 4. 角落偏好
        if (state.board[0][0] == state.max_tile) corner_value += 100.0;
        if (state.board[0][BOARD_SIZE-1] == state.max_tile) corner_value += 80.0;
        if (state.board[BOARD_SIZE-1][0] == state.max_tile) corner_value += 80.0;
        if (state.board[BOARD_SIZE-1][BOARD_SIZE-1] == state.max_tile) corner_value += 60.0;
        
        // 5. 综合评估（调整权重平衡）
        total_score = empty_count * EMPTY_WEIGHT +
                     monotonicity * MONOTONICITY_WEIGHT +
                     smoothness * SMOOTHNESS_WEIGHT +
                     corner_value * CORNER_WEIGHT +
                     state.max_tile * MAX_TILE_WEIGHT +
                     merge_potential * MERGE_POTENTIAL_WEIGHT;
        
        return total_score;
    }
    
    // 动态搜索深度调整
    int get_dynamic_depth(const GameState& state) {
        int empty_cells = state.count_empty_cells();
        
        if (empty_cells >= 10) return 3;      // 简单局面
        else if (empty_cells >= 6) return 4;   // 中等局面
        else if (empty_cells >= 3) return 5;  // 复杂局面
        else return 6;                        // 极复杂局面
    }
    
    // 修复：线程安全的Expectimax搜索
    double expectimax_search(GameState state, int depth, bool is_maximizing, double probability = 1.0) {
        if (depth == 0 || state.is_game_over()) {
            return evaluate_state(state);
        }
        
        if (probability < 0.001) {
            return evaluate_state(state);
        }
        
        if (is_maximizing) {
            double best_value = -std::numeric_limits<double>::max();
            bool found_valid_move = false;
            
            for (int move_dir = 0; move_dir < 4; move_dir++) {
                GameState new_state = state;
                if (move(new_state, move_dir, false)) {
                    double value = expectimax_search(new_state, depth - 1, false, probability);
                    if (value > best_value) {
                        best_value = value;
                    }
                    found_valid_move = true;
                }
            }
            
            return found_valid_move ? best_value : evaluate_state(state);
        } else {
            // 期望节点（随机方块生成）
            double expected_value = 0.0;
            int empty_count = state.count_empty_cells();
            
            if (empty_count == 0) {
                return expectimax_search(state, depth - 1, true, probability);
            }
            
            // 评估所有可能的随机方块生成
            int evaluations = 0;
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (state.board[i][j] == 0) {
                        // 生成2（90%概率）
                        GameState state_2 = state;
                        state_2.board[i][j] = 1;
                        state_2.update_max_tile();
                        double value_2 = expectimax_search(state_2, depth - 1, true, 
                                                          probability * 0.9 / empty_count);
                        
                        // 生成4（10%概率）
                        GameState state_4 = state;
                        state_4.board[i][j] = 2;
                        state_4.update_max_tile();
                        double value_4 = expectimax_search(state_4, depth - 1, true, 
                                                          probability * 0.1 / empty_count);
                        
                        expected_value += 0.9 * value_2 + 0.1 * value_4;
                        evaluations++;
                    }
                }
            }
            
            return (evaluations > 0) ? expected_value : evaluate_state(state);
        }
    }
    
    // 修复：完全线程安全的移动评估
    int find_best_move() {
        int depth = get_dynamic_depth(current_state);
        double best_value = -std::numeric_limits<double>::max();
        int best_move = 0;
        
        std::vector<std::future<std::pair<int, double>>> futures;
        
        // 为每个移动方向创建独立的状态拷贝
        for (int move_dir = 0; move_dir < 4; move_dir++) {
            // 创建当前状态的完整拷贝
            GameState state_copy = current_state;
            
            futures.push_back(std::async(std::launch::async, 
                [state_copy, move_dir, depth, this]() mutable {
                    double value = -std::numeric_limits<double>::max();
                    
                    if (this->move(state_copy, move_dir, false)) {
                        value = this->expectimax_search(state_copy, depth - 1, false);
                    }
                    
                    return std::make_pair(move_dir, value);
                }
            ));
        }
        
        // 收集结果
        for (auto& future : futures) {
            auto [move, value] = future.get();
            if (value > best_value) {
                best_value = value;
                best_move = move;
            }
        }
        
        return (best_value > -1e9) ? best_move : 0;
    }
    
    void play_game() {
        auto start_time = std::chrono::high_resolution_clock::now();
        int display_counter = 0;
        int last_max_tile = 0;
        
        std::cout << "🚀 修复版高性能2048 AI启动\n";
        std::cout << "🎯 目标: 10万分 + 4096方块\n";
        std::cout << "⚡ 修复了状态管理和多线程数据竞争问题\n\n";
        
        while (!current_state.is_game_over() && moves < 10000) {
            if (display_counter % 5 == 0) {
                display();
                
                // 检测最大方块异常变化
                if (last_max_tile > 0 && current_state.max_tile > 0) {
                    int current_val = 1 << current_state.max_tile;
                    int last_val = 1 << last_max_tile;
                    if (current_val < last_val / 2) {
                        std::cout << "⚠️  检测到最大方块异常变化: " << last_val 
                                  << " -> " << current_val << "\n";
                    }
                }
                last_max_tile = current_state.max_tile;
            }
            
            int best_move = find_best_move();
            
            if (move(current_state, best_move, true)) {
                moves++;
                add_random_tile(current_state);
                current_state.update_max_tile();
            }
            
            display_counter++;
            
            // 显示进度
            if (moves % 20 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);
                std::cout << "📊 进度: " << moves << " 步 | 时间: " 
                          << duration.count() << "秒 | 分数: " << current_state.score 
                          << " | 最大方块: " << (current_state.max_tile > 0 ? 
                             (1 << current_state.max_tile) : 0) << "\n";
            }
            
            if (current_state.has_won()) {
                std::cout << "🎉 达成4096目标！继续向更高分前进...\n";
            }
            
            if (current_state.score >= 100000 && current_state.max_tile >= TARGET_4096) {
                std::cout << "🎉 目标达成！分数超过10万，最大方块达到4096+\n";
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
        std::cout << "🏆 最终分数: " << current_state.score << "\n";
        std::cout << "💎 最大方块: " << (current_state.max_tile > 0 ? 
                   (1 << current_state.max_tile) : 0) << "\n";
        
        if (current_state.has_won()) {
            std::cout << "🎉 成功达到4096方块目标！\n";
        }
        if (current_state.score >= 100000) {
            std::cout << "🎉 达成10万分目标！\n";
        }
        std::cout << std::string(60, '=') << "\n";
    }
};

int main() {
    std::cout << "2048 AI 修复优化版 - 彻底解决状态异常问题\n";
    std::cout << "==========================================\n";
    std::cout << "主要修复: 多线程数据竞争、状态管理、移动逻辑\n";
    std::cout << "优化特性: 线程安全搜索、平衡评估函数、动态深度调整\n\n";
    
    try {
        HighPerformance2048AI game;
        game.play_game();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
