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
#include <bitset>

#define BOARD_SIZE 4
#define TARGET_TILE 16  // 2^16 = 65536

class HighPerformance2048AI {
private:
    // 使用位棋盘表示状态（64位）
    uint64_t bitboard;
    int score;
    int moves;
    int max_tile;
    std::mt19937 rng;
    
    // 查找表
    std::unordered_map<uint64_t, std::pair<uint64_t, int>> move_table;
    std::unordered_map<uint64_t, double> eval_table;
    
    // 启发式权重（基于nneonneo的优化）
    static constexpr double EMPTY_WEIGHT = 270000.0;
    static constexpr double MONOTONICITY_WEIGHT = 35.0;
    static constexpr double SMOOTHNESS_WEIGHT = 25.0;
    static constexpr double CORNER_WEIGHT = 50000.0;
    static constexpr double MAX_TILE_WEIGHT = 400.0;
    static constexpr double MERGE_POTENTIAL_WEIGHT = 15.0;
    static constexpr double EDGE_WEIGHT = 8.0;
    
    // 将位置转换为位棋盘索引
    constexpr int pos_to_idx(int row, int col) const {
        return (row * BOARD_SIZE + col) * 4;
    }
    
    // 从位棋盘获取格子值
    int get_tile(int row, int col) const {
        int idx = pos_to_idx(row, col);
        return (bitboard >> idx) & 0xF; // 每个格子用4位表示
    }
    
    // 设置格子值到位棋盘
    void set_tile(int row, int col, int value) {
        int idx = pos_to_idx(row, col);
        uint64_t mask = ~(0xFULL << idx);
        bitboard = (bitboard & mask) | (static_cast<uint64_t>(value) << idx);
    }
    
    // 初始化查找表
    void initialize_lookup_tables() {
        // 在实际实现中，这里会预计算各种棋盘状态的移动和评估值
        // 简化版本：在实际搜索过程中动态计算
    }
    
public:
    HighPerformance2048AI() : bitboard(0), score(0), moves(0), max_tile(0) {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        initialize();
        initialize_lookup_tables();
    }
    
    void initialize() {
        bitboard = 0;
        add_random_tile();
        add_random_tile();
        update_max_tile();
    }
    
    void update_max_tile() {
        max_tile = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int tile = get_tile(i, j);
                if (tile > max_tile) max_tile = tile;
            }
        }
    }
    
    // 高性能评估函数
    double evaluate_state() {
        // 检查缓存
        auto it = eval_table.find(bitboard);
        if (it != eval_table.end()) {
            return it->second;
        }
        
        if (is_game_over()) {
            eval_table[bitboard] = -1000000.0;
            return -1000000.0;
        }
        
        double total_score = 0.0;
        int empty_count = 0;
        double monotonicity = 0.0;
        double smoothness = 0.0;
        double corner_value = 0.0;
        double merge_potential = 0.0;
        double edge_penalty = 0.0;
        
        // 1. 空格子统计
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (get_tile(i, j) == 0) empty_count++;
            }
        }
        
        // 2. 单调性计算
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                int tile1 = get_tile(i, j);
                int tile2 = get_tile(i, j+1);
                if (tile1 != 0 && tile2 != 0) {
                    double current = std::log2(1 << tile1);
                    double next = std::log2(1 << tile2);
                    monotonicity -= std::abs(current - next);
                }
            }
        }
        
        // 3. 平滑度计算
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                int tile1 = get_tile(i, j);
                int tile2 = get_tile(i, j+1);
                if (tile1 != 0 && tile2 != 0) {
                    smoothness -= std::abs(tile1 - tile2);
                }
            }
        }
        
        // 4. 合并潜力评估
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                int tile1 = get_tile(i, j);
                int tile2 = get_tile(i, j+1);
                if (tile1 != 0 && tile2 != 0 && tile1 == tile2) {
                    merge_potential += (1 << tile1) * 3.0;
                }
            }
        }
        
        // 5. 角落偏好
        if (get_tile(0, 0) == max_tile) corner_value += 100.0;
        
        // 6. 边缘惩罚
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int tile = get_tile(i, j);
                if (tile > 0) {
                    int edge_dist = std::min(std::min(i, BOARD_SIZE-1-i), 
                                           std::min(j, BOARD_SIZE-1-j));
                    edge_penalty -= edge_dist * tile;
                }
            }
        }
        
        // 综合评估
        total_score = empty_count * EMPTY_WEIGHT +
                     monotonicity * MONOTONICITY_WEIGHT +
                     smoothness * SMOOTHNESS_WEIGHT +
                     corner_value * CORNER_WEIGHT +
                     max_tile * MAX_TILE_WEIGHT +
                     merge_potential * MERGE_POTENTIAL_WEIGHT +
                     edge_penalty * EDGE_WEIGHT;
        
        eval_table[bitboard] = total_score;
        return total_score;
    }
    
    int get_dynamic_depth() {
        int empty_cells = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (get_tile(i, j) == 0) empty_cells++;
            }
        }
        
        if (empty_cells >= 10) return 3;
        else if (empty_cells >= 6) return 4;
        else if (empty_cells >= 3) return 5;
        else return 6;
    }
    
    bool add_random_tile() {
        std::vector<std::pair<int, int>> empty_cells;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (get_tile(i, j) == 0) {
                    empty_cells.push_back({i, j});
                }
            }
        }
        
        if (empty_cells.empty()) return false;
        
        auto [x, y] = empty_cells[rng() % empty_cells.size()];
        set_tile(x, y, (rng() % 10 < 9) ? 1 : 2); // 90% 2, 10% 4
        update_max_tile();
        return true;
    }
    
    void display() {
        std::cout << "\n";
        std::cout << "╔════════╦════════╦════════╦════════╗\n";
        for (int i = 0; i < BOARD_SIZE; i++) {
            std::cout << "║";
            for (int j = 0; j < BOARD_SIZE; j++) {
                int tile = get_tile(i, j);
                if (tile == 0) {
                    std::cout << "        ║";
                } else {
                    int value = 1 << tile;
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
    
    // 移动函数（使用位操作优化）
    bool move_left(bool actual_move = true) {
        uint64_t old_bitboard = bitboard;
        int old_score = score;
        bool moved = false;
        int move_score = 0;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            // 提取行
            uint64_t row = (bitboard >> (i * 16)) & 0xFFFF;
            
            // 压缩非零元素
            uint64_t new_row = 0;
            int pos = 0;
            for (int j = 0; j < BOARD_SIZE; j++) {
                int tile = (row >> (j * 4)) & 0xF;
                if (tile != 0) {
                    new_row |= static_cast<uint64_t>(tile) << (pos * 4);
                    pos++;
                }
            }
            
            // 合并相同元素
            for (int j = 0; j < BOARD_SIZE - 1; j++) {
                int tile1 = (new_row >> (j * 4)) & 0xF;
                int tile2 = (new_row >> ((j+1) * 4)) & 0xF;
                
                if (tile1 != 0 && tile1 == tile2) {
                    int merged = tile1 + 1;
                    new_row &= ~(0xFULL << (j * 4));
                    new_row &= ~(0xFULL << ((j+1) * 4));
                    new_row |= static_cast<uint64_t>(merged) << (j * 4);
                    move_score += 1 << merged;
                    moved = true;
                    
                    // 移动剩余元素
                    for (int k = j+1; k < BOARD_SIZE - 1; k++) {
                        int next_tile = (new_row >> ((k+1) * 4)) & 0xF;
                        new_row &= ~(0xFULL << (k * 4));
                        new_row |= static_cast<uint64_t>(next_tile) << (k * 4);
                    }
                    new_row &= ~(0xFULL << ((BOARD_SIZE-1) * 4));
                }
            }
            
            // 更新位棋盘
            bitboard &= ~(0xFFFFULL << (i * 16));
            bitboard |= new_row << (i * 16);
        }
        
        if (moved && actual_move) {
            score += move_score;
        } else if (!actual_move) {
            bitboard = old_bitboard;
            score = old_score;
        }
        
        return moved;
    }
    
    void rotate_board() {
        uint64_t new_bitboard = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int tile = get_tile(BOARD_SIZE - j - 1, i);
                new_bitboard |= static_cast<uint64_t>(tile) << pos_to_idx(i, j);
            }
        }
        bitboard = new_bitboard;
    }
    
    bool move(int direction, bool actual_move = true) {
        uint64_t old_bitboard = bitboard;
        int old_score = score;
        bool moved = false;
        
        for (int i = 0; i < direction; i++) {
            rotate_board();
        }
        
        moved = move_left(actual_move);
        
        for (int i = 0; i < (4 - direction) % 4; i++) {
            rotate_board();
        }
        
        if (!actual_move && !moved) {
            bitboard = old_bitboard;
            score = old_score;
        }
        
        return moved;
    }
    
    bool is_game_over() {
        // 检查空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (get_tile(i, j) == 0) return false;
            }
        }
        
        // 检查可合并的相邻方块
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int current = get_tile(i, j);
                if ((j < BOARD_SIZE - 1 && current == get_tile(i, j+1)) ||
                    (i < BOARD_SIZE - 1 && current == get_tile(i+1, j))) {
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
                uint64_t old_bitboard = bitboard;
                int old_score = score;
                
                if (move(move_dir, false)) {
                    double value = expectimax_search(depth - 1, false, probability);
                    if (value > best_value) {
                        best_value = value;
                    }
                    found_valid = true;
                }
                
                bitboard = old_bitboard;
                score = old_score;
            }
            
            return found_valid ? best_value : evaluate_state();
        } else {
            double expected_value = 0.0;
            int empty_count = 0;
            
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (get_tile(i, j) == 0) empty_count++;
                }
            }
            
            if (empty_count == 0) {
                return expectimax_search(depth - 1, true, probability);
            }
            
            int evaluations = 0;
            for (int i = 0; i < BOARD_SIZE; i++) {
                for (int j = 0; j < BOARD_SIZE; j++) {
                    if (get_tile(i, j) == 0) {
                        // 90%概率生成2
                        uint64_t old_bitboard = bitboard;
                        set_tile(i, j, 1);
                        double value_2 = expectimax_search(depth - 1, true, probability * 0.9 / empty_count);
                        bitboard = old_bitboard;
                        
                        // 10%概率生成4
                        set_tile(i, j, 2);
                        double value_4 = expectimax_search(depth - 1, true, probability * 0.1 / empty_count);
                        bitboard = old_bitboard;
                        
                        expected_value += 0.9 * value_2 + 0.1 * value_4;
                        evaluations++;
                    }
                }
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
                    uint64_t board_backup = this->bitboard;
                    int score_backup = this->score;
                    double value = -1e9;
                    
                    if (this->move(move_dir, false)) {
                        value = this->expectimax_search(depth - 1, false);
                    }
                    
                    this->bitboard = board_backup;
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
        
        std::cout << "🚀 高性能2048 AI启动 - 基于nneonneo算法优化\n";
        std::cout << "🎯 目标: 10万分 + 65536方块\n";
        std::cout << "⚡ 使用位棋盘表示和并行计算\n\n";
        
        while (!is_game_over() && moves < 10000) {
            if (display_counter % 10 == 0) {
                display();
            }
            
            int best_move = find_best_move();
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
            
            if (has_won()) {
                std::cout << "🎉 达成65536目标！继续向更高分前进...\n";
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
        }
        std::cout << std::string(60, '=') << "\n";
    }
};

int main() {
    std::cout << "2048 AI 高性能优化版 - 基于nneonneo算法\n";
    std::cout << "==========================================\n";
    
    try {
        HighPerformance2048AI game;
        game.play_game();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
