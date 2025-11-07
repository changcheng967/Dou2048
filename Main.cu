// main.cu
#include "2048_ai_controller.h"
#include <chrono>
#include <iomanip>

class Game2048 {
private:
    GameState current_state;
    CUDA2048AI ai;
    int move_count;
    int target_tile;
    
public:
    Game2048(int target = TILE_65536) : move_count(0), target_tile(target) {
        initialize_game();
    }
    
    void initialize_game() {
        current_state = GameState();
        add_random_tile();
        add_random_tile();
        move_count = 0;
        
        std::cout << "游戏初始化完成，目标方块: 2^" << target_tile 
                  << " = " << (1 << target_tile) << std::endl;
    }
    
    void add_random_tile() {
        std::vector<std::pair<int, int>> empty_cells;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] == 0) {
                    empty_cells.push_back({i, j});
                }
            }
        }
        
        if (!empty_cells.empty()) {
            int index = rand() % empty_cells.size();
            int value = (rand() % 10 == 0) ? 2 : 1; // 4(10%)或2(90%)
            auto pos = empty_cells[index];
            current_state.board[pos.first][pos.second] = value;
        }
    }
    
    void display_board() {
        std::cout << "\n移动次数: " << move_count << " | 分数: " 
                  << current_state.score << "\n";
        std::cout << std::string(25, '-') << "\n";
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] == 0) {
                    std::cout << std::setw(4) << ".";
                } else {
                    std::cout << std::setw(4) << (1 << current_state.board[i][j]);
                }
            }
            std::cout << "\n";
        }
        std::cout << std::string(25, '-') << std::endl;
    }
    
    bool has_reached_target() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] >= target_tile) {
                    return true;
                }
            }
        }
        return false;
    }
    
    bool is_game_over() {
        // 检查是否还有空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] == 0) {
                    return false;
                }
            }
        }
        
        // 检查是否还有可合并的相邻方块[3](@ref)
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int current = current_state.board[i][j];
                if (j < BOARD_SIZE - 1 && current == current_state.board[i][j+1]) {
                    return false;
                }
                if (i < BOARD_SIZE - 1 && current == current_state.board[i+1][j]) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    void run_ai_game() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "开始AI自动游戏...\n";
        
        while (!is_game_over() && !has_reached_target()) {
            display_board();
            
            int best_move = ai.get_best_move(current_state);
            
            if (best_move == -1) {
                std::cout << "无法找到有效移动！游戏结束。\n";
                break;
            }
            
            // 执行移动
            if (execute_move(best_move)) {
                move_count++;
                add_random_tile();
            }
            
            // 每100步显示进度
            if (move_count % 100 == 0) {
                std::cout << "已进行 " << move_count << " 步移动...\n";
            }
            
            if (move_count > 10000) { // 防止无限循环
                std::cout << "移动次数过多，强制结束\n";
                break;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        display_final_results(duration.count());
    }
    
private:
    bool execute_move(int direction) {
        // 实现具体的移动执行逻辑
        // 这里需要调用相应的移动函数
        return true;
    }
    
    void display_final_results(long long seconds) {
        std::cout << "\n" << std::string(40, '=') << "\n";
        std::cout << "游戏结束！\n";
        std::cout << "总移动次数: " << move_count << "\n";
        std::cout << "最终分数: " << current_state.score << "\n";
        std::cout << "游戏时间: " << seconds << " 秒\n";
        
        int max_tile = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_state.board[i][j] > max_tile) {
                    max_tile = current_state.board[i][j];
                }
            }
        }
        
        std::cout << "最大方块: 2^" << max_tile << " = " << (1 << max_tile) << "\n";
        
        if (has_reached_target()) {
            std::cout << "🎉 成功达到目标方块 65536！🎉\n";
        }
        
        std::cout << std::string(40, '=') << std::endl;
    }
};

// CUDA设备检查
void check_cuda_device() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "错误: 未找到CUDA设备\n";
        exit(1);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "使用CUDA设备: " << prop.name << "\n";
    std::cout << "计算能力: " << prop.major << "." << prop.minor << "\n";
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "多处理器数量: " << prop.multiProcessorCount << "\n";
}

int main() {
    std::cout << "2048 AI with CUDA Acceleration - 目标: 65536\n";
    std::cout << "============================================\n";
    
    // 检查CUDA设备
    check_cuda_device();
    
    try {
        // 创建游戏实例
        Game2048 game(TILE_65536);
        
        // 运行AI游戏
        game.run_ai_game();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
