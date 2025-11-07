// 2048_cuda_kernels.cu
#include "2048_ai_cuda.h"

// 设备上的评估函数
__device__ float evaluate_state_device(const GameState& state) {
    float score = 0.0f;
    int empty_count = 0;
    int max_tile = 0;
    float smoothness = 0.0f;
    float monotonicity = 0.0f;
    
    // 统计空格子和最大方块
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (state.board[i][j] == 0) {
                empty_count++;
            } else {
                if (state.board[i][j] > max_tile) {
                    max_tile = state.board[i][j];
                }
            }
        }
    }
    
    // 平滑度计算（相邻方块差异）
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE - 1; j++) {
            if (state.board[i][j] != 0 && state.board[i][j+1] != 0) {
                smoothness -= abs(state.board[i][j] - state.board[i][j+1]);
            }
        }
    }
    
    for (int j = 0; j < BOARD_SIZE; j++) {
        for (int i = 0; i < BOARD_SIZE - 1; i++) {
            if (state.board[i][j] != 0 && state.board[i+1][j] != 0) {
                smoothness -= abs(state.board[i][j] - state.board[i+1][j]);
            }
        }
    }
    
    // 单调性计算
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE - 1; j++) {
            if (state.board[i][j] > state.board[i][j+1]) {
                monotonicity += state.board[i][j] - state.board[i][j+1];
            } else {
                monotonicity += state.board[i][j+1] - state.board[i][j];
            }
        }
    }
    
    // 综合评估函数[1,6](@ref)
    score = empty_count * 10.0f + 
            max_tile * 2.0f + 
            smoothness * 0.1f + 
            monotonicity * 0.05f;
    
    return score;
}

// 设备上的移动函数
__device__ bool move_left_device(GameState& state) {
    bool moved = false;
    for (int i = 0; i < BOARD_SIZE; i++) {
        int write_pos = 0;
        int last_value = 0;
        
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (state.board[i][j] != 0) {
                if (last_value == state.board[i][j]) {
                    state.board[i][write_pos - 1] = last_value + 1;
                    state.score += (1 << (last_value + 1));
                    last_value = 0;
                    moved = true;
                } else {
                    if (write_pos != j) moved = true;
                    state.board[i][write_pos] = state.board[i][j];
                    last_value = state.board[i][j];
                    write_pos++;
                }
            }
        }
        
        for (int j = write_pos; j < BOARD_SIZE; j++) {
            state.board[i][j] = 0;
        }
    }
    return moved;
}

// 旋转棋盘以便统一处理不同方向
__device__ void rotate_board_device(GameState& state) {
    GameState temp = state;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            state.board[i][j] = temp.board[BOARD_SIZE - j - 1][i];
        }
    }
}

// 设备上的移动方向处理
__device__ bool move_direction_device(GameState& state, int direction) {
    bool moved = false;
    
    // 通过旋转统一处理不同方向[4](@ref)
    for (int i = 0; i < direction; i++) {
        rotate_board_device(state);
    }
    
    moved = move_left_device(state);
    
    for (int i = 0; i < (4 - direction) % 4; i++) {
        rotate_board_device(state);
    }
    
    return moved;
}

// Expectimax搜索核函数
__global__ void expectimax_search_kernel(ExpectimaxNode* nodes, int node_count, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;
    
    ExpectimaxNode& node = nodes[idx];
    
    if (depth == 0) {
        node.expected_value = evaluate_state_device(node.state);
        return;
    }
    
    // 最大化节点（玩家回合）
    if (depth % 2 == 0) {
        float best_value = -FLT_MAX;
        int best_move = -1;
        
        for (int move = 0; move < 4; move++) {
            GameState new_state = node.state;
            if (move_direction_device(new_state, move)) {
                ExpectimaxNode child_node;
                child_node.state = new_state;
                child_node.depth = depth - 1;
                
                // 递归评估（简化版本，实际需要更复杂的并行处理）
                child_node.expected_value = evaluate_state_device(child_node.state);
                
                if (child_node.expected_value > best_value) {
                    best_value = child_node.expected_value;
                    best_move = move;
                }
            }
        }
        
        node.expected_value = (best_move != -1) ? best_value : 0.0f;
        node.best_move = best_move;
    } else {
        // 期望节点（随机方块生成）
        float total_value = 0.0f;
        int empty_count = 0;
        
        // 统计空格子
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (node.state.board[i][j] == 0) {
                    empty_count++;
                }
            }
        }
        
        if (empty_count == 0) {
            node.expected_value = evaluate_state_device(node.state);
            return;
        }
        
        // 为每个空格子生成2和4的情况[6](@ref)
        float expected = 0.0f;
        int valid_chances = 0;
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (node.state.board[i][j] == 0) {
                    // 生成2（90%概率）
                    GameState state_2 = node.state;
                    state_2.board[i][j] = 1; // 2 = 2^1
                    float eval_2 = evaluate_state_device(state_2);
                    expected += 0.9f * eval_2;
                    valid_chances++;
                    
                    // 生成4（10%概率）
                    GameState state_4 = node.state;
                    state_4.board[i][j] = 2; // 4 = 2^2
                    float eval_4 = evaluate_state_device(state_4);
                    expected += 0.1f * eval_4;
                    valid_chances++;
                }
            }
        }
        
        node.expected_value = (valid_chances > 0) ? expected / valid_chances : 0.0f;
    }
}
