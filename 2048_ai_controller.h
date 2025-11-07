// 2048_ai_controller.h
#pragma once
#include "2048_ai_cuda.h"
#include <vector>
#include <memory>

class CUDA2048AI {
private:
    ExpectimaxNode* d_nodes;  // 设备节点指针
    ExpectimaxNode* h_nodes;  // 主机节点指针
    int max_depth;
    int batch_size;
    
public:
    CUDA2048AI(int depth = 4, int batch = 1024) : max_depth(depth), batch_size(batch) {
        // 分配设备内存
        cudaMalloc(&d_nodes, batch_size * sizeof(ExpectimaxNode));
        h_nodes = new ExpectimaxNode[batch_size];
        
        std::cout << "CUDA 2048 AI 初始化完成 - 最大深度: " << max_depth 
                  << ", 批次大小: " << batch_size << std::endl;
    }
    
    ~CUDA2048AI() {
        cudaFree(d_nodes);
        delete[] h_nodes;
    }
    
    // 获取最佳移动[1,6](@ref)
    int get_best_move(const GameState& current_state) {
        std::vector<GameState> possible_states;
        
        // 生成所有可能的移动
        for (int move = 0; move < 4; move++) {
            GameState new_state = current_state;
            if (move_direction(new_state, move)) {
                possible_states.push_back(new_state);
            }
        }
        
        if (possible_states.empty()) return -1;
        
        // 准备设备数据
        int node_count = possible_states.size();
        for (int i = 0; i < node_count; i++) {
            h_nodes[i].state = possible_states[i];
            h_nodes[i].depth = max_depth;
        }
        
        cudaMemcpy(d_nodes, h_nodes, node_count * sizeof(ExpectimaxNode), 
                   cudaMemcpyHostToDevice);
        
        // 启动CUDA核函数
        int block_size = 256;
        int grid_size = (node_count + block_size - 1) / block_size;
        
        expectimax_search_kernel<<<grid_size, block_size>>>(d_nodes, node_count, max_depth);
        cudaDeviceSynchronize();
        
        // 拷贝结果回主机
        cudaMemcpy(h_nodes, d_nodes, node_count * sizeof(ExpectimaxNode), 
                   cudaMemcpyDeviceToHost);
        
        // 选择最佳移动
        float best_value = -FLT_MAX;
        int best_index = -1;
        
        for (int i = 0; i < node_count; i++) {
            if (h_nodes[i].expected_value > best_value) {
                best_value = h_nodes[i].expected_value;
                best_index = i;
            }
        }
        
        // 映射回原始移动方向
        if (best_index != -1) {
            int move_index = 0;
            for (int move = 0; move < 4; move++) {
                GameState test_state = current_state;
                if (move_direction(test_state, move)) {
                    if (move_index == best_index) {
                        return move;
                    }
                    move_index++;
                }
            }
        }
        
        return 0; // 默认返回上移
    }
    
private:
    // CPU版本的移动函数（用于验证和备用）
    bool move_direction(GameState& state, int direction) {
        // 实现类似于设备版本的移动逻辑
        // 这里简化实现，实际需要完整实现
        return move_left(state); // 简化示例
    }
    
    bool move_left(GameState& state) {
        bool moved = false;
        // 实现左移逻辑[4](@ref)
        for (int i = 0; i < BOARD_SIZE; i++) {
            // 实现具体的移动和合并逻辑
        }
        return moved;
    }
};
