// 2048_ai_cuda.h
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#define BOARD_SIZE 4
#define TILE_65536 16  // 2^16 = 65536

// 游戏状态结构体（CPU版本）
struct GameState {
    int board[BOARD_SIZE][BOARD_SIZE];
    int score;
    bool moved;
    
    __host__ __device__ GameState() : score(0), moved(false) {
        for (int i = 0; i < BOARD_SIZE; i++)
            for (int j = 0; j < BOARD_SIZE; j++)
                board[i][j] = 0;
    }
};

// CUDA加速的Expectimax节点
struct ExpectimaxNode {
    GameState state;
    float expected_value;
    int best_move;
    int depth;
    
    __host__ __device__ ExpectimaxNode() : expected_value(0.0f), best_move(-1), depth(0) {}
};
