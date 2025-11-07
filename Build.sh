#!/bin/bash
echo "🔨 Building 2048 AI Optimized for CPU..."

# 使用优化编译标志
g++ -O3 -std=c++17 -pthread -march=native -flto -DNDEBUG \
    -o 2048_ai 2048_ai_cpu.cpp

# 检查是否编译成功
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📊 Binary size: $(stat -c%s 2048_ai) bytes"
else
    echo "❌ Build failed!"
    exit 1
fi
