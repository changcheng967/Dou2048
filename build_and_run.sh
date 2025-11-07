#!/bin/bash
echo "🔨 编译高性能2048 AI..."

# 使用最高级别优化
g++ -O3 -std=c++17 -pthread -march=native -flto -DNDEBUG \
    -o 2048_ai_high_perf 2048_ai_optimized.cpp

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo "📊 二进制大小: $(stat -c%s 2048_ai_high_perf) bytes"
    echo "🚀 开始运行AI..."
    echo ""
    ./2048_ai_high_perf
else
    echo "❌ 编译失败！"
    exit 1
fi
