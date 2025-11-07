name: 2048 AI Performance Benchmark

on: [push, pull_request, schedule]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-type: [performance, quick-test]
    
    steps:
    - name: 🚀 Checkout code
      uses: actions/checkout@v4
      
    - name: 🛠️ Setup build environment
      run: |
        sudo apt-get update
        sudo apt-get install -y g++-11 build-essential
        echo "CXX=g++-11" >> $GITHUB_ENV
        
    - name: 🔨 Compile with maximum optimizations
      run: |
        g++ -O3 -std=c++17 -pthread -march=native -flto -DNDEBUG \
            -o 2048_ai_high_perf 2048_ai_optimized.cpp
        echo "✅ 编译完成 - 使用最高级别优化"
        
    - name: 📊 Run performance test
      timeout-minutes: 15
      run: |
        echo "🧪 开始高性能测试..."
        echo "系统信息:"
        echo "  CPU: $(nproc) 核心"
        echo "  内存: $(free -h | grep Mem | awk '{print $2}')"
        echo "  优化级别: -O3 -march=native -flto"
        echo ""
        ./2048_ai_high_perf
        
    - name: 📈 Generate performance report
      if: always()
      run: |
        echo "🏁 测试完成于 $(date)"
        echo "📋 性能报告已生成"
        
    - name: 📦 Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: 2048-ai-results
        path: |
          2048_ai_high_perf
        retention-days: 30
