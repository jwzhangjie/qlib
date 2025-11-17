<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-2xl font-bold text-gray-900">欢迎使用Qlib量化平台</h1>
      <p class="mt-2 text-gray-600">专业的量化投资分析工具，助您做出更明智的投资决策</p>
    </div>

    <!-- 统计卡片 -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <el-card class="hover:shadow-lg transition-shadow duration-300">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <el-icon class="h-8 w-8 text-blue-500" size="32">
              <TrendCharts />
            </el-icon>
          </div>
          <div class="ml-4">
            <p class="text-sm font-medium text-gray-500">股票数量</p>
            <p class="text-2xl font-semibold text-gray-900">{{ stats.stockCount }}</p>
          </div>
        </div>
      </el-card>

      <el-card class="hover:shadow-lg transition-shadow duration-300">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <el-icon class="h-8 w-8 text-green-500" size="32">
              <DataAnalysis />
            </el-icon>
          </div>
          <div class="ml-4">
            <p class="text-sm font-medium text-gray-500">训练模型</p>
            <p class="text-2xl font-semibold text-gray-900">{{ stats.modelCount }}</p>
          </div>
        </div>
      </el-card>

      <el-card class="hover:shadow-lg transition-shadow duration-300">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <el-icon class="h-8 w-8 text-purple-500" size="32">
              <TrendData />
            </el-icon>
          </div>
          <div class="ml-4">
            <p class="text-sm font-medium text-gray-500">回测次数</p>
            <p class="text-2xl font-semibold text-gray-900">{{ stats.backtestCount }}</p>
          </div>
        </div>
      </el-card>

      <el-card class="hover:shadow-lg transition-shadow duration-300">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <el-icon class="h-8 w-8 text-orange-500" size="32">
              <Clock />
            </el-icon>
          </div>
          <div class="ml-4">
            <p class="text-sm font-medium text-gray-500">今日收益</p>
            <p class="text-2xl font-semibold text-gray-900">{{ stats.todayReturn }}%</p>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 主要内容区域 -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- 热门股票 -->
      <el-card>
        <template #header>
          <div class="flex items-center justify-between">
            <span class="text-lg font-medium">热门股票</span>
            <el-button type="primary" link @click="$router.push('/stocks')">
              查看更多
            </el-button>
          </div>
        </template>
        <div class="space-y-4">
          <div
            v-for="stock in hotStocks"
            :key="stock.symbol"
            class="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
            @click="goToStock(stock.symbol)"
          >
            <div>
              <p class="font-medium text-gray-900">{{ stock.symbol }}</p>
              <p class="text-sm text-gray-500">{{ stock.name }}</p>
            </div>
            <div class="text-right">
              <p class="font-medium" :class="stock.change > 0 ? 'text-green-600' : 'text-red-600'">
                {{ stock.price }}
              </p>
              <p class="text-sm" :class="stock.change > 0 ? 'text-green-600' : 'text-red-600'">
                {{ stock.change > 0 ? '+' : '' }}{{ stock.change }}%
              </p>
            </div>
          </div>
        </div>
      </el-card>

      <!-- 最近回测 -->
      <el-card>
        <template #header>
          <div class="flex items-center justify-between">
            <span class="text-lg font-medium">最近回测</span>
            <el-button type="primary" link @click="$router.push('/backtest')">
              查看更多
            </el-button>
          </div>
        </template>
        <div class="space-y-4">
          <div
            v-for="backtest in recentBacktests"
            :key="backtest.id"
            class="p-3 bg-gray-50 rounded-lg"
          >
            <div class="flex items-center justify-between mb-2">
              <span class="font-medium text-gray-900">{{ backtest.name }}</span>
              <el-tag :type="backtest.status === 'completed' ? 'success' : 'info'" size="small">
                {{ backtest.status }}
              </el-tag>
            </div>
            <div class="text-sm text-gray-600">
              <p>收益: <span :class="backtest.totalReturn > 0 ? 'text-green-600' : 'text-red-600'">
                {{ backtest.totalReturn > 0 ? '+' : '' }}{{ backtest.totalReturn }}%
              </span></p>
              <p>夏普比率: {{ backtest.sharpeRatio }}</p>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 快速操作 -->
    <el-card>
      <template #header>
        <span class="text-lg font-medium">快速操作</span>
      </template>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <el-button
          type="primary"
          plain
          class="h-20"
          @click="$router.push('/stocks')"
        >
          <div class="flex flex-col items-center space-y-2">
            <el-icon size="24"><Search /></el-icon>
            <span>股票查询</span>
          </div>
        </el-button>

        <el-button
          type="success"
          plain
          class="h-20"
          @click="$router.push('/models')"
        >
          <div class="flex flex-col items-center space-y-2">
            <el-icon size="24"><TrendData /></el-icon>
            <span>模型训练</span>
          </div>
        </el-button>

        <el-button
          type="warning"
          plain
          class="h-20"
          @click="$router.push('/backtest')"
        >
          <div class="flex flex-col items-center space-y-2">
            <el-icon size="24"><DataAnalysis /></el-icon>
            <span>策略回测</span>
          </div>
        </el-button>

        <el-button
          type="info"
          plain
          class="h-20"
          @click="$router.push('/data')"
        >
          <div class="flex flex-col items-center space-y-2">
            <el-icon size="24"><Document /></el-icon>
            <span>数据管理</span>
          </div>
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useStockStore } from '@/stores/stock'
import {
  TrendCharts,
  DataAnalysis,
  TrendData,
  Clock,
  Search,
  Document
} from '@element-plus/icons-vue'

const router = useRouter()
const stockStore = useStockStore()

// 统计数据
const stats = ref({
  stockCount: 0,
  modelCount: 0,
  backtestCount: 0,
  todayReturn: 0
})

// 热门股票
const hotStocks = ref([
  { symbol: 'AAPL', name: '苹果公司', price: 150.25, change: 2.5 },
  { symbol: 'GOOGL', name: '谷歌', price: 2800.50, change: -1.2 },
  { symbol: 'MSFT', name: '微软', price: 305.75, change: 3.8 },
  { symbol: 'TSLA', name: '特斯拉', price: 850.20, change: -2.1 }
])

// 最近回测
const recentBacktests = ref([
  { id: 1, name: 'LSTM策略回测', status: 'completed', totalReturn: 15.2, sharpeRatio: 1.8 },
  { id: 2, name: 'XGBoost策略回测', status: 'running', totalReturn: 0, sharpeRatio: 0 },
  { id: 3, name: 'Transformer策略回测', status: 'completed', totalReturn: -3.5, sharpeRatio: 0.9 }
])

// 跳转到股票详情
const goToStock = (symbol: string) => {
  router.push(`/stocks/${symbol}`)
}

// 加载数据
const loadData = async () => {
  try {
    // 这里可以调用实际的API获取数据
    stats.value = {
      stockCount: 5000,
      modelCount: 25,
      backtestCount: 150,
      todayReturn: 2.3
    }
  } catch (error) {
    ElMessage.error('加载数据失败')
  }
}

onMounted(() => {
  loadData()
})
</script>