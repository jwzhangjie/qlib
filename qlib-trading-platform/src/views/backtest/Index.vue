<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold text-gray-900">回测分析</h1>
      <el-button type="primary" @click="showBacktestDialog = true">
        <el-icon class="mr-1"><Plus /></el-icon>
        新建回测
      </el-button>
    </div>

    <!-- 回测列表 -->
    <el-card>
      <el-table
        :data="backtests"
        v-loading="loading"
        style="width: 100%"
      >
        <el-table-column prop="name" label="回测名称" min-width="150" />
        <el-table-column label="回测周期" width="180">
          <template #default="{ row }">
            <div class="text-sm">
              <div>{{ row.startDate }} 至</div>
              <div>{{ row.endDate }}</div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag
              :type="getStatusType(row.status)"
              size="small"
            >
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="收益" width="100">
          <template #default="{ row }">
            <span v-if="row.results" :class="row.results.totalReturn > 0 ? 'text-green-600' : 'text-red-600'">
              {{ row.results.totalReturn > 0 ? '+' : '' }}{{ row.results.totalReturn.toFixed(2) }}%
            </span>
            <span v-else class="text-gray-400">--</span>
          </template>
        </el-table-column>
        <el-table-column label="夏普比率" width="100">
          <template #default="{ row }">
            <span v-if="row.results">{{ row.results.sharpeRatio.toFixed(2) }}</span>
            <span v-else class="text-gray-400">--</span>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="创建时间" width="160">
          <template #default="{ row }">
            {{ formatDate(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="row.status === 'completed'"
              type="primary"
              link
              @click="viewResults(row)"
            >
              查看结果
            </el-button>
            <el-button
              v-if="row.status === 'running'"
              type="info"
              link
              @click="checkStatus(row)"
            >
              查看进度
            </el-button>
            <el-button
              type="danger"
              link
              @click="handleDelete(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="flex justify-between items-center mt-4">
        <div class="text-sm text-gray-500">
          共 {{ pagination.total }} 条记录
        </div>
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.size"
          :total="pagination.total"
          :page-sizes="[10, 20, 50, 100]"
          layout="sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </el-card>

    <!-- 新建回测对话框 -->
    <el-dialog
      v-model="showBacktestDialog"
      title="新建回测"
      width="600px"
      @close="resetBacktestForm"
    >
      <el-form :model="backtestForm" :rules="backtestRules" ref="backtestFormRef" label-width="120px">
        <el-form-item label="回测名称" prop="name">
          <el-input v-model="backtestForm.name" placeholder="请输入回测名称" />
        </el-form-item>
        
        <el-form-item label="策略类型" prop="strategyType">
          <el-select v-model="backtestForm.strategyType" placeholder="请选择策略类型" style="width: 100%">
            <el-option label="TopK Dropout" value="TopkDropoutStrategy" />
            <el-option label="买入持有" value="BuyAndHoldStrategy" />
            <el-option label="均值回归" value="MeanReversionStrategy" />
            <el-option label="动量策略" value="MomentumStrategy" />
          </el-select>
        </el-form-item>

        <el-form-item label="回测周期" required>
          <el-date-picker
            v-model="backtestForm.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            format="YYYY-MM-DD"
            value-format="YYYY-MM-DD"
            style="width: 100%"
          />
        </el-form-item>

        <el-form-item label="策略配置" prop="strategyConfig">
          <el-input
            v-model="backtestForm.strategyConfig"
            type="textarea"
            :rows="8"
            placeholder="请输入策略配置 (JSON格式)"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showBacktestDialog = false">取消</el-button>
          <el-button type="primary" @click="handleCreateBacktest" :loading="creating">
            开始回测
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 回测结果对话框 -->
    <el-dialog
      v-model="showResultsDialog"
      title="回测结果"
      width="800px"
      :close-on-click-modal="false"
    >
      <div v-if="currentResults" class="space-y-6">
        <!-- 关键指标 -->
        <el-row :gutter="20">
          <el-col :span="6">
            <div class="text-center p-4 bg-gray-50 rounded-lg">
              <div class="text-2xl font-bold" :class="currentResults.totalReturn > 0 ? 'text-green-600' : 'text-red-600'">
                {{ currentResults.totalReturn > 0 ? '+' : '' }}{{ currentResults.totalReturn.toFixed(2) }}%
              </div>
              <div class="text-sm text-gray-600">总收益</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="text-center p-4 bg-gray-50 rounded-lg">
              <div class="text-2xl font-bold text-blue-600">
                {{ currentResults.annualizedReturn.toFixed(2) }}%
              </div>
              <div class="text-sm text-gray-600">年化收益</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="text-center p-4 bg-gray-50 rounded-lg">
              <div class="text-2xl font-bold text-red-600">
                {{ currentResults.maxDrawdown.toFixed(2) }}%
              </div>
              <div class="text-sm text-gray-600">最大回撤</div>
            </div>
          </el-col>
          <el-col :span="6">
            <div class="text-center p-4 bg-gray-50 rounded-lg">
              <div class="text-2xl font-bold text-purple-600">
                {{ currentResults.sharpeRatio.toFixed(2) }}
              </div>
              <div class="text-sm text-gray-600">夏普比率</div>
            </div>
          </el-col>
        </el-row>

        <!-- 收益曲线图 -->
        <div>
          <h3 class="text-lg font-medium mb-4">收益曲线</h3>
          <div class="h-64 bg-gray-100 rounded-lg flex items-center justify-center">
            <span class="text-gray-500">收益曲线图 (需要集成图表库)</span>
          </div>
        </div>

        <!-- 交易记录 -->
        <div>
          <h3 class="text-lg font-medium mb-4">交易记录</h3>
          <el-table :data="currentResults.trades" max-height="300">
            <el-table-column prop="date" label="日期" width="120" />
            <el-table-column prop="symbol" label="股票" width="100" />
            <el-table-column prop="action" label="操作" width="80">
              <template #default="{ row }">
                <el-tag :type="row.action === 'buy' ? 'success' : 'danger'" size="small">
                  {{ row.action === 'buy' ? '买入' : '卖出' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="price" label="价格" width="100" />
            <el-table-column prop="quantity" label="数量" width="80" />
            <el-table-column prop="profit" label="收益" width="100">
              <template #default="{ row }">
                <span :class="row.profit > 0 ? 'text-green-600' : 'text-red-600'">
                  {{ row.profit > 0 ? '+' : '' }}{{ row.profit.toFixed(2) }}
                </span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { Backtest, BacktestResults } from '@/types'
import { backtestAPI } from '@/api/backtest'
import { Plus } from '@element-plus/icons-vue'

// 状态
const backtests = ref<Backtest[]>([])
const loading = ref(false)
const creating = ref(false)

// 分页配置
const pagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 对话框
const showBacktestDialog = ref(false)
const showResultsDialog = ref(false)
const backtestFormRef = ref()

// 表单数据
const backtestForm = reactive({
  name: '',
  strategyType: 'TopkDropoutStrategy',
  dateRange: [] as string[],
  strategyConfig: JSON.stringify({
    topk: 50,
    n_drop: 5,
    risk_degree: 0.95,
    hold_thresh: 1
  }, null, 2)
})

// 当前结果
const currentResults = ref<BacktestResults | null>(null)

// 表单验证规则
const backtestRules = {
  name: [
    { required: true, message: '请输入回测名称', trigger: 'blur' }
  ],
  strategyType: [
    { required: true, message: '请选择策略类型', trigger: 'change' }
  ],
  strategyConfig: [
    { required: true, message: '请输入策略配置', trigger: 'blur' },
    {
      validator: (rule: any, value: string, callback: any) => {
        try {
          JSON.parse(value)
          callback()
        } catch {
          callback(new Error('请输入有效的JSON格式'))
        }
      },
      trigger: 'blur'
    }
  ]
}

// 获取状态类型
const getStatusType = (status: string) => {
  const typeMap: Record<string, string> = {
    'pending': 'info',
    'running': 'warning',
    'completed': 'success',
    'failed': 'danger'
  }
  return typeMap[status] || 'info'
}

// 获取状态文本
const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    'pending': '待运行',
    'running': '运行中',
    'completed': '已完成',
    'failed': '运行失败'
  }
  return textMap[status] || status
}

// 格式化日期
const formatDate = (date: string) => {
  return new Date(date).toLocaleString('zh-CN')
}

// 加载回测列表
const loadBacktests = async () => {
  loading.value = true
  try {
    const result = await backtestAPI.getBacktests(pagination.page, pagination.size)
    backtests.value = result.items
    pagination.total = result.total
  } catch (error) {
    ElMessage.error('加载回测列表失败')
  } finally {
    loading.value = false
  }
}

// 创建回测
const handleCreateBacktest = async () => {
  if (!backtestFormRef.value) return

  await backtestFormRef.value.validate(async (valid: boolean) => {
    if (valid && backtestForm.dateRange.length === 2) {
      creating.value = true
      try {
        const backtestData = {
          name: backtestForm.name,
          strategyConfig: {
            strategy_type: backtestForm.strategyType,
            ...JSON.parse(backtestForm.strategyConfig)
          },
          startDate: backtestForm.dateRange[0],
          endDate: backtestForm.dateRange[1]
        }

        await backtestAPI.createBacktest(backtestData)
        ElMessage.success('回测任务已创建')
        showBacktestDialog.value = false
        resetBacktestForm()
        loadBacktests()
      } catch (error) {
        ElMessage.error('创建回测任务失败')
      } finally {
        creating.value = false
      }
    }
  })
}

// 查看结果
const viewResults = async (backtest: Backtest) => {
  if (backtest.results) {
    currentResults.value = backtest.results
    showResultsDialog.value = true
  } else {
    try {
      const results = await backtestAPI.getBacktestResults(backtest.id)
      currentResults.value = results
      showResultsDialog.value = true
    } catch (error) {
      ElMessage.error('获取回测结果失败')
    }
  }
}

// 检查状态
const checkStatus = (backtest: Backtest) => {
  ElMessage.info('正在检查回测进度...')
  // 这里可以实现实时查看进度的逻辑
}

// 删除回测
const handleDelete = async (backtest: Backtest) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除回测 "${backtest.name}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    await backtestAPI.deleteBacktest(backtest.id)
    ElMessage.success('删除成功')
    loadBacktests()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 重置回测表单
const resetBacktestForm = () => {
  backtestForm.name = ''
  backtestForm.strategyType = 'TopkDropoutStrategy'
  backtestForm.dateRange = []
  backtestForm.strategyConfig = JSON.stringify({
    topk: 50,
    n_drop: 5,
    risk_degree: 0.95,
    hold_thresh: 1
  }, null, 2)
}

// 分页变化
const handlePageChange = (page: number) => {
  pagination.page = page
  loadBacktests()
}

// 每页条数变化
const handleSizeChange = (size: number) => {
  pagination.size = size
  pagination.page = 1
  loadBacktests()
}

onMounted(() => {
  loadBacktests()
})
</script>