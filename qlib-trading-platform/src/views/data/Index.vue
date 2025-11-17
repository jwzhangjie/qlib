<template>
  <div class="data-management">
    <!-- 页面标题 -->
    <div class="mb-6">
      <h1 class="text-2xl font-bold text-gray-900">数据管理</h1>
      <p class="text-gray-600">管理股票数据、更新数据源、查看数据状态</p>
    </div>

    <!-- 统计卡片 -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
      <el-card class="stat-card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">股票总数</p>
            <p class="text-2xl font-bold text-gray-900">{{ stats.totalStocks }}</p>
          </div>
          <el-icon class="text-blue-500" size="32">
            <Coin />
          </el-icon>
        </div>
      </el-card>
      
      <el-card class="stat-card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">数据更新任务</p>
            <p class="text-2xl font-bold text-gray-900">{{ stats.updateTasks }}</p>
          </div>
          <el-icon class="text-green-500" size="32">
            <Refresh />
          </el-icon>
        </div>
      </el-card>
      
      <el-card class="stat-card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">最后更新</p>
            <p class="text-2xl font-bold text-gray-900">{{ stats.lastUpdate }}</p>
          </div>
          <el-icon class="text-orange-500" size="32">
            <Clock />
          </el-icon>
        </div>
      </el-card>
      
      <el-card class="stat-card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">数据源状态</p>
            <p class="text-2xl font-bold text-green-600">{{ stats.dataSourceStatus }}</p>
          </div>
          <el-icon class="text-purple-500" size="32">
            <Connection />
          </el-icon>
        </div>
      </el-card>
    </div>

    <!-- 操作按钮 -->
    <div class="mb-6 flex justify-between items-center">
      <div class="flex space-x-4">
        <el-button type="primary" @click="showUpdateDialog = true">
          <el-icon class="mr-1"><Refresh /></el-icon>
          更新数据
        </el-button>
        <el-button @click="showImportDialog = true">
          <el-icon class="mr-1"><Upload /></el-icon>
          导入数据
        </el-button>
        <el-button @click="exportData">
          <el-icon class="mr-1"><Download /></el-icon>
          导出数据
        </el-button>
      </div>
      
      <div class="flex items-center space-x-2">
        <el-input
          v-model="searchQuery"
          placeholder="搜索股票代码或名称"
          style="width: 250px"
          clearable
          @clear="handleSearch"
          @keyup.enter="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        <el-button @click="handleSearch">搜索</el-button>
      </div>
    </div>

    <!-- 数据表格 -->
    <el-card>
      <template #header>
        <div class="flex justify-between items-center">
          <span>数据列表</span>
          <el-button type="text" @click="refreshData">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>
      
      <el-table
        :data="tableData"
        v-loading="loading"
        style="width: 100%"
        @selection-change="handleSelectionChange"
      >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="symbol" label="股票代码" sortable />
        <el-table-column prop="name" label="股票名称" />
        <el-table-column prop="market" label="市场" />
        <el-table-column prop="dataRange" label="数据范围" />
        <el-table-column prop="lastUpdate" label="最后更新" sortable />
        <el-table-column prop="dataQuality" label="数据质量">
          <template #default="{ row }">
            <el-tag :type="getQualityType(row.dataQuality)">
              {{ row.dataQuality }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="{ row }">
            <el-button type="text" size="small" @click="viewData(row)">
              查看
            </el-button>
            <el-button type="text" size="small" @click="updateStockData(row)">
              更新
            </el-button>
            <el-button type="text" size="small" @click="deleteData(row)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <div class="mt-4 flex justify-between items-center">
        <div class="flex items-center space-x-2">
          <el-button 
            type="danger" 
            size="small" 
            :disabled="selectedRows.length === 0"
            @click="batchDelete"
          >
            批量删除
          </el-button>
          <span class="text-sm text-gray-600">
            已选择 {{ selectedRows.length }} 项
          </span>
        </div>
        
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 更新数据对话框 -->
    <el-dialog
      v-model="showUpdateDialog"
      title="更新数据"
      width="600px"
    >
      <el-form :model="updateForm" label-width="100px">
        <el-form-item label="更新范围">
          <el-radio-group v-model="updateForm.scope">
            <el-radio label="all">全部股票</el-radio>
            <el-radio label="selected">选中股票</el-radio>
            <el-radio label="custom">自定义列表</el-radio>
          </el-radio-group>
        </el-form-item>
        
        <el-form-item label="数据类型" v-if="updateForm.scope !== 'selected'">
          <el-checkbox-group v-model="updateForm.dataTypes">
            <el-checkbox label="price">股价数据</el-checkbox>
            <el-checkbox label="volume">成交量数据</el-checkbox>
            <el-checkbox label="financial">财务数据</el-checkbox>
            <el-checkbox label="news">新闻数据</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="updateForm.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            style="width: 100%"
          />
        </el-form-item>
        
        <el-form-item label="更新频率">
          <el-select v-model="updateForm.frequency" style="width: 100%">
            <el-option label="日频" value="daily" />
            <el-option label="分钟频" value="minute" />
            <el-option label=" tick数据" value="tick" />
          </el-select>
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showUpdateDialog = false">取消</el-button>
        <el-button type="primary" @click="confirmUpdate" :loading="updateLoading">
          开始更新
        </el-button>
      </template>
    </el-dialog>

    <!-- 导入数据对话框 -->
    <el-dialog
      v-model="showImportDialog"
      title="导入数据"
      width="500px"
    >
      <el-upload
        class="upload-demo"
        drag
        action="/api/data/import"
        :headers="uploadHeaders"
        :before-upload="beforeUpload"
        :on-success="handleUploadSuccess"
        :on-error="handleUploadError"
        accept=".csv,.xlsx,.json"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽文件到此处或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 CSV, Excel, JSON 格式，文件大小不超过 10MB
          </div>
        </template>
      </el-upload>
    </el-dialog>

    <!-- 数据详情对话框 -->
    <el-dialog
      v-model="showDetailDialog"
      title="数据详情"
      width="800px"
    >
      <div v-if="currentData">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="股票代码">{{ currentData.symbol }}</el-descriptions-item>
          <el-descriptions-item label="股票名称">{{ currentData.name }}</el-descriptions-item>
          <el-descriptions-item label="市场">{{ currentData.market }}</el-descriptions-item>
          <el-descriptions-item label="数据范围">{{ currentData.dataRange }}</el-descriptions-item>
          <el-descriptions-item label="最后更新">{{ currentData.lastUpdate }}</el-descriptions-item>
          <el-descriptions-item label="数据质量">{{ currentData.dataQuality }}</el-descriptions-item>
          <el-descriptions-item label="状态">{{ currentData.status }}</el-descriptions-item>
          <el-descriptions-item label="记录数">{{ currentData.recordCount }}</el-descriptions-item>
        </el-descriptions>
        
        <div class="mt-4">
          <h4 class="font-medium mb-2">最近数据预览</h4>
          <el-table :data="previewData" style="width: 100%" max-height="300">
            <el-table-column prop="date" label="日期" />
            <el-table-column prop="open" label="开盘价" />
            <el-table-column prop="high" label="最高价" />
            <el-table-column prop="low" label="最低价" />
            <el-table-column prop="close" label="收盘价" />
            <el-table-column prop="volume" label="成交量" />
          </el-table>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Coin,
  Refresh,
  Clock,
  Connection,
  Search,
  Upload,
  Download,
  UploadFilled
} from '@element-plus/icons-vue'
import { dataAPI } from '@/api/data'

// 统计数据
const stats = reactive({
  totalStocks: 3856,
  updateTasks: 12,
  lastUpdate: '2024-01-15',
  dataSourceStatus: '正常'
})

// 搜索和分页
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(100)
const loading = ref(false)

// 表格数据
const tableData = ref([
  {
    symbol: '000001.SZ',
    name: '平安银行',
    market: '深交所',
    dataRange: '2015-01-01 至 2024-01-15',
    lastUpdate: '2024-01-15 09:30:00',
    dataQuality: '优秀',
    status: '正常',
    recordCount: 2184
  },
  {
    symbol: '000002.SZ',
    name: '万科A',
    market: '深交所',
    dataRange: '2015-01-01 至 2024-01-15',
    lastUpdate: '2024-01-15 09:30:00',
    dataQuality: '良好',
    status: '正常',
    recordCount: 2184
  },
  {
    symbol: '600000.SH',
    name: '浦发银行',
    market: '上交所',
    dataRange: '2015-01-01 至 2024-01-15',
    lastUpdate: '2024-01-14 15:00:00',
    dataQuality: '优秀',
    status: '正常',
    recordCount: 2184
  }
])

const selectedRows = ref([])
const currentData = ref(null)
const previewData = ref([])

// 对话框状态
const showUpdateDialog = ref(false)
const showImportDialog = ref(false)
const showDetailDialog = ref(false)
const updateLoading = ref(false)

// 表单数据
const updateForm = reactive({
  scope: 'all',
  dataTypes: ['price', 'volume'],
  dateRange: [],
  frequency: 'daily'
})

// 上传头部
const uploadHeaders = reactive({
  Authorization: `Bearer ${localStorage.getItem('token') || ''}`
})

// 获取质量标签类型
const getQualityType = (quality: string) => {
  switch (quality) {
    case '优秀': return 'success'
    case '良好': return 'info'
    case '一般': return 'warning'
    case '较差': return 'danger'
    default: return 'info'
  }
}

// 获取状态标签类型
const getStatusType = (status: string) => {
  switch (status) {
    case '正常': return 'success'
    case '更新中': return 'warning'
    case '异常': return 'danger'
    default: return 'info'
  }
}

// 搜索
const handleSearch = async () => {
  loading.value = true
  try {
    // 模拟搜索逻辑
    await new Promise(resolve => setTimeout(resolve, 500))
    ElMessage.success('搜索完成')
  } catch (error) {
    ElMessage.error('搜索失败')
  } finally {
    loading.value = false
  }
}

// 刷新数据
const refreshData = async () => {
  loading.value = true
  try {
    // 模拟刷新逻辑
    await new Promise(resolve => setTimeout(resolve, 1000))
    ElMessage.success('数据已刷新')
  } catch (error) {
    ElMessage.error('刷新失败')
  } finally {
    loading.value = false
  }
}

// 选择变化
const handleSelectionChange = (selection: any[]) => {
  selectedRows.value = selection
}

// 分页变化
const handleSizeChange = (val: number) => {
  pageSize.value = val
  refreshData()
}

const handleCurrentChange = (val: number) => {
  currentPage.value = val
  refreshData()
}

// 查看数据详情
const viewData = (row: any) => {
  currentData.value = row
  previewData.value = [
    { date: '2024-01-15', open: 10.25, high: 10.45, low: 10.15, close: 10.35, volume: 125600 },
    { date: '2024-01-14', open: 10.15, high: 10.30, low: 10.10, close: 10.25, volume: 98500 },
    { date: '2024-01-13', open: 10.20, high: 10.25, low: 10.05, close: 10.15, volume: 112300 }
  ]
  showDetailDialog.value = true
}

// 更新股票数据
const updateStockData = async (row: any) => {
  try {
    await ElMessageBox.confirm(`确定要更新 ${row.symbol} 的数据吗？`, '确认', {
      type: 'warning'
    })
    
    loading.value = true
    // 模拟更新逻辑
    await new Promise(resolve => setTimeout(resolve, 2000))
    ElMessage.success('数据更新成功')
    refreshData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('更新失败')
    }
  } finally {
    loading.value = false
  }
}

// 删除数据
const deleteData = async (row: any) => {
  try {
    await ElMessageBox.confirm(`确定要删除 ${row.symbol} 的数据吗？`, '确认删除', {
      type: 'warning'
    })
    
    loading.value = true
    // 模拟删除逻辑
    await new Promise(resolve => setTimeout(resolve, 1000))
    ElMessage.success('删除成功')
    refreshData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  } finally {
    loading.value = false
  }
}

// 批量删除
const batchDelete = async () => {
  if (selectedRows.value.length === 0) return
  
  try {
    await ElMessageBox.confirm(`确定要删除选中的 ${selectedRows.value.length} 项数据吗？`, '确认删除', {
      type: 'warning'
    })
    
    loading.value = true
    // 模拟批量删除逻辑
    await new Promise(resolve => setTimeout(resolve, 1500))
    ElMessage.success('批量删除成功')
    selectedRows.value = []
    refreshData()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('批量删除失败')
    }
  } finally {
    loading.value = false
  }
}

// 导出数据
const exportData = async () => {
  try {
    loading.value = true
    // 模拟导出逻辑
    await new Promise(resolve => setTimeout(resolve, 1500))
    ElMessage.success('数据导出成功')
  } catch (error) {
    ElMessage.error('导出失败')
  } finally {
    loading.value = false
  }
}

// 确认更新
const confirmUpdate = async () => {
  updateLoading.value = true
  try {
    // 模拟更新逻辑
    await new Promise(resolve => setTimeout(resolve, 3000))
    ElMessage.success('数据更新任务已提交')
    showUpdateDialog.value = false
    refreshData()
  } catch (error) {
    ElMessage.error('更新失败')
  } finally {
    updateLoading.value = false
  }
}

// 文件上传处理
const beforeUpload = (file: File) => {
  const isLt10M = file.size / 1024 / 1024 < 10
  if (!isLt10M) {
    ElMessage.error('文件大小不能超过 10MB')
  }
  return isLt10M
}

const handleUploadSuccess = (response: any) => {
  ElMessage.success('文件上传成功')
  showImportDialog.value = false
  refreshData()
}

const handleUploadError = (error: any) => {
  ElMessage.error('文件上传失败')
}

// 初始化
onMounted(() => {
  refreshData()
})
</script>

<style scoped>
.data-management {
  @apply p-6;
}

.stat-card {
  @apply hover:shadow-md transition-shadow;
}

.upload-demo {
  @apply w-full;
}
</style>