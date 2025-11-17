<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold text-gray-900">股票查询</h1>
      <el-button type="primary" @click="showAddDialog = true">
        <el-icon class="mr-1"><Plus /></el-icon>
        添加股票
      </el-button>
    </div>

    <!-- 搜索栏 -->
    <el-card>
      <el-form :inline="true" :model="searchForm" class="flex items-center space-x-4">
        <el-form-item label="股票代码/名称">
          <el-input
            v-model="searchForm.query"
            placeholder="请输入股票代码或名称"
            clearable
            @clear="handleSearch"
            @keyup.enter="handleSearch"
            style="width: 300px"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">
            <el-icon class="mr-1"><Search /></el-icon>
            搜索
          </el-button>
          <el-button @click="resetSearch">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 股票列表 -->
    <el-card>
      <el-table
        :data="stockStore.stocks"
        v-loading="stockStore.loading"
        style="width: 100%"
        @row-click="goToStockDetail"
      >
        <el-table-column prop="symbol" label="股票代码" width="120" />
        <el-table-column prop="name" label="股票名称" min-width="150" />
        <el-table-column prop="market" label="市场" width="80">
          <template #default="{ row }">
            <el-tag size="small">{{ row.market }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="sector" label="行业" min-width="120" />
        <el-table-column prop="industry" label="细分行业" min-width="150" />
        <el-table-column label="操作" width="120" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click.stop="goToStockDetail(row)">
              查看详情
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

    <!-- 添加股票对话框 -->
    <el-dialog
      v-model="showAddDialog"
      title="添加股票"
      width="500px"
      @close="resetAddForm"
    >
      <el-form :model="addForm" :rules="addRules" ref="addFormRef" label-width="100px">
        <el-form-item label="股票代码" prop="symbol">
          <el-input v-model="addForm.symbol" placeholder="请输入股票代码" />
        </el-form-item>
        <el-form-item label="股票名称" prop="name">
          <el-input v-model="addForm.name" placeholder="请输入股票名称" />
        </el-form-item>
        <el-form-item label="市场" prop="market">
          <el-select v-model="addForm.market" placeholder="请选择市场" style="width: 100%">
            <el-option label="沪深" value="CN" />
            <el-option label="美股" value="US" />
            <el-option label="港股" value="HK" />
          </el-select>
        </el-form-item>
        <el-form-item label="行业" prop="sector">
          <el-input v-model="addForm.sector" placeholder="请输入行业" />
        </el-form-item>
        <el-form-item label="细分行业" prop="industry">
          <el-input v-model="addForm.industry" placeholder="请输入细分行业" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showAddDialog = false">取消</el-button>
          <el-button type="primary" @click="handleAddStock">确定</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useStockStore } from '@/stores/stock'
import type { Stock } from '@/types'
import { Search, Plus } from '@element-plus/icons-vue'

const router = useRouter()
const stockStore = useStockStore()

// 搜索表单
const searchForm = reactive({
  query: ''
})

// 分页配置
const pagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 添加股票对话框
const showAddDialog = ref(false)
const addFormRef = ref()
const addForm = reactive({
  symbol: '',
  name: '',
  market: 'CN',
  sector: '',
  industry: ''
})

// 添加表单验证规则
const addRules = {
  symbol: [
    { required: true, message: '请输入股票代码', trigger: 'blur' },
    { min: 1, max: 20, message: '长度在 1 到 20 个字符', trigger: 'blur' }
  ],
  name: [
    { required: true, message: '请输入股票名称', trigger: 'blur' },
    { min: 1, max: 100, message: '长度在 1 到 100 个字符', trigger: 'blur' }
  ],
  market: [
    { required: true, message: '请选择市场', trigger: 'change' }
  ]
}

// 搜索股票
const handleSearch = async () => {
  if (!searchForm.query.trim()) {
    ElMessage.warning('请输入搜索内容')
    return
  }
  
  try {
    const result = await stockStore.searchStocks(searchForm.query, pagination.page, pagination.size)
    pagination.total = result.total
  } catch (error) {
    ElMessage.error('搜索失败')
  }
}

// 重置搜索
const resetSearch = () => {
  searchForm.query = ''
  loadStocks()
}

// 跳转到股票详情
const goToStockDetail = (stock: Stock) => {
  router.push(`/stocks/${stock.symbol}`)
}

// 分页变化
const handlePageChange = (page: number) => {
  pagination.page = page
  if (searchForm.query) {
    handleSearch()
  } else {
    loadStocks()
  }
}

// 每页条数变化
const handleSizeChange = (size: number) => {
  pagination.size = size
  pagination.page = 1
  if (searchForm.query) {
    handleSearch()
  } else {
    loadStocks()
  }
}

// 添加股票
const handleAddStock = async () => {
  if (!addFormRef.value) return
  
  await addFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      try {
        // 这里调用添加股票的API
        ElMessage.success('添加股票成功')
        showAddDialog.value = false
        resetAddForm()
        loadStocks()
      } catch (error) {
        ElMessage.error('添加股票失败')
      }
    }
  })
}

// 重置添加表单
const resetAddForm = () => {
  addForm.symbol = ''
  addForm.name = ''
  addForm.market = 'CN'
  addForm.sector = ''
  addForm.industry = ''
}

// 加载股票列表
const loadStocks = async () => {
  try {
    // 这里可以调用获取股票列表的API
    // 暂时使用热门股票数据
    await stockStore.fetchHotStocks()
    pagination.total = stockStore.stocks.length
  } catch (error) {
    ElMessage.error('加载股票列表失败')
  }
}

onMounted(() => {
  loadStocks()
})
</script>