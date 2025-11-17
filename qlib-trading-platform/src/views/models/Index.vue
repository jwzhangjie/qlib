<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-bold text-gray-900">模型训练</h1>
      <el-button type="primary" @click="showTrainDialog = true">
        <el-icon class="mr-1"><Plus /></el-icon>
        新建训练任务
      </el-button>
    </div>

    <!-- 模型列表 -->
    <el-card>
      <el-table
        :data="models"
        v-loading="loading"
        style="width: 100%"
      >
        <el-table-column prop="name" label="模型名称" min-width="150" />
        <el-table-column prop="modelType" label="模型类型" width="120">
          <template #default="{ row }">
            <el-tag size="small">{{ row.modelType }}</el-tag>
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
              @click="handlePredict(row)"
            >
              预测
            </el-button>
            <el-button
              v-if="row.status === 'running'"
              type="info"
              link
              @click="checkTrainingStatus(row)"
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

    <!-- 训练模型对话框 -->
    <el-dialog
      v-model="showTrainDialog"
      title="新建训练任务"
      width="600px"
      @close="resetTrainForm"
    >
      <el-form :model="trainForm" :rules="trainRules" ref="trainFormRef" label-width="120px">
        <el-form-item label="模型名称" prop="name">
          <el-input v-model="trainForm.name" placeholder="请输入模型名称" />
        </el-form-item>
        
        <el-form-item label="模型类型" prop="modelType">
          <el-select v-model="trainForm.modelType" placeholder="请选择模型类型" style="width: 100%">
            <el-option label="LSTM" value="LSTM" />
            <el-option label="GRU" value="GRU" />
            <el-option label="XGBoost" value="XGBoost" />
            <el-option label="LightGBM" value="LightGBM" />
            <el-option label="Transformer" value="Transformer" />
          </el-select>
        </el-form-item>

        <el-form-item label="训练参数" prop="config">
          <el-input
            v-model="trainForm.config"
            type="textarea"
            :rows="6"
            placeholder="请输入训练参数 (JSON格式)"
          />
        </el-form-item>

        <el-form-item label="数据集配置" prop="datasetConfig">
          <el-input
            v-model="trainForm.datasetConfig"
            type="textarea"
            :rows="4"
            placeholder="请输入数据集配置 (JSON格式)"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showTrainDialog = false">取消</el-button>
          <el-button type="primary" @click="handleTrainModel" :loading="training">
            开始训练
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 预测对话框 -->
    <el-dialog
      v-model="showPredictDialog"
      title="模型预测"
      width="500px"
    >
      <el-form :model="predictForm" ref="predictFormRef" label-width="80px">
        <el-form-item label="股票代码" prop="symbols">
          <el-select
            v-model="predictForm.symbols"
            multiple
            filterable
            allow-create
            placeholder="请输入股票代码"
            style="width: 100%"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="showPredictDialog = false">取消</el-button>
          <el-button type="primary" @click="handlePredictSubmit" :loading="predicting">
            预测
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import type { Model, ModelTrainingRequest } from '@/types'
import { modelAPI } from '@/api/model'
import { Plus } from '@element-plus/icons-vue'

const authStore = useAuthStore()

// 状态
const models = ref<Model[]>([])
const loading = ref(false)
const training = ref(false)
const predicting = ref(false)

// 分页配置
const pagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 对话框
const showTrainDialog = ref(false)
const showPredictDialog = ref(false)
const trainFormRef = ref()
const predictFormRef = ref()

// 表单数据
const trainForm = reactive({
  name: '',
  modelType: 'LSTM',
  config: JSON.stringify({
    d_feat: 20,
    d_model: 64,
    n_epochs: 200,
    lr: 0.001,
    early_stop: 10,
    batch_size: 800
  }, null, 2),
  datasetConfig: JSON.stringify({
    train: {
      start_time: "2020-01-01",
      end_time: "2021-12-31"
    },
    valid: {
      start_time: "2022-01-01", 
      end_time: "2022-06-30"
    },
    test: {
      start_time: "2022-07-01",
      end_time: "2022-12-31"
    }
  }, null, 2)
})

const predictForm = reactive({
  symbols: [] as string[],
  modelId: 0
})

// 表单验证规则
const trainRules = {
  name: [
    { required: true, message: '请输入模型名称', trigger: 'blur' }
  ],
  modelType: [
    { required: true, message: '请选择模型类型', trigger: 'change' }
  ],
  config: [
    { required: true, message: '请输入训练参数', trigger: 'blur' },
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
  ],
  datasetConfig: [
    { required: true, message: '请输入数据集配置', trigger: 'blur' },
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
    'pending': '待训练',
    'running': '训练中',
    'completed': '已完成',
    'failed': '训练失败'
  }
  return textMap[status] || status
}

// 格式化日期
const formatDate = (date: string) => {
  return new Date(date).toLocaleString('zh-CN')
}

// 加载模型列表
const loadModels = async () => {
  loading.value = true
  try {
    const result = await modelAPI.getModels(pagination.page, pagination.size)
    models.value = result.items
    pagination.total = result.total
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  } finally {
    loading.value = false
  }
}

// 训练模型
const handleTrainModel = async () => {
  if (!trainFormRef.value) return

  await trainFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      training.value = true
      try {
        const trainingRequest: ModelTrainingRequest = {
          name: trainForm.name,
          modelType: trainForm.modelType,
          config: JSON.parse(trainForm.config),
          datasetConfig: JSON.parse(trainForm.datasetConfig)
        }

        const result = await modelAPI.trainModel(trainingRequest)
        ElMessage.success('训练任务已提交，请稍后查看进度')
        showTrainDialog.value = false
        resetTrainForm()
        loadModels()
      } catch (error) {
        ElMessage.error('提交训练任务失败')
      } finally {
        training.value = false
      }
    }
  })
}

// 预测
const handlePredict = (model: Model) => {
  predictForm.modelId = model.id
  predictForm.symbols = []
  showPredictDialog.value = true
}

// 提交预测
const handlePredictSubmit = async () => {
  if (!predictFormRef.value) return

  await predictFormRef.value.validate(async (valid: boolean) => {
    if (valid) {
      predicting.value = true
      try {
        const predictions = await modelAPI.predict(predictForm.modelId, predictForm.symbols)
        ElMessage.success('预测完成')
        showPredictDialog.value = false
        // 这里可以显示预测结果
      } catch (error) {
        ElMessage.error('预测失败')
      } finally {
        predicting.value = false
      }
    }
  })
}

// 检查训练状态
const checkTrainingStatus = (model: Model) => {
  // 这里可以实现实时查看训练进度的逻辑
  ElMessage.info('正在检查训练进度...')
}

// 删除模型
const handleDelete = async (model: Model) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除模型 "${model.name}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    await modelAPI.deleteModel(model.id)
    ElMessage.success('删除成功')
    loadModels()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 重置训练表单
const resetTrainForm = () => {
  trainForm.name = ''
  trainForm.modelType = 'LSTM'
  trainForm.config = JSON.stringify({
    d_feat: 20,
    d_model: 64,
    n_epochs: 200,
    lr: 0.001,
    early_stop: 10,
    batch_size: 800
  }, null, 2)
  trainForm.datasetConfig = JSON.stringify({
    train: {
      start_time: "2020-01-01",
      end_time: "2021-12-31"
    },
    valid: {
      start_time: "2022-01-01",
      end_time: "2022-06-30"
    },
    test: {
      start_time: "2022-07-01",
      end_time: "2022-12-31"
    }
  }, null, 2)
}

// 分页变化
const handlePageChange = (page: number) => {
  pagination.page = page
  loadModels()
}

// 每页条数变化
const handleSizeChange = (size: number) => {
  pagination.size = size
  pagination.page = 1
  loadModels()
}

onMounted(() => {
  loadModels()
})
</script>