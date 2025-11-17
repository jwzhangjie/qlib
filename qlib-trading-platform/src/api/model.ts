import type { Model, ModelTrainingRequest, ModelPrediction, ApiResponse } from '@/types'
import request from '@/utils/request'

export const modelAPI = {
  // 获取模型列表
  getModels: async (page = 1, size = 20) => {
    return request.get('/models', {
      params: { page, size }
    })
  },

  // 获取模型详情
  getModel: async (modelId: number): Promise<Model> => {
    return request.get(`/models/${modelId}`)
  },

  // 训练模型
  trainModel: async (trainingRequest: ModelTrainingRequest): Promise<{ task_id: string }> => {
    return request.post('/models/train', trainingRequest)
  },

  // 获取训练状态
  getTrainingStatus: async (modelId: string) => {
    return request.get(`/models/${modelId}/status`)
  },

  // 模型预测
  predict: async (modelId: number, symbols: string[]): Promise<ModelPrediction[]> => {
    return request.get(`/models/${modelId}/predict`, {
      params: { symbols: symbols.join(',') }
    })
  },

  // 删除模型
  deleteModel: async (modelId: number): Promise<void> => {
    return request.delete(`/models/${modelId}`)
  },

  // 获取可用模型类型
  getModelTypes: async (): Promise<string[]> => {
    return request.get('/models/types')
  },

  // 获取模型配置模板
  getModelConfigTemplate: async (modelType: string) => {
    return request.get(`/models/config-template/${modelType}`)
  }
}