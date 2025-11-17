import request from '@/utils/request'
import type { ApiResponse } from '@/types'

export interface DataUpdateTask {
  id: string
  symbol: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  startTime: string
  endTime?: string
  message?: string
  dataType: 'price' | 'volume' | 'financial' | 'news'
  dateRange: [string, string]
}

export interface DataImportResult {
  success: boolean
  importedCount: number
  failedCount: number
  errors: string[]
  warnings: string[]
}

export interface DataQualityReport {
  symbol: string
  name: string
  totalRecords: number
  missingRecords: number
  dataQuality: 'excellent' | 'good' | 'fair' | 'poor'
  issues: string[]
  suggestions: string[]
}

export const dataAPI = {
  // 获取数据列表
  getDataList: (params: {
    page: number
    pageSize: number
    search?: string
    market?: string
    status?: string
  }) => {
    return request.get<ApiResponse<any>>('/api/data/list', { params })
  },

  // 获取数据统计
  getDataStats: () => {
    return request.get<ApiResponse<{
      totalStocks: number
      updateTasks: number
      lastUpdate: string
      dataSourceStatus: string
    }>>('/api/data/stats')
  },

  // 更新数据
  updateData: (data: {
    scope: 'all' | 'selected' | 'custom'
    symbols?: string[]
    dataTypes: string[]
    dateRange: [string, string]
    frequency: 'daily' | 'minute' | 'tick'
  }) => {
    return request.post<ApiResponse<DataUpdateTask>>('/api/data/update', data)
  },

  // 导入数据
  importData: (file: File, options: {
    format: 'csv' | 'excel' | 'json'
    dataType: string
    dateColumn?: string
    symbolColumn?: string
  }) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('format', options.format)
    formData.append('dataType', options.dataType)
    if (options.dateColumn) {
      formData.append('dateColumn', options.dateColumn)
    }
    if (options.symbolColumn) {
      formData.append('symbolColumn', options.symbolColumn)
    }
    
    return request.post<ApiResponse<DataImportResult>>('/api/data/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  // 导出数据
  exportData: (params: {
    symbols?: string[]
    dataTypes: string[]
    dateRange: [string, string]
    format: 'csv' | 'excel' | 'json'
  }) => {
    return request.post('/api/data/export', params, {
      responseType: 'blob'
    })
  },

  // 获取数据详情
  getDataDetail: (symbol: string) => {
    return request.get<ApiResponse<any>>(`/api/data/detail/${symbol}`)
  },

  // 删除数据
  deleteData: (symbols: string[]) => {
    return request.delete<ApiResponse<void>>('/api/data/delete', {
      data: { symbols }
    })
  },

  // 获取数据质量报告
  getDataQuality: (symbol: string) => {
    return request.get<ApiResponse<DataQualityReport>>(`/api/data/quality/${symbol}`)
  },

  // 获取更新任务列表
  getUpdateTasks: (params: {
    page: number
    pageSize: number
    status?: string
  }) => {
    return request.get<ApiResponse<DataUpdateTask[]>>(`/api/data/tasks`, { params })
  },

  // 获取更新任务详情
  getUpdateTaskDetail: (taskId: string) => {
    return request.get<ApiResponse<DataUpdateTask>>(`/api/data/tasks/${taskId}`)
  },

  // 取消更新任务
  cancelUpdateTask: (taskId: string) => {
    return request.post<ApiResponse<void>>(`/api/data/tasks/${taskId}/cancel`)
  },

  // 获取数据源状态
  getDataSourceStatus: () => {
    return request.get<ApiResponse<{
      status: 'connected' | 'disconnected' | 'error'
      lastSync: string
      availableSources: string[]
      activeConnections: number
    }>>('/api/data/source-status')
  },

  // 测试数据源连接
  testDataSource: (source: string) => {
    return request.post<ApiResponse<{
      success: boolean
      responseTime: number
      message: string
    }>>(`/api/data/test-connection`, { source })
  },

  // 获取数据预览
  getDataPreview: (symbol: string, limit: number = 10) => {
    return request.get<ApiResponse<any[]>>(`/api/data/preview/${symbol}`, {
      params: { limit }
    })
  },

  // 修复数据质量问题
  fixDataQuality: (symbol: string, issues: string[]) => {
    return request.post<ApiResponse<void>>(`/api/data/fix-quality/${symbol}`, { issues })
  }
}