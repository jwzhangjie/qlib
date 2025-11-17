import type { Backtest, BacktestResults, ApiResponse } from '@/types'
import request from '@/utils/request'

export const backtestAPI = {
  // 获取回测列表
  getBacktests: async (page = 1, size = 20) => {
    return request.get('/backtests', {
      params: { page, size }
    })
  },

  // 获取回测详情
  getBacktest: async (backtestId: number): Promise<Backtest> => {
    return request.get(`/backtests/${backtestId}`)
  },

  // 创建回测
  createBacktest: async (backtestData: {
    name: string
    strategyConfig: Record<string, any>
    startDate: string
    endDate: string
  }): Promise<{ task_id: string }> => {
    return request.post('/backtests', backtestData)
  },

  // 获取回测状态
  getBacktestStatus: async (backtestId: string) => {
    return request.get(`/backtests/${backtestId}/status`)
  },

  // 获取回测结果
  getBacktestResults: async (backtestId: number): Promise<BacktestResults> => {
    return request.get(`/backtests/${backtestId}/results`)
  },

  // 删除回测
  deleteBacktest: async (backtestId: number): Promise<void> => {
    return request.delete(`/backtests/${backtestId}`)
  },

  // 获取策略类型
  getStrategyTypes: async (): Promise<string[]> => {
    return request.get('/backtests/strategy-types')
  },

  // 获取策略配置模板
  getStrategyConfigTemplate: async (strategyType: string) => {
    return request.get(`/backtests/strategy-config/${strategyType}`)
  }
}