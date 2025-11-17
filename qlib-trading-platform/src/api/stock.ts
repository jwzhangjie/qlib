import type { Stock, StockData, TechnicalIndicator, PaginatedResponse, ApiResponse } from '@/types'
import request from '@/utils/request'

export const stockAPI = {
  // 搜索股票
  search: async (query: string, page = 1, size = 20): Promise<PaginatedResponse<Stock>> => {
    return request.get('/stocks/search', {
      params: { query, page, size }
    })
  },

  // 获取股票数据
  getData: async (symbol: string, startDate: string, endDate: string, period = '1d'): Promise<StockData> => {
    return request.get(`/stocks/${symbol}/data`, {
      params: { start_date: startDate, end_date: endDate, period }
    })
  },

  // 获取技术指标
  getTechnicalIndicators: async (symbol: string, indicators: string[]): Promise<TechnicalIndicator[]> => {
    return request.get(`/stocks/${symbol}/indicators`, {
      params: { indicators: indicators.join(',') }
    })
  },

  // 获取股票基本信息
  getStockInfo: async (symbol: string): Promise<Stock> => {
    return request.get(`/stocks/${symbol}`)
  },

  // 获取热门股票
  getHotStocks: async (): Promise<Stock[]> => {
    return request.get('/stocks/hot')
  },

  // 获取市场概览
  getMarketOverview: async () => {
    return request.get('/stocks/market-overview')
  }
}