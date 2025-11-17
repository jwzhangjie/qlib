import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Stock, StockData, TechnicalIndicator } from '@/types'
import { stockAPI } from '@/api/stock'

export const useStockStore = defineStore('stock', () => {
  // 状态
  const stocks = ref<Stock[]>([])
  const currentStock = ref<Stock | null>(null)
  const stockData = ref<Record<string, StockData>>({})
  const technicalIndicators = ref<Record<string, TechnicalIndicator[]>>({})
  const loading = ref(false)
  const error = ref<string | null>(null)

  // 计算属性
  const hasStocks = computed(() => stocks.value.length > 0)
  const hasCurrentStock = computed(() => !!currentStock.value)
  const currentStockData = computed(() => 
    currentStock.value ? stockData.value[currentStock.value.symbol] : null
  )

  // 搜索股票
  const searchStocks = async (query: string, page = 1, size = 20) => {
    loading.value = true
    error.value = null
    try {
      const response = await stockAPI.search(query, page, size)
      stocks.value = response.items
      return response
    } catch (err) {
      error.value = err instanceof Error ? err.message : '搜索失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 获取股票数据
  const fetchStockData = async (symbol: string, startDate: string, endDate: string, period = '1d') => {
    loading.value = true
    error.value = null
    try {
      const data = await stockAPI.getData(symbol, startDate, endDate, period)
      stockData.value[symbol] = data
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取股票数据失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 获取技术指标
  const fetchTechnicalIndicators = async (symbol: string, indicators: string[]) => {
    loading.value = true
    error.value = null
    try {
      const data = await stockAPI.getTechnicalIndicators(symbol, indicators)
      technicalIndicators.value[symbol] = data
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取技术指标失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 获取股票信息
  const fetchStockInfo = async (symbol: string) => {
    loading.value = true
    error.value = null
    try {
      const stock = await stockAPI.getStockInfo(symbol)
      currentStock.value = stock
      return stock
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取股票信息失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 获取热门股票
  const fetchHotStocks = async () => {
    loading.value = true
    error.value = null
    try {
      const data = await stockAPI.getHotStocks()
      stocks.value = data
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取热门股票失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 获取市场概览
  const fetchMarketOverview = async () => {
    loading.value = true
    error.value = null
    try {
      const data = await stockAPI.getMarketOverview()
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : '获取市场概览失败'
      throw err
    } finally {
      loading.value = false
    }
  }

  // 设置当前股票
  const setCurrentStock = (stock: Stock | null) => {
    currentStock.value = stock
  }

  // 清除错误
  const clearError = () => {
    error.value = null
  }

  return {
    // 状态
    stocks,
    currentStock,
    stockData,
    technicalIndicators,
    loading,
    error,
    
    // 计算属性
    hasStocks,
    hasCurrentStock,
    currentStockData,
    
    // 方法
    searchStocks,
    fetchStockData,
    fetchTechnicalIndicators,
    fetchStockInfo,
    fetchHotStocks,
    fetchMarketOverview,
    setCurrentStock,
    clearError
  }
})