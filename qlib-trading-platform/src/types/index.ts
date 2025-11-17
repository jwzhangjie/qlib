// 股票相关接口
export interface Stock {
  id: number
  symbol: string
  name: string
  market: string
  sector?: string
  industry?: string
  createdAt: string
}

export interface StockData {
  symbol: string
  data: {
    dates: string[]
    open: number[]
    high: number[]
    low: number[]
    close: number[]
    volume: number[]
  }
}

export interface TechnicalIndicator {
  name: string
  values: number[]
  dates: string[]
}

// 模型相关接口
export interface Model {
  id: number
  name: string
  modelType: string
  config: Record<string, any>
  status: 'pending' | 'running' | 'completed' | 'failed'
  userId: number
  createdAt: string
  updatedAt: string
}

export interface ModelTrainingRequest {
  name: string
  modelType: string
  config: Record<string, any>
  datasetConfig: Record<string, any>
}

export interface ModelPrediction {
  symbol: string
  prediction: number
  confidence: number
  date: string
}

// 回测相关接口
export interface Backtest {
  id: number
  name: string
  strategyConfig: Record<string, any>
  startDate: string
  endDate: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  results?: BacktestResults
  userId: number
  createdAt: string
  updatedAt: string
}

export interface BacktestResults {
  totalReturn: number
  annualizedReturn: number
  maxDrawdown: number
  sharpeRatio: number
  volatility: number
  winRate: number
  trades: Trade[]
  equityCurve: { date: string; value: number }[]
}

export interface Trade {
  date: string
  symbol: string
  action: 'buy' | 'sell'
  price: number
  quantity: number
  profit: number
}

// 用户相关接口
export interface User {
  id: number
  username: string
  email: string
  isActive: boolean
  createdAt: string
}

export interface LoginCredentials {
  username: string
  password: string
}

export interface AuthResponse {
  token: string
  user: User
}

// API响应接口
export interface ApiResponse<T> {
  success: boolean
  data: T
  message?: string
  error?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  size: number
  pages: number
}