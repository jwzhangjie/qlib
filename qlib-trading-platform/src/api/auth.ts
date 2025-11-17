import type { LoginCredentials, AuthResponse, User } from '@/types'
import request from '@/utils/request'

export const authAPI = {
  // 用户登录
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    return request.post('/auth/login', credentials)
  },

  // 用户注册
  register: async (userData: {
    username: string
    email: string
    password: string
  }): Promise<AuthResponse> => {
    return request.post('/auth/register', userData)
  },

  // 获取当前用户信息
  getCurrentUser: async (): Promise<User> => {
    return request.get('/auth/me')
  },

  // 刷新令牌
  refreshToken: async (): Promise<{ token: string }> => {
    return request.post('/auth/refresh')
  },

  // 修改密码
  changePassword: async (passwords: {
    currentPassword: string
    newPassword: string
  }): Promise<void> => {
    return request.post('/auth/change-password', passwords)
  },

  // 退出登录
  logout: async (): Promise<void> => {
    return request.post('/auth/logout')
  }
}