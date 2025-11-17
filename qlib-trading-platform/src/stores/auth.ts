import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginCredentials } from '@/types'
import { authAPI } from '@/api/auth'
import router from '@/router'

export const useAuthStore = defineStore('auth', () => {
  // 状态
  const token = ref<string>(localStorage.getItem('token') || '')
  const user = ref<User | null>(null)
  const loading = ref(false)

  // 计算属性
  const isAuthenticated = computed(() => !!token.value)
  const userRole = computed(() => user.value?.role || 'user')

  // 登录
  const login = async (credentials: LoginCredentials) => {
    loading.value = true
    try {
      const response = await authAPI.login(credentials)
      token.value = response.token
      user.value = response.user
      localStorage.setItem('token', response.token)
      
      // 跳转到首页
      router.push('/')
      
      return { success: true }
    } catch (error) {
      return { success: false, error }
    } finally {
      loading.value = false
    }
  }

  // 注册
  const register = async (userData: {
    username: string
    email: string
    password: string
  }) => {
    loading.value = true
    try {
      const response = await authAPI.register(userData)
      token.value = response.token
      user.value = response.user
      localStorage.setItem('token', response.token)
      
      // 跳转到首页
      router.push('/')
      
      return { success: true }
    } catch (error) {
      return { success: false, error }
    } finally {
      loading.value = false
    }
  }

  // 获取当前用户信息
  const fetchCurrentUser = async () => {
    if (!token.value) return
    
    try {
      const userData = await authAPI.getCurrentUser()
      user.value = userData
    } catch (error) {
      // 如果获取用户信息失败，清除token
      logout()
      throw error
    }
  }

  // 退出登录
  const logout = () => {
    token.value = ''
    user.value = null
    localStorage.removeItem('token')
    router.push('/login')
  }

  // 刷新令牌
  const refreshToken = async () => {
    try {
      const response = await authAPI.refreshToken()
      token.value = response.token
      localStorage.setItem('token', response.token)
    } catch (error) {
      // 刷新失败，退出登录
      logout()
      throw error
    }
  }

  // 修改密码
  const changePassword = async (passwords: {
    currentPassword: string
    newPassword: string
  }) => {
    await authAPI.changePassword(passwords)
  }

  return {
    // 状态
    token,
    user,
    loading,
    
    // 计算属性
    isAuthenticated,
    userRole,
    
    // 方法
    login,
    register,
    fetchCurrentUser,
    logout,
    refreshToken,
    changePassword
  }
})