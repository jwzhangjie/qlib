<template>
  <div class="min-h-screen bg-gray-50">
    <!-- 顶部导航栏 -->
    <nav class="bg-white shadow-sm border-b">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <!-- Logo和导航链接 -->
          <div class="flex items-center">
            <router-link to="/" class="flex items-center space-x-2">
              <el-icon class="text-blue-600" size="24">
                <TrendCharts />
              </el-icon>
              <span class="text-xl font-bold text-gray-900">Qlib量化平台</span>
            </router-link>
            
            <div class="hidden md:flex ml-10 space-x-8">
              <router-link
                v-for="item in navigation"
                :key="item.name"
                :to="item.path"
                :class="[
                  $route.path === item.path
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300',
                  'inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium'
                ]"
              >
                <el-icon class="mr-1" size="16">
                  <component :is="item.icon" />
                </el-icon>
                {{ item.name }}
              </router-link>
            </div>
          </div>

          <!-- 用户信息 -->
          <div class="flex items-center space-x-4">
            <el-dropdown>
              <span class="flex items-center space-x-2 cursor-pointer">
                <el-avatar :size="32" icon="UserFilled" />
                <span class="text-sm font-medium text-gray-700">{{ authStore.user?.username || '用户' }}</span>
                <el-icon class="text-gray-400" size="12">
                  <ArrowDown />
                </el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item>
                    <el-icon><User /></el-icon>
                    个人设置
                  </el-dropdown-item>
                  <el-dropdown-item>
                    <el-icon><Setting /></el-icon>
                    系统设置
                  </el-dropdown-item>
                  <el-dropdown-item divided @click="handleLogout">
                    <el-icon><SwitchButton /></el-icon>
                    退出登录
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
      </div>
    </nav>

    <!-- 主要内容区域 -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  TrendCharts,
  ArrowDown,
  User,
  Setting,
  SwitchButton,
  UserFilled,
  Search,
  DataAnalysis,
  TrendData,
  Document
} from '@element-plus/icons-vue'

const authStore = useAuthStore()
const router = useRouter()

const navigation = ref([
  { name: '仪表盘', path: '/', icon: 'DataAnalysis' },
  { name: '股票查询', path: '/stocks', icon: 'Search' },
  { name: '模型训练', path: '/models', icon: 'TrendData' },
  { name: '回测分析', path: '/backtest', icon: 'DataAnalysis' },
  { name: '数据管理', path: '/data', icon: 'Document' }
])

const handleLogout = async () => {
  try {
    await authStore.logout()
    ElMessage.success('退出登录成功')
  } catch (error) {
    ElMessage.error('退出登录失败')
  }
}
</script>