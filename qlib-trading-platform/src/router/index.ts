import { createRouter, createWebHistory } from 'vue-router'
import DefaultLayout from '@/layouts/DefaultLayout.vue'
import AuthLayout from '@/layouts/AuthLayout.vue'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: DefaultLayout,
      meta: { requiresAuth: true },
      children: [
        {
          path: '',
          name: 'dashboard',
          component: () => import('@/views/dashboard/Index.vue')
        },
        {
          path: '/stocks',
          name: 'stocks',
          component: () => import('@/views/stocks/Index.vue')
        },
        {
          path: '/stocks/:symbol',
          name: 'stock-detail',
          component: () => import('@/views/stocks/Detail.vue')
        },
        {
          path: '/models',
          name: 'models',
          component: () => import('@/views/models/Index.vue')
        },
        {
          path: '/backtest',
          name: 'backtest',
          component: () => import('@/views/backtest/Index.vue')
        },
        {
          path: '/data',
          name: 'data',
          component: () => import('@/views/data/Index.vue')
        }
      ]
    },
    {
      path: '/auth',
      component: AuthLayout,
      children: [
        {
          path: 'login',
          name: 'login',
          component: () => import('@/views/auth/Login.vue')
        },
        {
          path: 'register',
          name: 'register',
          component: () => import('@/views/auth/Register.vue')
        }
      ]
    }
  ],
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  // 检查是否需要认证
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    next('/auth/login')
    return
  }
  
  // 如果已登录，尝试获取用户信息
  if (authStore.isAuthenticated && !authStore.user) {
    try {
      await authStore.fetchCurrentUser()
    } catch (error) {
      // 获取用户信息失败，清除认证状态
      authStore.logout()
      next('/auth/login')
      return
    }
  }
  
  next()
})

export default router
