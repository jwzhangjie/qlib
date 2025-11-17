<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div class="text-center">
        <el-icon class="mx-auto h-12 w-12 text-blue-600" size="48">
          <TrendCharts />
        </el-icon>
        <h2 class="mt-6 text-3xl font-extrabold text-gray-900">
          Qlib量化平台
        </h2>
        <p class="mt-2 text-sm text-gray-600">
          专业的量化投资分析平台
        </p>
      </div>
      
      <router-view />
    </div>
  </div>
</template>

<script setup lang="ts">
import { TrendCharts } from '@element-plus/icons-vue'
</script>