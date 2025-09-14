import { ReactNode } from 'react'

interface StatsCardProps {
  label: string
  value: string | number
  icon?: ReactNode
  status?: 'success' | 'error' | 'warning' | 'default'
}

export function StatsCard({ label, value, icon, status = 'default' }: StatsCardProps) {
  const statusColors = {
    success: 'text-green-600 bg-green-50',
    error: 'text-red-600 bg-red-50',
    warning: 'text-yellow-600 bg-yellow-50',
    default: 'text-purple-600 bg-purple-50'
  }

  return (
    <div className={`p-4 rounded-xl ${statusColors[status]} transition-all hover:scale-105`}>
      <div className="flex items-center justify-between mb-2">
        {icon && <div className="opacity-60">{icon}</div>}
        <div className="text-2xl font-bold">
          {value}
        </div>
      </div>
      <div className="text-sm opacity-80 font-medium">
        {label}
      </div>
    </div>
  )
}