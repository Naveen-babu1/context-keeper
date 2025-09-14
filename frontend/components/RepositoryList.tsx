'use client'

import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, GitBranch } from 'lucide-react'
import { contextKeeperAPI } from '../lib/api'

interface RepositoryListProps {
  selectedRepo: string | null
  onSelectRepo: (repo: string | null) => void
}

export function RepositoryList({ selectedRepo, onSelectRepo }: RepositoryListProps) {
  const [isAddingRepo, setIsAddingRepo] = useState(false)
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['repositories'],
    queryFn: contextKeeperAPI.getRepositories,
  })

  const handleAddRepository = async () => {
    const repoPath = prompt('Enter repository path (e.g., D:/projects/my-repo):')
    if (!repoPath) return

    setIsAddingRepo(true)
    try {
      await contextKeeperAPI.startIngestion(repoPath)
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['repositories'] })
        queryClient.invalidateQueries({ queryKey: ['stats'] })
      }, 2000)
    } catch (error: any) {
      alert(`Error: ${error.message}`)
    } finally {
      setIsAddingRepo(false)
    }
  }

  const repositories = data?.repositories || {}

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <GitBranch className="w-5 h-5 text-emerald-400" />
        Repositories
      </h3>
      
      <div className="space-y-2 mb-4">
        {Object.entries(repositories).map(([path, repo]: [string, any]) => {
          const repoName = path.split(/[/\\]/).pop() || path
          const isSelected = selectedRepo === path

          return (
            <button
              key={path}
              onClick={() => onSelectRepo(isSelected ? null : path)}
              className={`w-full text-left p-3 rounded-xl transition-all ${
                isSelected
                  ? 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
                  : 'bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700 text-gray-300'
              }`}
            >
              <div className="font-medium">{repoName}</div>
              <div className="text-xs mt-1 opacity-70">
                {repo.commit_count || 0} commits
              </div>
            </button>
          )
        })}
      </div>

      <button
        onClick={handleAddRepository}
        disabled={isAddingRepo}
        className="w-full px-4 py-3 bg-gray-800/50 hover:bg-gray-800/70 border border-gray-700 text-gray-300 rounded-xl font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
      >
        <Plus className="w-4 h-4" />
        {isAddingRepo ? 'Adding Repository...' : 'Add Repository'}
      </button>
    </div>
  )
}