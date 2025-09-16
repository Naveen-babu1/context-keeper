'use client'

import { useState } from 'react'
import { X } from 'lucide-react'

interface AddRepositoryModalProps {
  isOpen: boolean
  onClose: () => void
  onAdd: (path: string) => Promise<void>
}

export function AddRepositoryModal({ isOpen, onClose, onAdd }: AddRepositoryModalProps) {
  const [repoPath, setRepoPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    if (!repoPath.trim()) {
      setError('Please enter a repository path')
      return
    }

    setIsLoading(true)
    setError('')
    
    try {
      await onAdd(repoPath)
      setRepoPath('')
      onClose()
    } catch (err: any) {
      setError(err.message || 'Failed to add repository')
    } finally {
      setIsLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-md shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <h3 className="text-xl font-semibold mb-2">Add Repository</h3>
        <p className="text-gray-400 text-sm mb-6">
          Enter the full path to your local repository
        </p>

        <input
          type="text"
          value={repoPath}
          onChange={(e) => setRepoPath(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          placeholder="e.g., D:/projects/my-repo"
          className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none transition-colors"
          autoFocus
        />

        {error && (
          <p className="mt-2 text-red-400 text-sm">{error}</p>
        )}

        <div className="flex gap-3 mt-6">
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Adding...' : 'Add Repository'}
          </button>
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors"
          >
            Cancel
          </button>
        </div>

        <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-400">
            💡 Tip: Make sure the repository has been initialized with git
          </p>
        </div>
      </div>
    </div>
  )
}