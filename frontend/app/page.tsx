'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Search, Brain, Loader2, GitBranch } from 'lucide-react'
import { contextKeeperAPI, QueryResponse } from '../lib/api'
import { QueryResults } from '../components/QueryResults'
import { RepositoryList } from '../components/RepositoryList'
import { BranchSelector } from '../components/BranchSelector'

export default function Home() {
  const [query, setQuery] = useState('')
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null)
  const [selectedBranches, setSelectedBranches] = useState<string[]>([])

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: contextKeeperAPI.getStats,
  })

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: contextKeeperAPI.getHealth,
  })

  const searchMutation = useMutation<QueryResponse, Error, string>({
    mutationFn: (searchQuery: string) => 
      contextKeeperAPI.query(searchQuery, selectedRepo || undefined),
  })

  const handleSearch = () => {
    if (query.trim()) {
      searchMutation.mutate(query)
    }
  }

  // Dynamic suggested queries based on selected branches
  const getSuggestedQueries = () => {
    const baseQueries = [
      'Tell me about this codebase',
      'What are the recent changes?',
      'Show the latest commit',
      'Explain the architecture'
    ];

    if (selectedBranches.length > 0) {
      const branchSpecific = [
        `What's happening on ${selectedBranches[0]} branch?`,
        selectedBranches.includes('main') || selectedBranches.includes('master') 
          ? 'What was merged to main recently?' 
          : 'Show critical bug fixes',
        selectedBranches.some(b => b.startsWith('feature/'))
          ? 'Show feature development progress'
          : 'Who has been most active?',
        selectedBranches.length > 1 
          ? 'Compare branches activity' 
          : 'Show recent commits'
      ];
      return branchSpecific;
    }

    return baseQueries;
  };

  const suggestedQueries = getSuggestedQueries();

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <Brain className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-xl font-semibold">Context Keeper</h1>
              <p className="text-sm text-gray-400">AI Memory Layer for Development Teams</p>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8 max-w-7xl">

        {/* Stats */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {stats?.events?.in_qdrant || 0}
            </div>
            <div className="text-sm text-gray-400 mt-1">Total Commits</div>
          </div>
          <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {stats?.repositories || 0}
            </div>
            <div className="text-sm text-gray-400 mt-1">Repositories</div>
          </div>
          <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 text-center">
            <div className="text-2xl font-bold">
              {health?.services?.ollama === 'healthy' ? 
                <span className="text-emerald-400">Online</span> : 
                <span className="text-red-400">Offline</span>
              }
            </div>
            <div className="text-sm text-gray-400 mt-1">AI Status</div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Repositories and Branches */}
          <div className="lg:col-span-1 space-y-6">
            <RepositoryList 
              selectedRepo={selectedRepo}
              onSelectRepo={setSelectedRepo}
            />
            
            {/* Branch Selector */}
            <BranchSelector
              selectedRepository={selectedRepo}
              selectedBranches={selectedBranches}
              onBranchesChange={setSelectedBranches}
            />
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3 space-y-6">
            {/* Search Section */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
              <div className="relative mb-6">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleSearch()
                  }}
                  placeholder={selectedRepo 
                    ? `Ask anything about ${selectedRepo.split('/').pop()}...` 
                    : "Select a repository and ask anything..."
                  }
                  className="w-full px-4 py-3 pr-12 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none transition-colors"
                  disabled={!selectedRepo}
                />
                <button
                  onClick={handleSearch}
                  disabled={searchMutation.isPending || !selectedRepo || !query.trim()}
                  className="absolute right-2 top-2 p-2 text-emerald-400 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {searchMutation.isPending ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                </button>
              </div>

              {/* Branch Context Display */}
              {selectedBranches.length > 0 && (
                <div className="flex items-center gap-2 mb-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <GitBranch className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-400">Filtering by:</span>
                  <div className="flex gap-1 flex-wrap">
                    {selectedBranches.map(branch => (
                      <span
                        key={branch}
                        className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs border border-blue-500/30"
                      >
                        {branch}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Suggested Queries */}
              {selectedRepo && (
                <div>
                  <div className="text-sm text-gray-500 mb-3 flex items-center gap-2">
                    <span>💡 Try asking:</span>
                    {selectedBranches.length > 0 && (
                      <span className="text-xs text-blue-400">
                        (filtered by {selectedBranches.length} branch{selectedBranches.length > 1 ? 'es' : ''})
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {suggestedQueries.map((sq, index) => (
                      <button
                        key={index}
                        onClick={() => {
                          setQuery(sq)
                          searchMutation.mutate(sq)
                        }}
                        disabled={searchMutation.isPending}
                        className="p-4 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700 hover:border-gray-600 rounded-xl text-left transition-all group disabled:opacity-50"
                      >
                        <div className="flex items-start gap-2">
                          <span className="text-yellow-400 text-sm">💡</span>
                          <p className="text-sm text-gray-300 group-hover:text-white transition-colors">
                            {sq}
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Repository Selection Prompt */}
              {!selectedRepo && (
                <div className="text-center py-8">
                  <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-6">
                    <Brain className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                    <h3 className="text-lg font-medium text-gray-300 mb-2">
                      Ready to Analyze Your Code
                    </h3>
                    <p className="text-gray-400 text-sm">
                      Select a repository from the sidebar to start querying your development history
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Query Results */}
            {searchMutation.data && (
              <QueryResults data={searchMutation.data} />
            )}

            {/* Error State */}
            {searchMutation.error && (
              <div className="bg-red-900/20 border border-red-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-red-400 mb-2">Query Failed</h3>
                <p className="text-red-300">
                  {searchMutation.error.message || 'An error occurred while processing your query'}
                </p>
                <button
                  onClick={() => searchMutation.reset()}
                  className="mt-3 text-sm text-red-400 hover:text-red-300 transition-colors"
                >
                  Try again
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}