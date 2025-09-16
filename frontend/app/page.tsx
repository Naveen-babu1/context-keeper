'use client'

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Search, Brain, Loader2 } from 'lucide-react'
import { contextKeeperAPI, QueryResponse } from '../lib/api'
import { QueryResults } from '../components/QueryResults'
import { RepositoryList } from '../components/RepositoryList'

export default function Home() {
  const [query, setQuery] = useState('')
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null)

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

  const suggestedQueries = [
    'Tell me about this codebase',
    'What are the recent changes?',
    'Show critical bug fixes',
    'Explain the architecture'
  ]

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
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Repositories Sidebar */}
          <div className="lg:col-span-1">
            <RepositoryList 
              selectedRepo={selectedRepo}
              onSelectRepo={setSelectedRepo}
            />
          </div>

          {/* Search and Results */}
          <div className="lg:col-span-2 space-y-6">
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
                  placeholder="Ask about your repositories, commits, or code..."
                  className="w-full px-4 py-3 pr-12 bg-gray-800/50 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:border-emerald-500 focus:outline-none transition-colors"
                />
                <button
                  onClick={handleSearch}
                  disabled={searchMutation.isPending}
                  className="absolute right-2 top-2 p-2 text-emerald-400 hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
                >
                  {searchMutation.isPending ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Search className="w-5 h-5" />
                  )}
                </button>
              </div>

              {/* Suggested Queries */}
              <div className="grid grid-cols-2 gap-3">
                {suggestedQueries.map((sq) => (
                  <button
                    key={sq}
                    onClick={() => {
                      setQuery(sq)
                      searchMutation.mutate(sq)
                    }}
                    className="p-4 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700 rounded-xl text-left transition-all group"
                  >
                    <span className="text-yellow-400 text-lg mb-1">💡</span>
                    <p className="text-sm text-gray-300 group-hover:text-white transition-colors">
                      {sq}
                    </p>
                  </button>
                ))}
              </div>
            </div>

            {/* Results */}
            {searchMutation.data && (
              <QueryResults data={searchMutation.data} />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}