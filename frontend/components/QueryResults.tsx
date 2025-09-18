'use client'
import { useState } from 'react'
import { QueryResponse } from '../lib/api'
import ReactMarkdown from 'react-markdown'
import { GitBranch, User, Calendar, FileText, GitMerge, ExternalLink, ChevronDown } from 'lucide-react'
import { sources } from 'next/dist/compiled/webpack/webpack'

interface QueryResultsProps {
  data: QueryResponse
}

export function QueryResults({ data }: QueryResultsProps) {
  const [expandedCommit, setExpandedCommit] = useState<string | null>(null);

  const getBranchColor = (branchType?: string) => {
    const colors = {
      main: 'text-green-400 bg-green-400/20 border-green-400/30',
      develop: 'text-blue-400 bg-blue-400/20 border-blue-400/30',
      feature: 'text-purple-400 bg-purple-400/20 border-purple-400/30',
      hotfix: 'text-red-400 bg-red-400/20 border-red-400/30',
      release: 'text-orange-400 bg-orange-400/20 border-orange-400/30',
      other: 'text-gray-400 bg-gray-400/20 border-gray-400/30'
    };
    return colors[branchType as keyof typeof colors] || colors.other;
  };

  const formatTimeAgo = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.ceil(diffDays / 7)} weeks ago`;
    return `${Math.ceil(diffDays / 30)} months ago`;
  };
  return (
    <div className="space-y-6">
      {/* AI Analysis */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <span>🤖</span> AI Analysis
          </h3>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-400">Confidence:</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all"
                  style={{ width: `${(data.confidence || 0) * 100}%` }}
                />
              </div>
              <span className="text-emerald-400 font-medium">
                {Math.round((data.confidence || 0) * 100)}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="prose prose-invert max-w-none">
          <div className="text-gray-300 leading-relaxed">
            <ReactMarkdown
              components={{
                h1: ({children}) => <h1 className="text-xl font-bold text-gray-100 mb-3">{children}</h1>,
                h2: ({children}) => <h2 className="text-lg font-semibold text-gray-200 mb-2 mt-4">{children}</h2>,
                strong: ({children}) => <strong className="text-gray-100 font-semibold">{children}</strong>,
                code: ({children}) => <code className="bg-gray-800/50 px-1.5 py-0.5 rounded text-emerald-400 text-sm font-mono">{children}</code>,
                li: ({children}) => <li className="mb-1">{children}</li>
              }}
            >
              {data.answer}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* Sources */}
      {data.sources && data.sources.length > 0 && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FileText className="w-5 h-5 text-gray-400" />
            Sources ({data.sources.length} commits)
          </h3>
          
          <div className="space-y-3">
            {data.sources.map((source, index) => (
              <div
                key={`${source.commit}-${index}`}
                className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-4 hover:bg-gray-800/50 transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    {/* Commit Header */}
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                      <span className="font-mono text-sm bg-gray-700/50 px-2 py-1 rounded text-emerald-400">
                        {source.commit}
                      </span>
                      
                      {/* Branch indicators */}
                      {source.primary_branch && (
                        <div className={`px-2 py-1 text-xs rounded border ${getBranchColor(source.branch_context?.branch_type || 'main')}`}>
                          {source.primary_branch}
                        </div>
                      )}
                      
                      {source.merge_commit && (
                        <div className="flex items-center gap-1 text-xs text-purple-400 bg-purple-400/20 px-2 py-1 rounded border border-purple-400/30">
                          <GitMerge className="w-3 h-3" />
                          Merge
                        </div>
                      )}
                    </div>

                    {/* Commit Message */}
                    <p className="text-gray-300 mb-3 leading-relaxed">
                      {source.content}
                    </p>

                    {/* Metadata Row */}
                    <div className="flex items-center gap-4 text-sm text-gray-400 mb-2">
                      <div className="flex items-center gap-1">
                        <User className="w-4 h-4" />
                        <span>{source.author.split('<')[0].trim()}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Calendar className="w-4 h-4" />
                        <span>{formatTimeAgo(source.timestamp)}</span>
                      </div>
                      {source.files_changed && source.files_changed.length > 0 && (
                        <div className="flex items-center gap-1">
                          <FileText className="w-4 h-4" />
                          <span>{source.files_changed.length} files</span>
                        </div>
                      )}
                    </div>

                    {/* All Branches (if multiple) */}
                    {source.all_branches && source.all_branches.length > 1 && (
                      <div className="flex items-center gap-2 mb-3">
                        <GitBranch className="w-4 h-4 text-gray-400" />
                        <span className="text-xs text-gray-400">Also in:</span>
                        <div className="flex gap-1 flex-wrap">
                          {source.all_branches
                            .filter(branch => branch !== source.primary_branch)
                            .slice(0, 3)
                            .map(branch => (
                              <span
                                key={branch}
                                className="px-1.5 py-0.5 text-xs bg-gray-600/30 text-gray-400 rounded"
                              >
                                {branch}
                              </span>
                            ))}
                          {source.all_branches.length > 4 && (
                            <span className="px-1.5 py-0.5 text-xs bg-gray-600/30 text-gray-400 rounded">
                              +{source.all_branches.length - 4}
                            </span>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Expandable Files List */}
                    {source.files_changed && source.files_changed.length > 0 && (
                      <div>
                        <button
                          onClick={() => setExpandedCommit(
                            expandedCommit === source.commit ? null : source.commit
                          )}
                          className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
                        >
                          <ChevronDown className={`w-3 h-3 transition-transform ${
                            expandedCommit === source.commit ? 'rotate-180' : ''
                          }`} />
                          {expandedCommit === source.commit ? 'Hide' : 'Show'} files ({source.files_changed.length})
                        </button>
                        
                        {expandedCommit === source.commit && (
                          <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
                            {source.files_changed.map((file, idx) => (
                              <div
                                key={idx}
                                className="text-xs text-gray-400 font-mono bg-gray-700/30 px-2 py-1 rounded"
                              >
                                {file}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Action Button */}
                  <button
                    onClick={() => {
                      console.log('View commit details:', source.commit);
                    }}
                    className="p-2 text-gray-400 hover:text-gray-300 transition-colors flex-shrink-0"
                    title="View detailed commit analysis"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}