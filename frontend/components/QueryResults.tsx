'use client'

import { QueryResponse } from '../lib/api'
import ReactMarkdown from 'react-markdown'

interface QueryResultsProps {
  data: QueryResponse
}

export function QueryResults({ data }: QueryResultsProps) {
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
          <div className="text-gray-300">
            <ReactMarkdown>
              {data.answer}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* Sources */}
      {data.sources && data.sources.length > 0 && (
        <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
          <h3 className="text-lg font-semibold mb-4">Sources</h3>
          <div className="space-y-3">
            {data.sources.slice(0, 5).map((source, idx) => (
              <div
                key={idx}
                className="p-4 bg-gray-800/30 border border-gray-700 rounded-xl"
              >
                <div className="flex items-start justify-between mb-2">
                  <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 text-xs font-mono rounded">
                    {source.commit}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(source.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-sm text-gray-300 font-medium mb-1">
                  {source.content}
                </p>
                <p className="text-xs text-gray-500">
                  {source.author?.split(' <')[0]}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}