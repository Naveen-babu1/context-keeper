"use client";

import { useState, useEffect } from "react";
import { GitBranch, X, Filter } from "lucide-react";
import { api } from "../lib/api";

interface Branch {
  name: string;
  commit_count: number;
  last_commit_date: string;
  branch_type: 'main' | 'develop' | 'feature' | 'hotfix' | 'release' | 'other';
}

interface BranchSelectorProps {
  selectedRepository: string | null;
  selectedBranches: string[];
  onBranchesChange: (branches: string[]) => void;
}

export function BranchSelector({ 
  selectedRepository, 
  selectedBranches, 
  onBranchesChange 
}: BranchSelectorProps) {
  const [branches, setBranches] = useState<Branch[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    if (selectedRepository) {
      loadBranches();
    } else {
      setBranches([]);
      onBranchesChange([]);
    }
  }, [selectedRepository]);

  const loadBranches = async () => {
    if (!selectedRepository) return;
    
    setLoading(true);
    try {
      const response = await api.get(`/api/repositories/${encodeURIComponent(selectedRepository)}/branches`);
      setBranches(response.data.branches || []);
    } catch (error) {
      console.error('Failed to load branches:', error);
      // Fallback - show basic branch options
      setBranches([
        { name: 'main', commit_count: 0, last_commit_date: '', branch_type: 'main' },
        { name: 'develop', commit_count: 0, last_commit_date: '', branch_type: 'develop' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getBranchColor = (type: string) => {
    const colors = {
      main: 'text-green-400 bg-green-400/20 border-green-400/30',
      develop: 'text-blue-400 bg-blue-400/20 border-blue-400/30',
      feature: 'text-purple-400 bg-purple-400/20 border-purple-400/30',
      hotfix: 'text-red-400 bg-red-400/20 border-red-400/30',
      release: 'text-orange-400 bg-orange-400/20 border-orange-400/30',
      other: 'text-gray-400 bg-gray-400/20 border-gray-400/30'
    };
    return colors[type as keyof typeof colors] || colors.other;
  };

  const toggleBranch = (branchName: string) => {
    const newSelection = selectedBranches.includes(branchName)
      ? selectedBranches.filter(b => b !== branchName)
      : [...selectedBranches, branchName];
    
    onBranchesChange(newSelection);
  };

  const clearSelection = () => {
    onBranchesChange([]);
  };

  const selectMainBranches = () => {
    const mainBranches = branches
      .filter(b => ['main', 'develop'].includes(b.branch_type))
      .map(b => b.name);
    onBranchesChange(mainBranches);
  };

  if (!selectedRepository) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
        <div className="text-center text-gray-500 py-4">
          <GitBranch className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p>Select a repository to view branches</p>
        </div>
      </div>
    );
  }

  const displayedBranches = showAll ? branches : branches.slice(0, 8);

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-blue-400" />
          Branches
        </h3>
        <div className="flex gap-2 text-sm">
          <button
            onClick={selectMainBranches}
            className="text-blue-400 hover:text-blue-300 transition-colors"
          >
            Main
          </button>
          <button
            onClick={clearSelection}
            className="text-gray-400 hover:text-gray-300 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Selected branches summary */}
      {selectedBranches.length > 0 && (
        <div className="mb-4 p-3 bg-gray-800/30 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">
              {selectedBranches.length} branch(es) selected
            </span>
            <button
              onClick={clearSelection}
              className="text-gray-400 hover:text-red-400 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex flex-wrap gap-1">
            {selectedBranches.map(branch => (
              <span
                key={branch}
                className="px-2 py-1 text-xs bg-blue-500/20 text-blue-400 rounded flex items-center gap-1"
              >
                {branch}
                <button
                  onClick={() => toggleBranch(branch)}
                  className="hover:text-red-400 transition-colors"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Branch list */}
      <div className="space-y-2 max-h-80 overflow-y-auto">
        {loading ? (
          <div className="text-center py-4 text-gray-500">
            <div className="animate-pulse">Loading branches...</div>
          </div>
        ) : branches.length === 0 ? (
          <div className="text-center py-4 text-gray-500">
            <GitBranch className="w-6 h-6 mx-auto mb-2 opacity-50" />
            <p>No branches found</p>
          </div>
        ) : (
          <>
            {displayedBranches.map(branch => (
              <div
                key={branch.name}
                className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedBranches.includes(branch.name)
                    ? 'bg-blue-500/20 border-blue-500/50'
                    : 'bg-gray-800/30 border-gray-700 hover:bg-gray-800/50'
                }`}
                onClick={() => toggleBranch(branch.name)}
              >
                <div className="flex items-center gap-3">
                  <div className={`px-2 py-1 rounded text-xs border ${getBranchColor(branch.branch_type)}`}>
                    {branch.branch_type}
                  </div>
                  <div>
                    <div className="font-medium text-gray-300">{branch.name}</div>
                    <div className="text-xs text-gray-500">
                      {branch.commit_count} commits
                      {branch.last_commit_date && ` • ${formatTimeAgo(branch.last_commit_date)}`}
                    </div>
                  </div>
                </div>
                
                <div className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-all ${
                  selectedBranches.includes(branch.name)
                    ? 'bg-blue-500 border-blue-500'
                    : 'border-gray-600 hover:border-gray-500'
                }`}>
                  {selectedBranches.includes(branch.name) && (
                    <div className="w-2 h-2 bg-white rounded-full" />
                  )}
                </div>
              </div>
            ))}
            
            {branches.length > 8 && (
              <button
                onClick={() => setShowAll(!showAll)}
                className="w-full text-center py-2 text-sm text-gray-400 hover:text-gray-300 transition-colors"
              >
                {showAll ? 'Show Less' : `Show All (${branches.length} branches)`}
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function formatTimeAgo(dateString: string): string {
  if (!dateString) return '';
  
  const date = new Date(dateString);
  const now = new Date();
  const diffTime = Math.abs(now.getTime() - date.getTime());
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  
  if (diffDays === 1) return '1d ago';
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.ceil(diffDays / 7)}w ago`;
  if (diffDays < 365) return `${Math.ceil(diffDays / 30)}mo ago`;
  return `${Math.ceil(diffDays / 365)}y ago`;
}