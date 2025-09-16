"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, GitBranch, Trash2, RefreshCw } from "lucide-react";
import { api, contextKeeperAPI } from "../lib/api";
import { AddRepositoryModal } from "./AddRepositoryModal";

interface RepositoryListProps {
  selectedRepo: string | null;
  onSelectRepo: (repo: string | null) => void;
}

export function RepositoryList({
  selectedRepo,
  onSelectRepo,
}: RepositoryListProps) {
  const [showAddModal, setShowAddModal] = useState(false);
  const queryClient = useQueryClient();

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["repositories"],
    queryFn: contextKeeperAPI.getRepositories,
  });

  const handleAddRepository = async (repoPath: string) => {
    await contextKeeperAPI.startIngestion(repoPath);
    setTimeout(() => {
      refetch();
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    }, 2000);
  };

  const handleDeleteRepository = async (repoPath: string) => {
    if (confirm(`Delete repository ${repoPath}?`)) {
      try {
        await api.delete(`/api/repositories/${encodeURIComponent(repoPath)}`);
        refetch();
        if (selectedRepo === repoPath) {
          onSelectRepo(null);
        }
      } catch (error) {
        console.error("Failed to delete repository:", error);
      }
    }
  };

  const handleClearAll = async () => {
    if (
      confirm(
        "Clear all repositories? This will remove tracking but keep indexed data."
      )
    ) {
      try {
        await api.delete("/api/repositories/clear");
        refetch();
        onSelectRepo(null);
      } catch (error) {
        console.error("Failed to clear repositories:", error);
      }
    }
  };

  const repositories = data?.repositories || {};

  // Filter out duplicates based on path
  const uniqueRepos = Object.entries(repositories).reduce(
    (acc, [path, repo]) => {
      const repoName = path.split(/[/\\]/).pop() || path;
      const existing = acc.find(([_, r]: any) => r.path === path);
      if (!existing || (repo as any).commit_count > 0) {
        acc.push([path, repo]);
      }
      return acc;
    },
    [] as Array<[string, any]>
  );

  const handleCleanup = async () => {
    try {
      const response = await api.post("/api/repositories/cleanup");
      const data = response.data;

      // Refresh stats after cleanup
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["repositories"] });

      // alert(`Cleanup complete: ${data.message}`);
    } catch (error) {
      console.error("Cleanup failed:", error);
    }
  };

  return (
    <>
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-emerald-400" />
            Repositories
          </h3>
          <button
            onClick={handleCleanup}
            className="text-gray-400 hover:text-emerald-400 transition-colors"
            title="Clean up orphaned data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleClearAll}
            className="text-gray-400 hover:text-red-400 transition-colors"
            title="Clear all repositories"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          {/* <button
            onClick={handleClearAll}
            className="text-gray-400 hover:text-red-400 transition-colors"
            title="Clear all repositories"
          >
            <Trash2 className="w-4 h-4" />
          </button> */}
        </div>

        <div className="space-y-2 mb-4 max-h-96 overflow-y-auto">
          {uniqueRepos.map(([path, repo]) => {
            const repoName = path.split(/[/\\]/).pop() || path;
            const isSelected = selectedRepo === path;
            const commitCount = repo.commit_count || 0;

            // Skip repos with 0 commits that are duplicates
            if (
              commitCount === 0 &&
              uniqueRepos.some(
                ([p, r]) =>
                  p !== path && p.includes(repoName) && r.commit_count > 0
              )
            ) {
              return null;
            }

            return (
              <div
                key={path}
                className={`group flex items-center justify-between p-3 rounded-xl transition-all ${
                  isSelected
                    ? "bg-emerald-500/20 border border-emerald-500/50"
                    : "bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700"
                }`}
              >
                <button
                  onClick={() => onSelectRepo(isSelected ? null : path)}
                  className="flex-1 text-left"
                >
                  <div
                    className={`font-medium ${
                      isSelected ? "text-emerald-400" : "text-gray-300"
                    }`}
                  >
                    {repoName}
                  </div>
                  <div className="text-xs mt-1 opacity-70">
                    {commitCount} commits
                  </div>
                </button>

                <button
                  onClick={() => handleDeleteRepository(path)}
                  className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-400 transition-all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>

        <button
          onClick={() => setShowAddModal(true)}
          className="w-full px-4 py-3 bg-gray-800/50 hover:bg-gray-800/70 border border-gray-700 text-gray-300 rounded-xl font-medium transition-all flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Repository
        </button>
      </div>

      <AddRepositoryModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAdd={handleAddRepository}
      />
    </>
  );
}
