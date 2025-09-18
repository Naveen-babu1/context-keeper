import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Repository {
  path: string;
  first_indexed: string;
  last_updated: string;
  commit_count: number;
  selected?: boolean;
  branch_count?: number;
  branch_stats?: {
    main_commits: number;
    feature_commits: number;
    total_branches: number;
    active_branches: number;
  };
}
export interface Branch {
  name: string;
  commit_count: number;
  last_commit_date: string;
  branch_type: 'main' | 'develop' | 'feature' | 'hotfix' | 'release' | 'other';
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
}

export interface Source {
  type: string;
  content: string;
  timestamp: string;
  author: string;
  commit: string;
  repository: string;
  files_changed: string[];
  primary_branch?: string;
  all_branches?: string[];
  merge_commit?: boolean;
  branch_context?: {
    branch_type: string;
    is_main_branch: boolean;
    is_feature_branch: boolean;
  };
}

export const contextKeeperAPI = {
  // Queries
  query: async (query: string, repository?: string, limit: number = 10): Promise<QueryResponse> => {
    const { data } = await api.post('/api/query', { query, repository, limit });
    return data;
  },

  queryWithBranches: async (
    query: string, 
    repository?: string, 
    branches?: string[], 
    limit: number = 10
  ): Promise<QueryResponse> => {
    const { data } = await api.post('/api/query', { 
      query, 
      repository, 
      branches: branches && branches.length > 0 ? branches : undefined,
      limit 
    });
    return data;
  },


  // Repositories
  getRepositories: async () => {
    const { data } = await api.get('/api/repositories');
    return data;
  },

  getRepositoryBranches: async (repoPath: string): Promise<{ repository: string; branches: Branch[]; total_branches: number }> => {
    const { data } = await api.get(`/api/repositories/${encodeURIComponent(repoPath)}/branches`);
    return data;
  },

  // Stats
  getStats: async () => {
    const { data } = await api.get('/api/stats');
    return data;
  },

  // Health
  getHealth: async () => {
    const { data } = await api.get('/health');
    return data;
  },

  // Ingestion
  startIngestion: async (repoPath: string) => {
    const { data } = await api.post('/api/ingest/start', { repo_path: repoPath });
    return data;
  },

  startMultiBranchIngestion: async (
    repoPath: string, 
    options: {
      branches?: string[];
      allBranches?: boolean;
      activeBranches?: boolean;
      days?: number;
    } = {}
  ) => {
    const { data } = await api.post('/api/ingest/multi-branch', {
      repo_path: repoPath,
      ...options
    });
    return data;
  },

  getIngestionStatus: async () => {
    const { data } = await api.get('/api/ingest/status');
    return data;
  },

  // Commit Analysis
  getCommitDetails: async (commitHash: string) => {
    const { data } = await api.get(`/api/commits/${commitHash}`);
    return data;
  },

  compareCommits: async (commit1: string, commit2: string) => {
    const { data } = await api.post('/api/commits/compare', {
      commit_1: commit1,
      commit_2: commit2
    });
    return data;
  },

  // Branch Analysis
  analyzeBranchActivity: async (repository: string, branches?: string[]) => {
    const { data } = await api.post('/api/analyze/branches', {
      repository,
      branches
    });
    return data;
  },
};