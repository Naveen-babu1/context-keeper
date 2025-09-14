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
}

export const contextKeeperAPI = {
  // Queries
  query: async (query: string, repository?: string, limit: number = 10): Promise<QueryResponse> => {
    const { data } = await api.post('/api/query', { query, repository, limit });
    return data;
  },

  // Repositories
  getRepositories: async () => {
    const { data } = await api.get('/api/repositories');
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

  getIngestionStatus: async () => {
    const { data } = await api.get('/api/ingest/status');
    return data;
  },
};