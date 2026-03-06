// API client for the duplicate video detector backend

const API_BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Scan endpoints
export const api = {
  // Scan
  startScan: (path: string, options?: any) =>
    request<{ id: number; status: string; message: string }>('/scan', {
      method: 'POST',
      body: JSON.stringify({ path, options }),
    }),

  getScanStatus: (scanId: number) =>
    request<any>(`/scan/${scanId}/status`),

  pauseScan: (scanId: number) =>
    request<any>(`/scan/${scanId}/pause`, { method: 'POST' }),

  resumeScan: (scanId: number) =>
    request<any>(`/scan/${scanId}/resume`, { method: 'POST' }),

  stopScan: (scanId: number) =>
    request<any>(`/scan/${scanId}/stop`, { method: 'POST' }),

  cancelScan: (scanId: number) =>
    request<any>(`/scan/${scanId}`, { method: 'DELETE' }),

  listScans: () =>
    request<any[]>('/scans'),

  // Duplicates
  listDuplicates: (params?: {
    page?: number;
    per_page?: number;
    sort_by?: string;
    min_similarity?: number;
    status?: string;
    scan_id?: number;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          searchParams.set(key, String(value));
        }
      });
    }
    return request<any>(`/duplicates?${searchParams.toString()}`);
  },

  getDuplicateGroup: (groupId: number) =>
    request<any>(`/duplicates/${groupId}`),

  resolveGroup: (groupId: number, keepIds: number[], deleteIds: number[], moveToTrash: boolean = true) =>
    request<any>(`/duplicates/${groupId}/resolve`, {
      method: 'POST',
      body: JSON.stringify({
        keep_file_ids: keepIds,
        delete_file_ids: deleteIds,
        move_to_trash: moveToTrash,
      }),
    }),

  // Actions
  deleteFiles: (fileIds: number[], moveToTrash: boolean = true) =>
    request<any>('/delete', {
      method: 'POST',
      body: JSON.stringify({ file_ids: fileIds, move_to_trash: moveToTrash }),
    }),

  autoClean: (moveToTrash: boolean = true, confirm: boolean = false) =>
    request<any>('/auto-clean', {
      method: 'POST',
      body: JSON.stringify({ move_to_trash: moveToTrash, confirm }),
    }),

  // Stats
  getStats: () =>
    request<any>('/stats'),

  // Settings
  getSettings: () =>
    request<any>('/settings'),

  updateSettings: (settings: any) =>
    request<any>('/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),

  // History
  getHistory: (page: number = 1, perPage: number = 50) =>
    request<any>(`/history?page=${page}&per_page=${perPage}`),

  undoDelete: (logId: number) =>
    request<any>(`/history/${logId}/undo`, { method: 'POST' }),

  clearHistory: () =>
    request<any>('/history', { method: 'DELETE' }),

  // GPU status
  getGpuStatus: () =>
    request<any>('/gpu-status'),

  // File browser
  browsePath: (path?: string) => {
    const params = path ? `?path=${encodeURIComponent(path)}` : '';
    return request<any>(`/browse${params}`);
  },
};

export default api;
