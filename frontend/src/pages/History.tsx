import { useState, useEffect } from 'react';
import api from '../services/api';
import type { DeletionLog } from '../types';
import ConfirmationModal from '../components/ConfirmationModal';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return 'N/A';
  return new Date(dateStr).toLocaleString();
}

export default function History() {
  const [logs, setLogs] = useState<DeletionLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [undoingId, setUndoingId] = useState<number | null>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [clearing, setClearing] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getHistory(1, 50);
      setLogs(data.items);
      setTotal(data.total);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleUndo = async (logId: number) => {
    setUndoingId(logId);
    try {
      await api.undoDelete(logId);
      loadHistory();
    } catch (e: any) {
      console.error('Undo failed:', e);
      alert(`Undo failed: ${e.message}`);
    }
    setUndoingId(null);
  };

  const handleClearHistory = async () => {
    setClearing(true);
    try {
      await api.clearHistory();
      setShowClearConfirm(false);
      loadHistory();
    } catch (e: any) {
      console.error('Clear history failed:', e);
      alert(`Clear history failed: ${e.message}`);
    }
    setClearing(false);
  };

  return (
    <div>
      <div className="page-header">
        <div className="flex items-center" style={{ justifyContent: 'space-between' }}>
          <div>
            <h2>📜 History & Logs</h2>
            <p>{total} deletion record{total !== 1 ? 's' : ''}</p>
          </div>
          {logs.length > 0 && (
            <button className="btn btn-danger btn-sm" onClick={() => setShowClearConfirm(true)}>
              🗑️ Clear History
            </button>
          )}
        </div>
      </div>

      {loading ? (
        <div className="empty-state">
          <div className="icon"><span className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>
          <h3>Loading history...</h3>
        </div>
      ) : logs.length === 0 ? (
        <div className="empty-state">
          <div className="icon">📜</div>
          <h3>No History Yet</h3>
          <p>Deletion history will appear here after you delete files.</p>
        </div>
      ) : (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>File</th>
                <th>Size</th>
                <th>Mode</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id}>
                  <td className="text-mono text-sm">{formatDate(log.deleted_at)}</td>
                  <td className="text-mono text-sm truncate" style={{ maxWidth: 400 }} title={log.original_path}>
                    {log.original_path.split(/[/\\]/).pop()}
                  </td>
                  <td className="text-mono">{formatBytes(log.file_size)}</td>
                  <td>
                    <span className={`badge ${log.deletion_mode === 'trash' ? 'badge-yellow' : 'badge-red'}`}>
                      {log.deletion_mode}
                    </span>
                  </td>
                  <td>
                    {log.is_undone ? (
                      <span className="badge badge-green">Restored</span>
                    ) : (
                      <span className="badge badge-muted">Deleted</span>
                    )}
                  </td>
                  <td>
                    {!log.is_undone && log.deletion_mode === 'trash' && (
                      <button
                        className="btn btn-success btn-sm"
                        onClick={() => handleUndo(log.id)}
                        disabled={undoingId === log.id}
                      >
                        {undoingId === log.id ? '...' : '↩ Undo'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmationModal
        isOpen={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        onConfirm={handleClearHistory}
        title="Clear History"
        message={`Delete all ${total} history record${total !== 1 ? 's' : ''}? This cannot be undone.`}
        confirmLabel={clearing ? 'Clearing...' : 'Clear All'}
        confirmVariant="danger"
      />
    </div>
  );
}
