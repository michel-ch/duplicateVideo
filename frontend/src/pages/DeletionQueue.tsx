import { useState, useEffect } from 'react';
import api from '../services/api';
import ConfirmationModal from '../components/ConfirmationModal';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function DeletionQueue() {
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [trashMode, setTrashMode] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    loadPreview();
  }, []);

  const loadPreview = async () => {
    setLoading(true);
    try {
      const data = await api.autoClean(trashMode, false);
      setPreview(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleExecute = async () => {
    setExecuting(true);
    try {
      const res = await api.autoClean(trashMode, true);
      setResult(res);
      setShowConfirm(false);
      loadPreview(); // Refresh
    } catch (e: any) {
      console.error(e);
    }
    setExecuting(false);
  };

  if (loading) {
    return (
      <div className="empty-state">
        <div className="icon"><span className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>
        <h3>Loading deletion queue...</h3>
      </div>
    );
  }

  const files = preview?.files_to_delete || [];
  const totalSpace = preview?.total_space || 0;

  return (
    <div>
      <div className="page-header">
        <h2>🗑️ Deletion Queue</h2>
        <p>Review files staged for deletion</p>
      </div>

      {/* Summary */}
      <div className="deletion-summary mb-xl">
        <div className="summary-item">
          <div className="summary-value text-red">{files.length}</div>
          <div className="summary-label">Files to Delete</div>
        </div>
        <div className="summary-item">
          <div className="summary-value text-cyan">{formatBytes(totalSpace)}</div>
          <div className="summary-label">Space to Recover</div>
        </div>
        <div style={{ flex: 1 }} />
        <div className="flex items-center gap-md">
          <div className="toggle-wrapper">
            <div
              className={`toggle ${trashMode ? 'active' : ''}`}
              onClick={() => setTrashMode(!trashMode)}
            />
            <span className="text-sm">{trashMode ? 'Move to Trash' : 'Permanent Delete'}</span>
          </div>
          <button
            className="btn btn-danger"
            onClick={() => setShowConfirm(true)}
            disabled={files.length === 0}
          >
            Confirm & Execute
          </button>
        </div>
      </div>

      {/* Result message */}
      {result && (
        <div className="card card-accent mb-lg" style={{ borderColor: 'var(--accent-green)' }}>
          <p className="text-green font-bold">
            ✅ Successfully deleted {result.deleted_count} files!
          </p>
          {result.errors?.length > 0 && (
            <p className="text-red mt-md">
              ⚠️ {result.errors.length} error(s) occurred
            </p>
          )}
        </div>
      )}

      {/* File list */}
      {files.length === 0 ? (
        <div className="empty-state">
          <div className="icon">✨</div>
          <h3>Queue is Empty</h3>
          <p>No files are pending deletion. Go to Duplicate Groups to review and select files.</p>
        </div>
      ) : (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>File Path</th>
                <th>Size</th>
                <th>Quality</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file: any, idx: number) => (
                <tr key={idx}>
                  <td className="text-mono text-sm truncate" style={{ maxWidth: 500 }} title={file.path}>
                    {file.path}
                  </td>
                  <td className="text-mono">{formatBytes(file.size)}</td>
                  <td>
                    <span className={`badge ${(file.quality_score || 0) < 40 ? 'badge-red' : 'badge-yellow'}`}>
                      {file.quality_score?.toFixed(1) || 'N/A'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmationModal
        isOpen={showConfirm}
        onClose={() => setShowConfirm(false)}
        onConfirm={handleExecute}
        title={trashMode ? '🗑️ Move to Trash' : '⚠️ Permanent Deletion'}
        message={
          trashMode
            ? `Move ${files.length} files to trash? You can undo this from the History page.`
            : `PERMANENTLY delete ${files.length} files? This cannot be undone!`
        }
        confirmLabel={executing ? 'Deleting...' : `Delete ${files.length} files`}
        confirmVariant="danger"
      />
    </div>
  );
}
