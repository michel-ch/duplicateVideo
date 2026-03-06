import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import api from '../services/api';
import type { DuplicateGroup } from '../types';
import ConfirmationModal from '../components/ConfirmationModal';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function DuplicatesList() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [groups, setGroups] = useState<DuplicateGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(() => Number(searchParams.get('page')) || 1);
  const [totalPages, setTotalPages] = useState(0);
  const [total, setTotal] = useState(0);
  const [sortBy, setSortBy] = useState(() => searchParams.get('sort') || 'wasted_space');
  const [statusFilter, setStatusFilter] = useState(() => searchParams.get('status') || 'pending');
  const [showAutoClean, setShowAutoClean] = useState(false);
  const [autoCleanPreview, setAutoCleanPreview] = useState<any>(null);
  const [cleaningInProgress, setCleaningInProgress] = useState(false);

  const loadGroups = async () => {
    setLoading(true);
    try {
      const params: any = { page, per_page: 20, sort_by: sortBy };
      if (statusFilter) params.status = statusFilter;
      const data = await api.listDuplicates(params);
      setGroups(data.items);
      setTotalPages(data.total_pages);
      setTotal(data.total);
    } catch (e) {
      console.error('Failed to load duplicates:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    const params: Record<string, string> = {};
    if (page > 1) params.page = String(page);
    if (sortBy !== 'wasted_space') params.sort = sortBy;
    if (statusFilter) params.status = statusFilter;
    setSearchParams(params, { replace: true });
    loadGroups();
  }, [page, sortBy, statusFilter]);

  const handleAutoClean = async () => {
    try {
      const preview = await api.autoClean(true, false);
      setAutoCleanPreview(preview);
      setShowAutoClean(true);
    } catch (e: any) {
      console.error('Auto-clean preview failed:', e);
    }
  };

  const confirmAutoClean = async () => {
    setCleaningInProgress(true);
    try {
      await api.autoClean(true, true);
      setShowAutoClean(false);
      setAutoCleanPreview(null);
      loadGroups();
    } catch (e: any) {
      console.error('Auto-clean failed:', e);
    }
    setCleaningInProgress(false);
  };

  return (
    <div>
      <div className="page-header">
        <h2>📋 Duplicate Groups</h2>
        <p>{total} duplicate group{total !== 1 ? 's' : ''} found</p>
      </div>

      {/* Filters & Actions */}
      <div className="filters-bar">
        <div className="flex items-center gap-md" style={{ flex: 1 }}>
          <label className="text-muted text-sm text-mono">Sort:</label>
          <select value={sortBy} onChange={(e) => { setSortBy(e.target.value); setPage(1); }}>
            <option value="wasted_space">Wasted Space</option>
            <option value="similarity">Similarity</option>
            <option value="file_count">File Count</option>
            <option value="date">Date</option>
          </select>

          <label className="text-muted text-sm text-mono" style={{ marginLeft: '16px' }}>Status:</label>
          <select value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}>
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="in_queue">In Queue</option>
            <option value="resolved">Resolved</option>
          </select>
        </div>

        <div className="flex gap-sm">
          <button className="btn btn-danger btn-sm" onClick={handleAutoClean}>
            🧹 Auto-Clean All
          </button>
        </div>
      </div>

      {/* Groups List */}
      {loading ? (
        <div className="empty-state">
          <div className="icon"><span className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>
          <h3>Loading duplicate groups...</h3>
        </div>
      ) : groups.length === 0 ? (
        <div className="empty-state">
          <div className="icon">📂</div>
          <h3>No Duplicates Found</h3>
          <p>Run a scan from the Dashboard to detect duplicate videos.</p>
        </div>
      ) : (
        <div className="flex-col gap-md">
          {groups.map((group) => {
            const bestVideo = group.videos.find((v) => v.is_best_quality);
            const thumbVideo = bestVideo || group.videos[0];
            return (
              <div
                key={group.id}
                className="group-card"
                onClick={() => navigate(`/duplicates/${group.id}`, { state: { listSearch: `?${searchParams.toString()}` } })}
              >
                <div className="group-thumb">
                  {thumbVideo?.thumbnail_path ? (
                    <img
                      src={thumbVideo.thumbnail_path}
                      alt="Preview"
                      loading="lazy"
                    />
                  ) : (
                    <span style={{ fontSize: '2rem', color: 'var(--text-muted)' }}>🎬</span>
                  )}
                </div>

                <div className="group-info">
                  <h3>
                    {thumbVideo?.file_name || `Group #${group.id}`}
                    {group.status === 'in_queue' && (
                      <span className="badge badge-yellow" style={{ marginLeft: 8 }}>In Queue</span>
                    )}
                    {group.status === 'resolved' && (
                      <span className="badge badge-green" style={{ marginLeft: 8 }}>Resolved</span>
                    )}
                  </h3>
                  <div className="group-meta">
                    <span>
                      📄 <strong>{group.file_count}</strong> files
                    </span>
                    <span>
                      💾 <strong className="text-red">{formatBytes(group.total_wasted_space)}</strong> wasted
                    </span>
                    <span>
                      🎯 <strong className="text-cyan">{group.similarity_score.toFixed(1)}%</strong> similar
                    </span>
                  </div>
                </div>

                <div className="group-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => navigate(`/duplicates/${group.id}`, { state: { listSearch: `?${searchParams.toString()}` } })}
                  >
                    Review →
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => setPage(page - 1)}>← Prev</button>
          {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
            const p = i + 1;
            return (
              <button
                key={p}
                className={p === page ? 'active' : ''}
                onClick={() => setPage(p)}
              >
                {p}
              </button>
            );
          })}
          {totalPages > 7 && <span className="text-muted">...</span>}
          <button disabled={page >= totalPages} onClick={() => setPage(page + 1)}>Next →</button>
        </div>
      )}

      {/* Auto-Clean Modal */}
      <ConfirmationModal
        isOpen={showAutoClean}
        onClose={() => setShowAutoClean(false)}
        onConfirm={confirmAutoClean}
        title="🧹 Auto-Clean All Duplicates"
        message={
          autoCleanPreview
            ? `This will delete ${autoCleanPreview.total_files} lower-quality files, freeing ${autoCleanPreview.message?.split('freeing ')[1] || formatBytes(autoCleanPreview.total_space)}. Files will be moved to trash.`
            : 'Preparing auto-clean preview...'
        }
        confirmLabel={cleaningInProgress ? 'Cleaning...' : 'Confirm Auto-Clean'}
        confirmVariant="danger"
      />
    </div>
  );
}
