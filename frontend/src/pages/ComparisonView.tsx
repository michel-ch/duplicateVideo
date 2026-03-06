import { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import api from '../services/api';
import type { DuplicateGroup, VideoFile } from '../types';
import VideoCard from '../components/VideoCard';
import MetadataTable from '../components/MetadataTable';
import ConfirmationModal from '../components/ConfirmationModal';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function ComparisonView() {
  const { groupId } = useParams<{ groupId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const listSearch = location.state?.listSearch || '';
  const [group, setGroup] = useState<DuplicateGroup | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedKeep, setSelectedKeep] = useState<Set<number>>(new Set());
  const [selectedDelete, setSelectedDelete] = useState<Set<number>>(new Set());
  const [showConfirm, setShowConfirm] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [expandedMeta, setExpandedMeta] = useState<number | null>(null);

  useEffect(() => {
    if (!groupId) return;
    loadGroup();
  }, [groupId]);

  const loadGroup = async () => {
    setLoading(true);
    try {
      const data = await api.getDuplicateGroup(Number(groupId));
      setGroup(data);
      // Auto-select best as keep
      const keepIds = new Set<number>();
      const deleteIds = new Set<number>();
      data.videos.forEach((v: VideoFile) => {
        if (v.is_best_quality) keepIds.add(v.id);
        else deleteIds.add(v.id);
      });
      setSelectedKeep(keepIds);
      setSelectedDelete(deleteIds);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const toggleKeep = (id: number) => {
    const newKeep = new Set(selectedKeep);
    const newDelete = new Set(selectedDelete);

    if (newKeep.has(id)) {
      newKeep.delete(id);
      newDelete.add(id);
    } else {
      newKeep.add(id);
      newDelete.delete(id);
    }

    // Must keep at least one
    if (newKeep.size === 0) return;

    setSelectedKeep(newKeep);
    setSelectedDelete(newDelete);
  };

  const handleKeepBestDeleteRest = () => {
    if (!group) return;
    const keep = new Set<number>();
    const del = new Set<number>();
    group.videos.forEach((v) => {
      if (v.is_best_quality) keep.add(v.id);
      else del.add(v.id);
    });
    setSelectedKeep(keep);
    setSelectedDelete(del);
    setShowConfirm(true);
  };

  const handleResolve = async () => {
    if (!groupId) return;
    setResolving(true);
    try {
      await api.resolveGroup(
        Number(groupId),
        Array.from(selectedKeep),
        Array.from(selectedDelete),
        true
      );
      navigate(`/duplicates${listSearch}`, { replace: true });
    } catch (e: any) {
      console.error('Resolve failed:', e);
    }
    setResolving(false);
    setShowConfirm(false);
  };

  if (loading) {
    return (
      <div className="empty-state">
        <div className="icon"><span className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>
        <h3>Loading comparison...</h3>
      </div>
    );
  }

  if (!group) {
    return (
      <div className="empty-state">
        <div className="icon">❌</div>
        <h3>Group Not Found</h3>
        <button className="btn btn-secondary" onClick={() => navigate(-1)}>← Back</button>
      </div>
    );
  }

  const wasteSaved = group.videos
    .filter((v) => selectedDelete.has(v.id))
    .reduce((sum, v) => sum + v.file_size, 0);

  return (
    <div>
      <div className="page-header">
        <div className="flex items-center gap-md mb-md">
          <button className="btn btn-ghost btn-sm" onClick={() => navigate(-1)}>
            ← Back
          </button>
          <span className="badge badge-cyan">{group.similarity_score.toFixed(1)}% Similar</span>
          <span className="badge badge-yellow">{group.file_count} files</span>
          {group.status === 'in_queue' && <span className="badge badge-yellow">In Queue</span>}
          {group.status === 'resolved' && <span className="badge badge-green">Resolved</span>}
        </div>
        <h2>🔍 Comparison View — Group #{group.id}</h2>
        <p>Compare duplicates side by side and choose which files to keep</p>
      </div>

      {/* Action bar */}
      <div className="filters-bar mb-lg">
        <div className="flex items-center gap-md" style={{ flex: 1 }}>
          <span className="text-sm text-muted">
            Keeping <strong className="text-green">{selectedKeep.size}</strong> · 
            Deleting <strong className="text-red">{selectedDelete.size}</strong> · 
            Saving <strong className="text-cyan">{formatBytes(wasteSaved)}</strong>
          </span>
        </div>
        <div className="flex gap-sm">
          <button className="btn btn-success btn-sm" onClick={handleKeepBestDeleteRest}>
            ⭐ Keep Best, Delete Rest
          </button>
          <button
            className="btn btn-danger btn-sm"
            onClick={() => setShowConfirm(true)}
            disabled={selectedDelete.size === 0}
          >
            🗑️ Confirm Selection
          </button>
        </div>
      </div>

      {/* Comparison Grid */}
      <div className="comparison-grid">
        {group.videos.map((video) => (
          <div key={video.id}>
            <VideoCard
              video={video}
              onKeep={() => toggleKeep(video.id)}
              onDelete={() => toggleKeep(video.id)}
              selectionMode={
                selectedKeep.has(video.id) ? 'keep' :
                selectedDelete.has(video.id) ? 'delete' :
                null
              }
            />
            <div style={{ marginTop: 'var(--space-sm)' }}>
              <button
                className="btn btn-ghost btn-sm"
                style={{ width: '100%' }}
                onClick={() => setExpandedMeta(expandedMeta === video.id ? null : video.id)}
              >
                {expandedMeta === video.id ? '▼ Hide' : '▶ Show'} Full Metadata
              </button>
              {expandedMeta === video.id && (
                <div className="card" style={{ marginTop: 'var(--space-sm)' }}>
                  <MetadataTable video={video} />
                </div>
              )}
            </div>
            <div style={{ textAlign: 'center', marginTop: 'var(--space-sm)' }}>
              {selectedKeep.has(video.id) ? (
                <span className="badge badge-green">✓ KEEPING</span>
              ) : (
                <span className="badge badge-red">✗ DELETING</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirm}
        onClose={() => setShowConfirm(false)}
        onConfirm={handleResolve}
        title="Confirm Deletion"
        message={`Delete ${selectedDelete.size} file(s) and save ${formatBytes(wasteSaved)}? Files will be moved to trash.`}
        confirmLabel={resolving ? 'Processing...' : 'Delete Selected'}
        confirmVariant="danger"
      />
    </div>
  );
}
