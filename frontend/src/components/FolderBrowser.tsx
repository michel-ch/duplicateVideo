import { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import type { BrowseEntry } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
}

export default function FolderBrowser({ isOpen, onClose, onSelect, initialPath }: Props) {
  const [currentPath, setCurrentPath] = useState('');
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const browse = useCallback(async (path?: string) => {
    setLoading(true);
    setError('');
    try {
      const data = await api.browsePath(path || undefined);
      setCurrentPath(data.current_path || '');
      setParentPath(data.parent_path);
      setEntries(data.entries || []);
    } catch (e: any) {
      setError(e.message || 'Failed to browse directory');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (isOpen) {
      browse(initialPath || undefined);
    }
  }, [isOpen, initialPath, browse]);

  if (!isOpen) return null;

  const handleNavigate = (path: string) => {
    browse(path);
  };

  const handleUp = () => {
    if (parentPath !== null) {
      browse(parentPath);
    } else {
      // Go to drive list on Windows
      browse(undefined);
    }
  };

  const handleSelect = () => {
    if (currentPath) {
      onSelect(currentPath);
      onClose();
    }
  };

  // Build breadcrumb segments
  const breadcrumbs: { label: string; path: string }[] = [];
  if (currentPath) {
    const isWindows = currentPath.includes('\\') || /^[A-Z]:/.test(currentPath);
    const sep = isWindows ? '\\' : '/';
    const parts = currentPath.split(sep).filter(Boolean);

    let accumulated = '';
    for (const part of parts) {
      if (!accumulated && isWindows) {
        accumulated = part + '\\';
      } else {
        accumulated = accumulated + (accumulated.endsWith(sep) ? '' : sep) + part;
      }
      breadcrumbs.push({ label: part, path: accumulated });
    }

    if (!isWindows && breadcrumbs.length === 0) {
      breadcrumbs.push({ label: '/', path: '/' });
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="folder-browser-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-title">Select Folder</div>

        {/* Breadcrumb */}
        <div className="folder-browser-breadcrumb">
          <button
            className="breadcrumb-segment breadcrumb-root"
            onClick={() => browse(undefined)}
            title="Go to root"
          >
            💻
          </button>
          {breadcrumbs.map((crumb, i) => (
            <span key={crumb.path} className="breadcrumb-part">
              <span className="breadcrumb-sep">/</span>
              <button
                className={`breadcrumb-segment ${i === breadcrumbs.length - 1 ? 'breadcrumb-active' : ''}`}
                onClick={() => handleNavigate(crumb.path)}
              >
                {crumb.label}
              </button>
            </span>
          ))}
        </div>

        {/* Current path display */}
        {currentPath && (
          <div className="folder-browser-path text-mono text-sm">
            {currentPath}
          </div>
        )}

        {/* Directory list */}
        <div className="folder-browser-list">
          {loading ? (
            <div className="folder-browser-loading">
              <span className="spinner" />
              Loading...
            </div>
          ) : error ? (
            <div className="folder-browser-error">
              {error}
            </div>
          ) : (
            <>
              {/* Up button */}
              {currentPath && (
                <button className="folder-browser-item folder-browser-up" onClick={handleUp}>
                  <span className="folder-icon">⬆️</span>
                  <span className="folder-name">..</span>
                </button>
              )}

              {entries.length === 0 && !currentPath ? (
                <div className="folder-browser-empty">No drives found</div>
              ) : entries.length === 0 ? (
                <div className="folder-browser-empty">No subdirectories</div>
              ) : (
                entries.map((entry) => (
                  <button
                    key={entry.path}
                    className="folder-browser-item"
                    onClick={() => handleNavigate(entry.path)}
                    title={entry.path}
                  >
                    <span className="folder-icon">📁</span>
                    <span className="folder-name">{entry.name}</span>
                  </button>
                ))
              )}
            </>
          )}
        </div>

        {/* Actions */}
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSelect}
            disabled={!currentPath}
          >
            Select This Folder
          </button>
        </div>
      </div>
    </div>
  );
}
