import { useState, useEffect } from 'react';
import api from '../services/api';
import type { AppSettings } from '../types';

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [newExtension, setNewExtension] = useState('');
  const [newProtectedPath, setNewProtectedPath] = useState('');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const data = await api.getSettings();
      setSettings(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleSave = async () => {
    if (!settings) return;
    setSaving(true);
    setSaved(false);
    try {
      await api.updateSettings(settings);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (e: any) {
      console.error(e);
    }
    setSaving(false);
  };

  const updateField = (field: keyof AppSettings, value: any) => {
    if (!settings) return;
    setSettings({ ...settings, [field]: value });
  };

  const addExtension = () => {
    if (!settings || !newExtension.trim()) return;
    const ext = newExtension.startsWith('.') ? newExtension : `.${newExtension}`;
    if (!settings.video_extensions.includes(ext)) {
      updateField('video_extensions', [...settings.video_extensions, ext]);
    }
    setNewExtension('');
  };

  const removeExtension = (ext: string) => {
    if (!settings) return;
    updateField('video_extensions', settings.video_extensions.filter((e) => e !== ext));
  };

  const addProtectedPath = () => {
    if (!settings || !newProtectedPath.trim()) return;
    if (!settings.protected_paths.includes(newProtectedPath)) {
      updateField('protected_paths', [...settings.protected_paths, newProtectedPath]);
    }
    setNewProtectedPath('');
  };

  const removeProtectedPath = (path: string) => {
    if (!settings) return;
    updateField('protected_paths', settings.protected_paths.filter((p) => p !== path));
  };

  if (loading || !settings) {
    return (
      <div className="empty-state">
        <div className="icon"><span className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} /></div>
        <h3>Loading settings...</h3>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2>⚙️ Settings</h2>
        <p>Configure duplicate detection parameters and preferences</p>
      </div>

      {/* Detection Settings */}
      <div className="card mb-lg">
        <h3 className="text-mono mb-lg" style={{ color: 'var(--accent-cyan)' }}>
          🎯 Detection Parameters
        </h3>

        <div className="scan-options-row" style={{ borderTop: 'none', paddingTop: 0, marginTop: 0 }}>
          <div className="input-group">
            <label className="input-label">Similarity Threshold ({settings.similarity_threshold}%)</label>
            <input
              type="range"
              className="range-slider"
              min={50}
              max={100}
              step={1}
              value={settings.similarity_threshold}
              onChange={(e) => updateField('similarity_threshold', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Duration Tolerance ({settings.duration_tolerance}s)</label>
            <input
              type="range"
              className="range-slider"
              min={0.5}
              max={10}
              step={0.5}
              value={settings.duration_tolerance}
              onChange={(e) => updateField('duration_tolerance', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Key Frames ({settings.key_frames_count})</label>
            <input
              type="range"
              className="range-slider"
              min={4}
              max={32}
              step={2}
              value={settings.key_frames_count}
              onChange={(e) => updateField('key_frames_count', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Hash Threshold ({settings.hash_threshold})</label>
            <input
              type="range"
              className="range-slider"
              min={1}
              max={30}
              step={1}
              value={settings.hash_threshold}
              onChange={(e) => updateField('hash_threshold', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Max Concurrent ({settings.max_concurrent})</label>
            <input
              type="range"
              className="range-slider"
              min={1}
              max={16}
              step={1}
              value={settings.max_concurrent}
              onChange={(e) => updateField('max_concurrent', Number(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Quality Weights */}
      <div className="card mb-lg">
        <h3 className="text-mono mb-lg" style={{ color: 'var(--accent-yellow)' }}>
          ⚖️ Quality Scoring Weights
        </h3>

        <div className="scan-options-row" style={{ borderTop: 'none', paddingTop: 0, marginTop: 0 }}>
          <div className="input-group">
            <label className="input-label">Resolution ({(settings.resolution_weight * 100).toFixed(0)}%)</label>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1}
              step={0.05}
              value={settings.resolution_weight}
              onChange={(e) => updateField('resolution_weight', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Bitrate ({(settings.bitrate_weight * 100).toFixed(0)}%)</label>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1}
              step={0.05}
              value={settings.bitrate_weight}
              onChange={(e) => updateField('bitrate_weight', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">Codec ({(settings.codec_weight * 100).toFixed(0)}%)</label>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1}
              step={0.05}
              value={settings.codec_weight}
              onChange={(e) => updateField('codec_weight', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">File Size ({(settings.file_size_weight * 100).toFixed(0)}%)</label>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1}
              step={0.05}
              value={settings.file_size_weight}
              onChange={(e) => updateField('file_size_weight', Number(e.target.value))}
            />
          </div>

          <div className="input-group">
            <label className="input-label">FPS ({(settings.fps_weight * 100).toFixed(0)}%)</label>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1}
              step={0.05}
              value={settings.fps_weight}
              onChange={(e) => updateField('fps_weight', Number(e.target.value))}
            />
          </div>
        </div>
      </div>

      {/* Deletion Settings */}
      <div className="card mb-lg">
        <h3 className="text-mono mb-lg" style={{ color: 'var(--accent-green)' }}>
          🗑️ Deletion Preferences
        </h3>

        <div className="toggle-wrapper mb-lg">
          <div
            className={`toggle ${settings.default_trash_mode ? 'active' : ''}`}
            onClick={() => updateField('default_trash_mode', !settings.default_trash_mode)}
          />
          <span>Default to Trash Mode (recommended)</span>
        </div>
      </div>

      {/* Video Extensions */}
      <div className="card mb-lg">
        <h3 className="text-mono mb-lg" style={{ color: 'var(--accent-purple)' }}>
          📎 Video Extensions
        </h3>
        <div className="flex gap-sm mb-md" style={{ flexWrap: 'wrap' }}>
          {settings.video_extensions.map((ext) => (
            <span key={ext} className="badge badge-purple" style={{ cursor: 'pointer' }} onClick={() => removeExtension(ext)}>
              {ext} ✕
            </span>
          ))}
        </div>
        <div className="input-row">
          <div className="input-group">
            <input
              type="text"
              className="input-field"
              placeholder=".mkv"
              value={newExtension}
              onChange={(e) => setNewExtension(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addExtension()}
            />
          </div>
          <button className="btn btn-secondary btn-sm" onClick={addExtension}>Add</button>
        </div>
      </div>

      {/* Protected Paths */}
      <div className="card mb-lg">
        <h3 className="text-mono mb-lg" style={{ color: 'var(--accent-red)' }}>
          🛡️ Protected Paths
        </h3>
        <p className="text-sm text-muted mb-md">Files in these directories will never be deleted.</p>
        {settings.protected_paths.length > 0 ? (
          <div className="flex-col gap-sm mb-md">
            {settings.protected_paths.map((p) => (
              <div key={p} className="flex items-center gap-md">
                <span className="text-mono text-sm" style={{ flex: 1, color: 'var(--text-secondary)' }}>{p}</span>
                <button className="btn btn-ghost btn-sm text-red" onClick={() => removeProtectedPath(p)}>✕</button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-muted text-sm mb-md">No protected paths configured.</p>
        )}
        <div className="input-row">
          <div className="input-group">
            <input
              type="text"
              className="input-field"
              placeholder="C:\Users\Important\Videos"
              value={newProtectedPath}
              onChange={(e) => setNewProtectedPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addProtectedPath()}
            />
          </div>
          <button className="btn btn-secondary btn-sm" onClick={addProtectedPath}>Add</button>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-between items-center" style={{ padding: 'var(--space-lg) 0' }}>
        {saved && (
          <span className="badge badge-green" style={{ animation: 'fadeIn 200ms ease' }}>
            ✓ Settings saved successfully
          </span>
        )}
        <div style={{ flex: 1 }} />
        <button
          className="btn btn-primary btn-lg"
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : '💾 Save Settings'}
        </button>
      </div>
    </div>
  );
}
