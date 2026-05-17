# Frontend

React 19 + Vite 7 + React Router 7. TypeScript. No global state library, no UI framework.

Source: [`frontend/src/`](../frontend/src/).

## Layout

```
App.tsx                  ─ BrowserRouter, sidebar, <main> with <Routes>
├── pages/
│   ├── Dashboard.tsx        ─ start scans, list scans, live progress, stats, GPU status
│   ├── DuplicatesList.tsx   ─ paginated list of duplicate groups, filters
│   ├── ComparisonView.tsx   ─ side-by-side review, choose what to keep/delete
│   ├── DeletionQueue.tsx    ─ batch process pending deletions
│   ├── History.tsx          ─ deletion log + undo
│   └── Settings.tsx         ─ thresholds, weights, video extensions, protected paths
├── components/
│   ├── VideoCard.tsx        ─ thumbnail + metadata block, used in ComparisonView
│   ├── ProgressTracker.tsx  ─ stage-by-stage scan UI with pause/resume/stop
│   ├── ConfirmationModal.tsx
│   ├── FolderBrowser.tsx    ─ uses /api/browse to pick a directory
│   ├── MetadataTable.tsx    ─ key/value rows of video properties
│   └── QualityBadge.tsx     ─ "Best" badge with score
├── hooks/
│   ├── useWebSocket.ts      ─ scan progress WS with reconnect
│   └── useScanProgress.ts   ─ thin wrapper combining HTTP polling + WS
├── services/
│   └── api.ts               ─ ALL fetch calls live here
└── types/
    └── index.ts             ─ TypeScript interfaces mirroring Pydantic schemas
```

Every component is a function component. No class components. All state is in `useState` or hooks; no Redux / Zustand / Context.

## Routing

```tsx
<Routes>
  <Route path="/"                   element={<Dashboard />} />
  <Route path="/duplicates"         element={<DuplicatesList />} />
  <Route path="/duplicates/:groupId" element={<ComparisonView />} />
  <Route path="/queue"              element={<DeletionQueue />} />
  <Route path="/history"            element={<History />} />
  <Route path="/settings"           element={<Settings />} />
</Routes>
```

The sidebar uses `<NavLink>` with `({ isActive })` callback styling for the current page.

When the backend is serving the SPA in production, the FastAPI catch-all route returns `index.html` for any unknown path (see `serve_frontend_fallback` in [`backend/main.py`](../backend/main.py)), so hard refreshing on `/duplicates/42` works.

## Real-time scan progress

`useScanProgress(scanId)` is the high-level hook used by Dashboard and ProgressTracker. It composes:

1. `useWebSocket(scanId)` — opens `ws://host/api/scan/{id}/ws` (or `wss://` under HTTPS), parses messages by `type` and dispatches:
   - `error_log` — appended to a capped (200) `errors[]` array. Persists across scan completion until a new scan starts, so the user can review failures from the just-finished run.
   - `pong` — heartbeat, ignored.
   - everything else — treated as `progress`.

   Exposes `{progress, errors, isConnected, clearErrors, disconnect}`. Reconnects with a 2s delay if the connection drops while the scan is non-terminal. The `errorsScanIdRef` gates the error reset so switching to a brand-new scanId clears the prior scan's errors automatically.

2. HTTP polling fallback for when the WS is briefly disconnected (so UI doesn't freeze).

The hook exposes `{scanStatus, wsProgress, errors, clearErrors, isComplete, error, reset}`. The `reset` function is called after completion to clear local state — but ONLY after `loadScans()` + `getStats()` have refreshed, to avoid the brief gap where the progress tracker would vanish before the new state arrives.

### Throttling

The backend sends WS updates only when `progress_percent` advances by ≥ 0.5% (or hits 100%). DB rows update every batch regardless, so `getScanStatus()` polling stays accurate.

## API client

[`services/api.ts`](../frontend/src/services/api.ts) is the **only** place in the codebase that calls `fetch`. Every page imports `api` from here. The base URL is just `/api` so the same code works in dev (Vite proxy) and prod (same-origin static serve).

```ts
const response = await fetch(`${API_BASE}${url}`, { ... });
if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `HTTP ${response.status}`);
}
return response.json();
```

Error envelope handling matches the backend's FastAPI `{ "detail": "..." }` format.

## Vite proxy

[`vite.config.ts`](../frontend/vite.config.ts):

```ts
server: {
  proxy: {
    '/api':       { target: 'http://localhost:9000', changeOrigin: true, ws: true },
    '/thumbnails':{ target: 'http://localhost:9000', changeOrigin: true },
  },
}
```

The `ws: true` flag is essential — without it, WebSocket connections to `/api/scan/{id}/ws` would 404.

## Build

`npm run build` runs:

```
tsc -b && vite build
```

`tsc -b` does a project-references type check (TS configs split into `tsconfig.app.json` and `tsconfig.node.json`); failures abort the Vite build. Output goes to `frontend/dist/` and is what the backend serves in production via the SPA fallback.

`npm run dev -- --port 3000` is the dev server with HMR. The script in `start.bat` launches both backend and frontend in separate windows.

## Common patterns

### Polling stats and scan list

```tsx
useEffect(() => {
  api.getStats().then(setStats).catch(() => {});
  loadScans();

  // Poll at 1.5 s for structural changes (new scans, queued→pending,
  // cancellations).  In-scan progress doesn't need the poll — it streams
  // in over WebSocket and patches the local scans array via a separate
  // useEffect on wsProgress.
  const interval = setInterval(() => {
    api.getStats().then(setStats).catch(() => {});
    loadScans();
  }, 1500);
  return () => clearInterval(interval);
}, [loadScans]);
```

`loadScans()` itself compares the new list against a `prevScansRef` and triggers an extra `getStats()` whenever any scan transitions from an active status to a terminal one — so the duplicate counter / recoverable space updates immediately at the end of a scan instead of waiting for the next poll tick.

It also guards every call with a monotonic sequence counter (`loadScansSeqRef`) and discards responses that arrive out-of-order. Without this, a slow `/api/scans` started at T=0 could land *after* a faster one started at T=1.5 s and overwrite it with stale data, making rows visually "rewind" or disappear.

The first mount uses a `scansLoaded` flag to render a `Loading scans…` placeholder card (spinner + text) until the first response lands. Two early-retry timeouts at 250 ms and 750 ms (both gated on `!scansLoaded`) close the gap between mount and the 1.5 s poll tick, so a hard refresh against a cold-starting backend doesn't sit on an empty page. All error catches in `loadScans`, `getStats`, and `getGpuStatus` log via `console.error` rather than swallowing silently.

### WebSocket-driven scan list patching

```tsx
useEffect(() => {
  if (!wsProgress?.scan_id) return;
  setScans(prev => prev.map(s =>
    s.id === wsProgress.scan_id
      ? { ...s, status: wsProgress.status ?? s.status,
              scanned_files: wsProgress.scanned_files ?? s.scanned_files,
              /* ... etc ... */ }
      : s
  ));
}, [wsProgress]);
```

Keeps the row for the active scan in the scan list (status badge, progress %, scanned/total counters) live without needing the polling tick. Missing fields fall through to the previous values, so an early progress message lacking `duplicate_groups_found` doesn't zero out a previously-set value.

### Scan deletion (history)

The Recent Scans list renders a per-row `✕` button and a `🗑 Clear All` button in the section header. Both hit `DELETE /api/scan/{id}` and `DELETE /api/scans` respectively, gated by `window.confirm` warnings that explicitly call out what gets removed (scan rows + duplicate groups) vs what's kept (the cross-scan `file_cache`, so future re-scans still skip work). Active scans are server-side-rejected with a 400 — for those the user must Stop first.

### Error panel

Dashboard renders a `⚠️ Scan Errors (N)` card under the progress tracker whenever `scanErrors.length > 0`. Each row shows the timestamp, a stage badge (`metadata` / `hashing` / `audio_fp` / `pipeline` / `cache_sweep`), the file path (truncated, full path in tooltip), and the error message. Stays visible briefly after the scan ends so the user can review what went wrong; a `Clear` button drops the in-memory list (does not affect the scan or the backend buffer).

### "Active" vs "terminal" status checks

```tsx
const TERMINAL = ['completed', 'failed', 'stopped'];
const ACTIVE   = ['pending', 'scanning', 'metadata', 'hashing', 'comparing', 'paused'];
```

Defined in `Dashboard.tsx`. The `paused` state is **active** for UI purposes — it's still showing a scan in progress.

## Styling

Plain CSS in `App.css` and `index.css`. No CSS-in-JS, no Tailwind. The sidebar layout is grid-based (see `.app-layout` in `App.css`).

## What's deliberately missing

- **No auth.** Single-user local app.
- **No global error boundary** — errors bubble to the page level and are shown inline.
- **No optimistic updates.** Every mutation waits for the server response before updating UI. Scans are slow but mutations (delete, undo) are fast enough that it doesn't matter.
- **No service worker.** Online-only.
