import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import DuplicatesList from './pages/DuplicatesList';
import ComparisonView from './pages/ComparisonView';
import DeletionQueue from './pages/DeletionQueue';
import History from './pages/History';
import Settings from './pages/Settings';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="logo-icon">🎬</div>
            <h1>VidDup<br/>Detector</h1>
          </div>

          <nav className="sidebar-nav">
            <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="nav-icon">⚡</span>
              <span>Dashboard</span>
            </NavLink>

            <NavLink to="/duplicates" className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="nav-icon">📋</span>
              <span>Duplicates</span>
            </NavLink>

            <NavLink to="/queue" className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="nav-icon">🗑️</span>
              <span>Delete Queue</span>
            </NavLink>

            <NavLink to="/history" className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="nav-icon">📜</span>
              <span>History</span>
            </NavLink>

            <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>
              <span className="nav-icon">⚙️</span>
              <span>Settings</span>
            </NavLink>
          </nav>

          <div className="sidebar-footer">
            v1.0.0 — VidDup
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/duplicates" element={<DuplicatesList />} />
            <Route path="/duplicates/:groupId" element={<ComparisonView />} />
            <Route path="/queue" element={<DeletionQueue />} />
            <Route path="/history" element={<History />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
