import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Image, Video, Layers, Sparkles, Grid3X3, FolderOpen, Settings, LogOut } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/generate-image', icon: Image, label: 'Generate Image' },
  { to: '/generate-video', icon: Video, label: 'Generate Video' },
  { to: '/concept-studio', icon: Layers, label: 'Concept Studio' },
  { to: '/brand-concepts', icon: Sparkles, label: 'Brand Concepts' },
  { to: '/products', icon: Grid3X3, label: 'Products' },
  { to: '/gallery', icon: FolderOpen, label: 'Gallery' },
]

export default function Sidebar() {
  const { user, signOut } = useAuth()
  const initials = user?.email?.[0]?.toUpperCase() || 'F'
  const displayName = user?.email?.split('@')[0] || 'Francis'

  return (
    <aside className="hidden lg:flex flex-col fixed left-0 top-0 bottom-0 w-64 bg-gradient-to-b from-[#0D0D14] to-[#06060A] border-r border-white/5 z-50">
      {/* Logo area */}
      <div className="flex flex-col items-center py-6 px-4">
        <img
          src="/logo.jpg"
          alt="Cherry River"
          className="h-28 w-full object-contain"
          style={{ filter: 'invert(1)', mixBlendMode: 'screen', opacity: 0.95 }}
        />
        <span className="text-xs font-semibold mt-2 tracking-widest uppercase" style={{ color: '#D4AF37' }}>AI Engine</span>
      </div>

      {/* Gradient divider */}
      <div className="mx-6 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent mb-2" />

      <nav className="flex-1 px-3 py-3 space-y-0.5">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'text-white'
                  : 'text-gray-500 hover:text-gray-200 hover:bg-white/5'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <>
                    {/* Left glow border */}
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 rounded-full bg-cherry shadow-[0_0_8px_#C41E3A]" />
                    {/* Active background */}
                    <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cherry/15 to-transparent" />
                  </>
                )}
                <Icon className={`relative w-5 h-5 flex-shrink-0 ${isActive ? 'text-cherry-light' : ''}`} />
                <span className="relative">{label}</span>
                {isActive && (
                  <div className="relative ml-auto w-1.5 h-1.5 rounded-full bg-cherry-light shadow-[0_0_6px_#E8294A]" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-4">
        <NavLink
          to="/settings"
          className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-gray-500 hover:text-gray-200 hover:bg-white/5 transition-all"
        >
          <Settings className="w-5 h-5" />
          Settings
        </NavLink>

        {/* Gradient divider */}
        <div className="mx-3 my-3 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />

        {/* User profile */}
        <div className="mx-3 flex items-center gap-3">
          <div className="relative">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cherry via-cherry-dark to-[#4a0010] flex items-center justify-center text-sm font-bold text-white shadow-[0_0_12px_rgba(196,30,58,0.3)]">
              {initials}
            </div>
            <div className="absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full bg-green-400 border-2 border-[#0D0D14]" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-white leading-tight truncate">{displayName}</p>
            <p className="text-xs text-gray-500 truncate">{user?.email}</p>
          </div>
          <button
            onClick={signOut}
            title="Sign out"
            className="text-gray-600 hover:text-gray-300 transition-colors flex-shrink-0"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  )
}
