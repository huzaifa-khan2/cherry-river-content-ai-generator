import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Image, Video, Layers, Sparkles, Grid3X3, FolderOpen } from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Home' },
  { to: '/generate-image', icon: Image, label: 'Image' },
  { to: '/generate-video', icon: Video, label: 'Video' },
  { to: '/concept-studio', icon: Layers, label: 'Studio' },
  { to: '/brand-concepts', icon: Sparkles, label: 'Brand' },
  { to: '/products', icon: Grid3X3, label: 'Products' },
  { to: '/gallery', icon: FolderOpen, label: 'Gallery' },
]

export default function MobileNav() {
  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-[#0D0D14]/95 backdrop-blur-2xl border-t border-white/5 z-50 px-2 pb-[env(safe-area-inset-bottom)]">
      <div className="flex justify-around py-1.5">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className="flex flex-col items-center gap-1 min-w-0 flex-1"
          >
            {({ isActive }) => (
              <div className={`flex flex-col items-center gap-1 px-3 py-1.5 rounded-xl transition-all w-full ${
                isActive ? '' : ''
              }`}>
                {/* Active dot indicator */}
                <div className={`w-1 h-1 rounded-full mb-0.5 transition-all duration-200 ${
                  isActive ? 'bg-cherry shadow-[0_0_6px_#C41E3A] opacity-100' : 'opacity-0'
                }`} />
                <div className={`relative flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200 ${
                  isActive
                    ? 'bg-cherry/15'
                    : 'hover:bg-white/5'
                }`}>
                  <Icon className={`w-5 h-5 transition-colors duration-200 ${
                    isActive ? 'text-cherry-light' : 'text-gray-500'
                  }`} />
                </div>
                <span className={`text-xs font-medium transition-colors duration-200 leading-none ${
                  isActive ? 'text-cherry-light' : 'text-gray-600'
                }`}>
                  {label}
                </span>
              </div>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
