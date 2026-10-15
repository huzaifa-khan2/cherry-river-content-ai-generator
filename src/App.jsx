import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import Layout from './components/Layout'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/Dashboard'
import GenerateImage from './pages/GenerateImage'
import GenerateVideo from './pages/GenerateVideo'
import ConceptStudio from './pages/ConceptStudio'
import BrandConcepts from './pages/BrandConcepts'
import Products from './pages/Products'
import Gallery from './pages/Gallery'

function PrivateRoute({ children }) {
  const { user, loading } = useAuth()
  if (loading) return (
    <div className="min-h-screen bg-[#06060A] flex items-center justify-center">
      <div className="w-6 h-6 border-2 border-cherry border-t-transparent rounded-full animate-spin" />
    </div>
  )
  return user ? children : <Navigate to="/login" replace />
}

function AppRoutes() {
  const { user, loading } = useAuth()
  if (loading) return (
    <div className="min-h-screen bg-[#06060A] flex items-center justify-center">
      <div className="w-6 h-6 border-2 border-cherry border-t-transparent rounded-full animate-spin" />
    </div>
  )
  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <Login />} />
      <Route path="/signup" element={user ? <Navigate to="/" replace /> : <Signup />} />
      <Route path="/" element={<PrivateRoute><Layout /></PrivateRoute>}>
        <Route index element={<Dashboard />} />
        <Route path="generate-image" element={<GenerateImage />} />
        <Route path="generate-video" element={<GenerateVideo />} />
        <Route path="concept-studio" element={<ConceptStudio />} />
        <Route path="brand-concepts" element={<BrandConcepts />} />
        <Route path="products" element={<Products />} />
        <Route path="gallery" element={<Gallery />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}

export default App
