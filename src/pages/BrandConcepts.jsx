import { useState, useEffect } from 'react'
import { Sparkles, Download, RotateCcw, Loader2, AlertCircle, Wine, Beer, Wand2, Info } from 'lucide-react'
import { apiFetch } from '../lib/apiFetch'

const API_URL = import.meta.env.VITE_API_URL || ''

const bottleShapes = [
  { key: 'bottle750', label: '750mL' },
  { key: 'bottle114', label: '1.14L' },
]

export default function BrandConcepts() {
  const [brands, setBrands] = useState([])
  const [loadingBrands, setLoadingBrands] = useState(true)
  const [brandsError, setBrandsError] = useState(null)

  const [brandId, setBrandId] = useState(null)
  const [productType, setProductType] = useState('can')
  const [bottleShapeKey, setBottleShapeKey] = useState('bottle750')
  const [flavorName, setFlavorName] = useState('')
  const [variantCount, setVariantCount] = useState(4)

  const [generating, setGenerating] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [images, setImages] = useState(null)
  const [enhancedPrompt, setEnhancedPrompt] = useState('')

  useEffect(() => {
    apiFetch(`${API_URL}/api/brands`)
      .then(r => r.json())
      .then(data => {
        setBrands(data)
        const withLogo = data.find(b => b.logo_url)
        if (withLogo) setBrandId(withLogo.id)
      })
      .catch(() => setBrandsError('Could not load brands'))
      .finally(() => setLoadingBrands(false))
  }, [])

  const selectedBrand = brands.find(b => b.id === brandId)
  const brandReady = !!selectedBrand?.logo_url
  const isApproximateShape = selectedBrand?.approximateBottleShapes?.includes(bottleShapeKey)

  const handleGenerate = async () => {
    if (!flavorName.trim()) { setError('Enter a flavor name first'); return }
    if (!brandId) { setError('Pick a brand first'); return }
    setGenerating(true); setProgress(0); setError(null); setImages(null); setEnhancedPrompt('')
    try {
      const interval = setInterval(() => {
        setProgress(prev => (prev >= 90 ? 90 : prev + Math.random() * 5 + 2))
      }, 600)
      const response = await apiFetch(`${API_URL}/api/generate-brand-concept`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          flavorName: flavorName.trim(),
          productType,
          brandId,
          bottleShapeKey: productType === 'bottle' ? bottleShapeKey : undefined,
          variantCount: productType === 'can' ? variantCount : undefined,
        }),
      })
      let data
      try {
        data = await response.json()
      } catch {
        clearInterval(interval)
        throw new Error(
          response.status === 504 || !response.ok
            ? 'Generation took too long and timed out on the server — try fewer design ideas (1-2) or a simpler flavor name, then try again.'
            : 'Generation failed — please try again.'
        )
      }
      if (!response.ok || !data.success) { clearInterval(interval); throw new Error(data.error || 'Generation failed') }
      clearInterval(interval)
      setProgress(100)
      setImages(data.images)
      if (data.enhancedPrompt) setEnhancedPrompt(data.enhancedPrompt)
    } catch (err) {
      setError(err.message)
    } finally {
      setGenerating(false)
    }
  }

  const handleReset = () => {
    setImages(null); setEnhancedPrompt(''); setProgress(0); setError(null); setFlavorName('')
  }

  const handleDownload = async (url, name) => {
    try {
      const response = await fetch(url)
      const blob = await response.blob()
      const blobUrl = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = blobUrl
      a.download = `${name || 'brand-concept'}-${Date.now()}.png`
      document.body.appendChild(a); a.click(); document.body.removeChild(a)
      URL.revokeObjectURL(blobUrl)
    } catch { window.open(url, '_blank') }
  }

  if (loadingBrands) {
    return <div className="flex items-center justify-center h-64"><Loader2 className="w-8 h-8 text-cherry-light animate-spin" /></div>
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-white">Brand Concepts</h1>
        <p className="text-gray-400 mt-1">Generate new flavor concepts using your real brand logo and identity</p>
      </div>

      {brandsError && (
        <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-400">{brandsError}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {!images && !generating && (
            <div className="bg-card/60 rounded-2xl p-6 border border-border/50 space-y-6">

              {/* Brand */}
              <div>
                <p className="text-sm font-medium text-gray-400 mb-3">Brand</p>
                <div className="flex flex-wrap gap-2">
                  {brands.map(b => (
                    <button
                      key={b.id}
                      onClick={() => setBrandId(b.id)}
                      className={`px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
                        brandId === b.id
                          ? 'border-cherry/50 bg-cherry/10 text-white shadow-[0_0_15px_rgba(196,30,58,0.15)]'
                          : 'border-border bg-dark text-gray-400 hover:border-gray-600'
                      }`}
                    >
                      {b.name}
                      {!b.logo_url && <span className="ml-2 text-[10px] text-amber-400 uppercase tracking-wider">No logo yet</span>}
                    </button>
                  ))}
                </div>
                {selectedBrand && !brandReady && (
                  <div className="flex items-center gap-2 mt-3 text-xs text-amber-400">
                    <Info className="w-3.5 h-3.5 shrink-0" />
                    No logo uploaded for {selectedBrand.name} yet — concepts can't be generated until one is added.
                  </div>
                )}
              </div>

              {/* Product type */}
              <div>
                <p className="text-sm font-medium text-gray-400 mb-3">Product Type</p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setProductType('can')}
                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl border text-sm font-medium transition-all ${
                      productType === 'can' ? 'border-cherry/50 bg-cherry/10 text-white' : 'border-border bg-dark text-gray-400 hover:border-gray-600'
                    }`}
                  >
                    <Beer className="w-4 h-4" /> Can
                  </button>
                  <button
                    onClick={() => setProductType('bottle')}
                    className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl border text-sm font-medium transition-all ${
                      productType === 'bottle' ? 'border-cherry/50 bg-cherry/10 text-white' : 'border-border bg-dark text-gray-400 hover:border-gray-600'
                    }`}
                  >
                    <Wine className="w-4 h-4" /> Bottle
                  </button>
                </div>
              </div>

              {/* Bottle type (bottles only — shape already fixed per product line) */}
              {productType === 'bottle' && (
                <div>
                  <p className="text-sm font-medium text-gray-400 mb-3">Bottle Type</p>
                  <div className="flex gap-2">
                    {bottleShapes.map(s => (
                      <button
                        key={s.key}
                        onClick={() => setBottleShapeKey(s.key)}
                        className={`flex-1 px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
                          bottleShapeKey === s.key ? 'border-cherry/50 bg-cherry/10 text-white' : 'border-border bg-dark text-gray-400 hover:border-gray-600'
                        }`}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                  {isApproximateShape && (
                    <div className="flex items-center gap-2 mt-3 text-xs text-amber-400">
                      <Info className="w-3.5 h-3.5 shrink-0" />
                      {selectedBrand.name}'s real {bottleShapes.find(s=>s.key===bottleShapeKey)?.label} bottle photo is out of date in the catalog — this shape is an approximation until an updated photo is supplied.
                    </div>
                  )}
                </div>
              )}

              {/* Flavor name */}
              <div>
                <p className="text-sm font-medium text-gray-400 mb-2">Flavor Name</p>
                <input
                  type="text"
                  value={flavorName}
                  onChange={(e) => setFlavorName(e.target.value)}
                  placeholder={productType === 'can' ? 'e.g. Cherry River Cola' : 'e.g. Apple Rum'}
                  className="w-full bg-dark rounded-xl px-4 py-3 text-white placeholder-gray-600 border border-border focus:border-cherry/40 focus:outline-none text-sm"
                />
              </div>

              {/* Variant count (cans only — no fixed shape per flavor, so we generate a few ideas) */}
              {productType === 'can' && (
                <div>
                  <p className="text-sm font-medium text-gray-400 mb-3">Design Ideas</p>
                  <div className="grid grid-cols-4 gap-2">
                    {[1, 2, 3, 4].map(n => (
                      <button
                        key={n}
                        onClick={() => setVariantCount(n)}
                        className={`px-4 py-2.5 rounded-xl border text-sm font-medium transition-all ${
                          variantCount === n ? 'border-cherry/50 bg-cherry/10 text-white' : 'border-border bg-dark text-gray-400 hover:border-gray-600'
                        }`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {error && (
                <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                  <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                  <div><p className="text-sm font-medium text-red-400">Generation failed</p><p className="text-xs text-red-400/70 mt-0.5">{error}</p></div>
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={generating || !flavorName.trim() || !brandReady}
                className="w-full bg-gradient-to-r from-cherry to-cherry-dark hover:opacity-90 text-white font-semibold py-4 rounded-xl flex items-center justify-center gap-2 transition-all shadow-[0_0_30px_rgba(196,30,58,0.2)] disabled:opacity-50"
              >
                <Sparkles className="w-5 h-5" /> Generate Concept{productType === 'can' && variantCount > 1 ? 's' : ''}
              </button>
            </div>
          )}

          {generating && (
            <div className="bg-[#11111C] rounded-2xl p-8 border border-white/5 flex flex-col items-center justify-center gap-4">
              <div className="w-16 h-16 rounded-full border-4 border-cherry/20 border-t-cherry animate-spin" />
              <div className="text-center"><p className="text-base font-semibold text-white">Generating your concept...</p><p className="text-sm text-gray-500 mt-1">This can take up to a minute, especially for multiple design ideas</p></div>
              <div className="w-56 h-1.5 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-gradient-to-r from-cherry-dark to-cherry rounded-full transition-all duration-300" style={{ width: `${Math.min(progress, 100)}%` }} /></div>
              <p className="text-sm text-gray-500">{Math.min(Math.round(progress), 99)}%</p>
            </div>
          )}

          {images && (
            <div className="bg-card/60 backdrop-blur-sm rounded-2xl p-6 border border-border/50 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">{flavorName || 'Concept'}</h3>
                <button onClick={handleReset} className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"><RotateCcw className="w-4 h-4" /> Generate New</button>
              </div>
              {enhancedPrompt && (
                <div className="bg-[#0D0D14] rounded-xl p-4 border border-cherry/20">
                  <div className="flex items-center gap-2 mb-2"><Wand2 className="w-3.5 h-3.5 text-cherry-light" /><span className="text-xs font-semibold text-cherry-light uppercase tracking-wider">Claude Enhanced Prompt</span></div>
                  <p className="text-xs text-gray-400 leading-relaxed">{enhancedPrompt}</p>
                </div>
              )}
              <div className={`grid gap-4 ${images.length > 1 ? 'grid-cols-2' : 'grid-cols-1'}`}>
                {images.map((url, i) => (
                  <div key={i} className="relative rounded-xl overflow-hidden group">
                    <img src={url} alt={`${flavorName} concept ${i + 1}`} className="w-full object-cover rounded-xl" />
                    <button onClick={() => handleDownload(url, flavorName)} className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 backdrop-blur-sm rounded-full p-2 hover:bg-black/80"><Download className="w-4 h-4 text-white" /></button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Summary */}
        <div className="space-y-4">
          <div className="bg-card/60 backdrop-blur-sm rounded-2xl p-5 border border-border/50 space-y-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Summary</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between"><span className="text-sm text-gray-500">Brand</span><span className="text-sm text-white">{selectedBrand?.name || <span className="text-gray-600">Not selected</span>}</span></div>
              <div className="h-px bg-border" />
              <div className="flex items-center justify-between"><span className="text-sm text-gray-500">Type</span><span className="text-sm text-white capitalize">{productType}{productType === 'bottle' ? ` (${bottleShapes.find(s=>s.key===bottleShapeKey)?.label})` : ''}</span></div>
              <div className="h-px bg-border" />
              <div><span className="text-sm text-gray-500">Flavor</span><p className="text-sm text-white mt-1">{flavorName || <span className="text-gray-600">Not written yet</span>}</p></div>
              {productType === 'can' && (
                <>
                  <div className="h-px bg-border" />
                  <div className="flex items-center justify-between"><span className="text-sm text-gray-500">Design ideas</span><span className="text-sm text-white">{variantCount}</span></div>
                </>
              )}
              <div className="h-px bg-border" />
              <div className="flex items-center justify-between"><span className="text-sm text-gray-500">Est. Cost</span><span className="text-sm font-semibold text-green-400">~${(0.055 * (productType === 'can' ? variantCount : 1)).toFixed(3)}</span></div>
            </div>
          </div>
          <div className="bg-[#11111C] rounded-2xl p-4 border border-cherry/10 space-y-3">
            <p className="text-xs font-semibold text-cherry-light uppercase tracking-wider">Real Brand Identity</p>
            <p className="text-xs text-gray-500 leading-relaxed">Unlike Concept Studio's blank-label mode, this applies your ACTUAL logo and brand identity to the mockup — for real packaging concepts ready to show, not placeholders for a designer to fill in.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
