// v5 - Cinematic redesign
import { useState, useEffect, useRef } from 'react'

const SEEDANCE_COST_PER_SECOND = 0.008
import { Search, Check, Play, Download, RotateCcw, ChevronDown, Loader2, Clock, Package, AlertCircle, ImageOff, Sparkles, Volume2, VolumeX, Zap, Film, ChevronRight } from 'lucide-react'
import { apiFetch } from '../lib/apiFetch'

const API_URL = import.meta.env.VITE_API_URL || ''

const durations = [
  { label: '5s',  value: 5  },
  { label: '10s', value: 10 },
  { label: '15s', value: 15 },
]

const videoFormats = [
  { label: '9:16',  desc: 'Reels',   icon: '▯', aspectClass: 'aspect-[9/16]',  wRatio: 9,  hRatio: 16 },
  { label: '1:1',   desc: 'Square',  icon: '□', aspectClass: 'aspect-square',   wRatio: 1,  hRatio: 1  },
  { label: '16:9',  desc: 'YouTube', icon: '▭', aspectClass: 'aspect-video',    wRatio: 16, hRatio: 9  },
]

const POLL_INTERVAL = 4000

export default function GenerateVideo() {
  const [step, setStep] = useState(1)
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [customPrompt, setCustomPrompt] = useState('')
  const [duration, setDuration] = useState(10)
  const [format, setFormat] = useState('9:16')
  const quality = 'Max'
  const [generating, setGenerating] = useState(false)
  const [generationStatus, setGenerationStatus] = useState('')
  const [videoUrl, setVideoUrl] = useState(null)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [products, setProducts] = useState([])
  const [loadingData, setLoadingData] = useState(true)
  const [muted, setMuted] = useState(true)

  const pollRef = useRef(null)
  const taskIdRef = useRef(null)   // tracks current task ID — updated on auto-retry
  const wmTaskIdRef = useRef(null) // tracks WM task ID across setInterval closure
  const savedRef = useRef(false)   // prevents duplicate DB saves on rapid poll ticks
  const selectedFormat = videoFormats.find(f => f.label === format) || videoFormats[0]

  useEffect(() => {
    apiFetch(`${API_URL}/api/products`)
      .then(r => r.json())
      .then(setProducts)
      .catch(console.error)
      .finally(() => setLoadingData(false))
  }, [])

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const readyProducts = products.filter(p => p.status === 'ready')
  const filteredProducts = readyProducts.filter(p =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.category.toLowerCase().includes(searchQuery.toLowerCase())
  )
  const groupedProducts = filteredProducts.reduce((acc, p) => {
    if (!acc[p.category]) acc[p.category] = []
    acc[p.category].push(p)
    return acc
  }, {})

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    wmTaskIdRef.current = null
  }

  const handleGenerate = async () => {
    setGenerating(true)
    setGenerationStatus('starting')
    setError(null)
    setVideoUrl(null)
    try {
      const startRes = await apiFetch(`${API_URL}/api/generate-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ productId: selectedProduct.id, customPrompt: customPrompt.trim() || null, duration, format, quality }),
      })
      const startData = await startRes.json()
      if (!startRes.ok) throw new Error(startData.error || 'Failed to start generation')
      const { predictionId } = startData
      taskIdRef.current = predictionId
      setGenerationStatus('processing')
      wmTaskIdRef.current = null
      pollRef.current = setInterval(async () => {
        try {
          const wmParam = wmTaskIdRef.current ? `?wm=${wmTaskIdRef.current}` : ''
          const statusRes = await apiFetch(`${API_URL}/api/video-status/${taskIdRef.current}${wmParam}`)
          const statusData = await statusRes.json()
          setGenerationStatus(statusData.status)
          // Track WM task ID so next poll includes it
          if (statusData.status === 'removing-watermark' && statusData.wmTaskId) {
            wmTaskIdRef.current = statusData.wmTaskId
          }
          if (statusData.status === 'retrying' && statusData.taskId) {
            taskIdRef.current = statusData.taskId
            setGenerationStatus('retrying')
          } else if (statusData.status === 'succeeded') {
            stopPolling(); setGenerating(false)
            if (!statusData.videoUrl) { setError('Video generated but failed to save. Please try again.'); return }
            setVideoUrl(statusData.videoUrl)
            if (!savedRef.current) {
              savedRef.current = true
              apiFetch(`${API_URL}/api/generations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  url: statusData.videoUrl,
                  productName: selectedProduct.name,
                  productColor: selectedProduct.color,
                  sceneName: customPrompt.trim() || 'Custom video',
                  type: 'video',
                  cost_usd: parseFloat((duration * SEEDANCE_COST_PER_SECOND).toFixed(3)),
                }),
              }).catch(console.error)
            }
          } else if (statusData.status === 'failed' || statusData.status === 'canceled') {
            stopPolling(); setGenerating(false)
            setError(statusData.error || 'Video generation failed')
          }
        } catch (pollErr) { console.error('Polling error:', pollErr) }
      }, POLL_INTERVAL)
    } catch (err) {
      setGenerating(false)
      setError(err.message)
    }
  }

  const handleReset = () => {
    stopPolling(); setVideoUrl(null); setError(null)
    setGenerationStatus(''); setGenerating(false)
    setCustomPrompt(''); setDuration(10); setStep(1); setSelectedProduct(null)
    savedRef.current = false
  }

  const handleDownload = async () => {
    if (!videoUrl) return
    try {
      const response = await fetch(videoUrl)
      const blob = await response.blob()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = `${selectedProduct?.name || 'video'}.mp4`
      a.click()
      URL.revokeObjectURL(a.href)
    } catch { window.open(videoUrl, '_blank') }
  }

  const statusLabel = {
    starting:             'Starting Seedance 2...',
    queued:               'Queued — starting soon...',
    processing:           'Generating your video...',
    in_progress:          'Rendering frames...',
    retrying:             'Flagged by Seedance — retrying with safer prompt...',
    'removing-watermark': 'Removing watermark...',
    succeeded:            'Done!',
    failed:               'Generation failed',
  }

  if (loadingData) return (
    <div className="flex items-center justify-center h-64">
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border border-cherry/20 animate-ping" style={{ animationDuration: '2s' }} />
        <div className="absolute inset-0 rounded-full border-2 border-transparent animate-spin" style={{ borderTopColor: 'rgba(196,30,58,0.8)', animationDuration: '1s' }} />
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* ── PAGE HEADER ── */}
      <div className="relative overflow-hidden rounded-2xl border border-white/5 bg-gradient-to-br from-[#16060A] via-[#0E0608] to-[#06060A] p-6">
        <div className="absolute inset-0 opacity-30" style={{ background: 'radial-gradient(ellipse at 0% 50%, rgba(196,30,58,0.25) 0%, transparent 60%), radial-gradient(ellipse at 100% 50%, rgba(80,0,20,0.15) 0%, transparent 60%)' }} />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-cherry/40 to-transparent" />
        <div className="relative flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-cherry/15 border border-cherry/25 flex items-center justify-center flex-shrink-0 shadow-[0_0_20px_rgba(196,30,58,0.2)]">
            <Film className="w-5 h-5 text-cherry-light" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">Generate Video</h1>
            <p className="text-sm text-gray-500 mt-0.5">Cinematic product films powered by Seedance 2 Max</p>
          </div>
          <div className="ml-auto hidden sm:flex items-center gap-1.5 bg-green-500/10 border border-green-500/20 rounded-full px-3 py-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse shadow-[0_0_6px_#22C55E]" />
            <span className="text-xs font-medium text-green-400">Seedance 2 Max</span>
          </div>
        </div>
      </div>

      {readyProducts.length === 0 ? (
        <div className="bg-[#0E0E18] rounded-2xl p-12 border border-white/5 flex flex-col items-center justify-center text-center">
          <Package className="w-12 h-12 text-gray-600 mb-4" />
          <h3 className="text-lg font-medium text-gray-400">No products available</h3>
          <p className="text-sm text-gray-600 mt-1 mb-4">Add a product first to start generating videos</p>
          <a href="/products" className="bg-cherry hover:bg-cherry-light text-white font-medium px-5 py-2.5 rounded-xl flex items-center gap-2 transition-colors">
            <Package className="w-4 h-4" /> Go to Products
          </a>
        </div>
      ) : (
        <>
          {/* ── STEPS ── */}
          <div className="flex items-center gap-0">
            {['Product', 'Prompt', 'Options'].map((label, i) => {
              const s = i + 1
              const isActive = step === s
              const isDone = step > s
              return (
                <button
                  key={s}
                  onClick={() => !generating && isDone ? setStep(s) : null}
                  className="flex items-center gap-0"
                >
                  <div className={`flex items-center gap-2.5 px-4 py-2 rounded-xl transition-all ${isActive ? 'bg-cherry/15 border border-cherry/30' : isDone ? 'border border-transparent hover:bg-white/4' : 'border border-transparent'}`}>
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all flex-shrink-0 ${
                      isActive ? 'bg-cherry text-white shadow-[0_0_12px_rgba(196,30,58,0.5)]' :
                      isDone   ? 'bg-cherry/20 text-cherry-light' :
                                 'bg-white/5 text-gray-600'
                    }`}>
                      {isDone ? <Check className="w-3 h-3" /> : s}
                    </div>
                    <span className={`text-sm font-medium hidden sm:block ${isActive ? 'text-white' : isDone ? 'text-gray-400' : 'text-gray-600'}`}>{label}</span>
                  </div>
                  {s < 3 && <ChevronRight className="w-4 h-4 text-gray-700 mx-1 flex-shrink-0" />}
                </button>
              )
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* ── LEFT — WIZARD ── */}
            <div className="lg:col-span-2 space-y-4">

              {/* STEP 1: Product */}
              {step === 1 && (
                <div className="relative rounded-2xl border border-white/6 bg-gradient-to-b from-[#12121E] to-[#0C0C16] p-6">
                  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
                  <h3 className="text-base font-semibold text-white mb-5 flex items-center gap-2">
                    <Package className="w-4 h-4 text-cherry-light" />
                    Select Product
                  </h3>
                  <div className="relative">
                    <button
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                      className={`w-full flex items-center justify-between bg-black/40 rounded-xl px-4 py-3.5 border transition-all ${dropdownOpen ? 'border-cherry/50 shadow-[0_0_20px_rgba(196,30,58,0.1)]' : 'border-white/8 hover:border-white/15'}`}
                    >
                      {selectedProduct ? (
                        <div className="flex items-center gap-3">
                          {selectedProduct.image
                            ? <img src={selectedProduct.image} alt={selectedProduct.name} className="w-9 h-9 rounded-lg object-cover ring-1 ring-white/10" />
                            : <div className="w-9 h-9 rounded-lg ring-1 ring-white/10" style={{ background: selectedProduct.color }} />
                          }
                          <div className="text-left">
                            <p className="text-sm font-medium text-white">{selectedProduct.name}</p>
                            <p className="text-xs text-gray-500">{selectedProduct.category}</p>
                          </div>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-500">Choose a product...</span>
                      )}
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                    </button>

                    {dropdownOpen && (
                      <div className="absolute top-full left-0 right-0 mt-2 bg-[#12121E] rounded-xl border border-white/8 shadow-2xl z-20 max-h-80 overflow-y-auto">
                        <div className="p-3 border-b border-white/5 sticky top-0 bg-[#12121E]">
                          <div className="relative">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
                            <input
                              type="text"
                              placeholder="Search products..."
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              className="w-full bg-black/40 rounded-lg pl-8 pr-3 py-2 text-sm text-white placeholder-gray-600 border border-white/8 focus:border-cherry/40 focus:outline-none"
                            />
                          </div>
                        </div>
                        {Object.entries(groupedProducts).map(([category, items]) => (
                          <div key={category}>
                            <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-widest px-4 py-2 bg-black/20">{category}</p>
                            {items.map(product => (
                              <button
                                key={product.id}
                                onClick={() => { setSelectedProduct(product); setDropdownOpen(false); setStep(2) }}
                                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/4 transition-colors"
                              >
                                {product.image
                                  ? <img src={product.image} alt={product.name} className="w-8 h-8 rounded-lg object-cover" />
                                  : <div className="w-8 h-8 rounded-lg" style={{ background: product.color }} />
                                }
                                <span className="text-sm text-white">{product.name}</span>
                                {!product.image && <span className="text-xs text-amber-500 ml-auto flex items-center gap-1"><ImageOff className="w-3 h-3" /> No image</span>}
                              </button>
                            ))}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {selectedProduct && !selectedProduct.image && (
                    <div className="mt-3 flex items-start gap-2 bg-amber-500/8 border border-amber-500/20 rounded-xl p-3">
                      <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                      <p className="text-xs text-amber-300/80">No product image — add one in Products for best bottle accuracy.</p>
                    </div>
                  )}

                  {selectedProduct && (
                    <button
                      onClick={() => setStep(2)}
                      className="mt-4 w-full bg-cherry/10 hover:bg-cherry/20 border border-cherry/25 hover:border-cherry/40 text-cherry-light font-medium py-3 rounded-xl flex items-center justify-center gap-2 transition-all text-sm"
                    >
                      Continue <ChevronRight className="w-4 h-4" />
                    </button>
                  )}
                </div>
              )}

              {/* STEP 2: Prompt */}
              {step === 2 && (
                <div className="relative overflow-hidden rounded-2xl border border-white/6 bg-gradient-to-b from-[#12121E] to-[#0C0C16] p-6 space-y-5">
                  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
                  <h3 className="text-base font-semibold text-white flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-cherry-light" />
                    Scene Description
                  </h3>

                  {(selectedProduct?.productType === 'can' || selectedProduct?.name?.toLowerCase().includes('can')) && (
                    <div className="flex items-start gap-2.5 bg-emerald-500/8 border border-emerald-500/20 rounded-xl px-4 py-3">
                      <Zap className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                      <p className="text-xs text-emerald-300 leading-relaxed">Flavor preset auto-applied — color world, scene, and props are pre-programmed for this product. Add a director's note below, or leave blank to use the preset.</p>
                    </div>
                  )}

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs text-gray-500">
                        {(selectedProduct?.productType === 'can' || selectedProduct?.name?.toLowerCase().includes('can'))
                          ? 'Optional director\'s note — Claude Opus will layer this on top of the preset'
                          : 'Describe what you want — Claude Opus will enhance it'}
                      </p>
                      <span className={`text-xs ${customPrompt.length > 3200 ? 'text-amber-400' : 'text-gray-600'}`}>{customPrompt.length} / 3500</span>
                    </div>
                    <textarea
                      rows={6}
                      placeholder={(selectedProduct?.productType === 'can' || selectedProduct?.name?.toLowerCase().includes('can'))
                        ? "Optional: e.g. \"focus on condensation droplets dripping\" or \"make the scene feel like golden hour at a festival\" — preset handles the rest"
                        : "e.g. Black void. Single sharp rim light cuts along the bottle like a blade. Bottle begins a slow cinematic 360-degree rotation, deep crimson liquid glowing from within..."}
                      value={customPrompt}
                      onChange={(e) => setCustomPrompt(e.target.value)}
                      maxLength={3500}
                      className="w-full bg-black/40 rounded-xl px-4 py-3.5 text-sm text-white placeholder-gray-600 border border-white/8 focus:border-cherry/40 focus:outline-none resize-none leading-relaxed transition-colors"
                    />
                  </div>

                  <div>
                    <p className="text-xs font-medium text-gray-500 mb-3 uppercase tracking-wider">Duration</p>
                    <div className="flex gap-2">
                      {durations.map(d => (
                        <button
                          key={d.value}
                          onClick={() => setDuration(d.value)}
                          className={`flex-1 flex items-center justify-center py-3 rounded-xl border transition-all ${
                            duration === d.value
                              ? 'border-cherry/50 bg-cherry/10 shadow-[0_0_15px_rgba(196,30,58,0.15)]'
                              : 'border-white/6 bg-black/30 hover:border-white/12'
                          }`}
                        >
                          <span className={`text-sm font-bold ${duration === d.value ? 'text-white' : 'text-gray-400'}`}>{d.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={() => setStep(3)}
                    disabled={!(selectedProduct?.productType === 'can' || selectedProduct?.name?.toLowerCase().includes('can')) && !customPrompt.trim()}
                    className="w-full bg-gradient-to-r from-cherry/80 to-cherry-dark/80 hover:from-cherry hover:to-cherry-dark disabled:opacity-30 disabled:cursor-not-allowed text-white font-semibold py-3.5 rounded-xl flex items-center justify-center gap-2 transition-all shadow-[0_4px_20px_rgba(196,30,58,0.2)] hover:shadow-[0_4px_30px_rgba(196,30,58,0.35)] text-sm"
                  >
                    <ChevronRight className="w-4 h-4" />
                    Continue to Options
                  </button>
                </div>
              )}

              {/* STEP 3: Format + Quality */}
              {step === 3 && (
                <div className="relative overflow-hidden rounded-2xl border border-white/6 bg-gradient-to-b from-[#12121E] to-[#0C0C16] p-6 space-y-6">
                  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
                  <h3 className="text-base font-semibold text-white flex items-center gap-2">
                    <Zap className="w-4 h-4 text-cherry-light" />
                    Output Settings
                  </h3>

                  {/* Format */}
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Format</p>
                    <div className="flex gap-3">
                      {videoFormats.map(f => (
                        <button
                          key={f.label}
                          onClick={() => setFormat(f.label)}
                          className={`flex-1 flex flex-col items-center gap-3 py-5 rounded-xl border transition-all ${
                            format === f.label
                              ? 'border-cherry/50 bg-cherry/8 shadow-[0_0_20px_rgba(196,30,58,0.12)]'
                              : 'border-white/6 bg-black/30 hover:border-white/12 hover:bg-white/3'
                          }`}
                        >
                          {/* Device shape */}
                          <div className="flex items-center justify-center h-10">
                            <div
                              className={`rounded border-2 transition-all ${format === f.label ? 'border-cherry-light shadow-[0_0_8px_rgba(196,30,58,0.4)]' : 'border-gray-600'}`}
                              style={{
                                width:  f.label === '9:16' ? '18px' : f.label === '1:1' ? '28px' : '40px',
                                height: f.label === '9:16' ? '32px' : f.label === '1:1' ? '28px' : '22px',
                              }}
                            />
                          </div>
                          <div className="text-center">
                            <p className={`text-xs font-bold ${format === f.label ? 'text-white' : 'text-gray-400'}`}>{f.label}</p>
                            <p className="text-[10px] text-gray-600 mt-0.5">{f.desc}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Quality */}
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Quality</p>
                    <div className="flex items-center justify-between bg-gradient-to-r from-green-500/8 to-emerald-500/5 border border-green-500/20 rounded-xl px-4 py-3.5">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-green-500/15 flex items-center justify-center">
                          <Zap className="w-4 h-4 text-green-400" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-white">Seedance 2 Max</p>
                          <p className="text-[10px] text-green-400/70">Highest quality — music + watermark-free</p>
                        </div>
                      </div>
                      <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_8px_#22C55E]" />
                    </div>
                  </div>

                  {/* Error */}
                  {error && (
                    <div className="flex items-start gap-3 bg-red-500/8 border border-red-500/25 rounded-xl p-4">
                      <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-red-300">
                          {error.startsWith('OutputAudioRisk') ? 'Audio flagged by Seedance' :
                           error.startsWith('OutputVideoRisk') ? 'Video content flagged' :
                           error.startsWith('Generation timed out') ? 'Seedance timeout' :
                           error.startsWith('Insufficient') ? 'Insufficient credits' :
                           'Generation failed'}
                        </p>
                        <p className="text-xs text-red-400/80 mt-1 leading-relaxed">
                          {error.includes(':') ? error.split(':').slice(1).join(':').trim() : error}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Generate button */}
                  <button
                    onClick={handleGenerate}
                    disabled={generating}
                    className="relative w-full overflow-hidden text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2.5 transition-all disabled:opacity-60 disabled:cursor-not-allowed group"
                    style={{ background: generating ? 'linear-gradient(135deg, #6B0000, #3D0000)' : 'linear-gradient(135deg, #C41E3A, #8B0000, #C41E3A)', backgroundSize: '200% 100%' }}
                  >
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: 'linear-gradient(135deg, rgba(255,255,255,0.05), transparent)' }} />
                    <div className="absolute inset-0 rounded-xl shadow-[0_0_40px_rgba(196,30,58,0.4)] group-hover:shadow-[0_0_60px_rgba(196,30,58,0.6)] transition-shadow" />
                    {generating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin relative z-10" />
                        <span className="relative z-10">{statusLabel[generationStatus] || 'Generating...'}</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 relative z-10" fill="white" />
                        <span className="relative z-10 text-base">Generate Video</span>
                      </>
                    )}
                  </button>
                  <p className="text-[11px] text-gray-600 text-center -mt-2">Powered by Seedance 2 — takes ~3–5 minutes</p>
                </div>
              )}
            </div>

            {/* ── RIGHT — SUMMARY + PREVIEW ── */}
            <div className="space-y-4">

              {/* Summary */}
              <div className="relative overflow-hidden rounded-2xl border border-white/6 bg-gradient-to-b from-[#12121E] to-[#0C0C16]">
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />

                {/* Product image hero */}
                {selectedProduct?.image && (
                  <div className="relative h-32 overflow-hidden">
                    <img src={selectedProduct.image} alt={selectedProduct.name} className="absolute inset-0 w-full h-full object-cover opacity-30 blur-md scale-110" />
                    <div className="absolute inset-0 bg-gradient-to-b from-black/20 to-[#0C0C16]" />
                    <img src={selectedProduct.image} alt={selectedProduct.name} className="absolute bottom-0 left-1/2 -translate-x-1/2 h-28 object-contain drop-shadow-2xl" />
                  </div>
                )}

                <div className="p-4 space-y-3">
                  <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-widest">Summary</p>
                  {[
                    { label: 'Product', value: selectedProduct?.name, dot: selectedProduct?.color },
                    { label: 'Duration', value: duration ? `${duration}s` : null },
                    { label: 'Format', value: format },
                    { label: 'Quality', value: 'Seedance 2 Max', green: true },
                    { label: 'Est. Cost', value: duration ? `~$${(duration * SEEDANCE_COST_PER_SECOND).toFixed(3)}` : null, green: true },
                  ].map(({ label, value, dot, green }) => (
                    <div key={label}>
                      <div className="h-px bg-white/4 mb-3" />
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs text-gray-600">{label}</span>
                        {value ? (
                          <div className="flex items-center gap-1.5">
                            {dot && <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: dot }} />}
                            <span className={`text-xs font-medium ${green ? 'text-green-400' : 'text-white'}`}>{value}</span>
                          </div>
                        ) : <span className="text-xs text-gray-700">—</span>}
                      </div>
                    </div>
                  ))}

                  {customPrompt && (
                    <>
                      <div className="h-px bg-white/4" />
                      <div>
                        <p className="text-[10px] text-gray-600 mb-1.5 uppercase tracking-wider">Prompt</p>
                        <p className="text-xs text-gray-400 leading-relaxed line-clamp-3">{customPrompt}</p>
                      </div>
                    </>
                  )}

                  {generating && (
                    <>
                      <div className="h-px bg-white/4" />
                      <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-cherry animate-pulse shadow-[0_0_6px_rgba(196,30,58,0.8)]" />
                        <span className="text-xs text-cherry-light font-medium">{statusLabel[generationStatus] || 'Processing...'}</span>
                      </div>
                      <div className="h-0.5 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ background: 'linear-gradient(90deg, rgba(196,30,58,0.7), rgba(220,60,80,1), rgba(196,30,58,0.7))', backgroundSize: '200% 100%', animation: 'shimmer 2s linear infinite', width: '65%' }} />
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Preview */}
              <div className="relative overflow-hidden rounded-2xl border border-white/6 bg-[#0C0C16]">
                <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/8 to-transparent" />
                <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                  <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-widest">Preview</p>
                  {videoUrl && (
                    <div className="flex items-center gap-1.5">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-400 shadow-[0_0_6px_#22C55E]" />
                      <span className="text-xs text-green-400 font-medium">Ready</span>
                    </div>
                  )}
                </div>

                <div className={`relative ${selectedFormat.aspectClass} bg-black overflow-hidden max-h-[55vw] lg:max-h-none`}>

                  {/* VIDEO READY */}
                  {videoUrl && (
                    <>
                      <video src={videoUrl} className="absolute inset-0 w-full h-full object-contain" autoPlay loop muted={muted} playsInline />
                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 via-black/40 to-transparent flex items-center gap-2">
                        <button onClick={handleDownload} className="flex-1 bg-white/10 backdrop-blur-sm hover:bg-white/20 rounded-lg py-2 text-xs font-semibold text-white transition-colors flex items-center justify-center gap-1.5 border border-white/10">
                          <Download className="w-3.5 h-3.5" /> MP4
                        </button>
                        <button onClick={() => setMuted(m => !m)} className="bg-white/10 backdrop-blur-sm hover:bg-white/20 rounded-lg p-2 text-white transition-colors border border-white/10">
                          {muted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5 text-green-300" />}
                        </button>
                        <button onClick={handleReset} className="bg-white/10 backdrop-blur-sm hover:bg-white/20 rounded-lg p-2 text-white transition-colors border border-white/10">
                          <RotateCcw className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </>
                  )}

                  {/* GENERATING */}
                  {generating && !videoUrl && (
                    <>
                      {selectedProduct?.image && (
                        <img src={selectedProduct.image} alt="" className="absolute inset-0 w-full h-full object-cover opacity-10 blur-xl scale-125" />
                      )}
                      <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse at center, rgba(196,30,58,0.08) 0%, black 70%)' }} />
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6">
                        {/* Animated rings */}
                        <div className="relative w-14 h-14 flex-shrink-0">
                          <div className="absolute inset-0 rounded-full border border-cherry/15 animate-ping" style={{ animationDuration: '2.5s' }} />
                          <div className="absolute inset-1 rounded-full border border-cherry/10 animate-ping" style={{ animationDuration: '2s', animationDelay: '0.3s' }} />
                          <div className="absolute inset-0 rounded-full border-2 border-transparent animate-spin" style={{ borderTopColor: 'rgba(196,30,58,0.9)', borderRightColor: 'rgba(196,30,58,0.3)', animationDuration: '1.1s' }} />
                          <div className="absolute inset-3 rounded-full bg-cherry/8 flex items-center justify-center">
                            <Sparkles className="w-4 h-4 text-cherry-light" />
                          </div>
                        </div>
                        <div className="text-center space-y-1">
                          <p className="text-xs font-semibold text-white">{statusLabel[generationStatus] || 'Processing...'}</p>
                          <p className="text-[10px] text-gray-600">~3–5 minutes remaining</p>
                        </div>
                        <div className="w-24 h-0.5 bg-white/6 rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ background: 'linear-gradient(90deg, rgba(196,30,58,0.5), rgba(220,80,100,1), rgba(196,30,58,0.5))', backgroundSize: '200% 100%', animation: 'shimmer 1.8s linear infinite', width: '60%' }} />
                        </div>
                      </div>
                    </>
                  )}

                  {/* EMPTY */}
                  {!generating && !videoUrl && (
                    <>
                      {selectedProduct?.image && (
                        <img src={selectedProduct.image} alt="" className="absolute inset-0 w-full h-full object-cover opacity-6 blur-lg scale-110" />
                      )}
                      <div className="absolute inset-0" style={{ background: 'radial-gradient(ellipse at center, rgba(196,30,58,0.03) 0%, black 65%)' }} />
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                        <div className="w-10 h-10 rounded-full bg-white/3 border border-white/6 flex items-center justify-center">
                          <Play className="w-4 h-4 text-gray-700 ml-0.5" />
                        </div>
                        <p className="text-[10px] text-gray-700 text-center px-4 leading-relaxed">
                          {selectedProduct ? 'Complete steps & generate' : 'Select a product to start'}
                        </p>
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black/50 backdrop-blur-sm rounded-full px-2 py-1 border border-white/5">
                        <span className="text-[10px] text-gray-600">{format}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <style>{`
        @keyframes shimmer {
          0%   { background-position: 200% center; }
          100% { background-position: -200% center; }
        }
      `}</style>
    </div>
  )
}
