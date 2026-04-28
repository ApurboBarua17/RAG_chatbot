import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/* ─── Types ──────────────────────────────────────────────────── */
interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  attachedFile?: string
}

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const FF       = "'Inter', system-ui, sans-serif"

/* Non-bouncy spring — settles fast like iOS */
const SP = { type: 'spring' as const, stiffness: 420, damping: 42 }
/* Standard ease for fades/slides */
const EA = { duration: 0.38, ease: [0.25, 0.1, 0.25, 1] as [number,number,number,number] }

const SUGGESTIONS = [
  { emoji: '🎓', label: 'Graduation Plan',  sub: 'Full CS degree requirements',     prompt: 'What courses do I need to graduate with a CS degree?' },
  { emoji: '🔗', label: 'Prerequisites',    sub: 'Prereq chain for CSC 345',        prompt: 'What are the prerequisites for CSC 345?' },
  { emoji: '⚙️', label: 'Systems Track',   sub: 'Electives that qualify',           prompt: 'Which electives count toward the systems track?' },
  { emoji: '🤖', label: 'BS AI Degree',    sub: 'AI vs CS degree breakdown',        prompt: 'What are the requirements for the BS in Artificial Intelligence?' },
]

/* ─── Shared glass surfaces ──────────────────────────────────── */
const glassWhite: React.CSSProperties = {
  backdropFilter: 'blur(28px) saturate(180%)',
  WebkitBackdropFilter: 'blur(28px) saturate(180%)',
  background: 'rgba(255,255,255,0.72)',
  border: '1px solid rgba(255,255,255,0.58)',
}
const glassNavy: React.CSSProperties = {
  backdropFilter: 'blur(36px) saturate(160%)',
  WebkitBackdropFilter: 'blur(36px) saturate(160%)',
  background: 'rgba(8, 22, 56, 0.80)',
  borderRight: '1px solid rgba(255,255,255,0.08)',
}

/* ─── CS Mark ────────────────────────────────────────────────── */
function CSMark({ size = 32, radius }: { size?: number; radius?: number }) {
  const r = radius ?? Math.round(size * 0.26)
  return (
    <div style={{
      width: size, height: size, borderRadius: r, flexShrink: 0,
      background: 'linear-gradient(148deg, #D4152B 0%, #AB0520 55%, #8A0018 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: '#fff', fontWeight: 900, fontSize: Math.round(size * 0.36),
      fontFamily: "'Arial Black', Arial, sans-serif", letterSpacing: '-0.5px',
      userSelect: 'none',
      boxShadow: `0 ${Math.round(size / 12)}px ${Math.round(size / 3)}px rgba(171,5,32,0.28),
                  inset 0 1px 0 rgba(255,255,255,0.26)`,
    }}>
      CS
    </div>
  )
}

/* ─── Icons ──────────────────────────────────────────────────── */
const PlusIcon       = () => <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
const PaperclipIcon  = () => <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
const SendIcon       = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
const CloseIcon      = () => <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>

/* ─── Typing dots (CSS pulse — no JS animation loop) ─────────── */
function TypingDots() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: '50%',
          background: 'rgba(100,116,139,0.65)',
          animation: `typingPulse 1.5s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
    </div>
  )
}

/* ─── Main ───────────────────────────────────────────────────── */
export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [file, setFile]         = useState<File | null>(null)

  const bottomRef   = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef     = useRef<HTMLInputElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  /* Auto-grow textarea */
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 180) + 'px'
  }, [input])

  const sendMessage = useCallback(async (question: string, attachment?: File | null) => {
    const q = question.trim()
    if (!q || loading) return
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: q }
    if (attachment) userMsg.attachedFile = attachment.name
    setMessages(p => [...p, userMsg])
    setInput(''); setFile(null); setLoading(true)
    try {
      let res: Response
      if (attachment) {
        const fd = new FormData(); fd.append('question', q); fd.append('file', attachment)
        res = await fetch(`${API_BASE}/chat`, { method: 'POST', body: fd })
      } else {
        res = await fetch(`${API_BASE}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) })
      }
      const data = await res.json()
      setMessages(p => [...p, { id: Date.now().toString(), role: 'assistant', content: data.answer, sources: data.sources ?? [] }])
    } catch {
      setMessages(p => [...p, { id: Date.now().toString(), role: 'assistant', content: 'Could not reach the server. Make sure the backend is running.' }])
    } finally { setLoading(false); textareaRef.current?.focus() }
  }, [loading])

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input, file) }
  }
  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (f) setFile(f); e.target.value = ''
  }
  const canSend = input.trim().length > 0 && !loading

  return (
    /*
     * Page-level gradient — this is what bleeds through the glass.
     * Keep it subtle; the glass effect only reads well against colour.
     */
    <div style={{
      display: 'flex', height: '100vh', overflow: 'hidden', fontFamily: FF,
      background: `
        radial-gradient(ellipse 75% 55% at 12% 12%, rgba(171,5,32,0.10)  0%, transparent 55%),
        radial-gradient(ellipse 65% 70% at 88% 88%, rgba(12,35,75,0.10)   0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 50% 45%, rgba(30,82,136,0.05)  0%, transparent 60%),
        linear-gradient(160deg, #f2f5fb 0%, #fdf5f6 45%, #f4f4fd 100%)
      `,
    }}>

      {/* ══ SIDEBAR ══════════════════════════════════════════════ */}
      <motion.aside
        initial={{ x: -256, opacity: 0 }}
        animate={{ x: 0,    opacity: 1 }}
        transition={{ ...EA, duration: 0.45 }}
        style={{
          width: 252, flexShrink: 0,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
          ...glassNavy,
          boxShadow: '4px 0 32px rgba(0,0,0,0.10), inset -1px 0 0 rgba(255,255,255,0.06)',
        }}
      >
        {/* Logo row */}
        <motion.div
          initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }}
          transition={{ ...EA, delay: 0.15 }}
          style={{ padding: '20px 16px 14px', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <CSMark size={36} radius={10} />
            <div>
              <div style={{ color: 'rgba(255,255,255,0.92)', fontWeight: 700, fontSize: 14, lineHeight: 1.2 }}>
                CS Degree Planner
              </div>
              <div style={{ color: 'rgba(255,255,255,0.30)', fontSize: 10.5, marginTop: 2, fontFamily: "'SF Mono', ui-monospace, monospace" }}>
                Powered by Groq
              </div>
            </div>
          </div>
        </motion.div>

        {/* New Chat */}
        <div style={{ padding: '14px 12px 10px' }}>
          <motion.button
            whileHover={{ filter: 'brightness(1.10)', scale: 1.015 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => setMessages([])}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '10px 14px',
              background: 'rgba(171,5,32,0.88)',
              backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
              border: '1px solid rgba(255,255,255,0.14)',
              borderRadius: 12, cursor: 'pointer', color: '#fff',
              fontSize: 13, fontWeight: 600, fontFamily: FF,
              boxShadow: '0 4px 16px rgba(171,5,32,0.25), inset 0 1px 0 rgba(255,255,255,0.18)',
            }}
          >
            <PlusIcon /> New Chat
          </motion.button>
        </div>

        {/* Empty history */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '20px 16px', gap: 8 }}>
          <div style={{ fontSize: 26, opacity: 0.14 }}>💬</div>
          <p style={{ color: 'rgba(255,255,255,0.17)', fontSize: 11.5, textAlign: 'center', lineHeight: 1.55, margin: 0 }}>
            No conversations yet.<br />Start by asking a question.
          </p>
        </div>

        {/* Model badge */}
        <div style={{ padding: '12px 14px', borderTop: '1px solid rgba(255,255,255,0.07)' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'rgba(255,255,255,0.055)',
            backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
            borderRadius: 12, padding: '9px 12px',
            border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.07)',
          }}>
            <div style={{ width: 28, height: 28, borderRadius: 8, background: 'rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 14 }}>⚡</div>
            <div>
              <div style={{ color: 'rgba(255,255,255,0.80)', fontSize: 12, fontWeight: 600 }}>Llama 3.3 70B</div>
              <div style={{ color: 'rgba(255,255,255,0.26)', fontSize: 10.5, marginTop: 1 }}>14 400 req / day · free</div>
            </div>
          </div>
        </div>
      </motion.aside>

      {/* ══ MAIN ═════════════════════════════════════════════════ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* Header — glass */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ ...EA, delay: 0.12 }}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            height: 54, padding: '0 24px', flexShrink: 0,
            ...glassWhite,
            borderLeft: 'none', borderRight: 'none', borderTop: 'none',
            borderBottom: '1px solid rgba(255,255,255,0.38)',
            boxShadow: '0 1px 0 rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.03)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 3, height: 18, borderRadius: 2, background: 'linear-gradient(180deg,#AB0520,#0C234B)', opacity: 0.75 }} />
            <span style={{ color: 'rgba(0,0,0,0.32)', fontSize: 13 }}>CS Degree Planner</span>
            <span style={{ color: 'rgba(0,0,0,0.14)' }}>/</span>
            <span style={{ color: '#0C234B', fontSize: 13, fontWeight: 600 }}>
              {messages.length === 0 ? 'New Chat' : 'Session'}
            </span>
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 5,
            background: 'rgba(34,197,94,0.10)', border: '1px solid rgba(34,197,94,0.22)',
            borderRadius: 20, padding: '3px 10px', fontSize: 11.5, color: '#15803D', fontWeight: 600,
          }}>
            <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#22C55E', animation: 'statusPulse 2.5s ease-in-out infinite' }} />
            Ready
          </div>
        </motion.div>

        {/* ── Scrollable area ── */}
        <div style={{ flex: 1, overflowY: 'auto', position: 'relative' }}>
          <AnimatePresence mode="wait">

            {/* ───── WELCOME ───── */}
            {messages.length === 0 && !loading ? (
              <motion.div key="welcome"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                transition={{ duration: 0.28 }}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '100%', textAlign: 'center', padding: '40px 24px' }}
              >
                {/* CS Logo */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.82, y: 8 }}
                  animate={{ opacity: 1, scale: 1,    y: 0 }}
                  transition={{ ...SP, delay: 0.08 }}
                  style={{ marginBottom: 22 }}
                >
                  <CSMark size={78} radius={22} />
                </motion.div>

                {/* Heading */}
                <motion.h1
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ ...EA, delay: 0.16 }}
                  style={{ color: '#0C234B', fontSize: 30, fontWeight: 800, margin: '0 0 10px', letterSpacing: '-0.5px', lineHeight: 1.15 }}
                >
                  What can I help you with?
                </motion.h1>

                {/* Subtitle */}
                <motion.p
                  initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ ...EA, delay: 0.22 }}
                  style={{ color: '#64748B', fontSize: 15, maxWidth: 420, margin: '0 auto 32px', lineHeight: 1.65 }}
                >
                  Ask about prerequisites, graduation requirements, track electives,
                  or anything in the CS catalog. Attach a PDF for custom context.
                </motion.p>

                {/* 2 × 2 glass suggestion cards */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, maxWidth: 540, margin: '0 auto' }}>
                  {SUGGESTIONS.map((s, i) => (
                    <motion.button key={s.prompt}
                      initial={{ opacity: 0, y: 14 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ ...EA, delay: 0.30 + i * 0.06 }}
                      whileHover={{
                        scale: 1.025,
                        boxShadow: '0 14px 40px rgba(0,0,0,0.10), 0 0 0 1px rgba(171,5,32,0.18), inset 0 1px 0 rgba(255,255,255,1)',
                        background: 'rgba(255,255,255,0.86)',
                      }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => sendMessage(s.prompt)}
                      style={{
                        textAlign: 'left', padding: '16px 18px',
                        ...glassWhite,
                        borderRadius: 16, cursor: 'pointer', fontFamily: FF,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.92)',
                        transition: 'background 0.2s',
                      }}
                    >
                      <div style={{ fontSize: 22, marginBottom: 10 }}>{s.emoji}</div>
                      <div style={{ color: '#0C234B', fontSize: 13.5, fontWeight: 700, marginBottom: 3 }}>{s.label}</div>
                      <div style={{ color: '#94A3B8', fontSize: 12 }}>{s.sub}</div>
                    </motion.button>
                  ))}
                </div>

                {/* Feature pills — glass */}
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  transition={{ ...EA, delay: 0.58 }}
                  style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 28, flexWrap: 'wrap' }}
                >
                  {['Source-grounded', 'No hallucination', 'PDF upload', 'RAG pipeline'].map(label => (
                    <span key={label} style={{
                      fontSize: 11.5, color: '#475569', fontWeight: 500,
                      backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                      background: 'rgba(255,255,255,0.62)',
                      border: '1px solid rgba(255,255,255,0.52)',
                      borderRadius: 20, padding: '4px 12px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                    }}>{label}</span>
                  ))}
                </motion.div>
              </motion.div>

            ) : (

              /* ───── MESSAGES ───── */
              <motion.div key="messages"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={{ duration: 0.22 }}
                style={{ maxWidth: 720, margin: '0 auto', padding: '28px 20px 8px', display: 'flex', flexDirection: 'column', gap: 22 }}
              >
                <AnimatePresence initial={false}>
                  {messages.map(msg =>
                    msg.role === 'user' ? (

                      /* User — navy glass bubble */
                      <motion.div key={msg.id}
                        initial={{ opacity: 0, x: 14, scale: 0.98 }}
                        animate={{ opacity: 1, x: 0,  scale: 1 }}
                        transition={SP}
                        style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}
                      >
                        <div style={{ maxWidth: '72%' }}>
                          {msg.attachedFile && (
                            <div style={{
                              display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6,
                              backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
                              background: 'rgba(254,242,242,0.80)', border: '1px solid rgba(254,202,202,0.65)',
                              borderRadius: 10, padding: '4px 10px', fontSize: 11.5, color: '#991B1B',
                              width: 'fit-content', marginLeft: 'auto',
                            }}>📄 {msg.attachedFile}</div>
                          )}
                          <div style={{
                            backdropFilter: 'blur(24px) saturate(160%)',
                            WebkitBackdropFilter: 'blur(24px) saturate(160%)',
                            background: 'rgba(10, 26, 70, 0.88)',
                            border: '1px solid rgba(255,255,255,0.11)',
                            color: 'rgba(255,255,255,0.94)',
                            borderRadius: 18, borderBottomRightRadius: 4,
                            padding: '11px 16px', fontSize: 14, lineHeight: 1.7, whiteSpace: 'pre-wrap',
                            boxShadow: '0 4px 20px rgba(10,26,70,0.18), inset 0 1px 0 rgba(255,255,255,0.10)',
                          }}>{msg.content}</div>
                        </div>
                        <div style={{
                          width: 30, height: 30, borderRadius: '50%', flexShrink: 0, alignSelf: 'flex-end',
                          backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                          background: 'rgba(241,245,249,0.82)', border: '1px solid rgba(255,255,255,0.68)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13,
                          boxShadow: '0 2px 8px rgba(0,0,0,0.07)',
                        }}>👤</div>
                      </motion.div>

                    ) : (

                      /* Assistant — plain text + CS avatar */
                      <motion.div key={msg.id}
                        initial={{ opacity: 0, x: -14, scale: 0.98 }}
                        animate={{ opacity: 1, x: 0,   scale: 1 }}
                        transition={SP}
                        style={{ display: 'flex', gap: 12 }}
                      >
                        <CSMark size={30} radius={9} />
                        <div style={{ flex: 1, minWidth: 0, paddingTop: 4 }}>
                          <p style={{ color: '#1E293B', fontSize: 14.5, lineHeight: 1.8, margin: 0, whiteSpace: 'pre-wrap' }}>
                            {msg.content}
                          </p>
                          {msg.sources && msg.sources.length > 0 && (
                            <motion.div
                              initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                              transition={{ ...EA, delay: 0.14 }}
                              style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}
                            >
                              {msg.sources.map(src => (
                                <span key={src} style={{
                                  display: 'inline-flex', alignItems: 'center', gap: 5,
                                  fontSize: 11, color: '#64748B',
                                  backdropFilter: 'blur(14px)', WebkitBackdropFilter: 'blur(14px)',
                                  background: 'rgba(248,250,252,0.80)',
                                  border: '1px solid rgba(226,232,240,0.75)',
                                  borderRadius: 20, padding: '3px 10px',
                                  fontFamily: "'SF Mono', ui-monospace, monospace",
                                  boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
                                }}>📄 {src}</span>
                              ))}
                            </motion.div>
                          )}
                        </div>
                      </motion.div>
                    )
                  )}
                </AnimatePresence>

                {/* Typing indicator */}
                <AnimatePresence>
                  {loading && (
                    <motion.div key="typing"
                      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.22 }}
                      style={{ display: 'flex', gap: 12 }}
                    >
                      <CSMark size={30} radius={9} />
                      <div style={{
                        ...glassWhite,
                        borderRadius: 14, padding: '12px 16px', alignSelf: 'flex-start',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.92)',
                      }}>
                        <TypingDots />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div ref={bottomRef} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Input bar — glass ── */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          transition={{ ...EA, delay: 0.18 }}
          style={{
            flexShrink: 0,
            ...glassWhite,
            borderLeft: 'none', borderRight: 'none', borderBottom: 'none',
            borderTop: '1px solid rgba(255,255,255,0.38)',
            padding: '14px 20px 18px',
            boxShadow: '0 -1px 0 rgba(0,0,0,0.04), 0 -12px 28px rgba(0,0,0,0.03)',
          }}
        >
          <div style={{ maxWidth: 720, margin: '0 auto' }}>

            {/* Attached file chip */}
            <AnimatePresence>
              {file && (
                <motion.div
                  initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                  animate={{ opacity: 1, height: 'auto', marginBottom: 10 }}
                  exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                  transition={{ duration: 0.22 }}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 7, overflow: 'hidden',
                    backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
                    background: 'rgba(254,242,242,0.82)', border: '1px solid rgba(254,202,202,0.65)',
                    borderRadius: 10, padding: '6px 12px',
                  }}
                >
                  <span style={{ fontSize: 15 }}>📄</span>
                  <span style={{ color: '#991B1B', fontSize: 12.5, fontWeight: 500 }}>{file.name}</span>
                  <button onClick={() => setFile(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#EF4444', display: 'flex', padding: 2, marginLeft: 2 }}><CloseIcon /></button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Textarea — glass container */}
            <div
              onFocusCapture={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor  = 'rgba(171,5,32,0.32)'
                el.style.boxShadow    = '0 0 0 3px rgba(171,5,32,0.08), 0 4px 20px rgba(0,0,0,0.07), inset 0 1px 0 rgba(255,255,255,0.92)'
              }}
              onBlurCapture={e => {
                const el = e.currentTarget as HTMLDivElement
                el.style.borderColor  = 'rgba(255,255,255,0.55)'
                el.style.boxShadow    = '0 4px 20px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.90)'
              }}
              style={{
                backdropFilter: 'blur(24px) saturate(180%)',
                WebkitBackdropFilter: 'blur(24px) saturate(180%)',
                background: 'rgba(255,255,255,0.80)',
                border: '1px solid rgba(255,255,255,0.55)',
                borderRadius: 18, overflow: 'hidden',
                boxShadow: '0 4px 20px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.90)',
                transition: 'border-color 0.2s, box-shadow 0.2s',
              }}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask about courses, prerequisites, degree requirements…"
                disabled={loading}
                rows={1}
                style={{
                  display: 'block', width: '100%', background: 'transparent',
                  border: 'none', outline: 'none', resize: 'none',
                  color: '#0F172A', fontSize: 14, lineHeight: 1.65,
                  padding: '14px 16px 6px', fontFamily: FF,
                  maxHeight: 180, overflowY: 'auto',
                }}
              />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 10px 10px' }}>
                <div>
                  <input ref={fileRef} type="file" accept=".pdf" onChange={handleFileChange} style={{ display: 'none' }} />
                  <motion.button
                    whileHover={{ color: '#AB0520', backgroundColor: 'rgba(171,5,32,0.07)' }}
                    whileTap={{ scale: 0.90 }}
                    onClick={() => fileRef.current?.click()}
                    title="Attach a PDF"
                    style={{ width: 32, height: 32, borderRadius: 8, background: 'transparent', border: 'none', cursor: 'pointer', color: '#94A3B8', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'color 0.15s' }}
                  ><PaperclipIcon /></motion.button>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ color: '#CBD5E1', fontSize: 11, fontFamily: "'SF Mono', monospace" }}>⏎ send · ⇧⏎ newline</span>
                  <motion.button
                    onClick={() => sendMessage(input, file)}
                    disabled={!canSend}
                    whileHover={canSend ? { scale: 1.06, boxShadow: '0 4px 16px rgba(171,5,32,0.28)' } : {}}
                    whileTap={canSend ? { scale: 0.93 } : {}}
                    animate={canSend
                      ? { backgroundColor: '#AB0520', color: '#fff', boxShadow: '0 2px 10px rgba(171,5,32,0.22), inset 0 1px 0 rgba(255,255,255,0.18)' }
                      : { backgroundColor: 'rgba(241,245,249,0.85)', color: '#CBD5E1', boxShadow: 'none' }
                    }
                    transition={{ duration: 0.2 }}
                    style={{ width: 33, height: 33, borderRadius: 10, border: 'none', cursor: canSend ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                  ><SendIcon /></motion.button>
                </div>
              </div>
            </div>

            <p style={{ textAlign: 'center', color: '#94A3B8', fontSize: 11, marginTop: 9, marginBottom: 0 }}>
              Powered by Groq + RAG · Grounded in source documents · No hallucination
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
