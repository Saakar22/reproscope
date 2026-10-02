import { useId, useRef, useState, type DragEvent } from 'react'
import { fmtBytes } from './ui'

interface Props {
  label: string
  hint: string
  accept: string // e.g. ".pdf"
  file: File | null
  onFile: (file: File | null) => void
  maxMb: number
  error?: string | null
}

export default function FileDrop({ label, hint, accept, file, onFile, maxMb, error }: Props) {
  const id = useId()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const take = (f: File | undefined) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith(accept)) {
      setLocalError(`Expected a ${accept} file, got "${f.name}".`)
      return
    }
    if (f.size > maxMb * 1024 * 1024) {
      setLocalError(`"${f.name}" is ${fmtBytes(f.size)}; the limit is ${maxMb} MB.`)
      return
    }
    setLocalError(null)
    onFile(f)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    take(e.dataTransfer.files?.[0])
  }

  const shownError = localError ?? error
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-ink">
        {label}
      </label>
      <div
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
          over ? 'border-accent bg-accent-soft' : shownError ? 'border-bad/50 bg-bad-soft/40' : 'border-rule hover:border-faint'
        }`}
      >
        {file ? (
          <div className="flex items-center gap-3">
            <span className="rounded bg-accent-soft px-2 py-1 font-mono text-xs text-accent">{accept.slice(1).toUpperCase()}</span>
            <div className="text-left">
              <div className="max-w-[18rem] truncate text-sm font-medium">{file.name}</div>
              <div className="text-xs text-faint">{fmtBytes(file.size)}</div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onFile(null)
                if (input.current) input.current.value = ''
              }}
              className="ml-2 rounded px-2 py-1 text-xs text-muted hover:bg-neutral-soft hover:text-ink"
            >
              Remove
            </button>
          </div>
        ) : (
          <>
            <div className="text-sm text-muted">
              <span className="font-medium text-accent">Choose a file</span> or drag it here
            </div>
            <div className="mt-1 text-xs text-faint">{hint}</div>
          </>
        )}
        <input
          id={id}
          ref={input}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => take(e.target.files?.[0])}
        />
      </div>
      {shownError && <div className="mt-1.5 text-xs text-bad">{shownError}</div>}
    </div>
  )
}
