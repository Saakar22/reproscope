import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../api'
import type { PipelineEvent } from '../types'

const TERMINAL = 'job_finished'

export type StreamState = 'connecting' | 'open' | 'finished' | 'error'

/**
 * Subscribes to a job's Server-Sent Events.
 * live:   streams stored history, then follows the job until it finishes.
 * replay: re-streams a finished job's stored events with scaled timing (no execution).
 */
export function useJobEvents(jobId: string | undefined, mode: 'live' | 'replay', speed = 4) {
  const queryClient = useQueryClient()
  const [events, setEvents] = useState<PipelineEvent[]>([])
  const [state, setState] = useState<StreamState>('connecting')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return
    let source: EventSource | null = null
    let cancelled = false
    const seen = new Set<number>()
    setEvents([])
    setState('connecting')
    setError(null)

    const open = (url: string) => {
      source = new EventSource(url)
      source.onopen = () => !cancelled && setState('open')
      source.addEventListener('pipeline', (msg) => {
        const event = JSON.parse((msg as MessageEvent).data) as PipelineEvent
        if (seen.has(event.seq)) return
        seen.add(event.seq)
        setEvents((prev) => [...prev, event])
        if (mode === 'live' && (event.level === 'stage' || event.message === TERMINAL || event.level === 'error')) {
          queryClient.invalidateQueries({ queryKey: ['job', jobId] })
          queryClient.invalidateQueries({ queryKey: ['jobs'] })
        }
        if (event.message === TERMINAL) {
          source?.close()
          setState('finished')
        }
      })
      source.onerror = () => {
        // EventSource reconnects by itself (resuming via Last-Event-ID). A closed stream
        // after the terminal event is expected; anything else is surfaced.
        if (source?.readyState === EventSource.CLOSED && !cancelled) {
          setState((s) => (s === 'finished' ? s : 'error'))
          setError((e) => e ?? 'Lost connection to the event stream.')
        }
      }
    }

    if (mode === 'replay') {
      api
        .replay(jobId, speed)
        .then((r) => !cancelled && open(r.events_url))
        .catch((e: Error) => {
          if (!cancelled) {
            setState('error')
            setError(e.message)
          }
        })
    } else {
      open(`/api/jobs/${jobId}/events`)
    }
    return () => {
      cancelled = true
      source?.close()
    }
  }, [jobId, mode, speed, queryClient])

  return { events, state, error }
}
