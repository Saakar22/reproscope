import { useQuery } from '@tanstack/react-query'
import { NavLink, Outlet } from 'react-router-dom'
import { api } from './api'
import { Badge } from './components/ui'

function HealthIndicators() {
  const { data, isError } = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000 })
  if (isError) return <Badge tone="bad">Backend offline</Badge>
  if (!data) return null
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge
        tone={data.llm_mode === 'live' ? 'ok' : 'warn'}
        title={
          data.llm_mode === 'live'
            ? `LLM: ${data.llm_provider} / ${data.llm_model}`
            : 'No LLM API key configured. LLM stages use labelled development samples, not real analysis.'
        }
      >
        LLM {data.llm_mode === 'live' ? 'live' : 'dev mode'}
      </Badge>
      <Badge
        tone={data.docker.available ? 'ok' : 'bad'}
        title={data.docker.available ? `Docker ${data.docker.version}` : data.docker.error}
      >
        Docker {data.docker.available ? 'ready' : 'unavailable'}
      </Badge>
    </div>
  )
}

const navClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
    isActive ? 'bg-accent-soft text-accent' : 'text-muted hover:text-ink'
  }`

export default function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-rule bg-paper/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="font-serif text-lg font-semibold tracking-tight">ReproScope</span>
          </NavLink>
          <nav className="flex gap-1">
            <NavLink to="/" end className={navClass}>
              New analysis
            </NavLink>
            <NavLink to="/sample-report" className={navClass}>
              Sample report
            </NavLink>
          </nav>
          <div className="ml-auto">
            <HealthIndicators />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6">
        <Outlet />
      </main>
      <footer className="border-t border-rule py-5 text-center text-xs text-faint">
        The LLM proposes; deterministic code verifies. Every number in a report links to its evidence.
      </footer>
    </div>
  )
}
