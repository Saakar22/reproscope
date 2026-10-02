import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './App'
import './index.css'
import NewAnalysis from './pages/NewAnalysis'
import NotFound from './pages/NotFound'
import Pipeline from './pages/Pipeline'
import ReportPage from './pages/ReportPage'

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
})

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <NewAnalysis /> },
      { path: 'jobs/:jobId', element: <Pipeline /> },
      { path: 'jobs/:jobId/report', element: <ReportPage source="job" /> },
      { path: 'sample-report', element: <ReportPage source="fixture" /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
