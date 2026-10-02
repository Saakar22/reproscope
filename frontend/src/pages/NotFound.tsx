import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="py-24 text-center">
      <h1 className="font-serif text-3xl font-semibold">Page not found</h1>
      <p className="mt-2 text-muted">That page doesn't exist.</p>
      <Link to="/" className="mt-6 inline-block text-accent underline underline-offset-4">
        Start a new analysis
      </Link>
    </div>
  )
}
