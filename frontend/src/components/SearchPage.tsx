import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'

interface SearchResult {
  text: string
  score: number
  document: string
  page_number: number
  section?: string
  embedding_score?: number
  lexical_score?: number
  reranker_score?: number
}

interface SearchResponse {
  results: SearchResult[]
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [searchTrigger, setSearchTrigger] = useState('')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['search', searchTrigger],
    queryFn: async () => {
      if (!searchTrigger) return { results: [] }
      const response = await axios.post<SearchResponse>('/search', {
        query: searchTrigger,
        top_k: 10,
        diversity_factor: 0.3
      })
      return response.data
    },
    enabled: !!searchTrigger,
  })

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      setSearchTrigger(query)
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSearch} className="flex gap-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search documents..."
          className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 px-4 py-2 border"
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {isError && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-red-700">
                Error searching: {error instanceof Error ? error.message : 'Unknown error'}
              </p>
            </div>
          </div>
        </div>
      )}

      {data && data.results.length === 0 && searchTrigger && !isLoading && (
        <div className="text-center py-12 text-gray-500">
          No results found for "{searchTrigger}"
        </div>
      )}

      {data && data.results.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-medium text-gray-900">
            Results for "{searchTrigger}"
          </h2>
          <ul className="space-y-4">
            {data.results.map((result, idx) => (
              <li key={idx} className="bg-white shadow overflow-hidden rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start">
                  <h3 className="text-sm font-medium text-indigo-600 truncate">
                    {result.document}
                    <span className="text-gray-500 font-normal ml-2">
                       (Page {result.page_number})
                    </span>
                  </h3>
                  <div className="flex flex-col items-end">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Score: {(result.score * 100).toFixed(0)}%
                    </span>
                    <div className="text-xs text-gray-400 mt-1 space-x-2">
                       {result.embedding_score && <span>Sem: {result.embedding_score.toFixed(2)}</span>}
                       {result.lexical_score && result.lexical_score > 0 && <span>Key: {result.lexical_score.toFixed(2)}</span>}
                       {result.reranker_score && <span>Rerank: {result.reranker_score.toFixed(2)}</span>}
                    </div>
                  </div>
                </div>
                {result.section && (
                   <p className="text-xs text-gray-500 mt-1 italic">
                      Section: {result.section}
                   </p>
                )}
                <div className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">
                  {result.text}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
