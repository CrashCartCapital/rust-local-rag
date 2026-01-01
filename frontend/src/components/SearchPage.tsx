import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface SearchResult {
  doc_id: string;
  chunk_id: string;
  score: number;
  text: string;
  metadata: Record<string, unknown>;
  keywords: string[];
}

interface SearchResponse {
  results: SearchResult[];
}

const searchDocuments = async (query: string): Promise<SearchResponse> => {
  const response = await axios.post('/search', {
    query,
    limit: 10
  });
  return response.data;
};

export const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  const { data, isLoading, error, isFetched } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => searchDocuments(debouncedQuery),
    enabled: !!debouncedQuery,
    retry: false,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setDebouncedQuery(query);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Rust Local RAG</h1>
          <p className="text-lg text-gray-600">Search your documents with semantic understanding</p>
        </div>

        <form onSubmit={handleSubmit} className="mb-8 relative">
          <div className="flex shadow-sm rounded-md">
            <input
              type="text"
              className="flex-1 min-w-0 block w-full px-4 py-3 rounded-l-md border border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              placeholder="Ask a question or search for a topic..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button
              type="submit"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-r-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Search
            </button>
          </div>
        </form>

        <div className="space-y-6">
          {isLoading && (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 text-center text-red-700">
              An error occurred while searching. Please try again.
            </div>
          )}

          {!isLoading && isFetched && data?.results.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              No results found for your query.
            </div>
          )}

          {data?.results.map((result) => (
            <div
              key={`${result.doc_id}-${result.chunk_id}`}
              className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex justify-between items-start mb-2">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Score: {result.score.toFixed(3)}
                </span>
                <span className="text-xs text-gray-400 font-mono">{result.doc_id}</span>
              </div>
              <p className="text-gray-800 text-sm leading-relaxed mb-3">{result.text}</p>
              {result.keywords && result.keywords.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {result.keywords.map((keyword, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
