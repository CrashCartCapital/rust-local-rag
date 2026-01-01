import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SearchPage } from './components/SearchPage';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SearchPage />
    </QueryClientProvider>
  );
}

export default App;
