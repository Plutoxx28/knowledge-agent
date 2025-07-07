
import React from 'react';
import { Link } from 'react-router-dom';
import { Brain } from 'lucide-react';

interface MainLayoutProps {
  children: React.ReactNode;
}



export function MainLayout({
  children
}: MainLayoutProps) {

  return <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand - Fixed width */}
            <div className="flex items-center min-w-[250px]">
              <Link to="/" className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Knowledge Agent
                  </h1>
                </div>
              </Link>
            </div>



            {/* Right side placeholder for balance - Fixed width */}
            <div className="flex items-center space-x-4 text-sm text-gray-600 min-w-[250px] justify-end">
              
            </div>
          </div>
        </div>
      </header>



      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="animate-fade-in">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur-md border-t border-gray-200/50 mt-16">
        
      </footer>
    </div>;
}
