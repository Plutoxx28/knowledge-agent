
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Brain, Database, GitBranch, Settings, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface MainLayoutProps {
  children: React.ReactNode;
}

const navigation = [{
  name: '处理中心',
  href: '/',
  icon: Brain
}, {
  name: '知识库',
  href: '/knowledge-base',
  icon: Database
}, {
  name: '概念图谱',
  href: '/concept-graph',
  icon: GitBranch
}, {
  name: '设置',
  href: '/settings',
  icon: Settings
}];

export function MainLayout({
  children
}: MainLayoutProps) {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  return <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand - Fixed width */}
            <div className="flex items-center min-w-[250px]">
              <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(!sidebarOpen)} className="md:hidden mr-2">
                <Menu className="h-5 w-5" />
              </Button>
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

            {/* Navigation - Centered */}
            <nav className="hidden md:flex space-x-1 absolute left-1/2 transform -translate-x-1/2">
              {navigation.map(item => {
                const isActive = location.pathname === item.href;
                return <Link key={item.name} to={item.href} className={`flex items-center px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${isActive ? 'bg-blue-100 text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <item.icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Link>;
              })}
            </nav>

            {/* Right side placeholder for balance - Fixed width */}
            <div className="flex items-center space-x-4 text-sm text-gray-600 min-w-[250px] justify-end">
              
            </div>
          </div>
        </div>
      </header>

      {/* Mobile sidebar */}
      {sidebarOpen && <div className="fixed inset-0 z-40 md:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setSidebarOpen(false)} />
          <div className="absolute left-0 top-0 h-full w-64 bg-white shadow-xl">
            <div className="p-4">
              <nav className="space-y-2">
                {navigation.map(item => {
                  const isActive = location.pathname === item.href;
                  return <Link key={item.name} to={item.href} onClick={() => setSidebarOpen(false)} className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-all duration-200 ${isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                      <item.icon className="h-5 w-5 mr-3" />
                      {item.name}
                    </Link>;
                })}
              </nav>
            </div>
          </div>
        </div>}

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
