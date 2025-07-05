
import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  variant?: 'primary' | 'success' | 'warning' | 'info';
  className?: string;
}

const variantColors = {
  primary: 'from-primary-500 to-primary-600 text-white',
  success: 'from-green-500 to-emerald-600 text-white',
  warning: 'from-yellow-500 to-orange-600 text-white',
  info: 'from-purple-500 to-violet-600 text-white',
};

export function StatCard({ title, value, icon, trend, variant = 'primary', className }: StatCardProps) {
  return (
    <Card className={cn(
      "bg-card/70 backdrop-blur-sm border-border/50 shadow-lg transition-all duration-300 hover:shadow-xl",
      "rounded-2xl overflow-hidden",
      className
    )}>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className={cn(
            "w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br shadow-sm",
            variantColors[variant]
          )}>
            {icon}
          </div>
          <div className="flex-1">
            <p className="text-2xl font-bold text-foreground">{value}</p>
            <p className="text-sm text-muted-foreground">{title}</p>
            {trend && (
              <div className={cn(
                "flex items-center mt-1 text-xs",
                trend.isPositive ? 'text-green-600' : 'text-red-600'
              )}>
                <span>{trend.isPositive ? '↗' : '↘'}</span>
                <span className="ml-1">{Math.abs(trend.value)}%</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
