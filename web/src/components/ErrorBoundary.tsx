import { Component, type ErrorInfo, type ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface diagnostic details in DevTools while keeping the UI recoverable
    console.error('GenEuclidean Studio caught render error:', error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="min-h-[40vh] flex items-center justify-center p-6">
          <div className="lab-card rounded-2xl max-w-lg w-full p-6 text-center space-y-3 border border-rose-500/30">
            <div className="text-rose-400 font-bold text-base">
              {this.props.fallbackTitle ?? 'Studio View Encountered an Error'}
            </div>
            <p className="text-xs text-[var(--text-secondary)] font-mono-num break-words">
              {this.state.error.message}
            </p>
            <button
              type="button"
              onClick={() => this.setState({ error: null })}
              className="px-4 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/20 border border-cyan-400/40 text-cyan-200 hover:bg-cyan-500/30 cursor-pointer"
            >
              Retry View
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
