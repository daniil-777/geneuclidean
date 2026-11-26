import { Check, Code2, Copy, Download, FileJson, ShieldCheck } from 'lucide-react';
import { useState } from 'react';
import type { CodegenResult } from '../../types/circuit';
import { copyToClipboard, downloadTextFile } from '../../lib/data';

interface CodeExportDrawerProps {
  circuitName: string;
  codegen: CodegenResult;
  onExportJson: () => void;
}

function highlightPythonLine(line: string): React.ReactNode {
  const trimmed = line.trimStart();
  if (trimmed.startsWith('#')) {
    return <span className="text-slate-500 dark:text-slate-400 italic">{line}</span>;
  }
  if (trimmed.startsWith('"""')) {
    return <span className="text-emerald-700 dark:text-emerald-300/90">{line}</span>;
  }

  // Lightweight token highlight for keywords, strings, ge. calls, and numbers
  const regex = /("(?:\\.|[^"\\])*"|\b(?:import|from|as|for|in|range|if|else|and|or|True|False|None|print)\b|\bge\.[A-Za-z0-9_]+\b|\b\d+(?:\.\d+)?\b)/g;
  const parts = line.split(regex);
  return parts.map((part, idx) => {
    if (!part) return null;
    if (part.startsWith('"') && part.endsWith('"')) {
      return (
        <span key={idx} className="text-emerald-700 dark:text-emerald-300">
          {part}
        </span>
      );
    }
    if (/^(import|from|as|for|in|range|if|else|and|or|True|False|None|print)$/.test(part)) {
      return (
        <span key={idx} className="text-violet-600 dark:text-violet-300 font-semibold">
          {part}
        </span>
      );
    }
    if (part.startsWith('ge.')) {
      return (
        <span key={idx} className="text-cyan-700 dark:text-cyan-300 font-semibold">
          {part}
        </span>
      );
    }
    if (/^\d+(?:\.\d+)?$/.test(part)) {
      return (
        <span key={idx} className="text-amber-600 dark:text-amber-300">
          {part}
        </span>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

export function CodeExportDrawer({
  circuitName,
  codegen,
  onExportJson,
}: CodeExportDrawerProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (await copyToClipboard(codegen.pythonCode)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  const handleDownloadPy = () => {
    const slug = circuitName.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
    downloadTextFile(`${slug || 'geneuclidean_circuit'}.py`, codegen.pythonCode + '\n', 'text/x-python');
  };

  const lines = codegen.pythonCode.split('\n');

  return (
    <div className="flex flex-col h-full">
      {/* Header Toolbar */}
      <div className="px-3.5 py-2.5 border-b border-[var(--circuit-node-border)] space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Code2 className="h-3.5 w-3.5 text-cyan-500" />
            <span className="text-xs font-bold text-[var(--text-primary)]">
              Live Python SDK
            </span>
          </div>
          <span className="inline-flex items-center gap-1 font-mono-num text-[9.5px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 border border-emerald-500/25">
            <ShieldCheck className="h-3 w-3" /> SDK v5.1
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="flex-1 h-7 px-2.5 rounded-lg text-[11px] font-semibold bg-cyan-500/14 text-cyan-700 dark:text-cyan-200 border border-cyan-500/35 hover:bg-cyan-500/22 flex items-center justify-center gap-1.5 cursor-pointer"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
            {copied ? 'Copied' : 'Copy Python'}
          </button>
          <button
            type="button"
            onClick={handleDownloadPy}
            className="h-7 px-2.5 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 text-[var(--text-primary)] flex items-center gap-1 cursor-pointer"
          >
            <Download className="h-3 w-3" /> .py
          </button>
          <button
            type="button"
            onClick={onExportJson}
            className="h-7 px-2.5 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 text-[var(--text-primary)] flex items-center gap-1 cursor-pointer"
          >
            <FileJson className="h-3 w-3" /> .json
          </button>
        </div>
      </div>

      {/* Syntax-Highlighted Code Scroll Area */}
      <div
        data-python-sdk-code="true"
        style={{ backgroundColor: 'var(--circuit-code-bg)' }}
        className="flex-1 overflow-auto p-3 font-mono-num text-[11px] leading-[1.6]"
      >
        <pre className="m-0">
          {lines.map((line, i) => (
            <div key={i} className="flex gap-3 hover:bg-slate-500/8 px-1 rounded">
              <span className="w-6 text-right select-none text-[var(--text-muted)]/60 shrink-0">
                {i + 1}
              </span>
              <code className="text-[var(--text-primary)] whitespace-pre">
                {highlightPythonLine(line)}
              </code>
            </div>
          ))}
        </pre>
      </div>

      {/* Footer Digest */}
      <div className="px-3.5 py-2 border-t border-[var(--circuit-node-border)] flex items-center justify-between font-mono-num text-[10px] text-[var(--text-muted)]">
        <span>{lines.length} lines · import geneuclidean as ge</span>
        <span>SHA {codegen.provenanceSha256.slice(0, 8)}</span>
      </div>
    </div>
  );
}
