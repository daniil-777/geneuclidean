import { Check, Code2, Copy, Download, FileJson, ShieldCheck, WrapText } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { CodegenResult } from '../../types/circuit';
import { copyToClipboard, downloadTextFile } from '../../lib/data';

interface CodeExportDrawerProps {
  circuitName: string;
  codegen: CodegenResult;
  selectedNodeId?: string | null;
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

  // Lightweight Xcode-style token highlight: strings, keywords, ge. symbols, numbers
  const regex =
    /("(?:\\.|[^"\\])*"|\b(?:import|from|as|for|in|range|if|else|and|or|True|False|None|print)\b|\bge\.[A-Za-z0-9_.]+\b|\b\d+(?:\.\d+)?\b)/g;
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
  selectedNodeId,
  onExportJson,
}: CodeExportDrawerProps) {
  const [copied, setCopied] = useState(false);
  const [softWrap, setSoftWrap] = useState(true);

  const handleCopy = async () => {
    if (await copyToClipboard(codegen.pythonCode)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  const handleDownloadPy = () => {
    const slug = circuitName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_|_$/g, '');
    downloadTextFile(
      `${slug || 'geneuclidean_circuit'}.py`,
      codegen.pythonCode + '\n',
      'text/x-python',
    );
  };

  const lines = useMemo(() => codegen.pythonCode.split('\n'), [codegen.pythonCode]);

  // Identify line range belonging to the currently selected block for instant visual spotlight
  const highlightedRange = useMemo(() => {
    if (!selectedNodeId) return null;
    const marker = `# [${selectedNodeId}]`;
    const start = lines.findIndex((l) => l.includes(marker));
    if (start < 0) return null;
    let end = start;
    for (let i = start + 1; i < lines.length; i += 1) {
      if (lines[i].trim() === '') break;
      end = i;
    }
    return { start, end };
  }, [lines, selectedNodeId]);

  return (
    <div className="flex flex-col h-full">
      {/* Header Toolbar */}
      <div className="px-3 py-2.5 border-b border-[var(--circuit-node-border)] space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Code2 className="h-3.5 w-3.5 text-cyan-500" />
            <span className="text-xs font-bold text-[var(--text-primary)]">Live Python SDK</span>
          </div>
          <span className="inline-flex items-center gap-1 font-mono-num text-[9.5px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 border border-emerald-500/25">
            <ShieldCheck className="h-3 w-3" /> SDK v5.1
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleCopy}
            className="flex-1 h-7 px-2 rounded-lg text-[11px] font-semibold bg-cyan-500/14 text-cyan-700 dark:text-cyan-200 border border-cyan-500/35 hover:bg-cyan-500/22 flex items-center justify-center gap-1.5 cursor-pointer"
          >
            {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button
            type="button"
            onClick={handleDownloadPy}
            className="h-7 px-2 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 text-[var(--text-primary)] flex items-center gap-1 cursor-pointer"
          >
            <Download className="h-3 w-3" /> .py
          </button>
          <button
            type="button"
            onClick={onExportJson}
            className="h-7 px-2 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 text-[var(--text-primary)] flex items-center gap-1 cursor-pointer"
          >
            <FileJson className="h-3 w-3" /> .json
          </button>
          <button
            type="button"
            onClick={() => setSoftWrap((w) => !w)}
            title={softWrap ? 'Disable soft line wrap' : 'Enable soft line wrap'}
            className={`h-7 w-7 rounded-lg border flex items-center justify-center cursor-pointer ${
              softWrap
                ? 'border-cyan-500/40 bg-cyan-500/12 text-cyan-600 dark:text-cyan-300'
                : 'border-[var(--circuit-node-border)] bg-[var(--bg-surface)] text-[var(--text-muted)]'
            }`}
          >
            <WrapText className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Syntax-Highlighted Code Scroll Area */}
      <div
        data-python-sdk-code="true"
        style={{ backgroundColor: 'var(--circuit-code-bg)' }}
        className="flex-1 overflow-auto p-2.5 font-mono-num text-[10.5px] leading-[1.58]"
      >
        <pre className="m-0">
          {lines.map((line, i) => {
            const isSpotlight =
              highlightedRange !== null &&
              i >= highlightedRange.start &&
              i <= highlightedRange.end;
            return (
              <div
                key={i}
                className={`flex gap-2.5 px-1.5 rounded-sm ${
                  isSpotlight
                    ? 'bg-cyan-500/12 border-l-2 border-cyan-500'
                    : 'hover:bg-slate-500/8 border-l-2 border-transparent'
                }`}
              >
                <span className="w-5 text-right select-none text-[var(--text-muted)]/55 shrink-0">
                  {i + 1}
                </span>
                <code
                  className={`text-[var(--text-primary)] min-w-0 flex-1 ${
                    softWrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'
                  }`}
                >
                  {highlightPythonLine(line)}
                </code>
              </div>
            );
          })}
        </pre>
      </div>

      {/* Footer Digest */}
      <div className="px-3 py-2 border-t border-[var(--circuit-node-border)] flex items-center justify-between font-mono-num text-[10px] text-[var(--text-muted)]">
        <span>{lines.length} lines · PEP-8 SDK</span>
        <span>SHA {codegen.provenanceSha256.slice(0, 8)}</span>
      </div>
    </div>
  );
}
