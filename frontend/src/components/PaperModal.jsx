import React from 'react';
import { X, ExternalLink } from 'lucide-react';

export default function PaperModal({ content, onClose }) {
  if (!content) return null;

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center p-6"
      style={{ background: 'rgba(5, 11, 20, 0.72)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="texture-paper w-full max-w-3xl max-h-[86vh] flex flex-col overflow-hidden relative shadow-[10px_10px_0_0_var(--ink)] border border-[var(--ink)] fade-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* corner ticks */}
        <div className="absolute top-0 left-0 w-3 h-3 border-l-2 border-t-2 border-[var(--vermillion)] pointer-events-none" />
        <div className="absolute top-0 right-0 w-3 h-3 border-r-2 border-t-2 border-[var(--vermillion)] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-3 h-3 border-l-2 border-b-2 border-[var(--vermillion)] pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-3 h-3 border-r-2 border-b-2 border-[var(--vermillion)] pointer-events-none" />

        {/* Header */}
        <div className="px-8 pt-7 pb-5 shrink-0 border-b border-[var(--rule-paper-2)] flex justify-between items-start">
          <div>
            <div className="eyebrow text-[var(--ink-dim)]">
              {content.type === 'abstract'    && 'Plate III · Monograph'}
              {content.type === 'bibtex'      && 'Plate III · Citation'}
              {content.type === 'loading_bib' && 'Dispatch · in transit'}
              {content.type === 'error'       && 'Erratum'}
            </div>
            <h3 className="font-display text-[28px] leading-none mt-1 text-[var(--ink)]">
              {content.type === 'abstract'    && 'Paper in Full'}
              {content.type === 'bibtex'      && 'BibTeX Record'}
              {content.type === 'loading_bib' && 'Fetching from the registry…'}
              {content.type === 'error'       && 'Something went amiss'}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="btn-chrome shrink-0"
            title="Close"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="px-8 py-7 overflow-y-auto flex-grow custom-scrollbar">
          {content.type === 'abstract' && (
            <div className="space-y-8">
              <div>
                <div className="eyebrow text-[var(--ink-dim)] mb-2">Title</div>
                <p className="font-display text-[32px] leading-[1.1] text-[var(--ink)]">
                  {content.paper.title}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-8 pt-5 border-t border-[var(--rule-paper)]">
                <div>
                  <div className="eyebrow text-[var(--ink-dim)] mb-2">Authors</div>
                  <p className="text-[16px] text-[var(--ink)] leading-snug">
                    {content.paper.author}
                  </p>
                </div>
                <div>
                  <div className="eyebrow text-[var(--ink-dim)] mb-2">Venue</div>
                  <p className="text-[16px] text-[var(--ink)] leading-snug italic">
                    {content.paper.journal}
                  </p>
                  <p className="font-mono tnum text-[13px] text-[var(--ink-dim)] mt-1">
                    {content.paper.date || content.paper.year}
                  </p>
                </div>
              </div>

              {content.paper.concepts && content.paper.concepts.length > 0 && (
                <div className="pt-5 border-t border-[var(--rule-paper)]">
                  <div className="eyebrow text-[var(--ink-dim)] mb-3">Concepts</div>
                  <div className="flex flex-wrap gap-x-5 gap-y-2">
                    {content.paper.concepts.map((concept, idx) => (
                      <span
                        key={idx}
                        className="text-[15px] text-[var(--ink)] leading-none border-b border-[var(--gold)] pb-0.5"
                      >
                        {concept}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-5 border-t border-[var(--rule-paper)]">
                <div className="eyebrow text-[var(--ink-dim)] mb-3">Abstract</div>
                <p
                  className="text-[16px] leading-[1.7] text-[var(--ink-soft)]"
                  style={{ textAlign: 'justify', hyphens: 'auto' }}
                >
                  {content.paper.abstract || (
                    <span className="text-[var(--ink-dim)] italic">
                      No abstract filed on record.
                    </span>
                  )}
                </p>
              </div>
            </div>
          )}

          {content.type === 'bibtex' && (
            <pre className="bg-[var(--night)] text-[var(--gold)] p-5 border border-[var(--night)] font-mono text-[13px] leading-[1.75] overflow-x-auto whitespace-pre-wrap">
              {content.content}
            </pre>
          )}

          {content.type === 'loading_bib' && (
            <div className="flex flex-col items-center justify-center py-14 gap-4">
              <div className="animate-spin w-10 h-10 border-[3px] border-[var(--ink)] border-t-transparent rounded-full" />
              <span className="text-[15px] text-[var(--ink-dim)] italic">
                the wire is humming…
              </span>
            </div>
          )}

          {content.type === 'error' && (
            <div className="text-[17px] text-[var(--vermillion)] leading-snug p-5 border-l-4 border-[var(--vermillion)] bg-[var(--paper-2)]">
              {content.content}
            </div>
          )}
        </div>

        {/* Footer */}
        {content.type === 'abstract' && (
          <div className="px-8 py-5 border-t border-[var(--rule-paper-2)] bg-[var(--paper-2)] shrink-0 flex justify-between items-center">
            <div className="font-mono tnum text-[13px] text-[var(--ink-dim)]">
              ID · {content.paper.id}
            </div>
            <a
              href={`https://openalex.org/${content.paper.id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="eyebrow text-[var(--ink)] hover:text-[var(--vermillion)] transition-colors flex items-center gap-2 border-b border-[var(--ink)] hover:border-[var(--vermillion)] pb-0.5"
            >
              Open in OpenAlex <ExternalLink size={13} />
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
