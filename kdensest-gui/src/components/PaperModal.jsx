import React from 'react';
import { X, BookOpen, Quote } from 'lucide-react';

export default function PaperModal({ content, onClose }) {
  if (!content) return null;
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
        <div className="px-6 py-4 border-b flex justify-between items-center bg-gray-50">
          <h3 className="font-bold text-gray-800 flex items-center gap-2">
            {content.type === 'abstract'     && <><BookOpen size={18}/> Paper Details</>}
            {content.type === 'bibtex'       && <><Quote size={18}/> BibTeX Export</>}
            {content.type === 'loading_bib'  && 'Fetching from DOI registry...'}
            {content.type === 'error'        && 'Error'}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 transition-colors"><X size={20}/></button>
        </div>
        <div className="p-6 overflow-y-auto">
          {content.type === 'abstract' && (
            <div className="space-y-4">
              <div><span className="font-semibold text-gray-900 block mb-1">Title</span><p className="text-gray-700">{content.paper.title}</p></div>
              <div className="grid grid-cols-2 gap-4">
                <div><span className="font-semibold text-gray-900 block mb-1">Authors</span><p className="text-gray-700 text-sm">{content.paper.author}</p></div>
                <div><span className="font-semibold text-gray-900 block mb-1">Venue</span><p className="text-gray-700 text-sm">{content.paper.journal} ({content.paper.year})</p></div>
              </div>
              <div><span className="font-semibold text-gray-900 block mb-1">Abstract</span><p className="text-gray-700 text-sm leading-relaxed text-justify">{content.paper.abstract}</p></div>
            </div>
          )}
          {content.type === 'bibtex'      && <pre className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs overflow-x-auto whitespace-pre-wrap">{content.content}</pre>}
          {content.type === 'loading_bib' && <div className="flex justify-center py-12"><div className="animate-spin w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full"></div></div>}
          {content.type === 'error'       && <div className="bg-red-50 text-red-700 p-4 rounded-lg border border-red-200">{content.content}</div>}
        </div>
      </div>
    </div>
  );
}
