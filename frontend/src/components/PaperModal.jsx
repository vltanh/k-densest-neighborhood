import React from 'react';
import { X, BookOpen, Quote, ExternalLink } from 'lucide-react';

export default function PaperModal({ content, onClose }) {
  if (!content) return null;
  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden">
        
        <div className="px-6 py-4 border-b flex justify-between items-center bg-gray-50 shrink-0">
          <h3 className="font-bold text-gray-800 flex items-center gap-2">
            {content.type === 'abstract'     && <><BookOpen size={18}/> Paper Details</>}
            {content.type === 'bibtex'       && <><Quote size={18}/> BibTeX Export</>}
            {content.type === 'loading_bib'  && 'Fetching from DOI registry...'}
            {content.type === 'error'        && 'Error'}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 transition-colors"><X size={20}/></button>
        </div>
        
        <div className="p-6 overflow-y-auto flex-grow custom-scrollbar">
          {content.type === 'abstract' && (
            <div className="space-y-6">
              <div>
                <span className="font-semibold text-gray-900 block mb-1 text-xs uppercase tracking-wider">Title</span>
                <p className="text-gray-800 font-medium text-lg leading-snug">{content.paper.title}</p>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <span className="font-semibold text-gray-900 block mb-1 text-xs uppercase tracking-wider">Authors</span>
                  <p className="text-gray-700 text-sm">{content.paper.author}</p>
                </div>
                <div>
                  <span className="font-semibold text-gray-900 block mb-1 text-xs uppercase tracking-wider">Venue</span>
                  <p className="text-gray-700 text-sm">{content.paper.journal}</p>
                  <p className="text-gray-500 text-xs mt-0.5">{content.paper.date || content.paper.year}</p>
                </div>
              </div>

              {content.paper.concepts && content.paper.concepts.length > 0 && (
                <div>
                  <span className="font-semibold text-gray-900 block mb-2 text-xs uppercase tracking-wider">Concepts / Topics</span>
                  <div className="flex flex-wrap gap-2">
                    {content.paper.concepts.map((concept, idx) => (
                       <span key={idx} className="bg-indigo-50 text-indigo-700 border border-indigo-100 px-2.5 py-1 rounded-md text-xs font-medium">
                         {concept}
                       </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <span className="font-semibold text-gray-900 block mb-2 text-xs uppercase tracking-wider">Abstract</span>
                <p className="text-gray-700 text-sm leading-relaxed text-justify bg-gray-50 p-4 rounded-lg border border-gray-100">
                  {content.paper.abstract || "No abstract available."}
                </p>
              </div>
            </div>
          )}
          
          {content.type === 'bibtex' && (
            <pre className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs overflow-x-auto whitespace-pre-wrap">{content.content}</pre>
          )}
          {content.type === 'loading_bib' && (
            <div className="flex justify-center py-12"><div className="animate-spin w-8 h-8 border-4 border-indigo-500 border-t-transparent rounded-full"></div></div>
          )}
          {content.type === 'error' && (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg border border-red-200">{content.content}</div>
          )}
        </div>

        {content.type === 'abstract' && (
          <div className="px-6 py-4 border-t bg-gray-50 shrink-0 flex justify-end gap-3">
             <div className="text-xs font-mono text-gray-500 flex-grow flex items-center">
               ID: {content.paper.id}
             </div>
             <a 
               href={`https://openalex.org/${content.paper.id}`}
               target="_blank" 
               rel="noopener noreferrer"
               className="flex items-center gap-2 px-4 py-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg text-sm font-medium transition-colors shadow-sm"
             >
               <ExternalLink size={16} /> Open in OpenAlex
             </a>
          </div>
        )}
      </div>
    </div>
  );
}