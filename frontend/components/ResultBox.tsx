"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  content: string;
  loading?: boolean;
  error?: string | null;
  mono?: boolean;
}

export default function ResultBox({ content, loading, error, mono }: Props) {
  if (loading) {
    return (
      <div className="result-box flex items-center gap-2 text-gray-400">
        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
        </svg>
        Running…
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-box border-red-200 bg-red-50 text-red-700">
        {error}
      </div>
    );
  }

  if (!content) return null;

  if (mono) {
    return <pre className="result-box">{content}</pre>;
  }

  return (
    <div className="result-box prose prose-sm max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
