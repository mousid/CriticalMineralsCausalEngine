import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Causal Engine",
  description: "Critical Minerals Causal Inference Engine",
};

const nav = [
  { href: "/", label: "Ask", icon: "💬" },
  { href: "/scenarios", label: "Scenarios", icon: "⚙️" },
  { href: "/rag", label: "Documents / RAG", icon: "📄" },
  { href: "/kg", label: "Knowledge Graph", icon: "🕸️" },
  { href: "/causal", label: "Causal Analysis", icon: "🔬" },
  { href: "/validate", label: "Validate", icon: "✅" },
  { href: "/pomdp", label: "POMDP", icon: "📡" },
  { href: "/three-layers", label: "Three Layers", icon: "🗂️" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex min-h-screen">
        {/* Sidebar */}
        <aside className="w-[220px] shrink-0 bg-white border-r border-gray-200 flex flex-col">
          <div className="px-5 py-4 border-b border-gray-100">
            <span className="text-base font-bold text-gray-900 tracking-tight">
              Causal Engine
            </span>
            <p className="text-xs text-gray-400 mt-0.5">Critical Minerals</p>
          </div>
          <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="px-4 py-3 border-t border-gray-100">
            <p className="text-xs text-gray-400">Backend: FastAPI + Python</p>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-8 py-8">{children}</div>
        </main>
      </body>
    </html>
  );
}
