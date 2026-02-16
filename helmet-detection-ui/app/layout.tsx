// app/layout.tsx
import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Helmet Detection Dashboard",
  description: "Production-grade UI for YOLOv8 helmet detection",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 antialiased">
        <div className="min-h-screen">
          <header className="bg-white shadow-sm">
            <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="rounded-md bg-gradient-to-br from-indigo-600 to-pink-500 text-white w-10 h-10 flex items-center justify-center font-bold">🪖</div>
                <div>
                  <h1 className="text-xl font-semibold">Helmet Detection</h1>
                  <p className="text-sm text-slate-500">YOLOv8 inference dashboard</p>
                </div>
              </div>
              <nav className="text-sm text-slate-600">
                <a className="hover:text-slate-900" href="/docs" target="_blank" rel="noreferrer">Docs</a>
              </nav>
            </div>
          </header>

          <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>

          <footer className="max-w-6xl mx-auto px-6 py-6 text-sm text-slate-500">
            © {new Date().getFullYear()} Helmet Detection — built with Ultralytics YOLOv8
          </footer>
        </div>
      </body>
    </html>
  );
}
