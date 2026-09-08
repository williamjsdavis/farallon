import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Farallon — Geological Hypothesis Lab',
  icons: { icon: '/favicon.svg' },
  description:
    'GPT-6 reads a geological map, writes an executable history, and tests the world it generates.',
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body>{children}</body>
    </html>
  );
}
