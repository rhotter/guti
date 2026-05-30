import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/react";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "A Unified Theory of Brain Sensing",
  description:
    "Computing the theoretical information limit of brain imaging modalities from first principles.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <div className="page-wrapper">
          <article>{children}</article>
        </div>
        <Analytics />
      </body>
    </html>
  );
}
