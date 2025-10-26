import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/react";
import "katex/dist/katex.min.css";

import "./globals.css";

export const metadata: Metadata = {
  title: "Experiments.house",
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="flex justify-center">
          <div className="font-serif container p-6 text-gray-900 max-w-xl">
            <div className="prose prose-sm">{children}</div>
          </div>
        </div>
        <Analytics />
      </body>
    </html>
  );
}
