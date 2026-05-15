import type { Metadata } from "next";
import { Syne } from "next/font/google";
import { DM_Mono } from "next/font/google";
import "./globals.css";

const syne = Syne({
  variable: "--font-syne",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

const dmMono = DM_Mono({
  variable: "--font-dm-mono",
  subsets: ["latin"],
  weight: ["300", "400", "500"],
});

export const metadata: Metadata = {
  title: "RAG Engine",
  description: "Retrieval-Augmented Generation Chatbot",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${syne.variable} ${dmMono.variable} h-full`}
    >
      <body className="min-h-full flex flex-col bg-[#0a0a0f]">{children}</body>
    </html>
  );
}