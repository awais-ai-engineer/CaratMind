import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "CaratMind | Diamond Intelligence",
  description:
    "Independent machine learning and deep learning diamond price estimates.",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
