import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Rewrite /api/* to the FastAPI backend so the same origin is used in production
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
