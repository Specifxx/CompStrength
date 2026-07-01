import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // The pipeline writes data/*.json to the monorepo root (two levels up
  // from this app). Vercel's build only traces files it can detect as
  // dependencies, so the JSON snapshot must be explicitly included in the
  // serverless function bundle for the API routes that read it.
  outputFileTracingRoot: path.join(__dirname, "..", ".."),
  outputFileTracingIncludes: {
    "/api/predict": ["../../data/*.json"],
    "/api/champions": ["../../data/*.json"],
    "/": ["../../data/*.json"],
    "/methodology": ["../../data/*.json"],
  },
};

export default nextConfig;
