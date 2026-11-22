/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",       // small Docker image: only the files the server needs
  poweredByHeader: false,
};
export default nextConfig;
