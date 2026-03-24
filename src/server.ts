import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export interface ServerOptions {
  port: number;
  dataDir: string;
  dbPath: string;
}

export async function startServer(opts: ServerOptions): Promise<http.Server> {
  // Resolve frontend dist path — try multiple locations
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [
    path.join(__dirname, "..", "frontend", "dist"),   // dev: from dist/server.js
    path.join(__dirname, "..", "..", "frontend", "dist"), // npx: from dist/server.js in package
  ];
  let frontendDir = candidates.find((p) => fs.existsSync(p)) ?? candidates[0];

  const metricsPath = path.join(opts.dataDir, "metrics.json");

  const mimeTypes: Record<string, string> = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
  };

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${opts.port}`);

    // CORS headers
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // API: refresh
    if (req.method === "POST" && url.pathname === "/api/refresh") {
      try {
        const { runPipeline } = await import("./index.js");
        const stats = await runPipeline({
          dbPath: opts.dbPath,
          dataDir: opts.dataDir,
        });
        const metrics = JSON.parse(fs.readFileSync(metricsPath, "utf-8"));
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ metrics, stats }));
      } catch (err: any) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // API: health check
    if (url.pathname === "/api/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok" }));
      return;
    }

    // Serve metrics.json from data dir
    if (url.pathname === "/metrics.json" || url.pathname === "/wrapped/metrics.json") {
      if (fs.existsSync(metricsPath)) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(fs.readFileSync(metricsPath));
      } else {
        res.writeHead(404);
        res.end("metrics.json not found");
      }
      return;
    }

    // Serve static files from frontend/dist
    let filePath = path.join(frontendDir, url.pathname);

    // Directory → index.html
    if (url.pathname.endsWith("/")) {
      filePath = path.join(filePath, "index.html");
    }
    // No extension → try adding /index.html (for /wrapped → /wrapped/index.html)
    if (!path.extname(filePath)) {
      const indexPath = path.join(filePath, "index.html");
      if (fs.existsSync(indexPath)) {
        filePath = indexPath;
      }
    }

    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      const mime = mimeTypes[ext] ?? "application/octet-stream";
      res.writeHead(200, { "Content-Type": mime });
      fs.createReadStream(filePath).pipe(res);
    } else {
      res.writeHead(404);
      res.end("Not found");
    }
  });

  return new Promise((resolve) => {
    server.listen(opts.port, "127.0.0.1", () => resolve(server));
  });
}
