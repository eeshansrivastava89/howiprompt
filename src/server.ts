import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execSync } from "node:child_process";
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
  let pipelineRunning = false;

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
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS");
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

    // API: pick directory (native macOS folder picker)
    if (req.method === "GET" && url.pathname === "/api/pick-directory") {
      try {
        const result = execSync(
          `osascript -e 'tell application "Finder" to activate' -e 'return POSIX path of (choose folder with prompt "Select directory to exclude")'`,
          { timeout: 60000 },
        ).toString().trim();
        // Remove trailing slash
        const dirPath = result.replace(/\/$/, "");
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ path: dirPath }));
      } catch {
        // User cancelled or osascript failed
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ path: null }));
      }
      return;
    }

    // API: count messages that would be excluded for a given path
    if (req.method === "POST" && url.pathname === "/api/exclusion-count") {
      let body = "";
      for await (const chunk of req) body += chunk;
      const { path: dirPath } = JSON.parse(body);

      // Convert to Claude's directory format: /path/to/project -> -path-to-project
      const claudeDir = dirPath.replace(/\//g, "-");
      const projectsDir = path.join(os.homedir(), ".claude", "projects");
      const targetDir = path.join(projectsDir, claudeDir);

      let messageCount = 0;
      if (fs.existsSync(targetDir)) {
        // Count lines in all JSONL files under this directory
        const countLines = (dir: string): number => {
          let total = 0;
          try {
            for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
              const full = path.join(dir, entry.name);
              if (entry.isDirectory()) total += countLines(full);
              else if (entry.name.endsWith(".jsonl")) {
                total += fs.readFileSync(full, "utf-8").split("\n").filter(Boolean).length;
              }
            }
          } catch { /* skip */ }
          return total;
        };
        messageCount = Math.round(countLines(targetDir) / 2); // rough human-only count
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ messageCount, claudeDir }));
      return;
    }

    // API: reset (delete metrics.json + reset setup flag)
    if (req.method === "POST" && url.pathname === "/api/reset") {
      if (fs.existsSync(metricsPath)) fs.unlinkSync(metricsPath);
      const { saveConfig } = await import("./pipeline/config.js");
      saveConfig(opts.dataDir, { hasCompletedSetup: false });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }

    // API: detect backends
    if (req.method === "GET" && url.pathname === "/api/detect") {
      const { detectAll } = await import("./pipeline/backends.js");
      const backends = detectAll();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ backends }));
      return;
    }

    // API: config
    if (url.pathname === "/api/config") {
      if (req.method === "GET") {
        const { loadConfig } = await import("./pipeline/config.js");
        const config = loadConfig(opts.dataDir);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(config));
        return;
      }
      if (req.method === "PUT") {
        let body = "";
        for await (const chunk of req) body += chunk;
        const updates = JSON.parse(body);
        const { saveConfig } = await import("./pipeline/config.js");
        saveConfig(opts.dataDir, updates);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
    }

    // API: pipeline stream (SSE)
    if (req.method === "GET" && url.pathname === "/api/pipeline/stream") {
      if (pipelineRunning) {
        res.writeHead(409, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Pipeline already running" }));
        return;
      }

      pipelineRunning = true;
      let clientDisconnected = false;
      req.on("close", () => { clientDisconnected = true; });

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      });

      function emit(type: string, data: Record<string, unknown>) {
        if (clientDisconnected) return;
        res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
      }

      try {
        const { runPipeline } = await import("./index.js");
        const stats = await runPipeline({
          dbPath: opts.dbPath,
          dataDir: opts.dataDir,
          onProgress: (stage: string, detail: string) => {
            emit("progress", { stage, detail });
          },
        });
        const metrics = fs.existsSync(metricsPath)
          ? JSON.parse(fs.readFileSync(metricsPath, "utf-8"))
          : null;
        emit("complete", { stats, metrics });
      } catch (err: any) {
        emit("pipeline_error", { message: err.message });
      } finally {
        pipelineRunning = false;
        res.end();
      }
      return;
    }

    // Serve metrics.json from data dir (no-cache to avoid stale data after reset)
    if (url.pathname === "/metrics.json" || url.pathname === "/wrapped/metrics.json") {
      if (fs.existsSync(metricsPath)) {
        res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
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
