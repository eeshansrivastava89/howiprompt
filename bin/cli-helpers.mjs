/**
 * Pure helpers for CLI — testable without side effects.
 */

import net from "node:net";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { exec } from "node:child_process";

/**
 * Resolve the data directory: ~/.howiprompt
 */
export function resolveDataDir() {
  return path.join(os.homedir(), ".howiprompt");
}

/**
 * Find a free TCP port. Tries preferred first, falls back to OS-assigned.
 */
export function findFreePort(preferred) {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(preferred ?? 0, "127.0.0.1", () => {
      const addr = srv.address();
      if (addr && typeof addr === "object") {
        const port = addr.port;
        srv.close(() => resolve(port));
      } else {
        reject(new Error("Could not determine port"));
      }
    });
    srv.on("error", (err) => {
      if (preferred && err.code === "EADDRINUSE") {
        findFreePort(null).then(resolve, reject);
      } else {
        reject(err);
      }
    });
  });
}

/**
 * Poll a URL until it responds with status < 500 or timeout.
 */
export function waitForServer(url, timeoutMs = 15_000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Server did not start within ${timeoutMs}ms`));
        return;
      }
      const req = http.get(url, (res) => {
        if (res.statusCode && res.statusCode < 500) {
          resolve();
        } else {
          setTimeout(check, 200);
        }
      });
      req.on("error", () => setTimeout(check, 200));
      req.end();
    };
    check();
  });
}

/**
 * Parse CLI arguments.
 */
export function parseArgs(argv) {
  const help = argv.includes("--help") || argv.includes("-h");
  const version = argv.includes("--version") || argv.includes("-v");
  const noOpen = argv.includes("--no-open");
  const portIdx = argv.indexOf("--port");
  const port = portIdx !== -1 ? Number(argv[portIdx + 1]) : null;
  return { port, noOpen, help, version };
}

/**
 * Open a URL in the default browser. Cross-platform.
 */
export async function openBrowser(url) {
  const { default: open } = await import("open");
  return open(url);
}
