import { describe, it, expect } from "vitest";
import { getAllBackends, getBackend, detectAll, getEnabledBackends } from "../src/pipeline/backends.js";
import { loadConfig, saveConfig, type Config } from "../src/pipeline/config.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

describe("Backend Registry", () => {
  it("returns all 4 backends", () => {
    const backends = getAllBackends();
    expect(backends).toHaveLength(4);
    expect(backends.map((b) => b.id)).toEqual([
      "claude_code",
      "codex",
      "copilot_chat",
      "cursor",
    ]);
  });

  it("finds backend by id", () => {
    expect(getBackend("claude_code")?.name).toBe("Claude Code");
    expect(getBackend("codex")?.name).toBe("Codex");
    expect(getBackend("copilot_chat")?.name).toBe("Copilot Chat");
    expect(getBackend("cursor")?.name).toBe("Cursor");
    expect(getBackend("nonexistent")).toBeUndefined();
  });

  it("detectAll returns BackendInfo for each", () => {
    const infos = detectAll();
    expect(infos).toHaveLength(4);
    for (const info of infos) {
      expect(info).toHaveProperty("id");
      expect(info).toHaveProperty("name");
      expect(info).toHaveProperty("supported");
      expect(info).toHaveProperty("detected");
      expect(info).toHaveProperty("sourcePath");
      expect(["available", "coming_soon", "not_found"]).toContain(info.status);
    }
  });

  it("copilot_chat and cursor have coming_soon or not_found status", () => {
    const infos = detectAll();
    const copilot = infos.find((i) => i.id === "copilot_chat")!;
    const cursor = infos.find((i) => i.id === "cursor")!;
    expect(copilot.supported).toBe(false);
    expect(cursor.supported).toBe(false);
    expect(["coming_soon", "not_found"]).toContain(copilot.status);
    expect(["coming_soon", "not_found"]).toContain(cursor.status);
  });

  it("copilot_chat and cursor parse returns empty", async () => {
    const copilot = getBackend("copilot_chat")!;
    const cursor = getBackend("cursor")!;
    const config = loadConfig();
    expect(await copilot.parse(config)).toEqual([]);
    expect(await cursor.parse(config)).toEqual([]);
  });

  it("getEnabledBackends filters by config and availability", () => {
    const config = loadConfig();
    // Disable codex
    config.backends = {
      claude_code: { enabled: true, exclusions: [] },
      codex: { enabled: false, exclusions: [] },
    };
    const enabled = getEnabledBackends(config);
    const ids = enabled.map((b) => b.id);
    expect(ids).not.toContain("codex");
    expect(ids).not.toContain("copilot_chat"); // coming_soon, never enabled
    expect(ids).not.toContain("cursor"); // coming_soon/not_found
  });
});

describe("Config expansion", () => {
  it("loadConfig populates backends from legacy agentCwds", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-config-"));
    // Write legacy config
    fs.writeFileSync(
      path.join(tmpDir, "config.json"),
      JSON.stringify({ agentCwds: ["/foo/bar"] }),
    );
    const config = loadConfig(tmpDir);
    expect(config.backends.claude_code).toEqual({
      enabled: true,
      exclusions: ["/foo/bar"],
    });
    expect(config.backends.codex).toEqual({ enabled: true, exclusions: [] });
    expect(config.backends.copilot_chat).toEqual({ enabled: false, exclusions: [] });
    expect(config.backends.cursor).toEqual({ enabled: false, exclusions: [] });
    expect(config.hasCompletedSetup).toBe(false);
    fs.rmSync(tmpDir, { recursive: true });
  });

  it("loadConfig preserves explicit backends field", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-config-"));
    fs.writeFileSync(
      path.join(tmpDir, "config.json"),
      JSON.stringify({
        backends: { claude_code: { enabled: false, exclusions: [] } },
        hasCompletedSetup: true,
      }),
    );
    const config = loadConfig(tmpDir);
    expect(config.backends.claude_code.enabled).toBe(false);
    expect(config.backends.codex.enabled).toBe(true);
    expect(config.backends.copilot_chat.enabled).toBe(false);
    expect(config.hasCompletedSetup).toBe(true);
    fs.rmSync(tmpDir, { recursive: true });
  });

  it("saveConfig merges into existing config", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-config-"));
    fs.writeFileSync(
      path.join(tmpDir, "config.json"),
      JSON.stringify({ agentCwds: ["/a"], hasCompletedSetup: false }),
    );
    saveConfig(tmpDir, { hasCompletedSetup: true });
    const raw = JSON.parse(fs.readFileSync(path.join(tmpDir, "config.json"), "utf-8"));
    expect(raw.hasCompletedSetup).toBe(true);
    expect(raw.agentCwds).toEqual(["/a"]); // preserved
    expect(raw.backends.codex).toEqual({ enabled: true, exclusions: [] });
    fs.rmSync(tmpDir, { recursive: true });
  });

  it("saveConfig creates config.json if none exists", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-config-"));
    saveConfig(tmpDir, { hasCompletedSetup: true });
    const raw = JSON.parse(fs.readFileSync(path.join(tmpDir, "config.json"), "utf-8"));
    expect(raw.hasCompletedSetup).toBe(true);
    expect(raw.backends.claude_code.enabled).toBe(true);
    expect(raw.backends.copilot_chat.enabled).toBe(false);
    fs.rmSync(tmpDir, { recursive: true });
  });
});
