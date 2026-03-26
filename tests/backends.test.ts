import { describe, it, expect } from "vitest";
import { getAllBackends, getBackend, detectAll, getEnabledBackends } from "../src/pipeline/backends.js";
import { loadConfig, saveConfig, type Config } from "../src/pipeline/config.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

describe("Backend Registry", () => {
  it("returns all 5 backends", () => {
    const backends = getAllBackends();
    expect(backends).toHaveLength(5);
    expect(backends.map((b) => b.id)).toEqual([
      "claude_code",
      "codex",
      "copilot_chat",
      "cursor",
      "lmstudio",
    ]);
  });

  it("finds backend by id", () => {
    expect(getBackend("claude_code")?.name).toBe("Claude Code");
    expect(getBackend("codex")?.name).toBe("Codex");
    expect(getBackend("copilot_chat")?.name).toBe("Copilot Chat");
    expect(getBackend("cursor")?.name).toBe("Cursor");
    expect(getBackend("lmstudio")?.name).toBe("LM Studio");
    expect(getBackend("nonexistent")).toBeUndefined();
  });

  it("detectAll returns BackendInfo for each", () => {
    const infos = detectAll();
    expect(infos).toHaveLength(5);
    for (const info of infos) {
      expect(info).toHaveProperty("id");
      expect(info).toHaveProperty("name");
      expect(info).toHaveProperty("supported");
      expect(info).toHaveProperty("detected");
      expect(info).toHaveProperty("sourcePath");
      expect(["available", "coming_soon", "not_found"]).toContain(info.status);
    }
  });

  it("new chat backends are first-class supported backends", () => {
    const infos = detectAll();
    for (const id of ["copilot_chat", "cursor", "lmstudio"]) {
      const info = infos.find((candidate) => candidate.id === id)!;
      expect(info.supported).toBe(true);
      expect(["available", "not_found"]).toContain(info.status);
    }
  });

  it("getEnabledBackends filters by config and availability", () => {
    const config = loadConfig();
    // Disable codex
    config.backends = {
      claude_code: { enabled: true, exclusions: [] },
      codex: { enabled: false, exclusions: [] },
      copilot_chat: { enabled: false, exclusions: [] },
      cursor: { enabled: false, exclusions: [] },
      lmstudio: { enabled: false, exclusions: [] },
    };
    const enabled = getEnabledBackends(config);
    const ids = enabled.map((b) => b.id);
    expect(ids).not.toContain("codex");
    expect(ids).not.toContain("copilot_chat");
    expect(ids).not.toContain("cursor");
    expect(ids).not.toContain("lmstudio");
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
    expect(config.backends.lmstudio).toEqual({ enabled: false, exclusions: [] });
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
    expect(config.backends.lmstudio.enabled).toBe(false);
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
    expect(raw.backends.lmstudio).toEqual({ enabled: false, exclusions: [] });
    fs.rmSync(tmpDir, { recursive: true });
  });

  it("saveConfig creates config.json if none exists", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-config-"));
    saveConfig(tmpDir, { hasCompletedSetup: true });
    const raw = JSON.parse(fs.readFileSync(path.join(tmpDir, "config.json"), "utf-8"));
    expect(raw.hasCompletedSetup).toBe(true);
    expect(raw.backends.claude_code.enabled).toBe(true);
    expect(raw.backends.copilot_chat.enabled).toBe(false);
    expect(raw.backends.lmstudio.enabled).toBe(false);
    fs.rmSync(tmpDir, { recursive: true });
  });
});
