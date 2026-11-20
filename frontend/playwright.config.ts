import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 60000,
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
    channel: process.env.PLAYWRIGHT_CHANNEL,
  },
  reporter: [
    ["list"],
    ["junit", { outputFile: "../artifacts/integration/browser_tests.xml" }],
  ],
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
  webServer: [
    {
      command:
        "..\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --app-dir .. --host 127.0.0.1 --port 8100",
      url: "http://127.0.0.1:8100/health",
      timeout: 240000,
      reuseExistingServer: false,
    },
    {
      command: "npm run start -- --port 3100",
      url: "http://127.0.0.1:3100",
      env: {
        API_BASE_URL: "http://127.0.0.1:8100",
        NEXT_TELEMETRY_DISABLED: "1",
      },
      timeout: 120000,
      reuseExistingServer: false,
    },
  ],
});
