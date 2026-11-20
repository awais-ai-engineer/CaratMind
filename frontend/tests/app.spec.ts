import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByText("Models connected", { exact: true }),
  ).toBeVisible();
});

test("real ML and ANN inference through the frontend proxy", async ({
  page,
}, testInfo) => {
  const consoleErrors: string[] = [];
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await expect(
    page.getByText("$271.77", { exact: true }).first(),
  ).toBeVisible();
  const mlResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/predict") &&
      response.request().method() === "POST",
  );
  const annResponse = page.waitForResponse((response) =>
    response.url().endsWith("/api/predict/ann"),
  );
  await page.getByRole("button", { name: "Estimate diamond price" }).click();
  const ml = await (await mlResponse).json();
  const ann = await (await annResponse).json();
  const usd = (value: number) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value);
  await expect(
    page.getByText(usd(ml.predicted_price_usd), { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(usd(ann.predicted_price_usd), { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(
      usd(Math.abs(ml.predicted_price_usd - ann.predicted_price_usd)),
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    page.getByText(/model disagreement, not prediction uncertainty/),
  ).toBeVisible();
  expect(consoleErrors).toEqual([]);
  const invalid = await page.request.post("/api/predict/ann", {
    data: { ...(await mlResponse).request().postDataJSON(), cut: "Unknown" },
  });
  expect(invalid.status()).toBe(422);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: `../artifacts/integration/${testInfo.project.name}.png`,
    fullPage: true,
  });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: `../artifacts/integration/${testInfo.project.name}_overview.png`,
  });
  await page
    .locator("#prediction")
    .screenshot({
      path: `../artifacts/integration/${testInfo.project.name}_studio.png`,
    });
  await testInfo.attach("actual-model-responses", {
    body: JSON.stringify({ ml, ann }, null, 2),
    contentType: "application/json",
  });
  writeFileSync(
    `../artifacts/integration/${testInfo.project.name}_responses.json`,
    JSON.stringify({ ml, ann }, null, 2),
  );
});

test("invalid numeric input is blocked, warnings remain visible for valid extrapolation", async ({
  page,
}) => {
  await page.getByLabel("Length x (mm)").fill("0");
  await page.getByRole("button", { name: "Estimate diamond price" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "must be a finite positive number" })).toContainText(
    "must be a finite positive number",
  );
  await page.getByLabel("Length x (mm)").fill("6.4");
  await page.getByLabel("Carat weight", { exact: false }).fill("5");
  await page.getByRole("button", { name: "Estimate diamond price" }).click();
  await expect(
    page.getByText(/carat is outside the training range/),
  ).toBeVisible();
});

test("loading state and backend validation errors are meaningful", async ({
  page,
}) => {
  // Deliberate failure injection exercises UI behavior; no fake result is shown.
  await page.route("**/api/predict{,/ann}", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 700));
    await route.fulfill({
      status: 422,
      contentType: "application/json",
      body: JSON.stringify({
        detail: [{ loc: ["body", "carat"], msg: "Invalid measurement" }],
      }),
    });
  });
  await page.getByRole("button", { name: "Estimate diamond price" }).click();
  await expect(
    page.getByRole("button", { name: "Estimating with both models…" }),
  ).toBeDisabled();
  await expect(page.getByRole("alert").first()).toContainText(
    "carat: Invalid measurement",
  );
  await expect(
    page.getByRole("button", { name: "Estimate diamond price" }),
  ).toBeEnabled();
});

test("a failed ANN request preserves the real primary estimate", async ({
  page,
}) => {
  await page.route("**/api/predict/ann", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "ANN unavailable" }),
    }),
  );
  await page.getByRole("button", { name: "Estimate diamond price" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "ANN estimator:" })).toContainText(
    "ANN estimator: ANN unavailable",
  );
  await expect(page.getByText("$6,136.93", { exact: true })).toBeVisible();
  await expect(page.getByText("Absolute model difference")).toHaveCount(0);
});

test("keyboard navigation, accessibility and responsive layout", async ({
  page,
}) => {
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to content" }),
  ).toBeFocused();
  await page.keyboard.press("Enter");
  await page.getByLabel("Carat weight", { exact: false }).focus();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("combobox", { name: "Cut", exact: true })).toBeFocused();
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(audit.violations).toEqual([]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
});

test("health failure clearly disables prediction", async ({ page }) => {
  await page.route("**/api/health", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Model is not ready" }),
    }),
  );
  await page.reload();
  await expect(page.getByRole("alert").filter({ hasText: "Backend unavailable" })).toContainText("Backend unavailable");
  await expect(
    page.getByRole("button", { name: "Estimate diamond price" }),
  ).toBeDisabled();
});
