type Context = { params: Promise<{ path: string[] }> };

async function forward(request: Request, context: Context) {
  const path = (await context.params).path.join("/");
  const reads = ["health", "model/info", "model/info/ann", "model/insights"];
  const predictions = ["predict", "predict/ann"];
  if (!reads.includes(path) && !predictions.includes(path))
    return Response.json({ detail: "Unknown API endpoint" }, { status: 404 });
  if (
    (request.method === "GET" && !reads.includes(path)) ||
    (request.method === "POST" && !predictions.includes(path))
  )
    return Response.json({ detail: "Method not allowed" }, { status: 405 });
  const backend = process.env.API_BASE_URL || "http://127.0.0.1:8000";
  try {
    const response = await fetch(`${backend.replace(/\/$/, "")}/${path}`, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: request.method === "POST" ? await request.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(25000),
    });
    return new Response(await response.text(), {
      status: response.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json(
      {
        detail:
          "Prediction backend is unavailable or timed out. Check the local FastAPI server.",
      },
      { status: 503 },
    );
  }
}

export const dynamic = "force-dynamic";
export const GET = forward;
export const POST = forward;
