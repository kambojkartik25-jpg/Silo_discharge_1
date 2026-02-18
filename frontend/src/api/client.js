const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

async function postJson(path, payload) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) {
    const detail =
      typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  return data;
}

export async function optimize(payload) {
  return postJson("/optimize", payload);
}

export async function simulate(payload) {
  return postJson("/simulate", payload);
}

export { API_BASE_URL };
