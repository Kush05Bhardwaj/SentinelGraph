/**
 * API client placeholder.
 *
 * Endpoints are implemented in later issues. This module establishes the
 * client shape so the frontend can be wired to the FastAPI backend
 * independently.
 */

export interface HealthResponse {
  status: string;
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${BASE_URL}/health`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}
