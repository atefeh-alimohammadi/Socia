const API_URL = "http://localhost:8000/api/v1"

export async function apiFetch(
    path: string,
    options: RequestInit = {}
) {
    const token = localStorage.getItem("token")

    const headers = {
        "Content-Type": "application/json",
        ...(token &&{
            Authorization: `Bearer ${token}`,
            }
        ),
        ...options.headers,
    }

    const response = await fetch(
        `${API_URL}${path}`,
        {
            ...options,
            headers,
        }
    )

    if (!response.ok) {
        const error = await response.json()

    throw new Error(
      error.detail || "API request failed"
    )
  }
    if(response.status === 204){
        return null
    }

  return response.json()
}