"use client"

import { useState } from "react"

export default function LoginPage() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [message, setMessage] = useState("")

  async function handleLogin(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()

    const formData = new FormData()
    formData.append("username", email)
    formData.append("password", password)

    try {
      const response = await fetch("http://localhost:8000/api/v1/auth/login", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        setMessage("Invalid email or password")
        return
      }

      const data = await response.json()

      localStorage.setItem("token", data.access_token)

      setMessage("Login successful 🎉")
    } catch (error) {
      console.error(error)
      setMessage("Server error")
    }
  }

  return (
    <main>
      <h1>Login to Socia</h1>

      <form onSubmit={handleLogin}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button type="submit">
          Login
        </button>
      </form>

      <p>{message}</p>
    </main>
  )
}