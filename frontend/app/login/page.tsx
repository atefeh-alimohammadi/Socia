"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"

export default function LoginPage() {

  const router = useRouter()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [message, setMessage] = useState("")


  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()

    const formData = new FormData()

    formData.append("username", email)
    formData.append("password", password)

    try {
      const response = await fetch(
        "http://localhost:8000/api/v1/auth/login",
        {
          method: "POST",
          body: formData,
        }
      )


      if (!response.ok) {
        setMessage("Invalid email or password")
        return
      }


      const data = await response.json()


      localStorage.setItem(
        "token",
        data.access_token
      )


      router.push("/dashboard")


    } catch (error) {
      setMessage("Server error")
    }
  }


  return (
    <main>
      <h1>Login to Socia</h1>

      <form onSubmit={handleLogin}>

        <input
          placeholder="Email"
          value={email}
          onChange={(e)=>setEmail(e.target.value)}
        />

        <input
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e)=>setPassword(e.target.value)}
        />

        <button>
          Login
        </button>

      </form>

      <p>{message}</p>

    </main>
  )
}