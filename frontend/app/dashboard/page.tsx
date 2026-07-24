"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"

export default function DashboardPage() {

  const router = useRouter()

  const [user, setUser] = useState<any>(null)

  useEffect(() => {

    console.log("Dashboard effect running")

    const token = localStorage.getItem("token")

    console.log("Dashboard token:", token)

    if (!token) {
      router.push("/login")
      return
    }


    const fetchUser = async () => {

      try {

        console.log("Fetching user...")

        const data = await apiFetch("/auth/me")

        console.log("USER DATA:", data)

        setUser(data)

      } catch (error) {

        console.log("Dashboard error:", error)

        localStorage.removeItem("token")
        router.push("/login")

      }

    }


    fetchUser()


  }, [router])


  return (
    <main className="max-w-2xl mx-auto p-6">

      <h1 className="text-3xl font-bold">
        Dashboard
      </h1>


      {user ? (
        <div className="mt-6">

          <p>
            Welcome back, {user.full_name || user.username} 👋
          </p>

          <p className="mt-2">
            Email: {user.email}
          </p>

        </div>

      ) : (

        <p className="mt-6">
          Loading...
        </p>

      )}


    </main>
  )
}