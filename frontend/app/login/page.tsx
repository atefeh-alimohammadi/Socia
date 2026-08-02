"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"

export default function LoginPage() {

    const router = useRouter()

    const [username, setUsername] = useState("")
    const [password, setPassword] = useState("")
    const [error, setError] = useState("")
    const [loading, setLoading] = useState(false)


    async function handleLogin(e: React.FormEvent) {
        e.preventDefault()

        setLoading(true)
        setError("")

        try {

            const formData = new FormData()

            formData.append("username", username)
            formData.append("password", password)


            const response = await fetch(
                "http://localhost:8000/api/v1/auth/login",
                {
                    method: "POST",
                    body: formData
                }
            )


            if (!response.ok) {
                throw new Error("Invalid username or password")
            }


            const data = await response.json()

            localStorage.setItem(
                "token",
                data.access_token
            )

            console.log("TOKEN:", data.access_token)
            console.log("LOCAL TOKEN:", localStorage.getItem("token"))

            router.push("/dashboard")


        } catch(err) {

            if(err instanceof Error){
                setError(err.message)
            } else {
                setError("Something went wrong")
            }

        } finally {
            setLoading(false)
        }
    }


    return (

        <main className="min-h-screen bg-slate-50 flex items-center justify-center">

            <form
                onSubmit={handleLogin}
                className="bg-white p-8 rounded-xl shadow-md w-full max-w-md"
            >

                <h1 className="text-3xl font-bold text-center mb-6">
                    Welcome back
                </h1>


                <input
                    className="border p-3 rounded w-full mb-3"
                    placeholder="Email or username"
                    value={username}
                    onChange={(e)=>setUsername(e.target.value)}
                />


                <input
                    className="border p-3 rounded w-full mb-3"
                    placeholder="Password"
                    type="password"
                    value={password}
                    onChange={(e)=>setPassword(e.target.value)}
                />


                {
                    error && (
                        <p className="text-red-500 mb-3">
                            {error}
                        </p>
                    )
                }


                <button
                    disabled={loading}
                    className="bg-indigo-600 text-white w-full py-3 rounded-lg"
                >
                    {
                        loading
                        ? "Logging in..."
                        : "Login"
                    }
                </button>


                <p className="text-center mt-4">

                    Don't have an account?{" "}

                    <button
                        type="button"
                        className="text-indigo-600"
                        onClick={() => router.push("/")}
                    >
                        Create one
                    </button>

                </p>


            </form>

        </main>

    )
}