"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"


export default function RegisterPage() {

    const router = useRouter()

    const [email, setEmail] = useState("")
    const [username, setUsername] = useState("")
    const [full_name, setFullName] = useState("")
    const [password, setPassword] = useState("")

    const [loading, setLoading] = useState(false)
    const [error, setError] = useState("")


    async function handleRegister(e: React.FormEvent) {

        e.preventDefault()

        setLoading(true)
        setError("")

        try {



            const registerResponse = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({
        email,
        username,
        full_name,
        password
    })
})



            const formData = new FormData()

formData.append("username", email)
formData.append("password", password)

const loginResponse = await fetch(
    "http://localhost:8000/api/v1/auth/login",
    {
        method: "POST",
        body: formData,
    }
)

if (!loginResponse.ok) {
    throw new Error("Registration succeeded but login failed")
}

const loginData: {
    access_token: string
    token_type: string
} = await loginResponse.json()

localStorage.setItem(
    "token",
    loginData.access_token
)




            router.push("/onboarding")


        } catch (err) {
    if (err instanceof Error) {
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
                onSubmit={handleRegister}
                className="bg-white p-8 rounded-xl shadow-md w-full max-w-md"
            >

                <h1 className="text-3xl font-bold text-center mb-6">
                    Create Account
                </h1>


                <input
                    className="border p-3 rounded w-full mb-3"
                    placeholder="Email"
                    value={email}
                    onChange={(e)=>setEmail(e.target.value)}
                />


                <input
                    className="border p-3 rounded w-full mb-3"
                    placeholder="Username"
                    value={username}
                    onChange={(e)=>setUsername(e.target.value)}
                />


                <input
                    className="border p-3 rounded w-full mb-3"
                    placeholder="Full name"
                    value={full_name}
                    onChange={(e)=>setFullName(e.target.value)}
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
                        ? "Creating..."
                        : "Create Account"
                    }
                </button>


                <p className="text-center mt-4">

                    Already have an account?{" "}

                    <button
                        type="button"
                        className="text-indigo-600"
                        onClick={() => router.push("/")}
                    >
                        Login
                    </button>

                </p>


            </form>


        </main>

    )
}