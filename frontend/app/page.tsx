"use client"

import { useRouter } from "next/navigation"

export default function HomePage() {

    const router = useRouter()

    return (
        <main className="min-h-screen bg-slate-50 flex items-center justify-center">

            <div className="max-w-md text-center p-8">

                <h1 className="text-5xl font-bold text-slate-800 mb-4">
                    Socia
                </h1>

                <p className="text-lg text-slate-600 mb-8">
                    Understand yourself.
                    <br />
                    Grow through reflection.
                </p>


                <div className="flex flex-col gap-4">

                    <button
                        className="bg-indigo-600 text-white py-3 rounded-lg hover:bg-indigo-700"
                        onClick={() => router.push("/register")}
                    >
                        Create Account
                    </button>


                    <button
                        className="border border-slate-300 text-slate-700 py-3 rounded-lg hover:bg-white"
                        onClick={() => router.push("/login")}
                    >
                        Log In
                    </button>

                </div>

            </div>

        </main>
    )
}