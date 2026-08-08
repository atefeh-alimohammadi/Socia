"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"


interface Journey {
    id: number
    title: string
    description: string
    status: string
    day_current: number
    day_total: number
    created_at: string
}


export default function JourneysPage() {

    const router = useRouter()

    const [journeys, setJourneys] = useState<Journey[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState("")


    useEffect(() => {

        const token = localStorage.getItem("token")

        if (!token) {
            router.push("/login")
            return
        }


        async function loadJourneys() {

            try {

                const data: Journey[] = await apiFetch("/journeys/")

                setJourneys(data)

            } catch (err) {

                console.error(err)
                setError("Failed to load journeys.")

            } finally {

                setLoading(false)

            }

        }


        loadJourneys()

    }, [router])



    if (loading) {

        return (
            <main className="min-h-screen flex items-center justify-center">
                <p className="text-slate-500">
                    Loading journeys...
                </p>
            </main>
        )

    }



    if (error) {

        return (
            <main className="min-h-screen flex items-center justify-center">
                <p className="text-red-500">
                    {error}
                </p>
            </main>
        )

    }



    return (

        <main className="min-h-screen bg-slate-50 p-8">

            <div className="max-w-5xl mx-auto">


                {/* Header */}

                <div className="flex justify-between items-center mb-8">


                    <div>

                        <h1 className="text-4xl font-bold text-slate-800">
                            My Journeys
                        </h1>


                        <p className="text-slate-500 mt-2">
                            Your journeys are created from patterns Socia has detected in your conversations.
                        </p>

                    </div>



                    <button
                        onClick={() => router.push("/dashboard")}
                        className="
                            px-5
                            py-2
                            rounded-lg
                            border
                            bg-white
                            hover:bg-slate-50
                        "
                    >
                        ← Dashboard
                    </button>


                </div>




                {
                    journeys.length === 0 ? (

                        <div
                            className="
                                bg-white
                                border
                                rounded-2xl
                                p-10
                                text-center
                            "
                        >

                            <p className="text-2xl mb-4">
                                🌱
                            </p>


                            <h2 className="font-semibold text-xl text-slate-800">
                                No journeys yet
                            </h2>


                            <p className="text-slate-500 mt-2">
                                Keep chatting with Socia. When patterns are detected,
                                you can start a journey from your Progress page.
                            </p>


                            <button
                                onClick={() => router.push("/progress")}
                                className="
                                    mt-6
                                    bg-indigo-600
                                    text-white
                                    px-6
                                    py-3
                                    rounded-xl
                                    hover:bg-indigo-700
                                "
                            >
                                Go to Progress →
                            </button>


                        </div>


                    ) : (


                        <div className="grid md:grid-cols-2 gap-6">


                            {
                                journeys.map((journey) => {


                                    const progress =
                                        journey.day_total > 0
                                            ? Math.round(
                                                ((journey.day_current - 1) /
                                                    journey.day_total) *
                                                100
                                            )
                                            : 0



                                    return (

                                        <div
                                            key={journey.id}
                                            className="
                                                bg-white
                                                border
                                                rounded-2xl
                                                p-6
                                                shadow-sm
                                            "
                                        >


                                            <h2 className="text-2xl font-bold text-slate-800">
                                                {journey.title}
                                            </h2>



                                            <p className="text-slate-600 mt-3 line-clamp-3">
                                                {journey.description}
                                            </p>



                                            <div className="mt-5">


                                                <p className="font-semibold text-slate-700">
                                                    Day {journey.day_current} / {journey.day_total}
                                                </p>



                                                <div
                                                    className="
                                                        w-full
                                                        bg-slate-100
                                                        rounded-full
                                                        h-2
                                                        mt-3
                                                    "
                                                >

                                                    <div
                                                        className="
                                                            bg-indigo-500
                                                            h-2
                                                            rounded-full
                                                            transition-all
                                                        "
                                                        style={{
                                                            width: `${progress}%`
                                                        }}
                                                    />

                                                </div>



                                                <p className="text-sm text-slate-500 mt-2">
                                                    {progress}% completed
                                                </p>


                                            </div>




                                            <button
                                                onClick={() =>
                                                    router.push(
                                                        `/journeys/${journey.id}`
                                                    )
                                                }
                                                className="
                                                    mt-6
                                                    bg-indigo-600
                                                    text-white
                                                    px-5
                                                    py-2
                                                    rounded-lg
                                                    hover:bg-indigo-700
                                                "
                                            >
                                                Continue →
                                            </button>



                                        </div>

                                    )


                                })
                            }


                        </div>


                    )
                }


            </div>

        </main>

    )

}