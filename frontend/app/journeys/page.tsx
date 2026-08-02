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


    useEffect(() => {

        const token = localStorage.getItem("token")

        if (!token) {
            router.push("/login")
            return
        }


        async function loadJourneys() {

            try {

                const data = await apiFetch("/journeys/")

                setJourneys(data)

            } catch(error) {

                console.error(error)

            } finally {

                setLoading(false)

            }

        }


        loadJourneys()


    }, [router])



    if (loading) {

        return (
            <main className="min-h-screen flex items-center justify-center">
                <p className="text-gray-500">
                    Loading journeys...
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

                        <h1 className="text-4xl font-bold">
                            My Journeys
                        </h1>


                        <p className="text-gray-600 mt-2">
                            Track your personal growth paths
                        </p>

                    </div>



                    <div className="flex gap-3">


                        <button
                            onClick={() => router.push("/dashboard")}
                            className="
                            px-5
                            py-2
                            rounded-lg
                            border
                            bg-white
                            "
                        >
                            ← Dashboard
                        </button>



                        <button
                            onClick={() => router.push("/journeys/new")}
                            className="
                            px-5
                            py-2
                            rounded-lg
                            bg-indigo-600
                            text-white
                            "
                        >
                            + Start New Journey
                        </button>


                    </div>


                </div>



                {
                    journeys.length === 0 ? (

                        <div className="
                            bg-white
                            rounded-2xl
                            border
                            p-8
                            text-center
                        ">

                            <p className="text-gray-600">
                                You haven't started any journeys yet.
                            </p>


                            <button
                                onClick={() => router.push("/journeys/new")}
                                className="
                                mt-5
                                bg-indigo-600
                                text-white
                                px-6
                                py-3
                                rounded-xl
                                "
                            >
                                Start your first journey
                            </button>


                        </div>


                    ) : (


                        <div className="grid md:grid-cols-2 gap-6">


                            {
                                journeys.map((journey) => {


                                    const progress = Math.round(
                                        ((journey.day_current - 1) /
                                        journey.day_total) * 100
                                    )


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


                                            <h2 className="text-2xl font-bold">
                                                {journey.title}
                                            </h2>


                                            <p className="text-gray-600 mt-2">
                                                {journey.description}
                                            </p>



                                            <div className="mt-5">

                                                <p className="font-semibold">
                                                    Day {journey.day_current} / {journey.day_total}
                                                </p>


                                                <div className="
                                                    w-full
                                                    bg-slate-100
                                                    rounded-full
                                                    h-2
                                                    mt-3
                                                ">

                                                    <div
                                                        className="
                                                        bg-indigo-500
                                                        h-2
                                                        rounded-full
                                                        "
                                                        style={{
                                                            width:`${progress}%`
                                                        }}
                                                    />

                                                </div>


                                                <p className="text-sm text-gray-500 mt-2">
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
                                                "
                                            >
                                                View Journey →
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