"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"

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


interface Challenge {
    id: number
    journey_id: number
    day_number: number
    title: string
    description: string
    status: string
    completed_at: string | null
}


interface JourneyDetail {
    journey: Journey
    challenges: Challenge[]
    today_challenge: Challenge | null
}


export default function JourneyDetailPage() {

    const router = useRouter()
    const params = useParams()

    const journeyId = Number(params.id)


    const [detail, setDetail] = useState<JourneyDetail | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState("")
    const [completing, setCompleting] = useState(false)



    useEffect(() => {

        if (!journeyId) return


        const token = localStorage.getItem("token")

        if (!token) {
            router.push("/login")
            return
        }


        async function loadJourney() {

            try {

                const data: JourneyDetail =
                    await apiFetch(`/journeys/${journeyId}`)

                setDetail(data)


            } catch{

                setError("Failed to load journey.")

            } finally {

                setLoading(false)

            }

        }


        loadJourney()


    }, [journeyId, router])




    async function completeChallenge() {

        setCompleting(true)


        try {

            const updated: JourneyDetail =
                await apiFetch(
                    `/journeys/${journeyId}/complete-challenge`,
                    {
                        method:"POST"
                    }
                )


            setDetail(updated)


        } catch{

            alert(
                "Failed to complete challenge. Please try again."
            )


        } finally {

            setCompleting(false)

        }

    }




    if(loading){

        return (
            <main className="min-h-screen flex items-center justify-center">
                Loading journey...
            </main>
        )

    }



    if(error || !detail){

        return (
            <main className="min-h-screen flex items-center justify-center">
                <p className="text-red-500">
                    {error || "Journey not found"}
                </p>
            </main>
        )

    }



    const progress =
        detail.journey.day_total > 0
        ?
        Math.round(
            (
                (detail.journey.day_current - 1)
                /
                detail.journey.day_total
            ) * 100
        )
        :
        0




    return (

        <main className="min-h-screen bg-slate-50 p-8">


            <div className="max-w-4xl mx-auto">


                <button
                    onClick={() => router.push("/journeys")}
                    className="
                    bg-white
                    border
                    px-4
                    py-2
                    rounded-lg
                    mb-6
                    "
                >
                    ← Back
                </button>



                <h1 className="text-4xl font-bold">
                    {detail.journey.title}
                </h1>



                <p className="text-slate-600 mt-3">
                    {detail.journey.description}
                </p>



                <div className="mt-8">


                    <div className="flex justify-between">

                        <p className="font-semibold">
                            Day {detail.journey.day_current}
                            {" / "}
                            {detail.journey.day_total}
                        </p>


                        <p className="text-slate-500">
                            {progress}%
                        </p>

                    </div>



                    <div className="
                        w-full
                        bg-slate-200
                        h-2
                        rounded-full
                        mt-3
                    ">

                        <div
                            className="
                            bg-indigo-600
                            h-2
                            rounded-full
                            "
                            style={{
                                width:`${progress}%`
                            }}
                        />

                    </div>


                </div>





                {
                    detail.journey.status === "completed"
                    ?

                    (

                    <div className="
                        bg-green-50
                        border
                        border-green-200
                        rounded-2xl
                        p-6
                        mt-8
                        text-center
                    ">

                        <p className="text-3xl">
                            🎉
                        </p>

                        <p className="
                            text-green-700
                            font-semibold
                            text-lg
                        ">
                            Journey completed!
                        </p>


                        <p className="text-slate-500 mt-2">
                            You've completed all {detail.journey.day_total} challenges.
                        </p>


                    </div>

                    )


                    :

                    detail.today_challenge &&

                    (

                    <div className="
                        bg-indigo-50
                        border
                        border-indigo-200
                        rounded-2xl
                        p-6
                        mt-8
                    ">


                        <p className="
                            text-xs
                            text-indigo-600
                            font-semibold
                            uppercase
                        ">
                            Day {detail.today_challenge.day_number} Challenge
                        </p>



                        <h3 className="
                            text-2xl
                            font-bold
                            mt-3
                        ">
                            {detail.today_challenge.title}
                        </h3>



                        <p className="
                            text-slate-600
                            mt-3
                        ">
                            {detail.today_challenge.description}
                        </p>



                        <button
                            onClick={completeChallenge}
                            disabled={completing}
                            className="
                            mt-6
                            bg-indigo-600
                            text-white
                            px-6
                            py-3
                            rounded-xl
                            disabled:opacity-50
                            "
                        >

                            {
                                completing
                                ?
                                "Saving..."
                                :
                                "Mark as Completed ✓"
                            }

                        </button>


                    </div>

                    )

                }






                <div className="mt-10">


                    <h3 className="
                        text-xl
                        font-bold
                        mb-5
                    ">
                        Challenge History
                    </h3>



                    <div className="flex flex-col gap-3">


                        {
                            detail.challenges.map((challenge)=>{


                                const isCompleted =
                                    challenge.status === "completed"


                                const isCurrent =
                                    challenge.day_number ===
                                    detail.journey.day_current


                                const isUpcoming =
                                    !isCompleted && !isCurrent



                                return (

                                    <div
                                        key={challenge.id}
                                        className={`
                                            flex
                                            gap-4
                                            p-4
                                            rounded-xl
                                            border
                                            ${
                                                isCurrent
                                                ?
                                                "bg-indigo-50 border-indigo-200"
                                                :
                                                "bg-white"
                                            }
                                        `}
                                    >


                                        <span className={`
                                            text-xl
                                            ${
                                                isCompleted
                                                ?
                                                "text-green-500"
                                                :
                                                isCurrent
                                                ?
                                                "text-indigo-600"
                                                :
                                                "text-slate-300"
                                            }
                                        `}>

                                            {
                                                isCompleted
                                                ?
                                                "✓"
                                                :
                                                isCurrent
                                                ?
                                                "→"
                                                :
                                                "○"
                                            }

                                        </span>



                                        <div className={
                                            isUpcoming
                                            ?
                                            "opacity-40"
                                            :
                                            ""
                                        }>


                                            <p className="font-semibold">

                                                Day {challenge.day_number}
                                                {" — "}
                                                {challenge.title}

                                            </p>



                                            {
                                                isCompleted &&
                                                challenge.completed_at &&

                                                <p className="text-xs text-slate-400 mt-1">

                                                    Completed{" "}
                                                    {
                                                        new Date(
                                                            challenge.completed_at
                                                        )
                                                        .toLocaleDateString()
                                                    }

                                                </p>

                                            }


                                        </div>


                                    </div>

                                )

                            })
                        }


                    </div>


                </div>



            </div>


        </main>

    )

}