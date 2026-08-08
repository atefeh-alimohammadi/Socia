"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"


interface User {
    id: number
    email: string
    username: string
    full_name: string | null
}


interface Journey {
    id: number
    title: string
    description: string
    status: string
    day_current: number
    day_total: number
}


interface Challenge {
    id: number
    title: string
    day_number: number
    description: string
    status: string
}


interface JourneyDetail {
    journey: Journey
    challenges: Challenge[]
    today_challenge: Challenge | null
}


interface Conversation {
    id: number
    title: string | null
    created_at: string
}

function getGreeting(): string {
    const hour = new Date().getHours()

    if (hour < 12) return "Good morning"
    if (hour < 17) return "Good afternoon"

    return "Good evening"
}

export default function HomePage() {


    const router = useRouter()


    const [user, setUser] =
        useState<User | null>(null)


    const [journey, setJourney] =
        useState<Journey | null>(null)


    const [journeyDetail, setJourneyDetail] =
        useState<JourneyDetail | null>(null)


    const [conversations, setConversations] =
        useState<Conversation[]>([])


    const [loading, setLoading] =
        useState(true)



    useEffect(() => {


        const token = localStorage.getItem("token")


        if (!token) {

            router.push("/login")
            return

        }



        async function loadUser() {


            try {


                const data =
                    await apiFetch("/auth/me")


                setUser(data)



                const journeys =
                    await apiFetch("/journeys/")



                if (journeys.length > 0) {


                    setJourney(journeys[0])


                    const detail: JourneyDetail =
                        await apiFetch(
                            `/journeys/${journeys[0].id}`
                        )


                    setJourneyDetail(detail)

                }



                const chats =
                    await apiFetch("/conversation/")


                setConversations(
                    chats.slice(0,3)
                )



            } catch{

                localStorage.removeItem("token")
                router.push("/login")


            } finally {


                setLoading(false)


            }


        }



        loadUser()



    }, [router])





    async function startNewConversation(){


        try{


            const newSession =
                await apiFetch(
                    "/conversation/",
                    {
                        method:"POST",
                        body:JSON.stringify({
                            title:null
                        })
                    }
                )



            router.push(
                `/conversations/${newSession.id}`
            )



        }catch{

        }


    }





    if(loading){

        return (

            <main className="min-h-screen flex items-center justify-center">

                <p>
                    Loading...
                </p>

            </main>

        )

    }




    return (

        <main className="min-h-screen bg-slate-50 p-8">


            <div className="max-w-5xl mx-auto">


                <div className="mb-8">


                    <h1 className="text-4xl font-bold">

                        {getGreeting()}, {user?.username || "there"} 👋

                    </h1>


                    <p className="text-xl text-gray-600 mt-2">

                        How are you feeling today?

                    </p>



                    <button
                        onClick={startNewConversation}
                        className="
                        mt-6
                        bg-indigo-600
                        hover:bg-indigo-700
                        text-white
                        px-8
                        py-4
                        rounded-xl
                        text-lg
                        font-semibold
                        "
                    >

                        💬 Start chatting with Socia

                    </button>


                </div>





                <div className="
                    grid
                    md:grid-cols-2
                    gap-6
                ">




                    {/* Journey */}


                    <div className="
                        bg-white
                        border
                        rounded-2xl
                        p-6
                        shadow-sm
                    ">


                        <h2 className="text-2xl font-bold mb-4">

                            Your Journey

                        </h2>



                        <h3 className="text-xl">

                            {journey?.title || "No active journey"}

                        </h3>



                        <p className="text-gray-500 mt-1">

                            Day {journey?.day_current || 0}
                            {" / "}
                            {journey?.day_total || 0}

                        </p>





                        {
                            journeyDetail?.today_challenge && (

                                <div className="mt-4">


                                    <p className="
                                        text-sm
                                        font-semibold
                                        text-slate-600
                                    ">

                                        Today's challenge:

                                    </p>



                                    <p className="
                                        text-slate-700
                                        text-sm
                                        mt-1
                                    ">

                                        {
                                            journeyDetail
                                            .today_challenge
                                            .title
                                        }

                                    </p>


                                </div>

                            )
                        }





                        <button
                            onClick={() => router.push("/journeys")}
                            className="
                            mt-6
                            bg-green-600
                            text-white
                            px-6
                            py-2
                            rounded-lg
                            "
                        >

                            Continue

                        </button>



                    </div>






                    {/* Recent Conversations */}


                    <div className="
                        bg-white
                        border
                        rounded-2xl
                        p-6
                        shadow-sm
                    ">


                        <h2 className="text-2xl font-bold mb-5">

                            Recent Conversations

                        </h2>



                        <div className="space-y-4">


                            {
                                conversations.length === 0 ? (

                                    <p className="text-gray-500">

                                        No conversations yet

                                    </p>


                                ) : (


                                    conversations.map((conversation)=>(


                            <div
                                key={conversation.id}
                                onClick={() =>
                                    router.push(
                                        `/conversations/${conversation.id}`
                                    )
                                }
                                className="
                                    flex
                                    items-center
                                    gap-3
                                    cursor-pointer
                                    hover:bg-gray-50
                                    p-2
                                    rounded-lg
                                "
                            >

                                <div
                                    className="
                                        w-3
                                        h-3
                                        rounded-full
                                        bg-purple-600
                                        flex-shrink-0
                                    "
                                />

                                <p className="text-sm">
                                    {conversation.title || "Untitled conversation"}
                                </p>

                        </div>


                                    ))


                                )
                            }


                        </div>


                        <button
                            onClick={() => router.push("/conversations")}
                            className="
                            mt-6
                            border
                            px-5
                            py-2
                            rounded-lg
                            "
                        >

                            View all

                        </button>


                    </div>






                    {/* Progress */}


                    <div className="
                        bg-white
                        border
                        rounded-2xl
                        p-6
                        shadow-sm
                    ">


                        <h2 className="text-2xl font-bold mb-5">

                            Your Progress

                        </h2>


                        <p className="text-gray-500">

                            Your progress insights will appear here.

                        </p>



                        <button
                            onClick={() => router.push("/progress")}
                            className="
                            mt-5
                            bg-indigo-600
                            text-white
                            px-6
                            py-2
                            rounded-lg
                            "
                        >

                            View Progress

                        </button>


                    </div>





{/* Profile */}

<div
    className="
        bg-white
        border
        rounded-2xl
        p-6
        shadow-sm
    "
>

    <h2 className="text-2xl font-bold mb-4">
        Your Profile
    </h2>

    <p className="text-gray-500">
        View your account information and what Socia knows about you.
    </p>


    <button
        onClick={() => router.push("/profile")}
        className="
            mt-6
            bg-indigo-600
            text-white
            px-6
            py-3
            rounded-lg
        "
    >
        View Profile →
    </button>

</div>



                </div>


            </div>


        </main>

    )


}