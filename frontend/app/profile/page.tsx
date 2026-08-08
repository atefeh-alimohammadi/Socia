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


interface Memory {
    id: number
    user_id: number
    memory_type: string
    content: string
    source: string | null
    tag: string | null
    created_at: string
}



export default function ProfilePage() {

    const router = useRouter()

    const [user, setUser] = useState<User | null>(null)
    const [preferences, setPreferences] = useState<Memory[]>([])
    const [loading, setLoading] = useState(true)



    useEffect(() => {

        const token = localStorage.getItem("token")

        if (!token) {
            router.push("/login")
            return
        }


        async function loadProfile() {

            try {

                const userData = await apiFetch(
                    "/auth/me"
                )

                setUser(userData)



                const memories = await apiFetch(
                    "/memory/?memory_type=preference"
                )


                setPreferences(memories)



            } catch{

                localStorage.removeItem("token")
                router.push("/login")


            } finally {

                setLoading(false)

            }

        }


        loadProfile()


    }, [router])




    function handleLogout() {

        localStorage.removeItem("token")

        router.push("/")

    }




    if (loading) {

        return (

            <main className="
                min-h-screen
                flex
                items-center
                justify-center
            ">

                <p className="text-slate-500">
                    Loading profile...
                </p>

            </main>

        )

    }




    return (

        <main className="
            min-h-screen
            bg-slate-50
            p-8
        ">


            <div className="
                max-w-3xl
                mx-auto
            ">


                <button

                    onClick={() => router.push("/dashboard")}

                    className="
                        mb-6
                        bg-white
                        border
                        px-4
                        py-2
                        rounded-lg
                    "

                >

                    ← Dashboard

                </button>



                <h1 className="
                    text-4xl
                    font-bold
                    mb-8
                ">

                    Profile

                </h1>





                {/* Account */}

                <div className="
                    bg-white
                    border
                    rounded-2xl
                    p-6
                ">


                    <h2 className="
                        text-xl
                        font-bold
                        mb-4
                    ">

                        Your Account

                    </h2>



                    <p>

                        <span className="text-slate-500">
                            Name:
                        </span>{" "}

                        {
                            user?.full_name
                            ||
                            user?.username
                        }

                    </p>



                    <p className="mt-2">

                        <span className="text-slate-500">
                            Email:
                        </span>{" "}

                        {user?.email}

                    </p>




                    <p className="mt-2">

                        <span className="text-slate-500">
                            Username:
                        </span>{" "}

                        @{user?.username}

                    </p>


                </div>






                {/* Preferences */}


                <div className="
                    bg-white
                    border
                    rounded-2xl
                    p-6
                    mt-6
                ">


                    <h2 className="
                        text-xl
                        font-bold
                        mb-4
                    ">

                        Your Goals & Preferences

                    </h2>



                    <p className="
                        text-slate-500
                        text-sm
                        mb-4
                    ">

                        From your onboarding conversation with Socia.

                    </p>





                    {
                        preferences.length === 0 ? (

                            <p className="text-slate-500">

                                No preferences saved yet.

                            </p>


                        ) : (


                            preferences.map((memory) => (

                                <div
                                    key={memory.id}
                                    className="mb-4"
                                >


                                    <p className="
                                        text-xs
                                        text-indigo-600
                                        font-semibold
                                        uppercase
                                    ">

                                        {
                                            memory.tag
                                            ?
                                            memory.tag.replace(/_/g, " ")
                                            :
                                            memory.memory_type
                                        }

                                    </p>



                                    <p className="
                                        text-slate-700
                                        mt-1
                                    ">

                                        {memory.content}

                                    </p>



                                </div>


                            ))


                        )

                    }


                </div>






                {/* Danger Zone */}


                <div className="
                    bg-white
                    border
                    border-red-100
                    rounded-2xl
                    p-6
                    mt-6
                ">


                    <h2 className="
                        text-xl
                        font-bold
                        text-red-600
                        mb-3
                    ">

                        Account

                    </h2>




                    <button

                        onClick={handleLogout}

                        className="
                            bg-red-50
                            text-red-600
                            border
                            border-red-200
                            px-4
                            py-2
                            rounded-lg
                        "

                    >

                        Log out

                    </button>



                </div>



            </div>


        </main>

    )

}