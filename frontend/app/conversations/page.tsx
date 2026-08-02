"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

import { apiFetch } from "@/lib/api"
import { relativeTime } from "@/lib/utils"


interface Conversation {
    id: number
    title: string | null
    created_at: string
}


export default function ConversationsPage() {

    const router = useRouter()

    const [conversations, setConversations] = useState<Conversation[]>([])
    const [deletingId, setDeletingId] = useState<number | null>(null)
    const [conversationToDelete, setConversationToDelete] = useState<number | null>(null)
    const [loading, setLoading] = useState(true)


    useEffect(() => {

        const token = localStorage.getItem("token")

        if (!token) {
            router.push("/login")
            return
        }


        async function loadConversations() {

            try {

                const data = await apiFetch("/conversation/")

                setConversations(data)

            } catch (error) {

                localStorage.removeItem("token")
                router.push("/login")

            } finally {

                setLoading(false)

            }

        }


        loadConversations()


    }, [router])



    async function createConversation() {

        try {

            const data = await apiFetch(
                "/conversation/",
                {
                    method: "POST",
                    body: JSON.stringify({
                        title: "New Conversation"
                    })
                }
            )


            router.push(
                `/conversations/${data.id}`
            )


        } catch {

            alert("Failed to create conversation")

        }

    }

   async function deleteConversation(id:number){

    try {

        setDeletingId(id)


        await apiFetch(
            `/conversation/${id}`,
            {
                method:"DELETE"
            }
        )


        setConversations(
            prev =>
            prev.filter(
                conversation =>
                conversation.id !== id
            )
        )


    } catch {

        alert("Failed to delete conversation")

    } finally {

        setDeletingId(null)
        setConversationToDelete(null)

    }

}



    if (loading) {

        return (
            <main className="min-h-screen flex items-center justify-center">
                <p className="text-gray-500">
                    Loading conversations...
                </p>
            </main>
        )

    }



    return (

        <main className="
            min-h-screen
            bg-slate-50
            p-6
        ">


            <div className="
                max-w-4xl
                mx-auto
            ">


                {/* Header */}

                <div className="
                    bg-white
                    rounded-2xl
                    shadow-sm
                    border
                    p-8
                    flex
                    justify-between
                    items-center
                ">


                    <div>

                        <h1 className="
                            text-4xl
                            font-bold
                            text-slate-900
                        ">
                            My Conversations
                        </h1>


                        <p className="
                            mt-2
                            text-slate-500
                        ">
                            Continue your past conversations or start a new one.
                        </p>


                    </div>



                    <button

                        onClick={createConversation}

                        className="
                            bg-indigo-600
                            hover:bg-indigo-700
                            text-white
                            px-6
                            py-3
                            rounded-xl
                            font-medium
                            transition
                        "
                    >

                        + New Conversation

                    </button>


                </div>




                {/* Back button */}

                <button

                    onClick={() => router.push("/dashboard")}

                    className="
                        mt-6
                        text-indigo-600
                        hover:text-indigo-800
                    "
                >

                    ← Back to Dashboard

                </button>




                {/* Conversation list */}


                <div className="
                    mt-8
                    space-y-4
                ">


                    {
                        conversations.length === 0 && (

                            <div className="
                                bg-white
                                rounded-2xl
                                border
                                p-10
                                text-center
                            ">

                                <div className="
                                    text-5xl
                                    mb-4
                                ">
                                    💬
                                </div>


                                <h2 className="
                                    font-semibold
                                    text-xl
                                ">
                                    No conversations yet
                                </h2>


                                <p className="
                                    text-gray-500
                                    mt-2
                                ">
                                    Start a conversation with Socia.
                                </p>


                            </div>

                        )
                    }




                    {
                        conversations.map((conversation) => (

                            <div

                                key={conversation.id}
                                className="
                                    bg-white
                                    border
                                    rounded-2xl
                                    p-5
                                    flex
                                    items-center
                                    justify-between
                                    cursor-pointer
                                    hover:shadow-md
                                    hover:border-indigo-200
                                    transition
                                "

                            >


                                <div className="
                                    flex
                                    items-center
                                    gap-4
                                "
                                     onClick={() =>
                                    router.push(
                                        `/conversations/${conversation.id}`
                                    )
                                }
                                >


                                    {/* Avatar */}

                                    <div className="
                                        w-12
                                        h-12
                                        rounded-full
                                        bg-indigo-100
                                        flex
                                        items-center
                                        justify-center
                                    ">


                                        <div className="
                                            w-5
                                            h-5
                                            rounded-full
                                            bg-indigo-600
                                        "/>


                                    </div>




                                    <div>

                                        <h2 className="
                                            font-semibold
                                            text-lg
                                            text-slate-900
                                        ">

                                            {
                                                conversation.title ||
                                                "Untitled conversation"
                                            }

                                        </h2>



                                        <p className="
                                            text-sm
                                            text-gray-500
                                            mt-1
                                        ">

                                            Conversation with Socia

                                        </p>


                                    </div>


                                </div>




                                <div className="
                                    flex
                                    items-center
                                    gap-4
                                ">


                                    <span className="
                                        text-sm
                                        text-gray-400
                                    ">

                                        {
                                            relativeTime(
                                                conversation.created_at
                                            )
                                        }

                                    </span>

                                    <button
                                        className="
                                        bg-red-500
                                        hover:bg-red-600
                                        text-white
                                        px-3
                                        py-1
                                        rounded-lg
                                        text-sm
                                        "
                                        onClick={(e)=>{

                                            e.stopPropagation()

                                            setConversationToDelete(
                                                conversation.id
                                            )
                                        }}
                                    >
                                        Delete
                                    </button>



                                    <span className="
                                        text-gray-400
                                        text-2xl
                                    ">
                                        ›
                                    </span>


                                </div>



                            </div>


                        ))
                    }



                </div>





                {/* Footer */}

                <div className="
                    text-center
                    mt-12
                    text-gray-400
                ">


                    <div className="text-3xl">
                        💜
                    </div>


                    <p className="mt-3">
                        Your conversations are private and secure.
                    </p>


                    <p>
                        We're here to support you.
                    </p>


                </div>



            </div>

{
conversationToDelete && (

<div
className="
fixed
inset-0
bg-black/40
flex
items-center
justify-center
z-50
"
>

<div
className="
bg-white
rounded-2xl
p-6
w-96
shadow-xl
"
>

<h2 className="
text-xl
font-bold
text-slate-900
">
Delete conversation?
</h2>


<p className="
mt-3
text-gray-500
">
Are you sure you want to delete this conversation?
</p>


<div className="
flex
justify-end
gap-3
mt-6
">


<button

className="
px-4
py-2
rounded-lg
bg-gray-200
"

onClick={() =>
setConversationToDelete(null)
}

>
Cancel
</button>



<button

className="
px-4
py-2
rounded-lg
bg-red-600
text-white
"

onClick={() =>
deleteConversation(
conversationToDelete
)
}

disabled={
deletingId === conversationToDelete
}

>

{
deletingId === conversationToDelete
?
"Deleting..."
:
"Delete"
}

</button>


</div>


</div>


</div>

)
}
        </main>


    )

}