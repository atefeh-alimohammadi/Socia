"use client"

import { useEffect, useState, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import { apiFetch } from "@/lib/api"


interface Session {
    id: number
    title: string | null
    created_at: string
}


interface Message {
    id: number
    session_id: number
    role: "user" | "assistant"
    content: string
    created_at: string
    rating?: "helpful" | "not_helpful" | null
}


export default function ChatPage() {

    const router = useRouter()
    const params = useParams()

    const sessionId = Number(params.id)


    const [session, setSession] =
        useState<Session | null>(null)

    const [messages, setMessages] =
        useState<Message[]>([])

    const [input, setInput] =
        useState("")

    const [loading, setLoading] =
        useState(false)

    const [submittingId, setSubmittingId] =
        useState<number | null>(null)

    const bottomRef =
        useRef<HTMLDivElement>(null)


    // ---------------------------------------------------------
    // Initialize chat
    // ---------------------------------------------------------

    useEffect(() => {

        const token =
            localStorage.getItem("token")


        if (!token) {

            router.push("/login")

            return
        }


        const initializeChat = async () => {

            try {

                const currentSession: Session =
                    await apiFetch(
                        `/conversation/${sessionId}`
                    )


                setSession(currentSession)


                // Load messages

                const chatMessages: Message[] =
                    await apiFetch(
                        `/conversation/${currentSession.id}/messages`
                    )





                setMessages(chatMessages)


            } catch {

                localStorage.removeItem("token")

                router.push("/login")

            }

        }


        initializeChat()

    }, [router, sessionId])


    // ---------------------------------------------------------
    // Auto scroll
    // ---------------------------------------------------------

    useEffect(() => {

        bottomRef.current?.scrollIntoView({
            behavior: "smooth",
        })

    }, [messages])


    // ---------------------------------------------------------
    // Send message
    // ---------------------------------------------------------

    async function sendMessage() {

        if (
            !session ||
            input.trim() === ""
        ) {
            return
        }


        setLoading(true)


        try {

            const response: Message[] =
                await apiFetch(
                    `/conversation/${session.id}/messages`,
                    {
                        method: "POST",
                        body: JSON.stringify({
                            content: input
                        })
                    }
                )





            setMessages((prev) => [
                ...prev,
                ...response
            ])


            setInput("")


        } catch {

            alert(
                "Failed to send message"
            )

        } finally {

            setLoading(false)

        }

    }


    // ---------------------------------------------------------
    // Submit message feedback
    // ---------------------------------------------------------

    async function submitFeedback(
        message: Message,
        rating: "helpful" | "not_helpful"
    ) {

        if (!session) {
            return
        }


        // Prevent duplicate submissions

        if (
            message.rating ||
            submittingId === message.id
        ) {
            return
        }




        setSubmittingId(message.id)


        try {

            await apiFetch(
                `/feedback/conversation/${session.id}/messages/${message.id}/feedback`,
                {
                    method: "POST",
                    body: JSON.stringify({
                        rating
                    })
                }
            )





            // Update local UI immediately

            setMessages((prev) =>
                prev.map((m) =>
                    m.id === message.id
                        ? {
                            ...m,
                            rating
                        }
                        : m
                )
            )


        } catch (error) {

            console.error(
                "FEEDBACK ERROR:",
                error
            )


            alert(
                "Failed to submit feedback"
            )

        } finally {

            setSubmittingId(null)

        }

    }


    // ---------------------------------------------------------
    // Render
    // ---------------------------------------------------------

    return (

        <main className="max-w-3xl mx-auto p-6">


            {/* Header */}

            <div className="
                flex
                justify-between
                items-center
                mb-6
            ">


                <h1 className="
                    text-3xl
                    font-bold
                ">
                    Socia Chat
                </h1>


                <button
                    className="
                        bg-gray-700
                        text-white
                        px-4
                        py-2
                        rounded
                    "
                    onClick={() =>
                        router.push("/conversations")
                    }
                >
                    Back
                </button>


            </div>


            {/* Messages */}

            <div
                className="
                    flex
                    flex-col
                    gap-4
                    h-96
                    overflow-y-auto
                    border
                    rounded
                    p-4
                "
            >


                {
                    messages.map((message) => {






                        return (

                            <div
                                key={message.id}
                                className={`
                                    flex
                                    flex-col
                                    gap-1
                                    ${
                                        message.role === "user"
                                            ? "items-end"
                                            : "items-start"
                                    }
                                `}
                            >


                                {/* Message bubble */}

                                <div
                                    className={
                                        message.role === "user"
                                            ? `
                                                bg-blue-600
                                                text-white
                                                px-4
                                                py-2
                                                rounded-lg
                                                max-w-xs
                                            `
                                            : `
                                                bg-gray-100
                                                text-gray-800
                                                px-4
                                                py-2
                                                rounded-lg
                                                max-w-xs
                                            `
                                    }
                                >
                                    {message.content}
                                </div>


                                {/* Assistant feedback */}

                                {
                                    message.role === "assistant" && (

                                        <div className="
                                            flex
                                            items-center
                                            gap-2
                                            px-1
                                            mt-1
                                        ">


                                            <span className="
                                                text-xs
                                                text-gray-400
                                                mr-1
                                            ">
                                                Helpful?
                                            </span>


                                            {/* Helpful */}

                                            <button
                                                type="button"
                                                onClick={() =>
                                                    submitFeedback(
                                                        message,
                                                        "helpful"
                                                    )
                                                }
                                                disabled={
                                                    !!message.rating ||
                                                    submittingId === message.id
                                                }
                                                className={`
                                                    px-2
                                                    py-1
                                                    rounded
                                                    text-sm
                                                    transition
                                                    ${
                                                        message.rating === "helpful"
                                                            ? "bg-green-100 opacity-100"
                                                            : "bg-gray-100 opacity-60 hover:opacity-100"
                                                    }
                                                    disabled:cursor-not-allowed
                                                `}
                                                aria-label="Helpful"
                                                title="Helpful"
                                            >
                                                👍
                                            </button>


                                            {/* Not helpful */}

                                            <button
                                                type="button"
                                                onClick={() =>
                                                    submitFeedback(
                                                        message,
                                                        "not_helpful"
                                                    )
                                                }
                                                disabled={
                                                    !!message.rating ||
                                                    submittingId === message.id
                                                }
                                                className={`
                                                    px-2
                                                    py-1
                                                    rounded
                                                    text-sm
                                                    transition
                                                    ${
                                                        message.rating === "not_helpful"
                                                            ? "bg-red-100 opacity-100"
                                                            : "bg-gray-100 opacity-60 hover:opacity-100"
                                                    }
                                                    disabled:cursor-not-allowed
                                                `}
                                                aria-label="Not helpful"
                                                title="Not helpful"
                                            >
                                                👎
                                            </button>


                                            {/* Confirmation */}

                                            {
                                                message.rating && (

                                                    <span className="
                                                        text-xs
                                                        text-gray-400
                                                        ml-1
                                                    ">
                                                        Thanks for your feedback
                                                    </span>

                                                )
                                            }


                                        </div>

                                    )
                                }


                            </div>

                        )

                    })
                }


                {/* Thinking indicator */}

                {
                    loading && (

                        <div className="
                            self-start
                            bg-gray-200
                            px-4
                            py-2
                            rounded-lg
                        ">
                            Socia is thinking...
                        </div>

                    )
                }


                <div ref={bottomRef} />


            </div>


            {/* Input */}

            <div className="
                flex
                gap-2
                mt-4
            ">


                <input
                    className="
                        border
                        rounded
                        p-2
                        w-full
                    "
                    placeholder="Type your message..."
                    value={input}
                    onChange={(e) =>
                        setInput(e.target.value)
                    }
                    onKeyDown={(e) => {

                        if (
                            e.key === "Enter" &&
                            !loading
                        ) {
                            sendMessage()
                        }

                    }}
                />


                <button
                    className="
                        bg-blue-600
                        text-white
                        px-4
                        py-2
                        rounded
                        disabled:opacity-50
                    "
                    onClick={sendMessage}
                    disabled={
                        loading ||
                        !input.trim()
                    }
                >
                    {
                        loading
                            ? "Sending..."
                            : "Send"
                    }
                </button>


            </div>


        </main>

    )

}