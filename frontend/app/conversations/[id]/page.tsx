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

function getConversationTitle(title: string | null) {
    const value = title?.trim()

    if (!value) {
        return "Conversation"
    }

    const lower = value.toLowerCase()

    const testTitles = [
        "new conversation",
        "untitled conversation",
        "dimensional emotion fallback test",
        "test conversation",
    ]

    if (testTitles.includes(lower)) {
        return "Conversation"
    }

    return value
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
        const token = localStorage.getItem("token")

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
    }, [messages, loading])

    // ---------------------------------------------------------
    // Send message
    // ---------------------------------------------------------

    async function sendMessage() {
        if (!session || input.trim() === "") {
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
                            content: input.trim(),
                        }),
                    }
                )

            setMessages((prev) => [
                ...prev,
                ...response,
            ])

            // Refresh the conversation after sending a message.
            // The backend may have generated a title from the
            // first message.
            const updatedSession: Session =
                await apiFetch(
                    `/conversation/${session.id}`
                )

            setSession(updatedSession)

            setInput("")
        } catch {
            alert("Failed to send message")
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
                        rating,
                    }),
                }
            )

            setMessages((prev) =>
                prev.map((m) =>
                    m.id === message.id
                        ? {
                            ...m,
                            rating,
                        }
                        : m
                )
            )
        } catch (error) {
            console.error(
                "FEEDBACK ERROR:",
                error
            )

            alert("Failed to submit feedback")
        } finally {
            setSubmittingId(null)
        }
    }

    // ---------------------------------------------------------
    // Render
    // ---------------------------------------------------------

    return (
        <main className="
            min-h-screen
            bg-slate-50
            px-4
            py-6
            sm:px-6
        ">
            <div className="
                max-w-5xl
                mx-auto
            ">

                {/* Header */}
                <div className="
                    flex
                    flex-col
                    gap-3
                    mb-5
                ">
                    <button
                        type="button"
                        className="
                            self-start
                            text-sm
                            font-medium
                            text-indigo-600
                            hover:text-indigo-800
                            transition
                        "
                        onClick={() =>
                            router.push("/conversations")
                        }
                    >
                        ← Back to Conversations
                    </button>

                    <div>
                        <h1 className="
                            text-3xl
                            font-bold
                            text-slate-900
                        ">
                            {getConversationTitle(
                                session?.title ?? null
                            )}
                        </h1>

                        <p className="
                            mt-1
                            text-sm
                            text-slate-500
                        ">
                            Chat with Socia
                        </p>
                    </div>
                </div>

                {/* Messages */}
                <div
                    className="
                        bg-white
                        border
                        border-slate-200
                        rounded-2xl
                        p-4
                        sm:p-6
                        h-[65vh]
                        min-h-[420px]
                        max-h-[720px]
                        overflow-y-auto
                    "
                >
                    <div className="
                        flex
                        flex-col
                        gap-5
                    ">
                        {messages.length === 0 && !loading && (
                            <div className="
                                flex
                                flex-col
                                items-center
                                justify-center
                                min-h-[300px]
                                text-center
                            ">
                                <div className="
                                    w-12
                                    h-12
                                    rounded-full
                                    bg-indigo-100
                                    flex
                                    items-center
                                    justify-center
                                    mb-4
                                ">
                                    <div className="
                                        w-5
                                        h-5
                                        rounded-full
                                        bg-indigo-600
                                    " />
                                </div>

                                <h2 className="
                                    text-lg
                                    font-semibold
                                    text-slate-900
                                ">
                                    Start a conversation
                                </h2>

                                <p className="
                                    mt-2
                                    text-sm
                                    text-slate-500
                                    max-w-md
                                ">
                                    Share what is on your mind,
                                    and Socia will respond.
                                </p>
                            </div>
                        )}

                        {messages.map((message) => (
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
                                    className={`
                                        px-4
                                        py-3
                                        rounded-2xl
                                        text-[15px]
                                        leading-6
                                        whitespace-pre-wrap
                                        break-words
                                        ${
                                            message.role === "user"
                                                ? `
                                                    bg-indigo-600
                                                    text-white
                                                    rounded-br-md
                                                `
                                                : `
                                                    bg-slate-100
                                                    text-slate-800
                                                    rounded-bl-md
                                                `
                                        }
                                    `}
                                    style={{
                                        maxWidth: "min(720px, 85%)",
                                    }}
                                >
                                    {message.content}
                                </div>

                                {/* Assistant feedback */}
                                {message.role === "assistant" && (
                                    <div className="
                                        flex
                                        items-center
                                        gap-2
                                        px-1
                                        mt-1
                                        flex-wrap
                                    ">
                                        <span className="
                                            text-xs
                                            text-slate-400
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
                                                rounded-lg
                                                text-sm
                                                transition
                                                disabled:cursor-not-allowed
                                                ${
                                                    message.rating === "helpful"
                                                        ? "bg-green-100 opacity-100"
                                                        : "bg-slate-100 opacity-60 hover:opacity-100"
                                                }
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
                                                rounded-lg
                                                text-sm
                                                transition
                                                disabled:cursor-not-allowed
                                                ${
                                                    message.rating === "not_helpful"
                                                        ? "bg-red-100 opacity-100"
                                                        : "bg-slate-100 opacity-60 hover:opacity-100"
                                                }
                                            `}
                                            aria-label="Not helpful"
                                            title="Not helpful"
                                        >
                                            👎
                                        </button>

                                        {message.rating && (
                                            <span className="
                                                text-xs
                                                text-slate-400
                                                ml-1
                                            ">
                                                Thanks for your feedback
                                            </span>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}

                        {/* Thinking indicator */}
                        {loading && (
                            <div className="flex items-start">
                                <div className="
                                    bg-slate-100
                                    text-slate-600
                                    px-4
                                    py-3
                                    rounded-2xl
                                    rounded-bl-md
                                    text-sm
                                ">
                                    Socia is thinking...
                                </div>
                            </div>
                        )}

                        <div ref={bottomRef} />
                    </div>
                </div>

                {/* Input */}
                <div className="
                    flex
                    gap-2
                    mt-4
                ">
                    <input
                        className="
                            flex-1
                            min-w-0
                            border
                            border-slate-200
                            bg-white
                            rounded-xl
                            px-4
                            py-3
                            text-slate-900
                            outline-none
                            focus:border-indigo-400
                            focus:ring-2
                            focus:ring-indigo-100
                        "
                        placeholder="Type your message..."
                        value={input}
                        onChange={(e) =>
                            setInput(e.target.value)
                        }
                        onKeyDown={(e) => {
                            if (
                                e.key === "Enter" &&
                                !e.shiftKey &&
                                !loading
                            ) {
                                e.preventDefault()
                                sendMessage()
                            }
                        }}
                    />

                    <button
                        type="button"
                        className="
                            bg-indigo-600
                            hover:bg-indigo-700
                            text-white
                            px-5
                            py-3
                            rounded-xl
                            font-medium
                            transition
                            disabled:opacity-50
                            disabled:cursor-not-allowed
                        "
                        onClick={sendMessage}
                        disabled={
                            loading ||
                            !input.trim()
                        }
                    >
                        {loading
                            ? "Sending..."
                            : "Send"}
                    </button>
                </div>
            </div>
        </main>
    )
}

