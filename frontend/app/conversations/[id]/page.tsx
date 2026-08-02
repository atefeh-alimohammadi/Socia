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
}

export default function ChatPage() {

    const router = useRouter()
    const params = useParams()

    const sessionId = Number(params.id)

    const [session, setSession] = useState<Session | null>(null)
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState("")
    const [loading, setLoading] = useState(false)
    const bottomRef = useRef<HTMLDivElement>(null)
    useEffect(() => {

            const token = localStorage.getItem("token")

            if (!token) {
                router.push("/login")
                return
            }

            const initializeChat = async () => {
                try {

                    const currentSession: Session = await apiFetch(
                    `/conversation/${sessionId}`
                        )

                    setSession(currentSession)


                    // Load messages
                    const chatMessages: Message[] = await apiFetch(
                        `/conversation/${currentSession.id}/messages`
                    )

                    setMessages(chatMessages)

                } catch {

                    localStorage.removeItem("token")
                    router.push("/login")

                }

            }

            initializeChat()

        }
        , [router, sessionId])

    useEffect(() => {
  bottomRef.current?.scrollIntoView({
    behavior: "smooth",
  })
}, [messages])

    async function sendMessage() {

        if (!session || input.trim() === "") return

        setLoading(true)

        try {

            const response: Message[] = await apiFetch(
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

            alert("Failed to send message")

        } finally {

            setLoading(false)

        }
    }

    return (
        <main className="max-w-3xl mx-auto p-6">

            <div className="flex justify-between items-center mb-6">

                <h1 className="text-3xl font-bold">
                    Socia Chat
                </h1>

                <button
                    className="bg-gray-700 text-white px-4 py-2 rounded"
                    onClick={() => router.push("/conversations")}
                >
                    Back
                </button>

            </div>

            <div
                className="flex flex-col gap-3 h-96 overflow-y-auto border rounded p-4"
            >

                {
                    messages.map((message) => (

                        <div
                            key={message.id}
                            className={
                                message.role === "user"
                                    ? "self-end bg-blue-600 text-white px-4 py-2 rounded-lg max-w-xs"
                                    : "self-start bg-gray-100 text-gray-800 px-4 py-2 rounded-lg max-w-xs"
                            }
                        >
                            {message.content}
                        </div>

                    ))

                }

                {
                    loading && (

                        <div
                            className="self-start bg-gray-200 px-4 py-2 rounded-lg"
                        >
                            Socia is thinking...
                        </div>

                    )
                }
                <div ref={bottomRef} />

            </div>

            <div className="flex gap-2 mt-4">

                <input
                    className="border rounded p-2 w-full"
                    placeholder="Type your message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter") {
                            sendMessage()
                        }
                    }}
                />

                <button
                    className="bg-blue-600 text-white px-4 py-2 rounded"
                    onClick={sendMessage}
                    disabled={loading}
                >
                    Send
                </button>

            </div>

        </main>
    )
}





