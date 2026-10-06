"use client"

import { useEffect, useState, useRef } from "react"
import { useRouter } from "next/navigation"
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
        const sessions: Session[] =
          await apiFetch("/conversation/")

        let currentSession: Session

        if (sessions.length === 0) {
          currentSession =
            await apiFetch(
              "/conversation/",
              {
                method: "POST",
                body: JSON.stringify({
                  title: null,
                }),
              }
            )
        } else {
          currentSession = sessions[0]
        }

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
  }, [router])

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
    const content = input.trim()

    if (!session || !content) {
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
              content,
            }),
          }
        )

      setMessages((prev) => [
        ...prev,
        ...response,
      ])

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
    } catch {
      alert("Failed to submit feedback")
    } finally {
      setSubmittingId(null)
    }
  }

  // ---------------------------------------------------------
  // Render
  // ---------------------------------------------------------

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-6 sm:px-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}

        <div className="mb-5">
          <button
            type="button"
            onClick={() =>
              router.push("/conversations")
            }
            className="
              text-sm
              font-medium
              text-indigo-600
              hover:text-indigo-700
              transition
            "
          >
            ← Back to Conversations
          </button>

          <h1 className="text-3xl font-bold text-slate-800 mt-4">
            Socia Chat
          </h1>

          <p className="text-sm text-slate-500 mt-1">
            Talk with Socia and reflect on whatever is on your mind.
          </p>
        </div>

        {/* Messages */}

        <div
          className="
            bg-white
            border
            border-slate-200
            rounded-2xl
            p-4
            sm:p-5
            h-[65vh]
            min-h-[420px]
            max-h-[720px]
            overflow-y-auto
          "
        >
          {messages.length === 0 && !loading && (
            <div className="h-full flex items-center justify-center px-6 text-center">
              <div>
                <p className="text-slate-600 font-medium">
                  Start a conversation with Socia.
                </p>

                <p className="text-sm text-slate-400 mt-1">
                  You can talk about a situation, something
                  you're thinking about, or simply check in.
                </p>
              </div>
            </div>
          )}

          <div className="flex flex-col gap-5">
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
                    text-sm
                    leading-relaxed
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
                    maxWidth:
                      "min(720px, 88%)",
                  }}
                >
                  {message.content}
                </div>

                {/* Assistant feedback */}

                {message.role === "assistant" && (
                  <div className="flex items-center gap-2 px-1 mt-1">
                    <span className="text-xs text-slate-400 mr-1">
                      Helpful?
                    </span>

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
                        rounded-md
                        text-sm
                        transition
                        ${
                          message.rating ===
                          "helpful"
                            ? "bg-green-100 opacity-100"
                            : "bg-slate-100 opacity-60 hover:opacity-100"
                        }
                        disabled:cursor-not-allowed
                        disabled:opacity-50
                      `}
                      aria-label="Helpful"
                      title="Helpful"
                    >
                      👍
                    </button>

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
                        rounded-md
                        text-sm
                        transition
                        ${
                          message.rating ===
                          "not_helpful"
                            ? "bg-red-100 opacity-100"
                            : "bg-slate-100 opacity-60 hover:opacity-100"
                        }
                        disabled:cursor-not-allowed
                        disabled:opacity-50
                      `}
                      aria-label="Not helpful"
                      title="Not helpful"
                    >
                      👎
                    </button>

                    {message.rating && (
                      <span className="text-xs text-slate-400 ml-1">
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
                <div
                  className="
                    bg-slate-100
                    text-slate-500
                    px-4
                    py-3
                    rounded-2xl
                    rounded-bl-md
                    text-sm
                  "
                >
                  Socia is thinking...
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input */}

        <div className="flex gap-2 mt-4">
          <input
            type="text"
            className="
              flex-1
              min-w-0
              bg-white
              border
              border-slate-300
              rounded-xl
              px-4
              py-3
              text-sm
              text-slate-800
              placeholder:text-slate-400
              outline-none
              focus:border-indigo-500
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
            disabled={loading}
          />

          <button
            type="button"
            className="
              bg-indigo-600
              text-white
              px-5
              py-3
              rounded-xl
              text-sm
              font-medium
              hover:bg-indigo-700
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
            {loading ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </main>
  )
}

