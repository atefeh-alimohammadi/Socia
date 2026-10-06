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

function getConversationTitle(title: string | null) {
    const value = title?.trim()

    if (!value) {
        return "Conversation"
    }

    const lower = value.toLowerCase()

    // Avoid exposing obvious seed/test placeholders in the UI.
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

function formatRelativeTime(date: string) {
    const value = relativeTime(date)

    // Avoid awkward grammar such as "1 months ago".
    if (value === "1 months ago") {
        return "1 month ago"
    }

    if (value === "0 months ago") {
        return "just now"
    }

    return value
}

export default function ConversationsPage() {
    const router = useRouter()

    const [conversations, setConversations] = useState<Conversation[]>([])
    const [deletingId, setDeletingId] = useState<number | null>(null)
    const [conversationToDelete, setConversationToDelete] =
        useState<number | null>(null)
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
            } catch {
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
            const data = await apiFetch("/conversation/", {
                method: "POST",
                body: JSON.stringify({
                    title: null,
                }),
            })

            router.push(`/conversations/${data.id}`)
        } catch {
            alert("Failed to create conversation")
        }
    }

    async function deleteConversation(id: number) {
        try {
            setDeletingId(id)

            await apiFetch(`/conversation/${id}`, {
                method: "DELETE",
            })

            setConversations((prev) =>
                prev.filter((conversation) => conversation.id !== id)
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
            <main className="min-h-screen flex items-center justify-center bg-slate-50">
                <p className="text-slate-500">
                    Loading conversations...
                </p>
            </main>
        )
    }

    return (
        <main className="min-h-screen bg-slate-50 px-6 py-8">
            <div className="max-w-5xl mx-auto">

                {/* Header */}
                <div className="bg-white rounded-2xl border border-slate-200 p-8 flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <h1 className="text-3xl font-bold text-slate-900">
                            My Conversations
                        </h1>

                        <p className="mt-2 text-slate-500">
                            Continue a previous conversation or start a new one.
                        </p>
                    </div>

                    <button
                        onClick={createConversation}
                        className="
                            bg-indigo-600
                            hover:bg-indigo-700
                            text-white
                            px-5
                            py-3
                            rounded-xl
                            font-medium
                            transition
                            self-start
                            sm:self-auto
                        "
                    >
                        + New Conversation
                    </button>
                </div>

                {/* Back navigation */}
                <button
                    type="button"
                    onClick={() => router.push("/dashboard")}
                    className="
                        mt-6
                        text-sm
                        font-medium
                        text-indigo-600
                        hover:text-indigo-800
                        transition
                    "
                >
                    ← Back to Dashboard
                </button>

                {/* Conversation list */}
                <div className="mt-8 space-y-4">
                    {conversations.length === 0 && (
                        <div className="bg-white rounded-2xl border border-slate-200 p-10 text-center">
                            <div className="text-4xl mb-4">
                                💬
                            </div>

                            <h2 className="font-semibold text-xl text-slate-900">
                                No conversations yet
                            </h2>

                            <p className="text-slate-500 mt-2">
                                Start a conversation with Socia.
                            </p>
                        </div>
                    )}

                    {conversations.map((conversation) => (
                        <div
                            key={conversation.id}
                            className="
                                bg-white
                                border
                                border-slate-200
                                rounded-2xl
                                p-5
                                flex
                                items-center
                                justify-between
                                gap-4
                                transition
                                hover:border-indigo-200
                                hover:shadow-sm
                            "
                        >
                            <button
                                type="button"
                                className="
                                    flex
                                    items-center
                                    gap-4
                                    min-w-0
                                    flex-1
                                    text-left
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
                                    shrink-0
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
                                    " />
                                </div>

                                <div className="min-w-0">
                                    <h2 className="
                                        font-semibold
                                        text-lg
                                        text-slate-900
                                        truncate
                                    ">
                                        {getConversationTitle(conversation.title)}
                                    </h2>

                                    <p className="text-sm text-slate-500 mt-1">
                                        Conversation with Socia
                                    </p>
                                </div>
                            </button>

                            <div className="
                                flex
                                items-center
                                gap-3
                                shrink-0
                            ">
                                <span className="
                                    hidden
                                    sm:inline
                                    text-sm
                                    text-slate-400
                                ">
                                    {formatRelativeTime(conversation.created_at)}
                                </span>

                                <button
                                    type="button"
                                    className="
                                        px-3
                                        py-2
                                        rounded-lg
                                        text-sm
                                        font-medium
                                        text-red-600
                                        hover:bg-red-50
                                        transition
                                    "
                                    onClick={(e) => {
                                        e.stopPropagation()
                                        setConversationToDelete(
                                            conversation.id
                                        )
                                    }}
                                >
                                    Delete
                                </button>

                                <span className="
                                    text-slate-400
                                    text-2xl
                                    hidden
                                    sm:inline
                                ">
                                    ›
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Footer */}
                <div className="
                    text-center
                    mt-12
                    text-slate-400
                    text-sm
                ">
                    <div className="text-2xl">
                        💜
                    </div>

                    <p className="mt-3">
                        Your conversations are private and secure.
                    </p>

                    <p>
                        Socia is here to support your reflection and practice.
                    </p>
                </div>
            </div>

            {/* Delete confirmation */}
            {conversationToDelete !== null && (
                <div
                    className="
                        fixed
                        inset-0
                        bg-black/40
                        flex
                        items-center
                        justify-center
                        z-50
                        px-4
                    "
                >
                    <div className="
                        bg-white
                        rounded-2xl
                        p-6
                        w-full
                        max-w-md
                        shadow-xl
                    ">
                        <h2 className="
                            text-xl
                            font-bold
                            text-slate-900
                        ">
                            Delete conversation?
                        </h2>

                        <p className="
                            mt-3
                            text-slate-500
                        ">
                            Are you sure you want to delete this conversation?
                            This action cannot be undone.
                        </p>

                        <div className="
                            flex
                            justify-end
                            gap-3
                            mt-6
                        ">
                            <button
                                type="button"
                                className="
                                    px-4
                                    py-2
                                    rounded-lg
                                    bg-slate-100
                                    hover:bg-slate-200
                                    text-slate-700
                                    transition
                                "
                                onClick={() =>
                                    setConversationToDelete(null)
                                }
                            >
                                Cancel
                            </button>

                            <button
                                type="button"
                                className="
                                    px-4
                                    py-2
                                    rounded-lg
                                    bg-red-600
                                    hover:bg-red-700
                                    text-white
                                    transition
                                    disabled:opacity-50
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
                                {deletingId === conversationToDelete
                                    ? "Deleting..."
                                    : "Delete"}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </main>
    )
}

