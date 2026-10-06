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
    source_memory_id: number | null
}

type DifficultyFeedback =
    | "too_easy"
    | "just_right"
    | "too_hard"

type SkipReason =
    | "busy"
    | "forgot"
    | "too_difficult"
    | "too_anxious"
    | "not_relevant"
    | "disliked_activity"
    | "situation_unavailable"
    | "other"

interface Challenge {
    id: number
    journey_id: number
    day_number: number
    title: string
    description: string
    status: string
    completed_at: string | null
    skip_reason: SkipReason | null
    skip_reason_detail: string | null
    difficulty_feedback: DifficultyFeedback | null
    emotional_response: string | null
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
    const [skipping, setSkipping] = useState(false)

    const [difficultyFeedback, setDifficultyFeedback] =
        useState<DifficultyFeedback | null>(null)

    const [skipReason, setSkipReason] =
        useState<SkipReason | "">("")

    const [skipReasonDetail, setSkipReasonDetail] =
        useState("")

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
            } catch {
                setError("Failed to load journey.")
            } finally {
                setLoading(false)
            }
        }

        loadJourney()
    }, [journeyId, router])

    async function completeChallenge() {
        if (!detail?.today_challenge) {
            return
        }

        setCompleting(true)

        try {
            const body: {
                difficulty_feedback?: DifficultyFeedback
            } = {}

            if (difficultyFeedback) {
                body.difficulty_feedback = difficultyFeedback
            }

            const updated: JourneyDetail =
                await apiFetch(
                    `/journeys/${journeyId}/complete-challenge`,
                    {
                        method: "POST",
                        body: JSON.stringify(body),
                    }
                )

            setDetail(updated)
            setDifficultyFeedback(null)
            setSkipReason("")
            setSkipReasonDetail("")
        } catch {
            alert(
                "Failed to complete challenge. Please try again."
            )
        } finally {
            setCompleting(false)
        }
    }

    async function skipChallenge() {
        if (!skipReason) {
            alert(
                "Please select a reason before skipping the challenge."
            )
            return
        }

        setSkipping(true)

        try {
            const body: {
                skip_reason: SkipReason
                skip_reason_detail?: string
            } = {
                skip_reason: skipReason,
            }

            if (skipReasonDetail.trim()) {
                body.skip_reason_detail =
                    skipReasonDetail.trim()
            }

            const updated: JourneyDetail =
                await apiFetch(
                    `/journeys/${journeyId}/skip-challenge`,
                    {
                        method: "POST",
                        body: JSON.stringify(body),
                    }
                )

            setDetail(updated)
            setDifficultyFeedback(null)
            setSkipReason("")
            setSkipReasonDetail("")
        } catch {
            alert(
                "Failed to skip challenge. Please try again."
            )
        } finally {
            setSkipping(false)
        }
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
                    Loading journey...
                </p>
            </main>
        )
    }

    if (error || !detail) {
        return (
            <main className="
                min-h-screen
                flex
                items-center
                justify-center
            ">
                <p className="text-red-500">
                    {error || "Journey not found"}
                </p>
            </main>
        )
    }

    const isCompleted =
        detail.journey.status === "completed"

    const progress = isCompleted
        ? 100
        : detail.journey.day_total > 0
            ? Math.min(
                100,
                Math.max(
                    0,
                    Math.round(
                        (
                            (detail.journey.day_current - 1) /
                            detail.journey.day_total
                        ) * 100
                    )
                )
            )
            : 0

    return (
        <main className="
            min-h-screen
            bg-slate-50
            p-8
        ">
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

                <h1 className="
                    text-4xl
                    font-bold
                    text-slate-800
                ">
                    {detail.journey.title}
                </h1>

                <p className="
                    text-slate-600
                    mt-3
                ">
                    {detail.journey.description}
                </p>

                {/* Associated Pulse */}

                {detail.journey.source_memory_id !== null && (
                    <div className="
                        mt-6
                        bg-white
                        border
                        border-slate-200
                        rounded-2xl
                        p-5
                    ">
                        <div className="
                            flex
                            items-center
                            gap-2
                        ">
                            <span className="
                                text-xs
                                font-semibold
                                uppercase
                                tracking-wide
                                text-slate-500
                            ">
                                Associated Pulse
                            </span>

                            <span className="
                                px-2
                                py-1
                                rounded-full
                                bg-indigo-100
                                text-indigo-700
                                text-xs
                                font-medium
                            ">
                                Pulse #{detail.journey.source_memory_id}
                            </span>
                        </div>

                        <p className="
                            text-slate-700
                            mt-3
                            text-sm
                        ">
                            This Journey was created from Pulse #
                            {detail.journey.source_memory_id}.
                        </p>
                    </div>
                )}

                {/* Journey Progress */}

                <div className="mt-8">
                    <div className="
                        flex
                        justify-between
                    ">
                        <p className="font-semibold">
                            Day{" "}
                            {isCompleted
                                ? detail.journey.day_total
                                : detail.journey.day_current
                            }
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
                                width: `${progress}%`,
                            }}
                        />
                    </div>
                </div>

                {/* Completed state */}

                {isCompleted ? (
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

                        <p className="
                            text-slate-500
                            mt-2
                        ">
                            You've completed all{" "}
                            {detail.journey.day_total} challenges.
                        </p>
                    </div>
                ) : (
                    detail.today_challenge && (
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
                                Day{" "}
                                {detail.today_challenge.day_number}
                                {" "}
                                Challenge
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

                            {/* Difficulty feedback */}

                            <div className="mt-6">
                                <p className="
                                    text-sm
                                    font-semibold
                                    text-slate-700
                                    mb-3
                                ">
                                    How did this challenge feel?
                                </p>

                                <div className="
                                    flex
                                    flex-wrap
                                    gap-2
                                ">
                                    <button
                                        type="button"
                                        onClick={() =>
                                            setDifficultyFeedback(
                                                "too_easy"
                                            )
                                        }
                                        className={`
                                            px-4
                                            py-2
                                            rounded-lg
                                            border
                                            text-sm
                                            ${
                                                difficultyFeedback ===
                                                "too_easy"
                                                    ? "bg-indigo-600 text-white border-indigo-600"
                                                    : "bg-white text-slate-700 border-slate-300"
                                            }
                                        `}
                                    >
                                        Too easy
                                    </button>

                                    <button
                                        type="button"
                                        onClick={() =>
                                            setDifficultyFeedback(
                                                "just_right"
                                            )
                                        }
                                        className={`
                                            px-4
                                            py-2
                                            rounded-lg
                                            border
                                            text-sm
                                            ${
                                                difficultyFeedback ===
                                                "just_right"
                                                    ? "bg-indigo-600 text-white border-indigo-600"
                                                    : "bg-white text-slate-700 border-slate-300"
                                            }
                                        `}
                                    >
                                        Just right
                                    </button>

                                    <button
                                        type="button"
                                        onClick={() =>
                                            setDifficultyFeedback(
                                                "too_hard"
                                            )
                                        }
                                        className={`
                                            px-4
                                            py-2
                                            rounded-lg
                                            border
                                            text-sm
                                            ${
                                                difficultyFeedback ===
                                                "too_hard"
                                                    ? "bg-indigo-600 text-white border-indigo-600"
                                                    : "bg-white text-slate-700 border-slate-300"
                                            }
                                        `}
                                    >
                                        Too hard
                                    </button>
                                </div>
                            </div>

                            {/* Complete */}

                            <button
                                onClick={completeChallenge}
                                disabled={
                                    completing ||
                                    skipping
                                }
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
                                {completing
                                    ? "Saving..."
                                    : "Mark as Completed"
                                }
                            </button>

                            {/* Skip */}

                            <div className="
                                mt-8
                                pt-6
                                border-t
                                border-indigo-200
                            ">
                                <p className="
                                    text-sm
                                    font-semibold
                                    text-slate-700
                                    mb-3
                                ">
                                    Need to skip this challenge?
                                </p>

                                <select
                                    value={skipReason}
                                    onChange={(event) =>
                                        setSkipReason(
                                            event.target.value as
                                                SkipReason | ""
                                        )
                                    }
                                    disabled={
                                        completing ||
                                        skipping
                                    }
                                    className="
                                        w-full
                                        bg-white
                                        border
                                        border-slate-300
                                        rounded-lg
                                        px-3
                                        py-2
                                        text-sm
                                    "
                                >
                                    <option value="">
                                        Select a reason
                                    </option>

                                    <option value="busy">
                                        Too busy
                                    </option>

                                    <option value="forgot">
                                        Forgot
                                    </option>

                                    <option value="too_difficult">
                                        Too difficult
                                    </option>

                                    <option value="too_anxious">
                                        Too anxious
                                    </option>

                                    <option value="not_relevant">
                                        Not relevant
                                    </option>

                                    <option value="disliked_activity">
                                        Didn't like the activity
                                    </option>

                                    <option value="situation_unavailable">
                                        Situation unavailable
                                    </option>

                                    <option value="other">
                                        Other
                                    </option>
                                </select>

                                {!skipReason && (
                                    <p className="
                                        text-xs
                                        text-slate-500
                                        mt-2
                                    ">
                                        Choose a reason to skip this challenge.
                                    </p>
                                )}

                                <textarea
                                    value={skipReasonDetail}
                                    onChange={(event) =>
                                        setSkipReasonDetail(
                                            event.target.value
                                        )
                                    }
                                    disabled={
                                        completing ||
                                        skipping
                                    }
                                    placeholder="Optional details"
                                    rows={3}
                                    className="
                                        w-full
                                        mt-3
                                        bg-white
                                        border
                                        border-slate-300
                                        rounded-lg
                                        px-3
                                        py-2
                                        text-sm
                                        resize-none
                                    "
                                />

                                <button
                                    type="button"
                                    onClick={skipChallenge}
                                    disabled={
                                        completing ||
                                        skipping ||
                                        !skipReason
                                    }
                                    className="
                                        mt-3
                                        bg-white
                                        border
                                        border-slate-300
                                        text-slate-700
                                        px-5
                                        py-2
                                        rounded-lg
                                        disabled:opacity-50
                                    "
                                >
                                    {skipping
                                        ? "Skipping..."
                                        : "Skip Challenge"
                                    }
                                </button>
                            </div>
                        </div>
                    )
                )}

                {/* Challenge History */}

                <div className="mt-10">
                    <h3 className="
                        text-xl
                        font-bold
                        mb-5
                    ">
                        Challenge History
                    </h3>

                    <div className="
                        flex
                        flex-col
                        gap-3
                    ">
                        {detail.challenges.map((challenge) => {
                            const isCompleted =
                                challenge.status === "completed"

                            const isCurrent =
                                challenge.day_number ===
                                detail.journey.day_current

                            const isUpcoming =
                                challenge.day_number >
                                detail.journey.day_current

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
                                                ? "bg-indigo-50 border-indigo-200"
                                                : "bg-white border-slate-200"
                                        }
                                    `}
                                >
                                    <span className={`
                                        text-xl
                                        ${
                                            isCompleted
                                                ? "text-green-500"
                                                : isCurrent
                                                    ? "text-indigo-600"
                                                    : "text-slate-300"
                                        }
                                    `}>
                                        {isCompleted
                                            ? "✓"
                                            : isCurrent
                                                ? "→"
                                                : "○"
                                        }
                                    </span>

                                    <div
                                        className={
                                            isUpcoming
                                                ? "opacity-40"
                                                : ""
                                        }
                                    >
                                        <p className="font-semibold">
                                            Day{" "}
                                            {challenge.day_number}
                                            {" — "}
                                            {challenge.title}
                                        </p>

                                        {isCompleted &&
                                            challenge.completed_at && (
                                                <p className="
                                                    text-xs
                                                    text-slate-400
                                                    mt-1
                                                ">
                                                    Completed{" "}
                                                    {new Date(
                                                        challenge.completed_at
                                                    ).toLocaleDateString()}
                                                </p>
                                            )}

                                        {challenge.status === "skipped" && (
                                            <p className="
                                                text-xs
                                                text-amber-600
                                                mt-1
                                            ">
                                                Skipped
                                                {challenge.skip_reason
                                                    ? ` — ${challenge.skip_reason.replaceAll("_", " ")}`
                                                    : ""
                                                }
                                            </p>
                                        )}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>

            </div>
        </main>
    )
}

