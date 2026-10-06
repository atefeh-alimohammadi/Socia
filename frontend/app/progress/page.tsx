"use client"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"

import { apiFetch } from "@/lib/api"

interface EmotionSeries {
  emotion: string
  color: string
  data: (number | null)[]
}

interface EmotionTimeline {
  weeks: string[]
  series: EmotionSeries[]
}

interface BehaviorPattern {
  id: number
  tag: string
  content: string
  evidence_count: number
  confidence: number | null
  created_at: string
}

interface EmotionTrend {
  user_id: number
  days: number
  average_valence: number | null
  average_arousal: number | null
}

interface RelatedEntity {
  entity_id: number
  name: string
  entity_type: string
  frequency: number
}

interface Journey {
  id: number
  source_memory_id: number | null
  status: string
}

const FALLBACK_LINE_COLORS = [
  "#4f46e5",
  "#0f766e",
  "#d97706",
  "#db2777",
]

export default function ProgressPage() {
  const router = useRouter()

  const [timeline, setTimeline] =
    useState<EmotionTimeline | null>(null)

  const [patterns, setPatterns] =
    useState<BehaviorPattern[]>([])

  const [trend, setTrend] =
    useState<EmotionTrend | null>(null)

  const [journeys, setJourneys] =
    useState<Journey[]>([])

  const [loading, setLoading] =
    useState(true)

  const [addingJourney, setAddingJourney] =
    useState<number | null>(null)

  const [reactivatingJourney, setReactivatingJourney] =
    useState<number | null>(null)

  const [expandedTag, setExpandedTag] =
    useState<string | null>(null)

  const [loadingEntities, setLoadingEntities] =
    useState<string | null>(null)

  const [relatedEntities, setRelatedEntities] =
    useState<Record<string, RelatedEntity[]>>({})

  useEffect(() => {
    const token = localStorage.getItem("token")

    if (!token) {
      router.push("/login")
      return
    }

    async function loadData() {
      try {
        const [
          timelineData,
          patternsData,
          trendData,
          journeysData,
        ] = await Promise.all([
          apiFetch("/analytics/emotion-timeline"),
          apiFetch("/analytics/behavior-patterns"),
          apiFetch("/analytics/emotion-trend"),
          apiFetch("/journeys/?status="),
        ])

        setTimeline(timelineData)
        setPatterns(patternsData)
        setTrend(trendData)
        setJourneys(journeysData)
      } catch (error) {
        console.error(error)
        router.push("/login")
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [router])

  function findJourneyForPattern(
    patternId: number
  ): Journey | undefined {
    return journeys.find(
      (journey) =>
        journey.source_memory_id === patternId
    )
  }

  async function handleAddToJourney(
    memoryId: number
  ) {
    setAddingJourney(memoryId)

    try {
      const journey = await apiFetch(
        "/journeys/from-pattern",
        {
          method: "POST",
          body: JSON.stringify({
            memory_id: memoryId,
          }),
        }
      )

      router.push(
        `/journeys/${journey.journey.id}`
      )
    } catch (error) {
      console.error(error)

      alert(
        "Failed to create journey. Please try again."
      )
    } finally {
      setAddingJourney(null)
    }
  }

  async function handleReactivateJourney(
    journeyId: number
  ) {
    setReactivatingJourney(journeyId)

    try {
      await apiFetch(
        `/journeys/${journeyId}/status?status=active`,
        {
          method: "PATCH",
        }
      )

      setJourneys((prev) =>
        prev.map((journey) =>
          journey.id === journeyId
            ? {
                ...journey,
                status: "active",
              }
            : journey
        )
      )
    } catch (error) {
      console.error(
        "Failed to reactivate journey:",
        error
      )

      alert(
        "Failed to reactivate journey. Please try again."
      )
    } finally {
      setReactivatingJourney(null)
    }
  }

  async function handleToggleRelatedEntities(
    tag: string
  ) {
    if (expandedTag === tag) {
      setExpandedTag(null)
      return
    }

    setExpandedTag(tag)

    if (relatedEntities[tag]) {
      return
    }

    setLoadingEntities(tag)

    try {
      const data: RelatedEntity[] =
        await apiFetch(
          `/analytics/related-entities?tag=${encodeURIComponent(tag)}`
        )

      setRelatedEntities((prev) => ({
        ...prev,
        [tag]: data,
      }))
    } catch (error) {
      console.error(error)

      setRelatedEntities((prev) => ({
        ...prev,
        [tag]: [],
      }))
    } finally {
      setLoadingEntities(null)
    }
  }

  /*
   * Keep the timeline readable.
   *
   * Showing every emotion can quickly make the chart noisy,
   * especially when several emotions only have sparse data.
   *
   * Prefer emotions with the most actual data points and show
   * at most four series.
   */
  const visibleTimelineSeries = useMemo(() => {
    if (!timeline?.series) {
      return []
    }

    return [...timeline.series]
      .sort((a, b) => {
        const aPoints = a.data.filter(
          (value) => value !== null
        ).length

        const bPoints = b.data.filter(
          (value) => value !== null
        ).length

        return bPoints - aPoints
      })
      .slice(0, 4)
  }, [timeline])

  const timelineHasEnoughData =
    visibleTimelineSeries.length > 0 &&
    visibleTimelineSeries.some(
      (series) =>
        series.data.filter(
          (value) => value !== null
        ).length >= 2
    )

  const chartData =
    timeline && visibleTimelineSeries.length > 0
      ? timeline.weeks.map(
          (week, weekIndex) => {
            const point: Record<
              string,
              string | number | null
            > = {
              week,
            }

            visibleTimelineSeries.forEach(
              (series) => {
                point[series.emotion] =
                  series.data[weekIndex]
              }
            )

            return point
          }
        )
      : []

  if (loading) {
    return (
      <main className="min-h-screen bg-slate-50 flex items-center justify-center">
        <p className="text-slate-500">
          Loading progress...
        </p>
      </main>
    )
  }

  /*
   * Completed journeys remain available on the Journeys page,
   * but their source patterns no longer need a Journey CTA here.
   */
  const visiblePatterns = patterns.filter(
    (pattern) => {
      const journey =
        findJourneyForPattern(pattern.id)

      return !(
        journey &&
        journey.status === "completed"
      )
    }
  )

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-8">
      <div className="max-w-5xl mx-auto">
        <button
          type="button"
          onClick={() =>
            router.push("/dashboard")
          }
          className="text-indigo-600 hover:text-indigo-700 text-sm font-medium mb-6"
        >
          ← Back to Dashboard
        </button>

        <h1 className="text-3xl font-bold text-slate-800">
          Your Progress
        </h1>

        <p className="text-slate-500 mt-2">
          See how your emotions and communication patterns
          have changed across conversations.
        </p>

        {/* =========================================================
            Emotion Timeline
        ========================================================= */}

        <section className="bg-white rounded-2xl border border-slate-200 p-6 mt-8">
          <h2 className="text-xl font-bold text-slate-800">
            Emotion Timeline
          </h2>

          <p className="text-sm text-slate-500 mt-1 mb-6">
            Your most frequently observed emotions over time.
          </p>

          {!timelineHasEnoughData ? (
            <div className="rounded-xl bg-slate-50 border border-slate-100 px-6 py-10 text-center">
              <p className="text-slate-600 font-medium">
                Not enough data to show a timeline yet.
              </p>

              <p className="text-sm text-slate-400 mt-1">
                Keep having conversations with Socia and
                your emotion trends will appear here.
              </p>
            </div>
          ) : (
            <>
              <ResponsiveContainer
                width="100%"
                height={300}
              >
                <LineChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#e2e8f0"
                  />

                  <XAxis
                    dataKey="week"
                    tick={{
                      fill: "#64748b",
                      fontSize: 12,
                    }}
                    axisLine={{
                      stroke: "#cbd5e1",
                    }}
                    tickLine={false}
                  />

                  <YAxis
                    domain={[0, 1]}
                    tickFormatter={(value) =>
                      `${Math.round(value * 100)}%`
                    }
                    tick={{
                      fill: "#64748b",
                      fontSize: 12,
                    }}
                    axisLine={false}
                    tickLine={false}
                  />

                  <Tooltip
                    formatter={(value) =>
                      value !== null
                        ? `${Math.round(
                            Number(value) * 100
                          )}%`
                        : "No data"
                    }
                  />

                  <Legend />

                  {visibleTimelineSeries.map(
                    (series, index) => (
                      <Line
                        key={series.emotion}
                        type="monotone"
                        dataKey={series.emotion}
                        stroke={
                          series.color ||
                          FALLBACK_LINE_COLORS[
                            index %
                              FALLBACK_LINE_COLORS.length
                          ]
                        }
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 5 }}
                        connectNulls={false}
                      />
                    )
                  )}
                </LineChart>
              </ResponsiveContainer>

              {timeline &&
                timeline.series.length > 4 && (
                  <p className="text-xs text-slate-400 mt-3">
                    Showing the four emotions with the most
                    available data.
                  </p>
                )}
            </>
          )}
        </section>

        {/* =========================================================
            Recent Mood
        ========================================================= */}

        <section className="bg-white rounded-2xl border border-slate-200 p-6 mt-8">
          <h2 className="text-xl font-bold text-slate-800">
            Recent Mood
          </h2>

          <p className="text-sm text-slate-500 mt-1 mb-6">
            A simple summary of the emotional tone and intensity
            detected in your recent conversations.
          </p>

          {trend &&
          (
            trend.average_valence !== null ||
            trend.average_arousal !== null
          ) ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="rounded-xl bg-slate-50 border border-slate-100 p-5">
                <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
                  Mood
                </p>

                <p className="text-2xl font-bold text-slate-800 mt-2">
                  {trend.average_valence !== null
                    ? trend.average_valence.toFixed(2)
                    : "—"}
                </p>

                <p className="text-xs text-slate-400 mt-1">
                  Ranges from -1 (more negative) to +1
                  (more positive).
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 border border-slate-100 p-5">
                <p className="text-xs text-slate-500 uppercase tracking-wide font-semibold">
                  Intensity
                </p>

                <p className="text-2xl font-bold text-slate-800 mt-2">
                  {trend.average_arousal !== null
                    ? trend.average_arousal.toFixed(2)
                    : "—"}
                </p>

                <p className="text-xs text-slate-400 mt-1">
                  Ranges from 0 (calm) to 1 (more intense).
                </p>
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-slate-50 border border-slate-100 px-6 py-8 text-center">
              <p className="text-slate-600 font-medium">
                Not enough recent emotional data yet.
              </p>

              <p className="text-sm text-slate-400 mt-1">
                More conversations will give Socia enough
                information to show this summary.
              </p>
            </div>
          )}
        </section>

        {/* =========================================================
            Behavior Patterns
        ========================================================= */}

        <section className="mt-10">
          <h2 className="text-2xl font-bold text-slate-800">
            What Socia Has Noticed
          </h2>

          <p className="text-sm text-slate-500 mt-1 mb-5">
            Recurring communication patterns based on your
            conversations.
          </p>

          {visiblePatterns.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center">
              <p className="text-slate-600 font-medium">
                Socia hasn't detected any recurring patterns yet.
              </p>

              <p className="text-sm text-slate-400 mt-1">
                Keep chatting to give Socia more context to work with.
              </p>
            </div>
          ) : (
            visiblePatterns.map((pattern) => {
              const existingJourney =
                findJourneyForPattern(pattern.id)

              return (
                <div
                  key={pattern.id}
                  className="bg-white border border-slate-200 rounded-2xl p-6 mb-4"
                >
                  <p className="text-xs text-indigo-600 font-semibold uppercase tracking-wide mb-2">
                    {pattern.tag.replace(/_/g, " ")}
                  </p>

                  <p className="text-slate-700 leading-relaxed">
                    {pattern.content}
                  </p>

                  <p className="text-xs text-slate-400 mt-3">
                    Based on {pattern.evidence_count}{" "}
                    observation
                    {pattern.evidence_count !== 1
                      ? "s"
                      : ""}
                    {pattern.confidence !== null &&
                      ` · ${Math.round(
                        pattern.confidence * 100
                      )}% model confidence`}
                  </p>

                  {/* =================================================
                      Journey Actions
                  ================================================= */}

                  <div className="flex flex-wrap gap-3 mt-4 items-center">
                    {existingJourney ? (
                      <>
                        {/* ================= ACTIVE ================= */}

                        {existingJourney.status ===
                          "active" && (
                          <button
                            type="button"
                            onClick={() =>
                              router.push(
                                `/journeys/${existingJourney.id}`
                              )
                            }
                            className="
                              border border-green-200
                              bg-green-50
                              text-green-700
                              px-3 py-1.5
                              rounded-full
                              text-sm
                              font-medium
                              hover:bg-green-100
                            "
                          >
                            Active
                          </button>
                        )}

                        {/* ================= PAUSED ================= */}

                        {existingJourney.status ===
                          "paused" && (
                          <>
                            <button
                              type="button"
                              onClick={() =>
                                router.push(
                                  `/journeys/${existingJourney.id}`
                                )
                              }
                              className="
                                border border-amber-200
                                bg-amber-50
                                text-amber-700
                                px-3 py-1.5
                                rounded-full
                                text-sm
                                font-medium
                                hover:bg-amber-100
                              "
                            >
                              Paused
                            </button>

                            <button
                              type="button"
                              onClick={() =>
                                handleReactivateJourney(
                                  existingJourney.id
                                )
                              }
                              disabled={
                                reactivatingJourney ===
                                existingJourney.id
                              }
                              className="
                                bg-indigo-600
                                text-white
                                px-4 py-2
                                rounded-lg
                                text-sm
                                font-medium
                                hover:bg-indigo-700
                                disabled:opacity-50
                              "
                            >
                              {reactivatingJourney ===
                              existingJourney.id
                                ? "Reactivating..."
                                : "Resume Journey"}
                            </button>
                          </>
                        )}

                        {/* ================= UNKNOWN ================= */}

                        {existingJourney.status !==
                          "active" &&
                          existingJourney.status !==
                            "paused" && (
                            <button
                              type="button"
                              onClick={() =>
                                router.push(
                                  `/journeys/${existingJourney.id}`
                                )
                              }
                              className="
                                bg-slate-100
                                text-slate-600
                                px-3 py-1.5
                                rounded-full
                                text-sm
                                font-medium
                              "
                            >
                              {existingJourney.status}
                            </button>
                          )}
                      </>
                    ) : (
                      /* ================= NO JOURNEY ================= */

                      <button
                        type="button"
                        disabled={
                          addingJourney === pattern.id
                        }
                        onClick={() =>
                          handleAddToJourney(pattern.id)
                        }
                        className="
                          bg-indigo-600
                          text-white
                          px-4 py-2
                          rounded-lg
                          text-sm
                          font-medium
                          hover:bg-indigo-700
                          disabled:opacity-50
                        "
                      >
                        {addingJourney === pattern.id
                          ? "Starting..."
                          : "Start a Journey"}
                      </button>
                    )}

                    {/* ================= VIEW EXAMPLES ================= */}

                    <button
                      type="button"
                      onClick={() =>
                        handleToggleRelatedEntities(
                          pattern.tag
                        )
                      }
                      className="
                        bg-white
                        border border-slate-300
                        text-slate-700
                        px-4 py-2
                        rounded-lg
                        text-sm
                        font-medium
                        hover:bg-slate-50
                      "
                    >
                      {expandedTag === pattern.tag
                        ? "Hide examples"
                        : "View examples"}
                    </button>
                  </div>

                  {/* =================================================
                      Related Entities
                  ================================================= */}

                  {expandedTag === pattern.tag && (
                    <div className="mt-4 border-t border-slate-100 pt-4">
                      {loadingEntities ===
                      pattern.tag ? (
                        <p className="text-sm text-slate-400">
                          Loading examples...
                        </p>
                      ) : relatedEntities[
                          pattern.tag
                        ] &&
                        relatedEntities[
                          pattern.tag
                        ].length > 0 ? (
                        <ul className="text-sm text-slate-600 space-y-2">
                          {relatedEntities[
                            pattern.tag
                          ].map((entity) => (
                            <li
                              key={entity.entity_id}
                            >
                              <span className="font-medium">
                                {entity.name}
                              </span>

                              <span className="text-slate-400 ml-1">
                                (
                                {entity.entity_type}
                                {" · "}
                                seen{" "}
                                {entity.frequency}{" "}
                                time
                                {entity.frequency !==
                                1
                                  ? "s"
                                  : ""}
                                )
                              </span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-sm text-slate-400">
                          No related examples found yet.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )
            })
          )}
        </section>
      </div>
    </main>
  )
}

