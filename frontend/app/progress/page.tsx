"use client"

import { useEffect, useState } from "react"
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

          // Empty status returns all journey statuses
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

      /*
       * IMPORTANT:
       * The backend expects `status` as a query parameter,
       * NOT inside the JSON request body.
       *
       * Backend:
       * PATCH /journeys/{journey_id}/status?status=active
       */

      await apiFetch(
        `/journeys/${journeyId}/status?status=active`,
        {
          method: "PATCH",
        }
      )


      /*
       * Update local state immediately.
       * This makes the UI change from:
       *
       * Paused + Reactivate
       *
       * to:
       *
       * Active
       */

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



  const chartData =
    timeline
      ? timeline.weeks.map(
          (week, weekIndex) => {

            const point:
              Record<string, string | number | null>
              = {
                week,
              }


            timeline.series.forEach(
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
      <div className="min-h-screen flex items-center justify-center">
        Loading progress...
      </div>
    )

  }



  /*
   * Completed journeys should no longer appear on the Progress page.
   *
   * Their journeys remain available on the Journeys page with
   * the "Completed" status.
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

    <main className="min-h-screen bg-slate-50 p-8">

      <button
        onClick={() =>
          router.push("/dashboard")
        }
        className="text-indigo-600 mb-6"
      >
        ← Back to Dashboard
      </button>


      <h1 className="text-3xl font-bold text-slate-800">
        Your Progress
      </h1>


      <p className="text-slate-500 mt-2">
        Socia tracks your emotional patterns across conversations over time.
      </p>



      {/* =========================================================
          Emotion Timeline
      ========================================================= */}

      <section className="bg-white rounded-2xl border border-slate-200 p-6 mt-8">

        <h2 className="text-xl font-bold text-slate-800 mb-6">
          Emotion Timeline
        </h2>


        {
          timeline ? (

            <ResponsiveContainer
              width="100%"
              height={300}
            >

              <LineChart data={chartData}>

                <CartesianGrid
                  strokeDasharray="3 3"
                />


                <XAxis
                  dataKey="week"
                />


                <YAxis
                  domain={[0, 1]}
                  tickFormatter={
                    (v) =>
                      `${Math.round(v * 100)}%`
                  }
                />


                <Tooltip
                  formatter={
                    (value) =>
                      value !== null
                        ? `${Math.round(Number(value) * 100)}%`
                        : "No data"
                  }
                />


                <Legend />


                {
                  timeline.series.map(
                    (series) => (

                      <Line
                        key={series.emotion}
                        type="monotone"
                        dataKey={series.emotion}
                        stroke={series.color}
                        strokeWidth={2}
                        connectNulls={false}
                      />

                    )
                  )
                }


              </LineChart>

            </ResponsiveContainer>

          ) : (

            <p className="text-slate-500">
              Keep having conversations with Socia to see your emotion trends here.
            </p>

          )
        }

      </section>



      {/* =========================================================
          Emotional Trend
      ========================================================= */}

      <section className="bg-white rounded-2xl border border-slate-200 p-6 mt-8">

        <h2 className="text-xl font-bold text-slate-800 mb-4">
          Emotional Trend (last {trend?.days ?? 14} days)
        </h2>


        {
          trend &&
          (
            trend.average_valence !== null ||
            trend.average_arousal !== null
          ) ? (

            <div className="flex gap-8">

              <div>

                <p className="text-xs text-slate-500 uppercase font-semibold">
                  Average Valence
                </p>


                <p className="text-2xl font-bold text-slate-800 mt-1">
                  {
                    trend.average_valence !== null
                      ? trend.average_valence.toFixed(2)
                      : "—"
                  }
                </p>


                <p className="text-xs text-slate-400 mt-1">
                  -1 (negative) to +1 (positive)
                </p>

              </div>



              <div>

                <p className="text-xs text-slate-500 uppercase font-semibold">
                  Average Arousal
                </p>


                <p className="text-2xl font-bold text-slate-800 mt-1">
                  {
                    trend.average_arousal !== null
                      ? trend.average_arousal.toFixed(2)
                      : "—"
                  }
                </p>


                <p className="text-xs text-slate-400 mt-1">
                  0 (calm) to 1 (intense)
                </p>

              </div>

            </div>

          ) : (

            <p className="text-slate-500">
              Not enough recent emotional data yet.
            </p>

          )
        }

      </section>



      {/* =========================================================
          Behavior Patterns
      ========================================================= */}

      <section className="mt-10">

        <h2 className="text-2xl font-bold text-slate-800 mb-4">
          What Socia Has Noticed
        </h2>


        {
          visiblePatterns.length === 0 ? (

            <div className="bg-slate-50 border rounded-xl p-6 text-center">

              <p className="text-slate-500">
                Socia hasn't detected any patterns yet.
                Keep chatting to discover recurring themes.
              </p>

            </div>

          ) : (

            visiblePatterns.map(
              (pattern) => {

                const existingJourney =
                  findJourneyForPattern(pattern.id)


                return (

                  <div
                    key={pattern.id}
                    className="bg-white border border-slate-200 rounded-2xl p-6 mb-4"
                  >

                    <p className="text-xs text-indigo-600 font-semibold uppercase mb-2">
                      {pattern.tag.replace(/_/g, " ")}
                    </p>


                    <p className="text-slate-700">
                      {pattern.content}
                    </p>


                    <p className="text-xs text-slate-400 mt-3">

                      Detected {pattern.evidence_count} time
                      {pattern.evidence_count !== 1 ? "s" : ""}


                      {
                        pattern.confidence !== null &&
                        ` · ${Math.round(pattern.confidence * 100)}% confidence`
                      }

                    </p>



                    {/* =================================================
                        Journey Actions
                    ================================================= */}

                    <div className="flex gap-3 mt-4 items-center">

                      {
                        existingJourney ? (

                          <>

                            {/* ================= ACTIVE ================= */}

                            {
                              existingJourney.status === "active" && (

                                <button
                                  onClick={() =>
                                    router.push(
                                      `/journeys/${existingJourney.id}`
                                    )
                                  }
                                  className="
                                    bg-green-100
                                    text-green-700
                                    px-4
                                    py-2
                                    rounded-lg
                                    text-sm
                                    font-medium
                                    hover:bg-green-200
                                  "
                                >
                                  Active
                                </button>

                              )
                            }



                            {/* ================= PAUSED ================= */}

                            {
                              existingJourney.status === "paused" && (

                                <>

                                  <button
                                    onClick={() =>
                                      router.push(
                                        `/journeys/${existingJourney.id}`
                                      )
                                    }
                                    className="
                                      bg-amber-100
                                      text-amber-700
                                      px-4
                                      py-2
                                      rounded-lg
                                      text-sm
                                      font-medium
                                      hover:bg-amber-200
                                    "
                                  >
                                    Paused
                                  </button>


                                  <button
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
                                      px-4
                                      py-2
                                      rounded-lg
                                      text-sm
                                      font-medium
                                      hover:bg-indigo-700
                                      disabled:opacity-50
                                    "
                                  >

                                    {
                                      reactivatingJourney ===
                                      existingJourney.id
                                        ? "Reactivating..."
                                        : "Reactivate"
                                    }

                                  </button>

                                </>

                              )
                            }



                            {/* ================= UNKNOWN ================= */}

                            {
                              existingJourney.status !== "active" &&
                              existingJourney.status !== "paused" && (

                                <button
                                  onClick={() =>
                                    router.push(
                                      `/journeys/${existingJourney.id}`
                                    )
                                  }
                                  className="
                                    bg-slate-100
                                    text-slate-600
                                    px-4
                                    py-2
                                    rounded-lg
                                    text-sm
                                    font-medium
                                  "
                                >
                                  {existingJourney.status}
                                </button>

                              )
                            }

                          </>

                        ) : (

                          /* ================= NO JOURNEY ================= */

                          <button
                            disabled={
                              addingJourney === pattern.id
                            }
                            onClick={() =>
                              handleAddToJourney(
                                pattern.id
                              )
                            }
                            className="
                              bg-indigo-600
                              text-white
                              px-4
                              py-2
                              rounded-lg
                              text-sm
                              disabled:opacity-50
                            "
                          >

                            {
                              addingJourney === pattern.id
                                ? "Creating journey..."
                                : "+ Add to Journey"
                            }

                          </button>

                        )
                      }



                      {/* ================= SHOW RELATED ================= */}

                      <button
                        onClick={() =>
                          handleToggleRelatedEntities(
                            pattern.tag
                          )
                        }
                        className="
                          bg-white
                          border
                          border-slate-300
                          text-slate-700
                          px-4
                          py-2
                          rounded-lg
                          text-sm
                        "
                      >

                        {
                          expandedTag === pattern.tag
                            ? "Hide related"
                            : "Show related"
                        }

                      </button>

                    </div>



                    {/* =================================================
                        Related Entities
                    ================================================= */}

                    {
                      expandedTag === pattern.tag && (

                        <div className="mt-4 border-t border-slate-100 pt-4">

                          {
                            loadingEntities === pattern.tag ? (

                              <p className="text-sm text-slate-400">
                                Loading related entities...
                              </p>

                            ) : (

                              relatedEntities[pattern.tag] &&
                              relatedEntities[pattern.tag].length > 0 ? (

                                <ul className="text-sm text-slate-600 space-y-1">

                                  {
                                    relatedEntities[
                                      pattern.tag
                                    ].map(
                                      (entity) => (

                                        <li
                                          key={entity.entity_id}
                                        >

                                          <span className="font-medium">
                                            {entity.name}
                                          </span>


                                          {" "}


                                          <span className="text-slate-400">

                                            (
                                            {entity.entity_type},
                                            {" "}
                                            seen{" "}
                                            {entity.frequency}
                                            {" "}
                                            time
                                            {
                                              entity.frequency !== 1
                                                ? "s"
                                                : ""
                                            }
                                            )

                                          </span>

                                        </li>

                                      )
                                    )
                                  }

                                </ul>

                              ) : (

                                <p className="text-sm text-slate-400">
                                  No related entities found yet.
                                </p>

                              )

                            )
                          }

                        </div>

                      )
                    }

                  </div>

                )

              }
            )

          )
        }

      </section>

    </main>

  )

}