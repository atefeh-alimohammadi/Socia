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


export default function ProgressPage() {

  const router = useRouter()


  const [timeline, setTimeline] =
    useState<EmotionTimeline | null>(null)

  const [patterns, setPatterns] =
    useState<BehaviorPattern[]>([])


  const [loading, setLoading] =
    useState(true)


  const [addingJourney, setAddingJourney] =
    useState<number | null>(null)



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
        ] = await Promise.all([
          apiFetch("/analytics/emotion-timeline"),
          apiFetch("/analytics/behavior-patterns"),
        ])


        setTimeline(timelineData)
        setPatterns(patternsData)


      } catch{

        console.error(
          router.push("/login")
        )

      } finally {

        setLoading(false)

      }

    }


    loadData()

  }, [router])



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


    } catch {

      alert(
        "Failed to create journey. Please try again."
      )

    } finally {

      setAddingJourney(null)

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



  return (

    <main className="min-h-screen bg-slate-50 p-8">

      <button
        onClick={() => router.push("/dashboard")}
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
                  domain={[0,1]}
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



      <section className="mt-10">

        <h2 className="text-2xl font-bold text-slate-800 mb-4">
          What Socia Has Noticed
        </h2>



        {
          patterns.length === 0 ? (

            <div className="bg-slate-50 border rounded-xl p-6 text-center">

              <p className="text-slate-500">
                Socia hasn't detected any patterns yet.
                Keep chatting to discover recurring themes.
              </p>

            </div>

          ) : (

            patterns.map(
              (pattern) => (

                <div
                  key={pattern.id}
                  className="bg-white border border-slate-200 rounded-2xl p-6 mb-4"
                >

                  <p className="text-xs text-indigo-600 font-semibold uppercase mb-2">
                    {pattern.tag.replace(/_/g," ")}
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



                  <button
                    disabled={
                      addingJourney === pattern.id
                    }
                    onClick={() =>
                      handleAddToJourney(pattern.id)
                    }
                    className="mt-4 bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                  >

                    {
                      addingJourney === pattern.id
                        ? "Creating journey..."
                        : "+ Add to Journey"
                    }

                  </button>


                </div>

              )
            )

          )
        }


      </section>


    </main>

  )

}