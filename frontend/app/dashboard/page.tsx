"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"

interface User {
  id: number
  email: string
  username: string
  full_name: string | null
  is_active: boolean
}

interface JournalEntry {
  id: number
  title: string
  content: string
  mood: string | null
  analysis_status: string
  created_at: string
}

export default function DashboardPage() {

  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [entries, setEntries] = useState<JournalEntry[]>([])
  const [title, setTitle] = useState("")
  const [content, setContent] = useState("")
  const [mood, setMood] = useState("")

  useEffect(() => {


    const token = localStorage.getItem("token")


    if (!token) {
      router.push("/login")
      return
    }


    const fetchUser = async () => {

      try {


        const data = await apiFetch("/auth/me")



        setUser(data)
        const journals = await apiFetch("/journal/")

        setEntries(journals)

      } catch (error) {


        localStorage.removeItem("token")
        router.push("/login")

      }

    }


    fetchUser()


  }, [router])

async function createJournal() {

  try {

    const newEntry = await apiFetch(
      "/journal/",
      {
        method: "POST",
        body: JSON.stringify({
          title,
          content,
          mood
        })
      }
    )

    setEntries([
      ...entries,
      newEntry
    ])

    setTitle("")
    setContent("")
    setMood("")

  } catch(error) {
    alert("Failed to create journal entry")
  }

}
  return (
    <main className="max-w-2xl mx-auto p-6">

      <h1 className="text-3xl font-bold">
        Dashboard
      </h1>


      {user ? (
        <div className="mt-6">

          <p>
            Welcome back, {user.full_name || user.username} 👋
          </p>

          <h2 className="text-2xl font-bold mt-8">
  Your Journals
</h2>


<div className="flex flex-col gap-4 mt-4">

{
entries.map((entry)=>(
  <div
    key={entry.id}
    className="border rounded p-4"
  >

    <h3 className="font-bold">
      {entry.title}
    </h3>

    <p>
      Mood: {entry.mood}
    </p>

    <p>
      {entry.content}
    </p>

    <p>
      Analysis: {entry.analysis_status}
    </p>
    <p>
  Date: {new Date(entry.created_at).toLocaleDateString()}
    </p>
  </div>
))
}

</div>
          <h2 className="text-2xl font-bold mt-8">
  Create Journal
</h2>


<input
 className="border rounded p-2 w-full mt-2"
 placeholder="Title"
 value={title}
 onChange={(e)=>setTitle(e.target.value)}
/>


<textarea
 className="border rounded p-2 w-full mt-2"
 placeholder="Content"
 value={content}
 onChange={(e)=>setContent(e.target.value)}
/>


<input
 className="border rounded p-2 w-full mt-2"
 placeholder="Mood"
 value={mood}
 onChange={(e)=>setMood(e.target.value)}
/>


<button
 className="bg-blue-600 text-white px-4 py-2 rounded mt-4"
 onClick={createJournal}
>
Create
</button>

          <div className="flex gap-2 mt-6">

  <button
    className="bg-green-600 text-white px-4 py-2 rounded"
    onClick={() => router.push("/chat")}
  >
    Open Chat
  </button>

  <button
    className="bg-red-600 text-white px-4 py-2 rounded"
    onClick={() => {
      localStorage.removeItem("token")
      router.push("/login")
    }}
  >
    Logout
  </button>

</div>

        </div>

      ) : (

        <p className="mt-6">
          Loading...
        </p>

      )}


    </main>
  )
}