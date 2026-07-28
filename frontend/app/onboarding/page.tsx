"use client"

import { useEffect, useRef, useState } from "react"
import {useRouter } from "next/navigation"
import { apiFetch } from "@/lib/api"

interface OnboardingStep{
    step: number
    question: string
    memory_type: string
    tag: string
}

interface ChatMessage {
    role: "socia" | "user"
    content: string
}

export default function OnboardingPage() {

    const router = useRouter()

    const [steps, setSteps] = useState<OnboardingStep[]>([])
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [currentStep, setCurrentStep] = useState(0)
    const [input, setInput] = useState("")
    const [loading, setLoading] = useState(false)
    const [completed, setCompleted] = useState(false)

    const bottomRef = useRef<HTMLDivElement>(null)

    useEffect(() => {

        const token = localStorage.getItem("token")

        if (!token){
            router.push("/login")
            return
        }

        const loadSteps = async () => {

            try{

                const data: OnboardingStep[] =
                    await apiFetch("/onboarding/steps")

                setSteps(data)

                setMessages(
                    [
                        {
                            role: "socia",
                            content: data[0].question
                        }
                    ]
                )
            } catch{
                router.push("/login")
            }
        }

        loadSteps()
        }, [router])
    useEffect(() => {
      bottomRef.current?.scrollIntoView({
      behavior: "smooth"
       })
    }, [messages])

    async function submitAnswer() {

    if (input.trim() === "" || loading) return

    const answer = input

    setInput("")

    setMessages((prev) => [
        ...prev,
        {
            role: "user",
            content: answer
        }
    ])

    setLoading(true)


    try {

        await apiFetch(
            "/onboarding/answer",
            {
                method: "POST",
                body: JSON.stringify({
                    step: currentStep + 1,
                    answer: answer
                })
            }
        )


        const nextStep = currentStep + 1


        if (nextStep < steps.length) {

            setCurrentStep(nextStep)

            setMessages((prev) => [
                ...prev,
                {
                    role: "socia",
                    content: steps[nextStep].question
                }
            ])

        } else {

            const result = await apiFetch(
                "/onboarding/complete",
                {
                    method: "POST"
                }
            )


            setMessages((prev) => [
                ...prev,
                {
                    role: "socia",
                    content: result.summary
                }
            ])


            setCompleted(true)

        }


    } catch{
        alert("Something went wrong")

    } finally {

        setLoading(false)

    }

}
return (
    <main className="bg-slate-50 min-h-screen p-6">

        <div className="max-w-lg mx-auto">

            <h1 className="text-2xl font-bold mb-6">
                Socia
            </h1>


            <div className="flex flex-col gap-4 h-96 overflow-y-auto">


                {messages.map((message, index) => (

                    <div
                        key={index}
                        className={
                            message.role === "socia"
                            ?
                            "self-start bg-white border border-slate-200 text-slate-700 px-4 py-3 rounded-lg max-w-sm"
                            :
                            "self-end bg-blue-600 text-white px-4 py-3 rounded-lg max-w-sm"
                        }
                    >

                        {
                            message.role === "socia" &&
                            <p className="text-xs font-bold mb-1">
                                Socia
                            </p>
                        }

                        {message.content}

                    </div>

                ))}

                  <div ref={bottomRef} />


            </div>


            {
                !completed && (

                    <div className="flex gap-2 mt-4">

                        <input
                            className="border rounded p-2 flex-1"
                            placeholder="Write your answer..."
                            value={input}
                            onChange={(e)=>setInput(e.target.value)}
                            onKeyDown={(e)=>{
                                if(e.key==="Enter"){
                                    submitAnswer()
                                }
                            }}
                        />


                        <button
                            className="bg-indigo-600 text-white px-4 rounded"
                            onClick={submitAnswer}
                            disabled={loading}
                        >
                            Send
                        </button>


                    </div>

                )
            }


            {
                completed && (

                    <button
                        className="bg-indigo-600 text-white px-4 py-2 rounded mt-6"
                        onClick={()=>router.push("/dashboard")}
                    >
                        Begin your journey →
                    </button>

                )
            }


        </div>

    </main>
)
}