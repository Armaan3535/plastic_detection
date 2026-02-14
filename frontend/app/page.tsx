"use client"

import { useRef, useState, useEffect } from "react"
import Link from "next/link";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 }
        })

        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (err) {
        console.error("Camera access denied:", err)
      }
    }

    startCamera()
  }, [])

  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current) return

    setLoading(true)

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext("2d")

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    context?.drawImage(video, 0, 0)

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg")
    )

    if (!blob) return

    const formData = new FormData()
    formData.append("file", blob, "capture.jpg")

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error("Prediction error:", error)
    }

    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-black text-white p-10">
      
      {/* Header */}
      <div className="flex justify-between items-center mb-10">
        <div>
          <h1 className="text-3xl font-bold">
            Live Plastic Verification
          </h1>
          <p className="text-slate-400 text-sm">
            Camera • AI classification • Blockchain logging
          </p>
        </div>

        <div className="bg-green-600/20 text-green-400 px-4 py-2 rounded-full text-sm">
          ● System online
        </div>
      </div>

      <div className="grid grid-cols-2 gap-8">

        {/* LEFT PANEL */}
        <div className="bg-slate-800/60 backdrop-blur-lg rounded-2xl p-6 border border-slate-700 shadow-xl">
          <h2 className="text-sm uppercase tracking-wide text-slate-400 mb-4">
            Current Frame
          </h2>

          <video
            ref={videoRef}
            autoPlay
            className="rounded-xl w-full"
          />

          <button
            onClick={captureAndPredict}
            className="mt-6 w-full bg-emerald-600 hover:bg-emerald-700 transition-all duration-300 p-3 rounded-xl font-semibold"
          >
            {loading ? "Processing..." : "Scan Material"}
          </button>

          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* RIGHT PANEL */}
        <div className="bg-slate-800/60 backdrop-blur-lg rounded-2xl p-6 border border-slate-700 shadow-xl">
          <h2 className="text-xs text-slate-400 uppercase tracking-wide">
            Identified Material
          </h2>

          {result ? (
            <>
              <h3 className="text-4xl font-bold mt-2">
                {result.plastic_type} Plastic
              </h3>

              {/* Confidence */}
              <div className="mt-8">
                <p className="text-sm text-slate-400">
                  AI confidence score
                </p>

                <div className="w-full bg-slate-700 rounded-full h-3 mt-2 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-3 rounded-full transition-all duration-700"
                    style={{
                      width: `${result.confidence * 100}%`
                    }}
                  />
                </div>

                <p className="mt-2 text-sm font-medium">
                  {(result.confidence * 100).toFixed(2)}%
                </p>
              </div>

              {/* Verification */}
              <div className="mt-8 bg-emerald-900/20 border border-emerald-600 p-5 rounded-xl">
                <p className="text-emerald-400 font-semibold">
                  ✔ Blockchain verification complete
                </p>

                <p className="text-sm text-slate-400 mt-2">
                  Hash: {result.image_hash.slice(0, 20)}...
                </p>

                <p className="text-sm text-slate-400">
                  Status: {result.verification_status}
                </p>

                <p className="text-xs text-slate-500 mt-1">
                  Request ID: {result.request_id}
                </p>
              </div>
            </>
          ) : (
            <div className="mt-10 text-slate-500">
              No scan yet. Click “Scan Material” to begin.
            </div>
          )}
        </div>

      </div>

      <div className="flex justify-end">
      <Link href="/history">
      <button
      className="mt-10 w-l bg-emerald-600 hover:bg-emerald-700 transition-all duration-300 p-3 rounded-xl font-semibold"
      >
          History
      </button>
      </Link>
      </div>

    </div>
  )
}
