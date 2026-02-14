"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface Block {
  index: number;
  timestamp: number;
  request_id: string;
  plastic_type: string;
  confidence: number;
  image_hash: string;
  verification_status: string;
  previous_hash: string;
  hash: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<Block[]>([]);

  const fetchHistory = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/chain");
      const data: Block[] = await res.json();

      // Remove genesis block
      const filtered = data.filter((block) => block.index !== 0);

      // Take last 10 and reverse
      const reversed = filtered.reverse();

      setHistory(reversed);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  useEffect(() => {
    fetchHistory();
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-2">

      <Link href="/">
      <button className="mt-2 ml-2  bg-emerald-600 hover:bg-emerald-700 text-xs px-3 py-2 mb-2 rounded-md transition">
          BACK
      </button>
      </Link>

      <div className="max-w-6xl mx-auto">

        <h1 className="text-3xl font-bold mb-2">
          Blockchain Scan History
        </h1>
        <p className="text-slate-400 mb-8">
          Past Verified Plastic Classifications
        </p>

        <div className="grid gap-6 md:grid-cols-2">
          {history.map((block) => {
            const confidencePercent = (block.confidence * 100).toFixed(2);
            const date = new Date(block.timestamp * 1000);

            return (
              <div
                key={block.hash}
                className="bg-slate-800 p-6 rounded-xl shadow-lg hover:scale-[1.02] transition"
              >
                <div className="flex justify-between items-center">
                  <h2 className="text-xl font-semibold">
                    {block.plastic_type}
                  </h2>

                  <span
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      block.verification_status === "verified"
                        ? "bg-green-600"
                        : "bg-red-600"
                    }`}
                  >
                    {block.verification_status.toUpperCase()}
                  </span>
                </div>

                <div className="mt-4">
                  <p className="text-sm text-slate-400">
                    Confidence: {confidencePercent}%
                  </p>

                  <div className="w-full bg-slate-700 h-2 rounded mt-2">
                    <div
                      className="bg-green-500 h-2 rounded"
                      style={{ width: `${confidencePercent}%` }}
                    ></div>
                  </div>
                </div>

                <p className="text-sm text-slate-400 mt-4">
                  {date.toLocaleString()}
                </p>

                <p className="text-xs text-slate-500 mt-2 break-all">
                  Hash: {block.hash.substring(0, 25)}...
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}