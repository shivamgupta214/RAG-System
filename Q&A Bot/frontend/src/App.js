// import React, { useState } from "react";

// function ChatBox() {
//   const [query, setQuery] = useState("");
//   const [response, setResponse] = useState("");
//   const [loading, setLoading] = useState(false);
//   const socketRef = React.useRef<WebSocket | null>(null);

//   const handleAsk = () => {
//     setLoading(true);
//     setResponse("");

//     socketRef.current = new WebSocket("ws://localhost:8000/ws/ask");

//     socketRef.current.onopen = () => {
//       socketRef.current?.send(query);
//     };

//     socketRef.current.onmessage = (event) => {
//       if (event.data === "loading...") {
//         setResponse("Thinking...");
//       } else {
//         setLoading(false);
//         setResponse(event.data);
//         socketRef.current?.close();
//       }
//     };

//     socketRef.current.onerror = (err) => {
//       console.error("WebSocket error:", err);
//       setLoading(false);
//       setResponse("Something went wrong.");
//     };
//   };

//   return (
//     <div>
//       <textarea value={query} onChange={(e) => setQuery(e.target.value)} />
//       <button onClick={handleAsk} disabled={loading}>
//         Ask
//       </button>
//       {loading && <p>Loading...</p>}
//       <p><strong>Response:</strong> {response}</p>
//     </div>
//   );
// }

// export default ChatBox;

import React, { useState, useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";
import "./App.css";

const WEBSOCKET_URL = "ws://localhost:8000/ws/ask";

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const socketRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendQuery = () => {
    if (!query.trim()) return;
    const newMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, newMsg]);
    setLoading(true);

    socketRef.current = new WebSocket(WEBSOCKET_URL);

    socketRef.current.onopen = () => {
      socketRef.current.send(query);
    };

    socketRef.current.onmessage = (event) => {
      if (event.data === "loading...") return;

      const botReply = { role: "bot", content: event.data };
      setMessages((prev) => [...prev, botReply]);
      setLoading(false);
      socketRef.current.close();
    };

    socketRef.current.onerror = () => {
      setLoading(false);
      setMessages((prev) => [...prev, { role: "bot", content: "⚠️ Error occurred." }]);
    };

    setQuery("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  };

  return (
    <div className="app">
      <h2>Salesforce Q&A Assistant 🤖</h2>

      <div className="chat-box">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} />
        ))}
        {loading && <ChatMessage message={{ role: "bot", content: "⏳ Thinking..." }} />}
        <div ref={bottomRef} />
      </div>

      <div className="input-container">
        <textarea
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button onClick={sendQuery} disabled={loading || !query.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
