// import React from "react";
// import "./App.css";

// function ChatMessage({ message }) {
//   const isUser = message.role === "user";

//   const renderFormattedText = (text) => {
//     const lines = text.split("\n").filter(Boolean);

//     return (
//       <div>
//         {lines.map((line, idx) => {
//           const trimmed = line.trim();

//           if (/^\d+\.\s/.test(trimmed)) {
//             return <li key={idx}>{trimmed}</li>;
//           } else if (trimmed.startsWith("- ") || trimmed.startsWith("• ")) {
//             return <li key={idx}>{trimmed.slice(2)}</li>;
//           } else {
//             return <p key={idx}>{trimmed}</p>;
//           }
//         })}
//       </div>
//     );
//   };

//   return (
//     <div className={`chat-message ${isUser ? "user" : "bot"}`}>
//       <div className="bubble">
//         {renderFormattedText(message.content)}
//       </div>
//     </div>
//   );
// }

// export default ChatMessage;

import ReactMarkdown from "react-markdown";

function ChatMessage({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`chat-message ${isUser ? "user" : "bot"}`}>
      <div className="bubble">
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
    </div>
  );
}

export default ChatMessage;